#include <cstddef>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <vector>

#include "kmeans.hpp"

using namespace sycl;

inline size_t assign_point_to_centroid(const Point &p, const accessor<Point, 1, access::mode::read> &centroids) {
  double min_val = std::numeric_limits<double>::max();
  size_t min_idx = 0;

  for (size_t i = 0; i < centroids.size(); i++) {
    const auto dist = squared_distance(p, centroids[i]);
    if (dist < min_val) {
      min_val = dist;
      min_idx = i;
    }
  }

  return min_idx;
}

kmeans_cluster_t kmeans(size_t k, const vector<Point> &points) {
  if (k > points.size()) {
    throw std::invalid_argument("Number of clusters must be less than or equal "
                                "to the number of points");
  }

  auto q = queue{};

  // Step 0: Initialize centroids

  // For simplicity, let's assume the first k points are the initial centroids.
  auto host_centroids = vector<Point>{k};
  std::copy_n(points.begin(), k, host_centroids.begin());

  // Create the relevant buffers to be used in the computation
  auto points_b        = buffer{points};
  auto centroids_b     = buffer{host_centroids};
  auto assoc_b         = buffer<size_t>{points.size()};
  auto new_centroids_b = buffer<Point>{k};
  auto converged       = bool{false};

  // main cycle: assign points to clusters, calculate new centroids, check for
  // convergence
  while (!converged) {

    // Step 1: Assign points to clusters
    q.submit([&](handler &h) {
      const auto points_acc    = points_b.get_access<access::mode::read>(h);
      const auto centroids_acc = centroids_b.get_access<access::mode::read>(h);
      const auto assoc_acc     = assoc_b.get_access<access::mode::write>(h);

      // for each point, calculate the distance to each centroid and assign the
      // point to the closest one

      h.parallel_for<class assign_kernel>(points_acc.size(), [=](const size_t item) {
        assoc_acc[item] = assign_point_to_centroid(points_acc[item], centroids_acc);
      });
    });

    // calculate new centroids by averaging the points in each cluster.

    q.submit([&](handler &h) {
      const auto points_acc = points_b.get_access<access::mode::read>(h);
      const auto assoc_acc  = assoc_b.get_access<access::mode::read>(h);
      const auto centr_acc  = new_centroids_b.get_access<access::mode::write>(h);

      // for each cluster, calculate the new centroid by averaging the points
      // associated with it
      h.parallel_for<class new_means_kernel>(k, [=](const size_t item) {
        Point new_centroid{0, 0};

        auto count = size_t{0};
        for (size_t i = 0; i < points_acc.size(); i++) {
          if (assoc_acc[i] == item) {
            new_centroid.x += points_acc[i].x;
            new_centroid.y += points_acc[i].y;
            count++;
          }
        }

        if (count > 0) {
          new_centroid.x /= static_cast<float>(count);
          new_centroid.y /= static_cast<float>(count);
        }

        centr_acc[item] = new_centroid;
      });
    });

    // check for convergence
    {
      // we need to use an int as atomic operations do not support a bool
      auto converged_i = int{true};

      auto event = q.submit([&](handler &h) {
        const auto new_centroids_a = new_centroids_b.get_access<access::mode::read>(h);
        const auto centroids_a     = centroids_b.get_access<access::mode::read>(h);
        const auto converged_ref   = atomic_ref<int, memory_order::relaxed, memory_scope::device>{converged_i};

        h.parallel_for<class is_converged_kernel>(new_centroids_a.size(), [=](const auto item) {
          if (squared_distance(new_centroids_a[item], centroids_a[item]) >= 0.0001) {
            converged_ref.store(false);
          }
        });
      });

      event.wait(); // is this wait necessary?

      // update the global state
      converged = converged_i;
    }

    // if not converged, update the centroids
    if (!converged) {
      q.submit([&](handler &h) {
        const auto new_acc = new_centroids_b.get_access<access::mode::read>(h);
        const auto acc     = centroids_b.get_access<access::mode::write>(h);

        h.parallel_for<class assign_kernel>(k, [=](auto item) { acc[item] = new_acc[item]; });
      });
    }
  }

  // return the final centroids
  auto centroids_a    = centroids_b.get_host_access();
  auto associations_a = assoc_b.get_host_access();

  auto clusters = std::vector<std::vector<Point>>{centroids_a.size()};
  for (size_t i = 0; i < k; i++) {
    clusters[i] = std::vector<Point>{};
  }

  for (size_t i = 0; i < points.size(); i++) {
    clusters[associations_a[i]].push_back(points[i]);
  }

  // copy from accessors to host memory
  // TODO: maybe we can avoid this copy
  host_centroids = std::vector<Point>{centroids_a.begin(), centroids_a.end()};

  return kmeans_cluster_t{
      host_centroids,
      clusters,
  };
}
