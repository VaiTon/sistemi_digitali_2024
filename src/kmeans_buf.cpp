#include <cstddef>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <vector>

#include "kmeans.hpp"
#include "kmeans_sycl.hpp"

using namespace sycl;

kmeans_cluster_t kmeans_buf(queue q, size_t k, const vector<Point> &points, size_t max_iter, float tol) {
  if (k > points.size()) {
    throw std::invalid_argument("Number of clusters must be less than or equal "
                                "to the number of points");
  }

  // tolerance must be squared
  tol *= tol;

  // Step 0: Initialize centroids
  // For simplicity, let's assume the first k points are the initial centroids.
  auto centroids_h = vector<Point>{k};
  std::copy_n(points.begin(), k, centroids_h.begin());

  // Create the relevant buffers to be used in the computation
  auto points_b        = buffer{points};
  auto centroids_b     = buffer{centroids_h};
  auto new_centroids_b = buffer<Point>{k};
  auto assoc_b         = buffer<size_t>{points.size()};

  // main cycle: assign points to clusters, calculate new centroids, check for
  // convergence
  auto converged = bool{false};
  auto iter      = size_t{0};

  // return the final centroids
  while (!converged && iter < max_iter) {

    // Step 1: Assign points to clusters
    q.submit([&](handler &h) {
      const auto points_acc    = points_b.get_access<access::mode::read>(h);
      const auto centroids_acc = centroids_b.get_access<access::mode::read>(h);
      const auto assoc_acc     = assoc_b.get_access<access::mode::write>(h);

      // for each point, calculate the distance to each centroid and assign the
      // point to the closest one

      h.parallel_for(points_acc.size(), [=](const size_t item) {
        double min_val = std::numeric_limits<double>::max();
        size_t min_idx = 0;

        for (size_t i = 0; i < centroids_acc.size(); i++) {
          const auto dist = squared_distance(points_acc[item], centroids_acc[i]);
          if (dist < min_val) {
            min_val = dist;
            min_idx = i;
          }
        }

        assoc_acc[item] = min_idx;
      });
    });

    // calculate new centroids by averaging the points in each cluster
    q.submit([&](handler &h) {
      const auto points_acc = points_b.get_access<access::mode::read>(h);
      const auto assoc_acc  = assoc_b.get_access<access::mode::read>(h);
      const auto centr_acc  = new_centroids_b.get_access<access::mode::write>(h);

      // for each cluster, calculate the new centroid by averaging the points
      // associated with it
      h.parallel_for(k, [=](const size_t item) {
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

    auto converged_h = int{true};                    // local variable to initialize the buffer
    auto converged_b = buffer<int>{&converged_h, 1}; // buffer to store the convergence status

    q.submit([&](handler &h) {
      auto new_centroids_a = new_centroids_b.get_access<access::mode::read>(h);
      auto centroids_a     = centroids_b.get_access<access::mode::read>(h);

      const auto converged_reduction = reduction(converged_b, h, logical_and{});

      h.parallel_for(new_centroids_a.size(), converged_reduction, [=](const auto item, auto &converged_ref) {
        auto dist = squared_distance(new_centroids_a[item], centroids_a[item]);
        converged_ref.combine(dist < tol);
      });
    });

    converged = converged_b.get_host_access()[0];

    // if not converged, update the centroids
    if (!converged) {
      q.submit([&](handler &h) {
        const auto new_acc = new_centroids_b.get_access<access::mode::read>(h);
        const auto acc     = centroids_b.get_access<access::mode::write>(h);

        h.parallel_for(k, [=](auto item) { acc[item] = new_acc[item]; });
      });
    }

    iter++;
  }
  const auto centroids_a    = centroids_b.get_host_access();
  const auto associations_a = assoc_b.get_host_access();

  auto clusters = std::vector<std::vector<Point>>{centroids_a.size()};
  for (size_t i = 0; i < k; i++) {
    clusters[i] = std::vector<Point>{};
  }

  for (size_t i = 0; i < points.size(); i++) {
    clusters[associations_a[i]].push_back(points[i]);
  }

  // copy from accessors to host memory
  // TODO: maybe we can avoid this copy
  centroids_h = std::vector<Point>{centroids_a.begin(), centroids_a.end()};

  return kmeans_cluster_t{
      centroids_h,
      clusters,
  };
}
