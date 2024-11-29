#include <cstddef>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <vector>

#include "kmeans.hpp"

using namespace sycl;

class BufAssignPointsKernel;
class BufNewCentroidsKernel;
class BufConvergedKernel;
class BufUpdateCentroidsKernel;

kmeans_cluster_t kmeans_sycl_buf(queue q, size_t k, const std::vector<Point> &points, size_t max_iter, float tol) {
  if (k > points.size()) {
    throw std::invalid_argument("Number of clusters must be less than or equal "
                                "to the number of points");
  }

  // tolerance must be squared
  tol *= tol;

  const auto points_n = points.size();

  // Step 0: Initialize centroids
  // For simplicity, let's assume the first k points are the initial centroids.
  auto centroids_h = std::vector<Point>{k};
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
      const auto points_a    = points_b.get_access<access::mode::read>(h);
      const auto centroids_a = centroids_b.get_access<access::mode::read>(h);
      const auto assoc_a     = assoc_b.get_access<access::mode::write>(h);

      // for each point, calculate the distance to each centroid and assign the
      // point to the closest one

      h.parallel_for<BufAssignPointsKernel>(points_n, [=](const size_t item) {
        double min_val = std::numeric_limits<double>::max();
        size_t min_idx = 0;

        for (size_t i = 0; i < k; i++) {
          const auto dist = squared_distance(points_a[item], centroids_a[i]);
          if (dist < min_val) {
            min_val = dist;
            min_idx = i;
          }
        }

        assoc_a[item] = min_idx;
      });
    });

    // Step 2: calculate new centroids by averaging the points in each cluster
    q.submit([&](handler &h) {
      const auto points_a        = points_b.get_access<access::mode::read>(h);
      const auto assoc_a         = assoc_b.get_access<access::mode::read>(h);
      const auto centroids_a     = centroids_b.get_access<access::mode::read>(h);
      const auto new_centroids_a = new_centroids_b.get_access<access::mode::write>(h);

      // for each cluster, calculate the new centroid by averaging the points
      // associated with it
      h.parallel_for<BufNewCentroidsKernel>(k, [=](const size_t item) {
        auto new_centroid = Point{0, 0};

        auto count = size_t{0};
        for (size_t i = 0; i < points_n; i++) {
          if (assoc_a[i] == item) {
            new_centroid.x += points_a[i].x;
            new_centroid.y += points_a[i].y;
            count++;
          }
        }

        if (count > 0) {
          new_centroid.x /= static_cast<float>(count);
          new_centroid.y /= static_cast<float>(count);
          new_centroids_a[item] = new_centroid;
        } else {
          new_centroids_a[item] = centroids_a[item];
        }
      });
    });

    auto converged_b = buffer<int>{1};

    converged_b.get_host_access()[0] = true;

    q.submit([&](handler &h) {
      auto new_centroids_a = new_centroids_b.get_access<access::mode::read>(h);
      auto centroids_a     = centroids_b.get_access<access::mode::read>(h);

      const auto converged_reduction = reduction(converged_b, h, logical_and{});

      h.parallel_for<BufConvergedKernel>(k, converged_reduction, [=](const auto item, auto &converged_ref) {
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

        h.parallel_for<BufUpdateCentroidsKernel>(k, [=](auto item) { acc[item] = new_acc[item]; });
      });
    }

    iter++;
  }

  const auto centroids_a    = centroids_b.get_host_access();
  const auto associations_a = assoc_b.get_host_access();

  auto final_clusters = std::vector<std::vector<Point>>{centroids_a.size()};
  for (size_t i = 0; i < k; i++) {
    final_clusters[i] = std::vector<Point>{};
  }

  for (size_t i = 0; i < points.size(); i++) {
    final_clusters[associations_a[i]].push_back(points[i]);
  }

  // copy from accessors to host memory
  auto final_centroids = std::vector<Point>{centroids_a.begin(), centroids_a.end()};

  return kmeans_cluster_t{
      final_centroids,
      final_clusters,
  };
}
