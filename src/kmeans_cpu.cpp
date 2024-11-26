#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "kmeans.hpp"

kmeans_cluster_t kmeans_cpu(const size_t k, const vector<Point> &points, const size_t max_iter, float tol) {
  if (k > points.size()) {
    throw std::invalid_argument("Number of clusters must be less than or equal "
                                "to the number of points");
  }

  tol *= tol; // we use squared distance for convergence check

  auto centroids     = vector<Point>(k);              // Centroids
  auto new_centroids = vector<Point>(k);              // Updated centroids after each iteration
  auto assoc         = vector<size_t>(points.size()); // Association of each point to a cluster
  auto converged     = bool{false};                   // Convergence flag
  auto iter          = size_t{0};                     // Iteration counter

  // Initialization: choose k centroids (Forgy, Random Partition, etc.)
  // For simplicity, let's assume the first k points are the initial centroids
  std::copy_n(points.begin(), k, centroids.begin());

  while (!converged && iter < max_iter) {

    // Assign each point to the "closest" centroid
    for (auto i = size_t{0}; i < points.size(); i++) {
      auto min_distance = std::numeric_limits<double>::max();
      auto min_idx      = size_t{0};

      for (auto j = size_t{0}; j < k; j++) {
        const auto distance = squared_distance(points[i], centroids[j]);
        if (distance < min_distance) {
          min_distance = distance;
          min_idx      = j;
        }
      }

      assoc[i] = min_idx;
    }

    // Calculate new centroids
    for (auto centroid_idx = size_t{0}; centroid_idx < k; centroid_idx++) {
      auto new_centroid = Point{0, 0};

      auto count = size_t{0};
      for (size_t point_idx = 0; point_idx < points.size(); point_idx++) {
        if (assoc[point_idx] == centroid_idx) {
          new_centroid.x += points[point_idx].x;
          new_centroid.y += points[point_idx].y;
          count++;
        }
      }

      if (count > 0) {
        new_centroid.x /= static_cast<float>(count);
        new_centroid.y /= static_cast<float>(count);
      }

      new_centroids[centroid_idx] = new_centroid;
    }

    converged = true;
    for (size_t i = 0; i == 0 || (i < k && !converged); ++i) {
      converged = squared_distance(new_centroids[i], centroids[i]) < tol;
    }

    centroids = new_centroids;
    iter++;
  }

  auto clusters = std::vector<std::vector<Point>>{centroids.size()};
  for (size_t i = 0; i < k; i++) {
    clusters[i] = std::vector<Point>{};
  }

  for (size_t i = 0; i < points.size(); i++) {
    clusters[assoc[i]].push_back(points[i]);
  }

  return kmeans_cluster_t{centroids, clusters};
}
