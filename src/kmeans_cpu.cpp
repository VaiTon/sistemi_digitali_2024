#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "kmeans.hpp"

kmeans_cluster_t kmeans_cpu(const size_t k, const std::vector<Point> &points, const size_t max_iter, float tol) {
  if (k > points.size()) {
    throw std::invalid_argument("Number of clusters must be less than or equal "
                                "to the number of points");
  }

  tol *= tol; // we use squared distance for convergence check

  auto centroids     = std::vector<Point>(k);              // Centroids
  auto new_centroids = std::vector<Point>(k);              // Updated centroids after each iteration
  auto assoc_len_d   = std::vector<size_t>(k);             // Number of points in each cluster
  auto assoc         = std::vector<size_t>(points.size()); // Association of each point to a cluster
  auto converged     = bool{false};                        // Convergence flag
  auto iter          = size_t{0};                          // Iteration counter

  // Initialization: choose k centroids (Forgy, Random Partition, etc.)
  // For simplicity, let's assume the first k points are the initial centroids
  std::copy_n(points.begin(), k, centroids.begin());

  while (!converged && iter < max_iter) {

    // Assign each point to the "closest" centroid
#ifdef USE_OPENMP
#pragma omp parallel for default(none) shared(points, centroids, assoc, k) firstprivate(tol)
#endif
    for (auto p_idx = size_t{0}; p_idx < points.size(); p_idx++) {
      auto min_distance = std::numeric_limits<double>::max();
      auto min_idx      = size_t{0};

      for (auto c_idx = size_t{0}; c_idx < k; c_idx++) {
        const auto distance = squared_distance(points[p_idx], centroids[c_idx]);
        if (distance < min_distance) {
          min_distance = distance;
          min_idx      = c_idx;
        }
      }

      assoc[p_idx] = min_idx;
    }

    // Step 2: Calculate new centroids
    // - Initialize new centroids to (0, 0)
    for (auto c_idx = size_t{0}; c_idx < k; c_idx++) {
      new_centroids[c_idx] = Point{0, 0};
      assoc_len_d[c_idx]   = 0;
    }

    // - For each point, add the point's coordinates to the corresponding centroid
    for (auto p_idx = size_t{0}; p_idx < points.size(); p_idx++) {
      const auto centroid_idx = assoc[p_idx];
      new_centroids[centroid_idx].x += points[p_idx].x;
      new_centroids[centroid_idx].y += points[p_idx].y;
      assoc_len_d[centroid_idx]++;
    }
    // - Divide the sum of coordinates by the number of points in the cluster
    for (auto c_idx = size_t{0}; c_idx < k; c_idx++) {
      if (assoc_len_d[c_idx] > 0) {
        new_centroids[c_idx].x /= static_cast<float>(assoc_len_d[c_idx]);
        new_centroids[c_idx].y /= static_cast<float>(assoc_len_d[c_idx]);
      }
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
