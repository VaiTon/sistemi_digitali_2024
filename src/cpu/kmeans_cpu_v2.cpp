#include "kmeans_cpu.hpp"

#include <algorithm>
#include <limits>

kmeans_cluster_t kmeans_cpu_v2::cluster(const size_t max_iter, double tol) {
  tol = tol * tol; // we use squared distance for convergence check

  auto centroids = std::vector<point_t>(num_centroids); // Centroids
  auto new_centroids =
      std::vector<point_t>(num_centroids);                 // Updated centroids after each iteration
  auto clusters_size = std::vector<size_t>(num_centroids); // Number of points in each cluster
  auto associations  = std::vector<size_t>(points.size()); // Association of each point to a cluster
  bool converged     = false;                              // Convergence flag

  // Initialization: choose centroids as the first k points
  std::copy_n(points.begin(), num_centroids, centroids.begin());

  for (iter = 0; iter < max_iter; iter++) {
    // Assign each point to the "closest" centroid
    for (size_t p_idx = 0; p_idx < points.size(); p_idx++) {
      auto min_distance = std::numeric_limits<float>::max();
      auto min_idx      = size_t{0};

      for (auto c_idx = size_t{0}; c_idx < num_centroids; c_idx++) {
        const auto distance = squared_distance(points[p_idx], centroids[c_idx]);
        if (distance < min_distance) {
          min_distance = distance;
          min_idx      = c_idx;
        }
      }

      associations[p_idx] = min_idx;
    }

    // Step 2: Calculate new centroids
    // - Initialize new centroids to (0, 0)
    for (auto c_idx = size_t{0}; c_idx < num_centroids; c_idx++) {
      new_centroids[c_idx] = point_t{0, 0};
      clusters_size[c_idx] = 0;
    }

    // - For each point, add the point's coordinates to the corresponding centroid
    for (auto p_idx = size_t{0}; p_idx < points.size(); p_idx++) {
      const auto centroid_idx = associations[p_idx];
      new_centroids[centroid_idx].x += points[p_idx].x;
      new_centroids[centroid_idx].y += points[p_idx].y;
      clusters_size[centroid_idx]++;
    }

    // - Divide the sum of coordinates by the number of points in the cluster
    for (auto c_idx = size_t{0}; c_idx < num_centroids; c_idx++) {
      if (clusters_size[c_idx] > 0) {
        new_centroids[c_idx].x /= static_cast<float>(clusters_size[c_idx]);
        new_centroids[c_idx].y /= static_cast<float>(clusters_size[c_idx]);
      }
    }

    // Check for convergence
    converged = true;
    for (size_t i = 0; i < num_centroids; ++i) {
      converged &= squared_distance(new_centroids[i], centroids[i]) < tol;

      if (!converged) {
        break; // no need to check further
      }
    }

    // Update centroids
    std::copy_n(new_centroids.begin(), num_centroids, centroids.begin());

    if (converged) {
      break;
    }
  }

  auto clusters = std::vector<std::vector<point_t>>{centroids.size()};
  for (size_t i = 0; i < num_centroids; i++) {
    clusters[i] = std::vector<point_t>{};
  }

  for (size_t i = 0; i < points.size(); i++) {
    clusters[associations[i]].push_back(points[i]);
  }

  return kmeans_cluster_t{centroids, clusters};
}
