#include "kmeans_cpu.hpp"

#include <iostream>

kmeans_cluster_t kmeans_cpu_v1::cluster(const size_t max_iter, double tol) {
  tol = tol * tol; // we use squared distance for convergence check

  auto centroids     = std::vector<point_t>(num_centroids);
  auto new_centroids = std::vector<point_t>(num_centroids);
  auto clusters      = std::vector<std::vector<point_t>>(num_centroids);
  bool converged     = false;

  // Initialization: choose centroids as the first k points
  std::copy_n(points.begin(), num_centroids, centroids.begin());

  for (iter = 0; iter < max_iter; iter++) {

    // Assign each point to the "closest" centroid
    for (auto point : points) {
      auto   min_distance = std::numeric_limits<double>::max();
      size_t min_idx      = 0;

      for (auto c_idx = size_t{0}; c_idx < num_centroids; c_idx++) {
        const auto distance = squared_distance(point, centroids[c_idx]);
        if (distance < min_distance) {
          min_distance = distance;
          min_idx      = c_idx;
        }
      }

      clusters[min_idx].push_back(point);
    }

    // Step 2: Calculate new centroids
    for (size_t c_idx = 0; c_idx < num_centroids; c_idx++) {

      // - Initialize new centroids to (0, 0)
      const auto size = static_cast<float>(clusters[c_idx].size());
      if (size == 0) {
        new_centroids[c_idx] = centroids[c_idx];
        continue;
      }

      float new_x = 0.0;
      float new_y = 0.0;

      for (auto [x, y] : clusters[c_idx]) {
        new_x += x;
        new_y += y;
      }

      new_centroids[c_idx] = point_t{new_x / size, new_y / size};
    }

    // Check for convergence
    converged = true;
    for (size_t i = 0; i < num_centroids && converged; ++i) {
      converged &= squared_distance(new_centroids[i], centroids[i]) < tol;
    }

    // Update centroids
    std::ranges::copy(new_centroids, centroids.begin());

    if (converged) {
      break;
    }

    clusters = std::vector<std::vector<point_t>>(num_centroids);
  }

  return kmeans_cluster_t{centroids, clusters};
}
