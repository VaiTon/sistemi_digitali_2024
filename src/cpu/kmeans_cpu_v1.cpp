#include "kmeans_cpu.hpp"

kmeans_cluster_t kmeans_cpu_v1::cluster(const size_t max_iter, double tol) {
  tol = tol * tol; // we use squared distance for convergence check

  auto centroids     = std::vector<point_t>(k);
  auto new_centroids = std::vector<point_t>(k);
  auto clusters      = std::vector<std::vector<point_t>>(k);
  bool converged     = false;

  // Initialization: choose centroids as the first k points
  std::copy_n(points.begin(), k, centroids.begin());

  for (iter = 0; iter < max_iter; iter++) {

    // Assign each point to the "closest" centroid
    for (auto point : points) {
      auto   min_distance = std::numeric_limits<double>::max();
      size_t min_idx      = 0;

      for (auto c_idx = size_t{0}; c_idx < k; c_idx++) {
        const auto distance = squared_distance(point, centroids[c_idx]);
        if (distance < min_distance) {
          min_distance = distance;
          min_idx      = c_idx;
        }
      }

      clusters[min_idx].push_back(point);
    }

    // Step 2: Calculate new centroids
    for (size_t c_idx = 0; c_idx < k; c_idx++) {
      // - Initialize new centroids to (0, 0)
      new_centroids[c_idx] = point_t{0, 0};

      for (auto [x, y] : clusters[c_idx]) {
        new_centroids[c_idx].x += x;
        new_centroids[c_idx].y += y;
      }

      auto size = static_cast<float>(clusters[c_idx].size());
      new_centroids[c_idx].x /= size;
      new_centroids[c_idx].y /= size;
    }

    // Check for convergence
    converged = true;
    for (size_t i = 0; i < k; ++i) {
      converged &= squared_distance(new_centroids[i], centroids[i]) < tol;

      if (!converged) {
        break; // no need to check further
      }
    }

    // Update centroids
    std::ranges::copy(new_centroids, centroids.begin());

    if (converged) {
      break;
    }

    clusters = std::vector<std::vector<point_t>>(k);
  }

  return kmeans_cluster_t{centroids, clusters};
}
