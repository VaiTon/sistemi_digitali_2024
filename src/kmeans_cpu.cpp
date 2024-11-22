#include <algorithm>
#include <stdexcept>
#include <vector>

#include "kmeans.hpp"

template <typename T, typename A> size_t arg_min(std::vector<T, A> const &vec) {
  return std::distance(vec.begin(), std::min_element(vec.begin(), vec.end()));
}

Point calculate_centroid(const std::vector<Point> &cluster) {
  float sum_x = 0, sum_y = 0;
  for (const auto &[x, y] : cluster) {
    sum_x += x;
    sum_y += y;
  }

  return {sum_x / static_cast<float>(cluster.size()), sum_y / static_cast<float>(cluster.size())};
}

kmeans_cluster_t kmeans(const size_t k, const vector<Point> &points) {
  if (k > points.size()) {
    throw std::invalid_argument("Number of clusters must be less than or equal "
                                "to the number of points");
  }

  std::vector<Point> centroids(k);
  // Initialization: choose k centroids (Forgy, Random Partition, etc.)
  // For simplicity, let's assume the first k points are the initial centroids
  std::copy_n(points.begin(), k, centroids.begin());

  std::vector<std::vector<Point>> clusters(k);
  bool                            converged = false;

  while (!converged) {
    // Clear previous clusters
    for (auto &cluster : clusters) {
      cluster.clear();
    }

    // Assign each point to the "closest" centroid
    for (const auto &point : points) {
      std::vector<double> distances_to_each_centroid(k);
      for (size_t i = 0; i < k; ++i) {
        distances_to_each_centroid[i] = squared_distance(point, centroids[i]);
      }

      const auto cluster_assignment = arg_min(distances_to_each_centroid);
      clusters[cluster_assignment].push_back(point);
    }

    // Calculate new centroids
    std::vector<Point> new_centroids(k);
    for (size_t i = 0; i < k; ++i) {
      new_centroids[i] = calculate_centroid(clusters[i]);
    }

    converged = new_centroids == centroids;
    centroids = new_centroids;
  }

  return kmeans_cluster_t{centroids, clusters};
}
