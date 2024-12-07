#include "kmeans.hpp"
#include <iostream>

int main(const int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <data.csv> <k> [output.json]" << std::endl;
    return EXIT_FAILURE;
  }

  const auto k              = std::stoi(argv[2]);
  const auto input_filename = std::string{argv[1]};

  const auto data = get_data(input_filename);
  std::clog << "--- Data size: " << data.size() << std::endl;

  const auto [centroids, clusters] = kmeans_cpu(k, data, 1000, 1e-4);

  const auto q                               = sycl::queue{};
  const auto [centroids_sycl, clusters_sycl] = kmeans_sycl(q, k, data, 1000, 1e-4);

  // compare results
  // calculate distance between centroids
  double max_distance = 0.0;

  for (int i = 0; i < k; ++i) {
    const auto distance = squared_distance(centroids[i], centroids_sycl[i]);
    max_distance        = std::max(max_distance, distance);
  }

  std::clog << "--- Max distance between centroids (" << SYCL_TYPE << "): " << max_distance << std::endl;

  std::cout << "[";
  for (size_t i = 0; i < k; ++i) {
    std::cout << "{\"c\": [" << centroids[i].x << ", " << centroids[i].y << "], \"a\": [";
    for (size_t j = 0; j < clusters[i].size(); ++j) {
      std::cout << "[" << clusters[i][j].x << ", " << clusters[i][j].y << "]";
      if (j != clusters[i].size() - 1) {
        std::cout << ", ";
      }
    }

    std::cout << (i == k - 1 ? "]}" : "]}, ") << std::endl;
  }
  std::cout << "]" << std::endl;

  return EXIT_SUCCESS;
}
