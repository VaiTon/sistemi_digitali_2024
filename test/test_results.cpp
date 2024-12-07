#include "kmeans.hpp"
#include <iostream>
#include <string_view>

int main(const int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <data.csv> <k> <cpu|sycl>";
    return EXIT_FAILURE;
  }

  const auto k              = std::stoi(argv[2]);
  const auto input_filename = std::string{argv[1]};
  const auto mode           = std::string{argv[3]};

  const auto data = get_data(input_filename);
  std::clog << "--- Data size: " << data.size() << std::endl;

  const auto [centroids, clusters] = kmeans_cpu(k, data, 1000, 1e-4);

  const auto q = sycl::queue{};
  std::clog << "--- Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  kmeans_cluster_t res;
  if (mode == "sycl") {
    res = kmeans_sycl(q, k, data, 1000, 1e-4);
  } else if (mode == "cpu") {
    res = kmeans_cpu(k, data, 1000, 1e-4);
  } else {
    std::clog << "Invalid mode: " << mode << ". Please use one of: cpu, sycl" << std::endl;
  }

  std::cout << "[";
  for (int i = 0; i < k; ++i) {
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
