#include "util.hpp"

#include <iosfwd>
#include <iostream>
#include <istream>

#include "kmeans.hpp"

#include "cpu/kmeans_cpu.hpp"
#include "simd/kmeans_simd.hpp"
#include "usm/kmeans_usm.hpp"

template <typename T>
auto do_work_sycl(T km, const std::vector<point_t> &centroids, const int max_iter, const double tol,
                  const int k) -> decltype(km.cluster(max_iter, tol), void()) {
  const auto [centroids_sycl, clusters_sycl] = km.cluster(max_iter, tol);

  // compare results
  // calculate distance between centroids
  double max_distance = 0.0;
  double tot_distance = 0.0;

  for (int i = 0; i < k; ++i) {
    const auto distance = squared_distance(centroids[i], centroids_sycl[i]);
    max_distance        = std::max(max_distance, distance);
    tot_distance += distance;
  }

  logger::info() << "Max distance between centroids: " << max_distance << std::endl;
  logger::info() << "Cumulative distance between centroids: " << tot_distance << std::endl;

  return;
}

int main(const int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <data.csv> <k> [output.json]" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr int  max_iter = 1000;
  constexpr auto tol      = 1e-4;

  const auto k              = std::stoi(argv[2]);
  const auto input_filename = std::string{argv[1]};

  const auto data = get_data(input_filename);
  logger::info() << "Data size: " << data.size() << std::endl;

  auto k_cpu = kmeans_cpu{static_cast<size_t>(k), data};

  const auto [centroids, clusters] = k_cpu.cluster(max_iter, tol);

  logger::info() << "Running kmeans on SIMD\n";
  const auto k_simd = kmeans_simd{static_cast<size_t>(k), data};
  do_work_sycl(k_simd, centroids, max_iter, tol, k);

  const auto q = sycl::queue{};

  logger::info() << "Running kmeans on USM\n";
  const auto k_usm = kmeans_usm{q, static_cast<size_t>(k), data};
  do_work_sycl(k_usm, centroids, max_iter, tol, k);

  return EXIT_SUCCESS;
}
