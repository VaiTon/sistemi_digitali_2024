#include "util.hpp"

#include <iosfwd>
#include <iostream>
#include <istream>
#include <ostream>
#include <sstream>

#include "kmeans.hpp"

#include "buf/kmeans_buf.hpp"
#include "cpu/kmeans_cpu.hpp"
#include "simd/kmeans_simd.hpp"
#include "usm/kmeans_usm.hpp"

void save_results(const std::string &path, const kmeans_cluster_t &res) {
  // open file
  std::ofstream file{path};

  file << "[";
  for (size_t i = 0; i < res.centroids.size(); ++i) {
    file << "{\"c\": [" << res.centroids[i].x << ", " << res.centroids[i].y << "], \"a\": [";
    for (size_t j = 0; j < res.clusters[i].size(); ++j) {
      file << "[" << res.clusters[i][j].x << ", " << res.clusters[i][j].y << "]";
      if (j != res.clusters[i].size() - 1) {
        file << ", ";
      }
    }

    file << (i == res.clusters.size() - 1 ? "]}" : "]}, ") << std::endl;
  }
  file << "]" << std::endl;
}

template <typename T>
auto test(const std::string &name, const std::string &filename, T &km, size_t max_iter, double tol)
    -> decltype(km.cluster(max_iter, tol), void()) {

  const std::string class_name = typeid(T).name();
  logger::info() << "Running kmeans on " << name << " <class " << class_name << ">\n";
  auto res = km.cluster(max_iter, tol);

  logger::info() << "Saving results to " << filename << "\n";
  save_results(filename, res);

  return;
}

int main(const int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <data.csv> <k> <output_dir>";
    return EXIT_FAILURE;
  }

  const auto input_filename = std::string{argv[1]};
  const auto k              = static_cast<size_t>(std::stoi(argv[2]));
  const auto output_dir     = std::string{argv[3]};

  logger::info() << "Input file: " << input_filename << "\n";
  logger::info() << "Loading data\n";
  const auto data = get_data(input_filename);
  logger::info() << "Data size: " << data.size() << "\n";

  constexpr size_t max_iter = 1000;
  constexpr double tol      = 1e-4;
  {
    auto kmeans = kmeans_cpu_v1{k, data};
    test("CPU", output_dir + "/cpu_v1.json", kmeans, max_iter, tol);
  }
  {
    auto kmeans = kmeans_cpu_v2{k, data};
    test("CPU (v2)", output_dir + "/cpu_v2.json", kmeans, max_iter, tol);
  }
  {
    auto kmeans = kmeans_cpu_v3{k, data};
    test("CPU (v3)", output_dir + "/cpu_v3.json", kmeans, max_iter, tol);
  }

  {
    auto kmeans = kmeans_simd{k, data};
    test("CPU (SIMD)", output_dir + "/simd.json", kmeans, max_iter, tol);
  }

  const auto q = sycl::queue{};
  logger::info() << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  {
    auto kmeans = kmeans_usm_v2{q, k, data};
    test("USM v2", output_dir + "/usm_v2.json", kmeans, max_iter, tol);
  }
  {
    auto kmeans = kmeans_usm_v3{q, k, data};
    test("USM v3", output_dir + "/usm_v3.json", kmeans, max_iter, tol);
  }

  {
    auto kmeans = kmeans_buf{q, k, data};
    test("Buffer", output_dir + "/buf.json", kmeans, max_iter, tol);
  }

  return EXIT_SUCCESS;
}
