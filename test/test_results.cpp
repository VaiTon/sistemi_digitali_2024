#include "util.hpp"

#include <iosfwd>
#include <iostream>
#include <istream>
#include <ostream>
#include <sstream>

#include "kmeans.hpp"

#include "buf/kmeans_buf.hpp"
#include "cpu/kmeans_cpu.hpp"
#ifdef USE_CUDA
#include "cuda/kmeans_cuda.hpp"
#endif
#include "ocv/kmeans_ocv.hpp"
#include "omp/kmeans_omp.hpp"
#include "simd/kmeans_simd.hpp"
#include "tbb/kmeans_tbb.hpp"
#include "usm/kmeans_usm.hpp"

void save_results(std::string const &path, kmeans_cluster_t const &res) {
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

template <typename T, std::enable_if_t<std::is_base_of_v<kmeans, T>> * = nullptr>
void test(std::string const &name, std::string const &filename, T &km, size_t max_iter,
          double tol) {

  logger::info() << "Running backend " << name << "\n";
  auto res = km.cluster(max_iter, tol);
  logger::info() << "Iterations: " << km.get_iters() << "\n";

  logger::info() << "Saving results to " << filename << "\n";
  save_results(filename, res);

  return;
}

int main(int const argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <data.csv> <k> <output_dir>";
    return EXIT_FAILURE;
  }

  auto const input_filename = std::string{argv[1]};
  auto const k              = static_cast<size_t>(std::stoi(argv[2]));
  auto const output_dir     = std::string{argv[3]};

  logger::info() << "Input file: " << input_filename << "\n", logger::info() << "Loading data...\n";
  auto const data = get_data(input_filename);

  logger::info() << "Data size: " << data.size() << ", clusters: " << k << "\n";

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
    auto kmeans = kmeans_omp{k, data};
    test("OpenMP", output_dir + "/omp.json", kmeans, max_iter, tol);
  }
  {
    auto kmeans = kmeans_tbb{k, data};
    test("TBB", output_dir + "/tbb.json", kmeans, max_iter, tol);
  }

  {
    auto kmeans = kmeans_simd{k, data};
    test("CPU (SIMD)", output_dir + "/simd.json", kmeans, max_iter, tol);
  }
  {
    auto kmeans = kmeans_ocv{k, data};
    test("OpenCV", output_dir + "/ocv.json", kmeans, max_iter, tol);
  }

#ifdef USE_CUDA
  {
    auto kmeans = kmeans_cuda{k, data};
    test("CUDA", output_dir + "/cuda.json", kmeans, max_iter, tol);
  }
#endif

  const auto q = sycl::queue{};
  logger::info() << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  {
    auto kmeans = kmeans_usm_v1{q, k, data};
    test("USM v1", output_dir + "/usm_v1.json", kmeans, max_iter, tol);
  }
  {
    auto kmeans = kmeans_usm_v2{q, k, data};
    test("USM v2", output_dir + "/usm_v2.json", kmeans, max_iter, tol);
  }
  {
    auto kmeans = kmeans_usm_v3{q, k, data};
    test("USM v3", output_dir + "/usm_v3.json", kmeans, max_iter, tol);
  }
  {
    auto kmeans = kmeans_usm_v4{q, k, data};
    test("USM v4", output_dir + "/usm_v4.json", kmeans, max_iter, tol);
  }

  {
    auto kmeans = kmeans_buf{q, k, data};
    test("Buffer", output_dir + "/buf.json", kmeans, max_iter, tol);
  }

  return EXIT_SUCCESS;
}
