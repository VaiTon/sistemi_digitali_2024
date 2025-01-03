#include "util.hpp"

#include <cstdlib>
#include <exception>
#include <fmt/core.h>
#include <iosfwd>
#include <iostream>
#include <omp.h>
#include <ostream>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "buf/kmeans_buf.hpp"
#include "cpu/kmeans_cpu.hpp"
#include "ocv/kmeans_ocv.hpp"
#include "omp/kmeans_omp.hpp"
#include "simd/kmeans_simd.hpp"
#include "usm/kmeans_usm.hpp"

#ifdef USE_TBB
#include "tbb/kmeans_tbb.hpp"
#endif
#ifdef USE_CUDA
#include "cuda/kmeans_cuda.hpp"
#endif

template <typename T>
auto time_and_print(std::string const &name, T &km, size_t max_iter, double tol,
                    size_t const data_size, long const comp_time = 0,
                    size_t const computation_units = 0)
    -> decltype(km.cluster(max_iter, tol), long()) {

  auto const start_time = std::chrono::high_resolution_clock::now();
  try {
    km.cluster(max_iter, tol);
  } catch (std::exception const &e) {
    logger::raw() << "Exception: " << e.what() << std::endl;
    return -1;
  }
  auto const end_time = std::chrono::high_resolution_clock::now();

  long const time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  auto const throughput = static_cast<float>(data_size) * 1e3 / static_cast<float>(time);

  logger::info() << fmt::format(" -> {:20.20}", name)  //
                 << fmt::format("time: {:5} ms", time) //
                 << fmt::format(", iterations: {:3}", km.get_iters())
                 << fmt::format(", throughput: {:5.2f} GB/s", throughput / 1e6);

  if (comp_time > 0) {
    float speedup = static_cast<float>(comp_time) / static_cast<float>(time);

    logger::raw() << fmt::format(", speedup: {:5.2f}", speedup);

    if (computation_units > 0) {
      float efficiency = speedup / static_cast<float>(computation_units);

      logger::raw() << fmt::format(", units: {:2}", computation_units) //
                    << fmt::format(", efficiency: {:4.2f}", efficiency);
    }
  }

  logger::raw() << std::endl;
  return time;
}

int main(int const argc, char **argv) {

  auto types = std::vector<std::string>{"cpu", "omp", "simd", "ocv", "sycl", "sycl-inorder"};
#ifdef USE_TBB
  types.push_back("tbb");
#endif
#ifdef USE_CUDA
  types.push_back("cuda");
#endif

  if (argc < 3) {
    logger::raw() << "Usage: " << argv[0]
                  << " <data.csv> <k> [type]\n"
                     "  type: ";
    for (auto const &t : types) {
      logger::raw() << t << " ";
    }
    logger::raw() << std::endl;
    return EXIT_FAILURE;
  }

  auto run_type = std::string{};
  if (argc == 4) {
    run_type = argv[3];
    std::transform(run_type.begin(), run_type.end(), run_type.begin(),
                   [](unsigned char const c) { return std::tolower(c); });
  }

  if (!run_type.empty() && std::find(types.begin(), types.end(), run_type) == types.end()) {
    logger::error() << "Invalid type: " << run_type << std::endl;
    logger::error() << "Valid types: ";
    for (auto const &t : types) {
      logger::raw() << t << " ";
    }
    logger::raw() << std::endl;
    return EXIT_FAILURE;
  }

  constexpr double tol      = 1e-4;
  constexpr size_t max_iter = 1000;

  auto const input_filename = std::string{argv[1]};
  auto const k              = static_cast<size_t>(std::stoi(argv[2]));

  // read data from csv file
  auto const   data      = get_data(input_filename);
  size_t const data_size = data.size() * sizeof(point_t);

  logger::info() << "Data size: " << data.size() << " (" << static_cast<double>(data_size) / 1e6
                 << " MB), clusters: " << k << std::endl;

  long ref_time;

  logger::info() << "Running benchmarks..." << "\n";

  {
    // run CPU version to get baseline
    auto kmeans = kmeans_cpu_v1{k, data};
    ref_time    = time_and_print("CPU (v1)", kmeans, max_iter, tol, data_size);
  }

  if (run_type.empty() || run_type == "cpu") {
    auto kmeans = kmeans_cpu_v2{k, data};
    time_and_print("CPU (v2)", kmeans, max_iter, tol, data_size, ref_time);
  }
  if (run_type.empty() || run_type == "omp") {
    auto kmeans = kmeans_omp{k, data};
    time_and_print("OpenMP", kmeans, max_iter, tol, data_size, ref_time, omp_get_max_threads());
  }
#ifdef USE_TBB
  if (run_type.empty() || run_type == "tbb") {
    auto kmeans = kmeans_tbb{k, data};
    time_and_print("TBB", kmeans, max_iter, tol, data_size, ref_time);
  }
#endif
  if (run_type.empty() || run_type == "simd") {
    {
      auto kmeans = kmeans_simd{k, data};
      time_and_print("CPU (SIMD)", kmeans, max_iter, tol, data_size, ref_time);
    }
    {
      auto kmeans = kmeans_ocv{k, data};
      time_and_print("OpenCV UI", kmeans, max_iter, tol, data_size, ref_time);
    }
  }

#ifdef USE_CUDA
  if (run_type.empty() || run_type == "cuda") {
    auto kmeans = kmeans_cuda{k, data};
    time_and_print("CUDA", kmeans, max_iter, tol, ref_time);
  }
#endif

  auto test_sycl = [&](sycl::queue const &q, size_t const compute_units) {

#ifndef USE_CUDA // Issue with CUDA and SYCL buffers
    {
      auto kmeans = kmeans_buf{q, k, data};
      time_and_print("BUF", kmeans, max_iter, tol, data_size, ref_time, compute_units);
    }
#endif

    {
      auto kmeans = kmeans_usm_v1{q, k, data};
      time_and_print("USMv1", kmeans, max_iter, tol, data_size, ref_time, compute_units);
    }

    {
      auto kmeans = kmeans_usm_v2{q, k, data};
      time_and_print("USMv2", kmeans, max_iter, tol, data_size, ref_time, compute_units);
    }

    {
      auto kmeans = kmeans_usm_v3{q, k, data};
      time_and_print("USMv3", kmeans, max_iter, tol, data_size, ref_time, compute_units);
    }
    {
      auto kmeans = kmeans_usm_v4{q, k, data};
      time_and_print("USMv4", kmeans, max_iter, tol, data_size, ref_time, compute_units);
    }
  };

  if (run_type.empty() || run_type == "sycl") {

    auto queue = sycl::queue{};

    auto const device_name   = queue.get_device().get_info<sycl::info::device::name>();
    auto const compute_units = queue.get_device().get_info<sycl::info::device::max_compute_units>();

    logger::info() << "Running on SYCL: " << device_name << " with " << compute_units
                   << " compute units" << std::endl;

    test_sycl(queue, compute_units);
  }
  if (run_type.empty() || run_type == "sycl-inorder") {

    auto queue = sycl::queue{sycl::property::queue::in_order{}};

    auto const device_name   = queue.get_device().get_info<sycl::info::device::name>();
    auto const compute_units = queue.get_device().get_info<sycl::info::device::max_compute_units>();

    logger::info() << "Running on SYCL (in-order): " << device_name << " with " << compute_units
                   << " compute units" << std::endl;

    test_sycl(queue, compute_units);
  }

  return EXIT_SUCCESS;
}
