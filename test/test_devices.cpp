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

#ifdef USE_CUDA
#include "cuda/kmeans_cuda.hpp"
#endif

template <typename T>
auto time_and_print(std::string const &name, T &km, size_t max_iter, double tol,
                    long const comp_time = 0, size_t const computation_units = 1)
    -> decltype(km.cluster(max_iter, tol), long()) {

  logger::info() << fmt::format(" -> {:20.20}", name);

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

  logger::raw() << fmt::format("time: {:5} ms", time) //
                << fmt::format(", iterations: {:3}", km.get_iters());

  if (comp_time > 0) {
    float speedup = static_cast<float>(comp_time) / static_cast<float>(time);
    logger::raw() << fmt::format(", speedup: {:5.2f}", speedup);

    if (computation_units > 1) {
      float efficiency = speedup / static_cast<float>(computation_units);

      logger::raw() << fmt::format(", units: {:2}", computation_units) //
                    << fmt::format(", efficiency: {:4.2f}", efficiency);
    }
  }

  logger::raw() << std::endl;
  return time;
}

int main(int const argc, char **argv) {
  if (argc < 3) {
    logger::raw() << "Usage: " << argv[0] << " <data.csv> <k>" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr double tol      = 1e-4;
  constexpr size_t max_iter = 1000;

  auto const input_filename = std::string{argv[1]};
  auto const k              = static_cast<size_t>(std::stoi(argv[2]));

  // read data from csv file
  auto const data = get_data(input_filename);
  logger::info() << "Data size: " << data.size() << std::endl;

  auto devices = sycl::device::get_devices();
  logger::info() << "Found " << devices.size() << " devices" << std::endl;
  for (auto const &device : devices) {
    auto dev_name   = device.get_info<sycl::info::device::name>();
    auto dev_vendor = device.get_info<sycl::info::device::vendor>();
    logger::info() << "  " << dev_name << " (" << dev_vendor << ")" << std::endl;
  }

  long ref_time;

  logger::info() << "Running benchmarks" << std::endl;

  {
    auto kmeans = kmeans_cpu_v1{k, data};
    ref_time    = time_and_print("CPU (v1)", kmeans, max_iter, tol);
  }
  {
    auto kmeans = kmeans_cpu_v2{k, data};
    time_and_print("CPU (v2)", kmeans, max_iter, tol, ref_time);
  }
  {
    auto const max_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    {
      auto kmeans = kmeans_omp{k, data};
      time_and_print("OpenMP", kmeans, max_iter, tol, ref_time, omp_get_max_threads());
    }

    omp_set_num_threads(max_threads);
    {
      auto kmeans = kmeans_omp{k, data};
      time_and_print("OpenMP", kmeans, max_iter, tol, ref_time, omp_get_max_threads());
    }
  }
  {
    auto kmeans = kmeans_simd{k, data};
    time_and_print("CPU (SIMD)", kmeans, max_iter, tol, ref_time);
  }
  {
    auto kmeans = kmeans_ocv{k, data};
    time_and_print("OpenCV UI", kmeans, max_iter, tol, ref_time);
  }

#ifdef USE_CUDA
  {
    auto kmeans = kmeans_cuda{k, data};
    time_and_print("CUDA", kmeans, max_iter, tol, ref_time);
  }
#endif

  auto test_sycl = [&](const sycl::queue &q, const size_t compute_units) {
#ifndef USE_CUDA // Issue with CUDA and SYCL buffers
    {
      auto kmeans = kmeans_buf{q, k, data};
      time_and_print("BUF", kmeans, max_iter, tol, ref_time, compute_units);
    }
#endif

    {
      auto kmeans = kmeans_usm_v1{q, k, data};
      time_and_print("USMv1", kmeans, max_iter, tol, ref_time, compute_units);
    }

    {
      auto kmeans = kmeans_usm_v2{q, k, data};
      time_and_print("USMv2", kmeans, max_iter, tol, ref_time, compute_units);
    }

    {
      auto kmeans = kmeans_usm_v3{q, k, data};
      time_and_print("USMv3", kmeans, max_iter, tol, ref_time, compute_units);
    }
    {
      auto kmeans = kmeans_usm_v4{q, k, data};
      time_and_print("USMv4", kmeans, max_iter, tol, ref_time, compute_units);
    }
  };

  // with every device
  for (auto &device : devices) {

    auto const device_name   = device.get_info<sycl::info::device::name>();
    auto const compute_units = device.get_info<sycl::info::device::max_compute_units>();

    logger::info() << "Running on SYCL: " << device_name << " with " << compute_units
                   << " compute units" << std::endl;

    auto queue = sycl::queue{device};
    test_sycl(queue, compute_units);

    // === in-order queue ===
    logger::info() << "Running on SYCL (in-order): " << device_name << " with " << compute_units
                   << " compute units" << std::endl;

    queue = sycl::queue{device, sycl::property::queue::in_order{}};
    test_sycl(queue, compute_units);
  }

  return EXIT_SUCCESS;
}
