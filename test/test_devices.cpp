#include "util.hpp"

#include <cstdlib>
#include <iosfwd>
#include <iostream>
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

template <typename T>
auto time_and_print(const std::string &name, T &km, size_t max_iter, double tol, long comp_time = 0,
                    size_t computation_units = 1) -> decltype(km.cluster(max_iter, tol), long()) {

  const std::string class_name = typeid(T).name();
  logger::info() << std::format("{:60.60}", name)              //
                 << std::format("<class {:10.10}", class_name) //
                 << ">\t";

  const auto start_time = std::chrono::high_resolution_clock::now();
  km.cluster(max_iter, tol);
  const auto end_time = std::chrono::high_resolution_clock::now();

  const long time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  std::cerr << std::format("time: {:5} ms", time) //
            << std::format(", iterations: {:3}", km.get_iters());

  if (comp_time > 0) {
    float speedup = static_cast<float>(comp_time) / static_cast<float>(time);
    std::cerr << std::format(", speedup: {:5.2f}", speedup);

    if (computation_units > 1) {
      float efficiency = speedup / static_cast<float>(computation_units);
      std::cerr << std::format(", efficiency: {:5.2f}", efficiency);
    }
  }

  std::cerr << std::endl;
  return time;
}

int main(const int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <data.csv> <k>" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr double tol      = 1e-4;
  constexpr size_t max_iter = 1000;

  const auto input_filename = std::string{argv[1]};
  const auto k              = static_cast<size_t>(std::stoi(argv[2]));

  // read data from csv file
  const auto data = get_data(input_filename);
  logger::info() << "Data size: " << data.size() << std::endl;

  auto devices = sycl::device::get_devices();
  logger::info() << "Found " << devices.size() << " devices" << std::endl;
  for (const auto &device : devices) {
    auto dev_name   = device.get_info<sycl::info::device::name>();
    auto dev_vendor = device.get_info<sycl::info::device::vendor>();
    logger::info() << "  - " << dev_name << " (" << dev_vendor << ")" << std::endl;
  }

  long ref_time;

  {
    auto kmeans = kmeans_cpu_v1{k, data};
    ref_time    = time_and_print("CPU (v1)", kmeans, max_iter, tol);
  }
  {
    auto kmeans = kmeans_cpu_v2{k, data};
    time_and_print("CPU (v2)", kmeans, max_iter, tol, ref_time);
  }
  {
    auto kmeans      = kmeans_omp{k, data};
    auto max_threads = omp_get_max_threads();

    omp_set_num_threads(1);
    time_and_print("OpenMP (1 thread)", kmeans, max_iter, tol, ref_time);

    omp_set_num_threads(max_threads);
    time_and_print("OpenMP (" + std::to_string(max_threads) + " threads)", kmeans, max_iter, tol,
                   ref_time, max_threads);
  }
  {
    auto kmeans = kmeans_simd{k, data};
    time_and_print("CPU (SIMD)", kmeans, max_iter, tol, ref_time);
  }
  {
    auto kmeans = kmeans_ocv{k, data};
    time_and_print("OpenCV Universal Intrinsics", kmeans, max_iter, tol, ref_time);
  }

  // with every device
  for (auto &device : devices) {
    auto device_name = device.get_info<sycl::info::device::name>();

    const auto q = sycl::queue{device};

    {
      auto kmeans = kmeans_usm_v2{q, k, data};
      time_and_print(device_name + " (USM, v2)", kmeans, max_iter, tol, ref_time,
                     omp_get_max_threads());
    }
    {
      auto kmeans = kmeans_usm_v3{q, k, data};
      time_and_print(device_name + " (USM, v3)", kmeans, max_iter, tol, ref_time,
                     omp_get_max_threads());
    }

    {
      auto kmeans = kmeans_buf{q, k, data};
      time_and_print(device_name + " (Buf)", kmeans, max_iter, tol, ref_time,
                     omp_get_max_threads());
    }

    auto inorder_q = sycl::queue{device, sycl::property::queue::in_order{}};

    {
      auto kmeans = kmeans_usm_v2{inorder_q, k, data};
      time_and_print(device_name + " (USM, v2, in-order)", kmeans, max_iter, tol, ref_time,
                     omp_get_max_threads());
    }
    {
      auto kmeans = kmeans_usm_v3{inorder_q, k, data};
      time_and_print(device_name + " (USM, v3, in-order)", kmeans, max_iter, tol, ref_time,
                     omp_get_max_threads());
    }

    {
      auto kmeans = kmeans_buf{inorder_q, k, data};
      time_and_print(device_name + " (Buf, in-order)", kmeans, max_iter, tol, ref_time,
                     omp_get_max_threads());
    }
  }

  return EXIT_SUCCESS;
}
