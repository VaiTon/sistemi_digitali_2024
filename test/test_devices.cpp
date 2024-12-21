#include "util.hpp"

#include <cstdlib>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "buf/kmeans_buf.hpp"
#include "cpu/kmeans_cpu.hpp"
#include "simd/kmeans_simd.hpp"
#include "usm/kmeans_usm.hpp"

template <typename T>
auto time_and_print(const std::string &name, T &km, size_t max_iter, double tol)
    -> decltype(km.cluster(max_iter, tol), void()) {

  const std::string class_name = typeid(T).name();
  logger::info() << "Running kmeans on " << name << " <class " << class_name << ">\n";

  const auto start_time = std::chrono::high_resolution_clock::now();
  km.cluster(max_iter, tol);
  const auto end_time = std::chrono::high_resolution_clock::now();

  const long time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  std::cerr << "\ttime: " << time << "ms" << ", iterations: " << km.get_iters() << std::endl;
  return;
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
  {
    auto kmeans = kmeans_omp{k, data};
    time_and_print("CPU", kmeans, max_iter, tol);
  }

  {
    auto kmeans = kmeans_simd{k, data};
    time_and_print("CPU (SIMD)", kmeans, max_iter, tol);
  }

  // with every device
  for (auto &device : devices) {
    auto device_name = device.get_info<sycl::info::device::name>();
    // auto device_vendor = device.get_info<sycl::info::device::vendor>();

    const auto q = sycl::queue{device};

    {
      auto kmeans = kmeans_usm_v2{q, k, data};
      time_and_print(device_name + " (USM, v2)", kmeans, max_iter, tol);
    }
    {
      auto kmeans = kmeans_usm_v3{q, k, data};
      time_and_print(device_name + " (USM, v3)", kmeans, max_iter, tol);
    }

    // {
    //   auto kmeans = kmeans_buf{q, k, data};
    //   time_and_print(device_name, kmeans, max_iter, tol);
    // }

    auto inorder_q = sycl::queue{device, sycl::property::queue::in_order{}};

    {
      auto kmeans = kmeans_usm_v2{inorder_q, k, data};
      time_and_print(device_name + " (USM, v2, in-order)", kmeans, max_iter, tol);
    }
    {
      auto kmeans = kmeans_usm_v3{inorder_q, k, data};
      time_and_print(device_name + " (USM, v3, in-order)", kmeans, max_iter, tol);
    }

    // {
    //   auto kmeans = kmeans_buf{inorder_q, k, data};
    //   time_and_print(device_name + " (Buf, in-order)", kmeans, max_iter, tol);
    // }
  }

  return EXIT_SUCCESS;
}
