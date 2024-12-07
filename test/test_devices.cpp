#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#include "../src/kmeans.hpp"

int main(const int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <data.csv> <k> [CPU/GPU]" << std::endl;
    return EXIT_FAILURE;
  }

  const auto input_filename = std::string{argv[1]};
  const auto k              = std::stoi(argv[2]);

  const auto device_type = argc == 4 ? std::string{argv[3]} : "";
  if (!device_type.empty() && device_type != "CPU" && device_type != "GPU") {
    std::cerr << "Invalid device type: " << device_type << std::endl;
    return EXIT_FAILURE;
  }

  // read data from csv file
  const auto data = get_data(input_filename);
  std::clog << "--- Data size: " << data.size() << std::endl;

  auto devices = sycl::device::get_devices();
  std::cerr << "--- Found " << devices.size() << " devices" << std::endl;
  for (const auto &device : devices) {
    auto dev_name   = device.get_info<sycl::info::device::name>();
    auto dev_vendor = device.get_info<sycl::info::device::vendor>();
    std::cerr << "  - " << dev_name << " (" << dev_vendor << ")" << std::endl;
  }

  if (device_type.empty() || device_type == "CPU") {
    std::cerr << "--- Running kmeans on CPU" << std::endl;
    const auto start_time = std::chrono::high_resolution_clock::now();
    kmeans_cpu(k, data, 1000, 1e-4);
    const auto end_time = std::chrono::high_resolution_clock::now();

    std::cerr << " Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
              << "ms" << std::endl;
  }

  // with every device
  for (auto &device : devices) {
    auto device_name   = device.get_info<sycl::info::device::name>();
    auto device_vendor = device.get_info<sycl::info::device::vendor>();

    const auto q = sycl::queue{device};

    std::cerr << "--- Running kmeans on SYCL (" << SYCL_TYPE << ") on device: " << device_name << " (" << device_vendor
              << ")" << std::endl;

    const auto start_time = std::chrono::high_resolution_clock::now();
    kmeans_sycl(q, k, data, 1000, 1e-4);
    const auto end_time = std::chrono::high_resolution_clock::now();

    std::cerr << " Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
              << "ms" << std::endl;
  }

  return EXIT_SUCCESS;
}
