#include <cstddef>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <vector>

#include "kmeans.hpp"

using namespace sycl;

class BufAssignPointsKernel;
class BufPartialReductionKernel;
class BufFinalReductionKernel;
class BufConvergedKernel;
class BufUpdateCentroidsKernel;

kmeans_cluster_t kmeans_sycl(queue q, size_t k, const std::vector<Point> &points, size_t max_iter, float tol) {
  if (k > points.size()) {
    throw std::invalid_argument("Number of clusters must be less than or equal "
                                "to the number of points");
  }

  // tolerance must be squared
  tol *= tol;

  const auto points_n = points.size();

  // Step 0: Initialize centroids
  // For simplicity, let's assume the first k points are the initial centroids.
  auto centroids_h = std::vector<Point>{k};
  std::copy_n(points.begin(), k, centroids_h.begin());

  // Create the relevant buffers to be used in the computation
  auto points_b        = buffer{points};
  auto centroids_b     = buffer{centroids_h};
  auto new_centroids_b = buffer<Point>{k};
  auto assoc_b         = buffer<size_t>{points.size()};

  const size_t work_group_size = q.get_device().get_info<info::device::max_work_group_size>();
  const size_t num_work_groups = (points_n + work_group_size - 1) / work_group_size;

  auto partial_sums_x = buffer<double>{num_work_groups * k};
  auto partial_sums_y = buffer<double>{num_work_groups * k};
  auto partial_counts = buffer<size_t>{num_work_groups * k};

  // main cycle: assign points to clusters, calculate new centroids, check for
  // convergence
  auto converged = bool{false};
  auto iter      = size_t{0};

  while (!converged && iter < max_iter) {

    // Step 1: Assign points to clusters
    q.submit([&](handler &h) {
      const auto points_a    = points_b.get_access<access::mode::read>(h);
      const auto centroids_a = centroids_b.get_access<access::mode::read>(h);
      const auto assoc_a     = assoc_b.get_access<access::mode::write>(h);

      // for each point, calculate the distance to each centroid and assign the
      // point to the closest one

      h.parallel_for<BufAssignPointsKernel>(points_n, [=](const size_t item) {
        double min_val = std::numeric_limits<double>::max();
        size_t min_idx = 0;

        for (size_t i = 0; i < k; i++) {
          const auto dist = squared_distance(points_a[item], centroids_a[i]);
          if (dist < min_val) {
            min_val = dist;
            min_idx = i;
          }
        }

        assoc_a[item] = min_idx;
      });
    });

    // Step 2.1: Parallel reduction over points
    q.submit([&](handler &h) {
      auto global_range = range{points_n};
      if (points_n % work_group_size != 0) { // the global range must be a multiple of the work group size
        global_range = range{(points_n / work_group_size + 1) * work_group_size};
      }
      const auto execution_range = nd_range{global_range, range(work_group_size)};

      const auto assoc_a  = assoc_b.get_access<access_mode::read>();
      const auto points_a = points_b.get_access<access_mode::read>();

      const auto partial_sums_x_a = partial_sums_x.get_access<access_mode::read_write>();
      const auto partial_sums_y_a = partial_sums_y.get_access<access_mode::read_write>();
      const auto partial_counts_a = partial_counts.get_access<access_mode::read_write>();

      h.parallel_for<BufPartialReductionKernel>(execution_range, [=](const nd_item<> &item) {
        const auto p_idx = item.get_global_id(0);
        if (p_idx >= points_n)
          return; // Out-of-bounds guard

        const auto wg_idx = item.get_group(0); // Workgroup index
        const auto k_idx  = assoc_a[p_idx];    // Cluster index

        // Use atomic operations to update partial results
        const auto x_atomic =
            atomic_ref<double, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(
                partial_sums_x_a[wg_idx * k + k_idx]);
        const auto y_atomic =
            atomic_ref<double, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(
                partial_sums_y_a[wg_idx * k + k_idx]);
        const auto count_atomic =
            atomic_ref<size_t, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(
                partial_counts_a[wg_idx * k + k_idx]);

        x_atomic += static_cast<double>(points_a[p_idx].x);
        y_atomic += static_cast<double>(points_a[p_idx].y);
        count_atomic += size_t{1};
      });
    });

    // Step 2.2: Final reduction and compute centroids
    q.submit([&](handler &h) {
      const auto partial_sums_x_a = partial_sums_x.get_access<access_mode::read>();
      const auto partial_sums_y_a = partial_sums_y.get_access<access_mode::read>();
      const auto partial_counts_a = partial_counts.get_access<access_mode::read>();

      const auto centroids_a     = centroids_b.get_access<access_mode::read>();
      const auto new_centroids_a = new_centroids_b.get_access<access_mode::write>();

      h.parallel_for<BufFinalReductionKernel>(range(k), [=](const id<> k_idx) {
        auto final_x     = double{0.0};
        auto final_y     = double{0.0};
        auto final_count = size_t{0};

        // Accumulate results from all workgroups
        for (auto wg = size_t{0}; wg < num_work_groups; ++wg) {
          final_x += partial_sums_x_a[wg * k + k_idx];
          final_y += partial_sums_y_a[wg * k + k_idx];
          final_count += partial_counts_a[wg * k + k_idx];
        }

        // Compute the final centroid
        if (final_count > 0) {
          final_x /= static_cast<double>(final_count);
          final_y /= static_cast<double>(final_count);
          new_centroids_a[k_idx] = Point{static_cast<float>(final_x), static_cast<float>(final_y)};
        } else {
          // No points in cluster, centroid remains unchanged
          new_centroids_a[k_idx] = centroids_a[k_idx];
        }
      });
    });

    auto converged_b = buffer<int>{1};

    converged_b.get_host_access()[0] = true;

    q.submit([&](handler &h) {
      auto new_centroids_a = new_centroids_b.get_access<access::mode::read>(h);
      auto centroids_a     = centroids_b.get_access<access::mode::read>(h);

      const auto converged_reduction = reduction(converged_b, h, logical_and{});

      h.parallel_for<BufConvergedKernel>(k, converged_reduction, [=](const auto item, auto &converged_ref) {
        auto dist = squared_distance(new_centroids_a[item], centroids_a[item]);
        converged_ref.combine(dist < tol);
      });
    });

    converged = converged_b.get_host_access()[0];

    // if not converged, update the centroids
    if (!converged) {
      q.submit([&](handler &h) {
        const auto new_acc = new_centroids_b.get_access<access::mode::read>(h);
        const auto acc     = centroids_b.get_access<access::mode::write>(h);

        h.parallel_for<BufUpdateCentroidsKernel>(k, [=](auto item) { acc[item] = new_acc[item]; });
      });
    }

    iter++;
  }

  const auto centroids_a    = centroids_b.get_host_access();
  const auto associations_a = assoc_b.get_host_access();

  auto final_clusters = std::vector<std::vector<Point>>{centroids_a.size()};
  for (size_t i = 0; i < k; i++) {
    final_clusters[i] = std::vector<Point>{};
  }

  for (size_t i = 0; i < points.size(); i++) {
    final_clusters[associations_a[i]].push_back(points[i]);
  }

  // copy from accessors to host memory
  auto final_centroids = std::vector<Point>{centroids_a.begin(), centroids_a.end()};

  return kmeans_cluster_t{
      .centroids = final_centroids,
      .clusters  = final_clusters,
  };
}
