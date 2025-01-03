#include "kmeans_buf.hpp"
#include "kmeans.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace sycl;

kmeans_cluster_t kmeans_buf::cluster(size_t const max_iter, double const tol) {
  if (num_centroids > points.size()) {
    throw std::invalid_argument("Number of clusters must be less than or equal "
                                "to the number of points");
  }

  auto const points_n      = points.size();
  auto const num_centroids = this->num_centroids;

  // Step 0: Initialize centroids
  // For simplicity, let's assume the first k points are the initial centroids.
  auto centroids_h = std::vector<point_t>{num_centroids};
  std::copy_n(points.begin(), num_centroids, centroids_h.begin());

  // Create the relevant buffers to be used in the computation
  auto points_buf        = buffer{points};
  auto centroids_buf     = buffer{centroids_h};
  auto new_centroids_buf = buffer<point_t>{num_centroids};
  auto assoc_buf         = buffer<size_t>{points.size()};
  auto converged_buf     = buffer<int>{1};

  size_t const work_group_size = q.get_device().get_info<info::device::max_work_group_size>();
  size_t const num_work_groups = (points_n + work_group_size - 1) / work_group_size;

  auto partial_sums_x_d = buffer<double>{num_work_groups * num_centroids};
  auto partial_sums_y_d = buffer<double>{num_work_groups * num_centroids};
  auto partial_counts_d = buffer<size_t>{num_work_groups * num_centroids};

  // main cycle: assign points to clusters, calculate new centroids, check for
  // convergence
  for (iter = 0; iter < max_iter; iter++) {

    // Step 1: Assign points to clusters
    q.submit([&](handler &h) {
      auto const assoc     = assoc_buf.get_access<access_mode::write>(h);
      auto const points    = points_buf.get_access<access_mode::read>(h);
      auto const centroids = centroids_buf.get_access<access_mode::read>(h);

      // for each point, calculate the distance to each centroid and assign the
      // point to the closest one

      h.parallel_for<class kmeans_assign_points>(points_n, [=](size_t const item) {
        auto   min_val = std::numeric_limits<double>::max();
        size_t min_idx = 0;

        for (size_t i = 0; i < num_centroids; i++) {
          auto const dist = squared_distance(points[item], centroids[i]);
          if (dist < min_val) {
            min_val = dist;
            min_idx = i;
          }
        }

        assoc[item] = min_idx;
      });
    });

    q.submit([&](handler &h) {
      auto acc = partial_sums_x_d.get_access<access_mode::write>(h);
      h.fill<double>(acc, 0.0);
    });
    q.submit([&](handler &h) {
      h.fill<double>(partial_sums_y_d.get_access<access_mode::write>(h), 0.0);
    });
    q.submit([&](handler &h) { //
      h.fill<size_t>(partial_counts_d.get_access<access_mode::write>(h), 0);
    });

    q.wait();

    // Step 2.1: Parallel reduction over points
    auto parallel_reduction_ev = q.submit([&](handler &h) {
      auto global_range = range{points_n};
      if (points_n % work_group_size !=
          0) { // the global range must be a multiple of the work group size
        global_range = range{(points_n / work_group_size + 1) * work_group_size};
      }
      auto const execution_range = nd_range{global_range, range(work_group_size)};

      auto const assoc  = assoc_buf.get_access<access_mode::read>();
      auto const points = points_buf.get_access<access_mode::read>();

      auto const partial_sums_x = partial_sums_x_d.get_access<access_mode::read_write>();
      auto const partial_sums_y = partial_sums_y_d.get_access<access_mode::read_write>();
      auto const partial_counts = partial_counts_d.get_access<access_mode::read_write>();

      h.parallel_for<class kmeans_partial_reduction>(execution_range, [=](nd_item<> const &item) {
        auto const p_idx = item.get_global_id(0);
        if (p_idx >= points_n) {
          return; // Out-of-bounds guard
        }

        size_t const wg_idx   = item.get_group(0);              // Workgroup index
        size_t const k_idx    = assoc[p_idx];                   // Cluster index
        size_t const sums_idx = wg_idx * num_centroids + k_idx; // Index in partial sums

        constexpr auto mem_order = memory_order::relaxed;
        constexpr auto mem_scope = memory_scope::device;
        constexpr auto mem_space = access::address_space::global_space;

        using double_atomic_ref = atomic_ref<double, mem_order, mem_scope, mem_space>;
        using size_t_atomic_ref = atomic_ref<size_t, mem_order, mem_scope, mem_space>;

        // Use atomic operations to update partial results

        auto const atom_x = double_atomic_ref(partial_sums_x[sums_idx]);
        auto const atom_y = double_atomic_ref(partial_sums_y[sums_idx]);
        auto const atom_c = size_t_atomic_ref(partial_counts[sums_idx]);

        atom_x += static_cast<double>(points[p_idx].x);
        atom_y += static_cast<double>(points[p_idx].y);
        atom_c += size_t{1};
      });
    });

    q.wait();

    // Step 2.2: Final reduction and compute centroids
    q.submit([&](handler &h) {
      auto const partial_sums_x = partial_sums_x_d.get_access<access_mode::read>();
      auto const partial_sums_y = partial_sums_y_d.get_access<access_mode::read>();
      auto const partial_counts = partial_counts_d.get_access<access_mode::read>();
      auto const centroids      = centroids_buf.get_access<access_mode::read>();

      auto const new_centroids = new_centroids_buf.get_access<access_mode::write>();

      h.parallel_for<class kmeans_final_reduction>(num_centroids, [=](size_t const k_idx) {
        double final_x     = 0.0;
        double final_y     = 0.0;
        size_t final_count = 0;

        // Accumulate results from all workgroups
        for (size_t wg = 0; wg < num_work_groups; ++wg) {
          final_x += partial_sums_x[wg * num_centroids + k_idx];
          final_y += partial_sums_y[wg * num_centroids + k_idx];
          final_count += partial_counts[wg * num_centroids + k_idx];
        }

        if (final_count <= 0) {
          // No points in cluster, centroid remains unchanged
          new_centroids[k_idx] = centroids[k_idx];
          return;
        }

        // Compute the final centroid
        final_x /= static_cast<double>(final_count);
        final_y /= static_cast<double>(final_count);

        new_centroids[k_idx] = point_t{
            static_cast<float>(final_x),
            static_cast<float>(final_y),
        };
      });
    });

    // Check for convergence
    converged_buf.get_host_access()[0] = true;

    q.wait();

    q.submit([&](handler &h) {
      auto const new_centroids = new_centroids_buf.get_access<access_mode::read>(h);
      auto const centroids     = centroids_buf.get_access<access_mode::read>(h);

      auto const converged_reduction = reduction(converged_buf, h, logical_and{});

      auto reduction_kernel = [=](size_t const item, auto &converged_ref) {
        auto const dist = squared_distance(new_centroids[item], centroids[item]);
        converged_ref.combine(dist < tol);
      };

      h.parallel_for<class kmeans_converged>(num_centroids, converged_reduction, reduction_kernel);
    });

    // Update centroids
    q.submit([&](handler &h) {
      auto const new_centroids = new_centroids_buf.get_access<access::mode::read>(h);
      auto const centroids     = centroids_buf.get_access<access::mode::write>(h);

      h.parallel_for<class kmeans_update_centroids>(
          num_centroids, [=](size_t const idx) { centroids[idx] = new_centroids[idx]; });
    });

    if (converged_buf.get_host_access()[0]) {
      break;
    }

    q.wait();
  }

  q.wait_and_throw();
  auto const centroids_acc    = centroids_buf.get_host_access();
  auto const associations_acc = assoc_buf.get_host_access();

  auto final_clusters = std::vector<std::vector<point_t>>{centroids_acc.size()};
  for (size_t i = 0; i < num_centroids; i++) {
    final_clusters[i] = std::vector<point_t>{};
  }

  for (size_t i = 0; i < points.size(); i++) {
    auto cluster_idx = associations_acc[i];
    final_clusters[cluster_idx].push_back(points[i]);
  }

  // copy from accessors to host memory
  auto final_centroids = std::vector<point_t>{centroids_acc.begin(), centroids_acc.end()};

  return kmeans_cluster_t{final_centroids, final_clusters};
}
