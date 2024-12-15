#include "kmeans_usm.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace sycl;

#ifdef ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
constexpr auto WARP_SIZE = 32;
#else
constexpr auto WARP_SIZE = 128;
#endif

/// Divide n by m and round up to the nearest integer by excess
size_t divceil(size_t n, size_t m) { return (n + m - 1) / m; }

/// Returns the smallest multiple of m greater or equal to n
/// i.e., pad_to_multiple_of(5, 3) == 6
size_t pad_to_multiple_of(size_t n, size_t m) { return divceil(n, m) * m; }

struct assign_points_to_clusters_kernel {
  const point_t *points;
  const size_t   num_points;

  const point_t *centroids;
  const size_t   num_centroids;

  size_t *assoc;

  void operator()(const nd_item<> &item) const {
    const auto num_points = this->num_points;      // copy to register
    const auto p_idx      = item.get_global_id(0); // point index
    const auto k          = this->num_centroids;   // copy to register

    if (p_idx >= num_points) {
      return; // Out-of-bounds guard
    }

    auto   min_val = std::numeric_limits<double>::max();
    size_t min_idx = 0;

    const auto point = points[p_idx];

    for (size_t c_idx = 0; c_idx < k; c_idx++) {

      const auto centroid = centroids[c_idx];
      const auto dist     = squared_distance(point, centroid);

      if (dist < min_val) {
        min_val = dist;
        min_idx = c_idx;
      }
    }

    assoc[p_idx] = min_idx;
  }

private:
  static inline float squared_distance(const point_t &lhs, const point_t &rhs) {
    const auto dx = lhs.x - rhs.x;
    const auto dy = lhs.y - rhs.y;

    return dx * dx + dy * dy;
  }
};

struct partial_reduction_kernel {

  const size_t num_centroids;
  const size_t num_points;

  const point_t *points;
  const size_t  *associations;

  double *partial_sums_x;
  double *partial_sums_y;
  size_t *partial_counts;

  // Local storage for partial sums
  sycl::local_accessor<double> local_sums_x;
  sycl::local_accessor<double> local_sums_y;
  sycl::local_accessor<size_t> local_counts;

  void operator()(const nd_item<> &item) const {
    const auto p_idx      = item.get_global_id(0);
    const auto local_id   = item.get_local_id(0);
    const auto group_id   = item.get_group(0);
    const auto group_size = item.get_local_range(0);

    if (p_idx >= num_points) {
      return;
    }

    const auto centroid_idx   = associations[p_idx];
    const auto local_sums_idx = local_id * num_centroids + centroid_idx;

    // Initialize local sums
    if (local_id == 0) {
      for (size_t i = 0; i < group_size * num_centroids; ++i) {
        local_sums_x[i] = 0.0;
        local_sums_y[i] = 0.0;
        local_counts[i] = 0;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    const auto point  = points[p_idx];
    const auto [x, y] = point;

    // Local reduction
    local_sums_x[local_sums_idx] += static_cast<double>(x);
    local_sums_y[local_sums_idx] += static_cast<double>(y);
    local_counts[local_sums_idx]++;

    item.barrier(sycl::access::fence_space::local_space);

    // Write local sums to global memory (only one thread per local sum)
    if (local_id == 0) {
      const auto global_sums_idx = group_id * num_centroids + centroid_idx;

      constexpr auto order = memory_order::relaxed;
      constexpr auto scope = memory_scope::device;
      constexpr auto as    = access::address_space::global_space;

      auto atom_x     = atomic_ref<double, order, scope, as>{partial_sums_x[global_sums_idx]};
      auto atom_y     = atomic_ref<double, order, scope, as>{partial_sums_y[global_sums_idx]};
      auto atom_count = atomic_ref<size_t, order, scope, as>{partial_counts[global_sums_idx]};

      atom_x += local_sums_x[local_sums_idx];
      atom_y += local_sums_y[local_sums_idx];
      atom_count += local_counts[local_sums_idx];
    }
  }
};

struct final_reduction_kernel {
  size_t num_work_groups;
  size_t num_centroids;

  double *partial_sums_x;
  double *partial_sums_y;
  size_t *partial_counts;

  point_t *new_centroids;
  point_t *centroids;

  void operator()(const size_t k_idx) const {
    const auto num_centroids   = this->num_centroids;   // copy to register
    const auto num_work_groups = this->num_work_groups; // copy to register

    double final_x     = 0;
    double final_y     = 0;
    size_t final_count = 0;

    // Accumulate results from all workgroups
    for (size_t wg = 0; wg < num_work_groups; ++wg) {
      const auto idx = wg * num_centroids + k_idx;
      final_x += partial_sums_x[idx];
      final_y += partial_sums_y[idx];
      final_count += partial_counts[idx];
    }

    if (final_count == 0) {
      // No points in cluster, centroid remains unchanged
      new_centroids[k_idx] = centroids[k_idx];
      return;
    }

    // Compute the final centroid
    final_x /= static_cast<double>(final_count);
    final_y /= static_cast<double>(final_count);

    // Cast again to float
    new_centroids[k_idx] = point_t{static_cast<float>(final_x), static_cast<float>(final_y)};
  }
};

struct check_convergence_kernel {
  size_t k;
  double tol;

  bool    *converged;
  point_t *centroids;
  point_t *new_centroids;

  void operator()() const {

    // tolerance must be squared
    const auto tol_sq = tol * tol;

    bool conv = true;
    for (size_t i = 0; i < k; i++) {
      if (squared_distance(centroids[i], new_centroids[i]) >= tol_sq) {
        conv = false;
        break;
      }
    }

    *converged = conv;
  }
};

kmeans_cluster_t kmeans_usm::cluster(const size_t max_iter, const double tol) {

  if (k > points.size()) {
    throw std::invalid_argument("Number of clusters (" + std::to_string(k) +
                                ") must be less than the number of points (" +
                                std::to_string(points.size()) + ")");
  }

  const auto points_n = points.size(); // size() should not be expensive, but cache it anyway
  const auto k        = this->k;       // copy to stack

  // Points
  const auto points_d = malloc_device<point_t>(points.size(), q);
  assert(points_d != nullptr);

  // Centroids
  const auto centroids_d = malloc_device<point_t>(k, q);
  assert(centroids_d != nullptr);

  // Updated centroids after each iteration
  const auto new_centroids_d = malloc_device<point_t>(k, q);
  assert(new_centroids_d != nullptr);

  // Association of each point to a cluster
  const auto assoc_d = malloc_device<size_t>(points.size(), q);
  assert(assoc_d != nullptr);

  const auto converged = malloc_host<bool>(1, q); // Convergence flag
  assert(converged != nullptr);

  const size_t max_work_group_size = q.get_device().get_info<info::device::max_work_group_size>();

  // Partial reduction kernel execution parameters
  size_t kernel_partial_local_size;
  {
    // try warp_size threads per workgroup
    size_t nsight_wg_size = 256;

    size_t local_size = points_n / WARP_SIZE; // try to use a warp per workgroup
    local_size        = std::min(local_size, max_work_group_size); // limit to device max wg size
    local_size        = std::min(local_size, nsight_wg_size);      // limit to nsight wg size

    kernel_partial_local_size = local_size;
  }

  // number of workgroups
  const size_t kernel_partial_wgs = divceil(points_n, kernel_partial_local_size);

  // allocate shared memory for partial sums and counts
  const auto partial_sums_x = malloc_shared<double>(kernel_partial_wgs * k, q);
  const auto partial_sums_y = malloc_shared<double>(kernel_partial_wgs * k, q);
  const auto partial_counts = malloc_shared<size_t>(kernel_partial_wgs * k, q);
  assert(partial_sums_x != nullptr && partial_sums_y != nullptr && partial_counts != nullptr);

  // == Kernel ranges

  // copy points to device memory
  // consider the first k points as the initial centroids
  q.copy(points.data(), points_d, points.size());
  q.copy(points.data(), centroids_d, k);
  *converged = false;
  q.wait();

  // main cycle: assign points to clusters, calculate new centroids, check for
  // convergence

  for (iter = 0; iter < max_iter; iter++) {
    // Step 1: Assign points to clusters
    // For each point, calculate the distance to each centroid and assign the point to the closest
    // one

    {
      constexpr auto wg_size = 256;
      const auto     g_size  = pad_to_multiple_of(points_n, wg_size);

      const auto exec_range = nd_range{range{g_size}, range{wg_size}};

      auto kernel = assign_points_to_clusters_kernel{points_d, points_n, centroids_d, k, assoc_d};
      q.parallel_for(exec_range, kernel).wait();
    }

    // Step 2: calculate new centroids by averaging the points in each cluster
    //
    // HAHA: SE NON CASTIAMO A DOUBLE, IL RISULTATO E' SBAGLIATO
    // q.fill<double>(partial_sums_x, 0.0, num_work_groups * k);
    q.memset(partial_sums_x, 0, kernel_partial_wgs * k * sizeof(double));
    // q.fill<double>(partial_sums_y, 0.0, num_work_groups * k);
    q.memset(partial_sums_y, 0, kernel_partial_wgs * k * sizeof(double));
    // q.fill<size_t>(partial_counts, 0, num_work_groups * k);
    q.memset(partial_counts, 0, kernel_partial_wgs * k * sizeof(size_t));
    q.wait();

    // Step 2.1: Parallel reduction over points
    {

      auto exec_range = nd_range{
          range{pad_to_multiple_of(points_n, kernel_partial_local_size)}, // global
          range{kernel_partial_local_size},                               // local
      };

      q.submit([&](handler &h) {
         local_accessor<double> local_sums_x{kernel_partial_local_size * k, h};
         local_accessor<double> local_sums_y{kernel_partial_local_size * k, h};
         local_accessor<size_t> local_counts{kernel_partial_local_size * k, h};

         auto kernel = partial_reduction_kernel{
             .num_centroids = k,
             .num_points    = points_n,

             .points       = points_d,
             .associations = assoc_d,

             .partial_sums_x = partial_sums_x,
             .partial_sums_y = partial_sums_y,
             .partial_counts = partial_counts,

             .local_sums_x = local_sums_x,
             .local_sums_y = local_sums_y,
             .local_counts = local_counts,
         };
         h.parallel_for(exec_range, kernel);
       }).wait();
    }

    // Step 2.2: Final reduction and compute centroids
    {
      auto kernel = final_reduction_kernel{
          .num_work_groups = kernel_partial_wgs,
          .num_centroids   = k,
          .partial_sums_x  = partial_sums_x,
          .partial_sums_y  = partial_sums_y,
          .partial_counts  = partial_counts,
          .new_centroids   = new_centroids_d,
          .centroids       = centroids_d,
      };
      q.parallel_for(k, kernel).wait();
    }

    // Step 3: Check for convergence
    {
      auto kernel = check_convergence_kernel{k, tol, converged, centroids_d, new_centroids_d};
      q.single_task(kernel).wait();
    }

    q.copy(new_centroids_d, centroids_d, k).wait();

    if (*converged) {
      break;
    }
  }

  auto centroids_h    = std::vector<point_t>(k);
  auto associations_h = std::vector<size_t>(points.size());

  // copy results back to host
  q.copy(centroids_d, centroids_h.data(), k);
  q.copy(assoc_d, associations_h.data(), points.size());

  q.wait_and_throw();

  // free device memory
  sycl::free(points_d, q);
  sycl::free(centroids_d, q);
  sycl::free(new_centroids_d, q);
  sycl::free(assoc_d, q);
  sycl::free(converged, q);
  sycl::free(partial_sums_x, q);
  sycl::free(partial_sums_y, q);
  sycl::free(partial_counts, q);

  auto clusters_h = std::vector<std::vector<point_t>>{centroids_h.size()};
  std::fill(clusters_h.begin(), clusters_h.end(), std::vector<point_t>{});

  // group points by cluster
  for (size_t i = 0; i < points.size(); i++) {
    clusters_h[associations_h[i]].push_back(points[i]);
  }

  return kmeans_cluster_t{
      .centroids = centroids_h,
      .clusters  = clusters_h,
  };
}
