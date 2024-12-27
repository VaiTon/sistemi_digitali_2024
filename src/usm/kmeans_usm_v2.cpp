#include "kmeans_usm.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace sycl;

struct assign_points_to_clusters_kernel_v2 {

  const point_t *points;
  const point_t *centroids;
  const size_t   num_centroids;
  size_t        *associations;

  void operator()(const size_t p_idx) const {
    auto   min_val = std::numeric_limits<double>::max();
    size_t min_idx = 0;

    for (size_t c_idx = 0; c_idx < num_centroids; c_idx++) {
      const auto dist = squared_distance(points[p_idx], centroids[c_idx]);

      if (dist < min_val) {
        min_val = dist;
        min_idx = c_idx;
      }
    }

    associations[p_idx] = min_idx;
  }
};

struct partial_reduction_kernel_v2 {

  const size_t   num_centroids;
  const size_t   num_points;
  const point_t *points;
  const size_t  *associations;

  double *partial_sums_x;
  double *partial_sums_y;
  size_t *partial_counts;

  void operator()(const nd_item<> &item) const {
    const auto p_idx = item.get_global_id(0); // Point index

    if (p_idx >= num_points) {
      return; // Out-of-bounds guard
    }

    const auto wg_idx   = item.get_group(0);              // Workgroup index
    const auto k_idx    = associations[p_idx];            // Cluster index
    const auto sums_idx = wg_idx * num_centroids + k_idx; // Index in partial sums

    const auto custom_atomic_ref = []<typename T>(T &ptr) {
      constexpr auto mem_order = memory_order::relaxed;
      constexpr auto mem_scope = memory_scope::device; // work_group is not supported for SM_35
      constexpr auto mem_space = access::address_space::global_space;

      return atomic_ref<T, mem_order, mem_scope, mem_space>{ptr};
    };

    // Use atomic operations to update partial results
    const auto atom_x = custom_atomic_ref(partial_sums_x[sums_idx]);
    const auto atom_y = custom_atomic_ref(partial_sums_y[sums_idx]);
    const auto atom_c = custom_atomic_ref(partial_counts[sums_idx]);

    atom_x += static_cast<double>(points[p_idx].x);
    atom_y += static_cast<double>(points[p_idx].y);
    atom_c += size_t{1};
  }
};

struct final_reduction_kernel_v2 {
  const size_t num_work_groups;
  const size_t num_centroids;

  const double  *partial_sums_x;
  const double  *partial_sums_y;
  const size_t  *partial_counts;
  const point_t *centroids;

  point_t *new_centroids;

  void operator()(const size_t k_idx) const {
    double final_x     = 0;
    double final_y     = 0;
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

    // Cast again to float
    new_centroids[k_idx] = point_t{
        .x = static_cast<float>(final_x),
        .y = static_cast<float>(final_y),
    };
  }
};

struct check_convergence_kernel_v2 {
  const size_t   num_centroids;
  const double   tol;
  const point_t *centroids;
  const point_t *new_centroids;

  bool *converged;

  void operator()() const {

    // tolerance must be squared
    const auto tol = this->tol * this->tol;

    bool conv = true;
    for (size_t i = 0; i == 0 || i < num_centroids; i++) {
      conv &= squared_distance(centroids[i], new_centroids[i]) < tol;

      if (!conv) {
        break;
      }
    }

    *converged = conv; // access pointer one time
  }
};

kmeans_cluster_t kmeans_usm_v2::cluster(const size_t max_iter, const double tol) {
  const auto num_points    = points.size();
  const auto num_centroids = this->num_centroids;

  // Points
  const auto dev_points        = malloc_device<point_t>(points.size(), q);
  // Centroids
  const auto dev_centroids     = malloc_device<point_t>(num_centroids, q);
  // Updated centroids after each iteration
  const auto dev_new_centroids = malloc_device<point_t>(num_centroids, q);
  // Association of each point to a cluster
  const auto dev_associations  = malloc_device<size_t>(points.size(), q);
  const auto converged         = malloc_host<bool>(1, q); // Convergence flag

  const size_t work_group_size = q.get_device().get_info<info::device::max_work_group_size>();
  const size_t num_work_groups = (num_points + work_group_size - 1) / work_group_size;

  const auto partial_sums_x = malloc_device<double>(num_work_groups * num_centroids, q);
  const auto partial_sums_y = malloc_device<double>(num_work_groups * num_centroids, q);
  const auto partial_counts = malloc_device<size_t>(num_work_groups * num_centroids, q);

  assert(dev_points != nullptr);
  assert(dev_centroids != nullptr);
  assert(dev_new_centroids != nullptr);
  assert(dev_associations != nullptr);
  assert(converged != nullptr);
  assert(partial_sums_x != nullptr);
  assert(partial_sums_y != nullptr);
  assert(partial_counts != nullptr);

  // Define the global and local ranges for the partial reduction kernel
  auto partial_reduction_global_range = range{num_points};
  if (num_points % work_group_size != 0) {
    // the global range must be a multiple of the work group size
    partial_reduction_global_range = range{(num_points / work_group_size + 1) * work_group_size};
  }
  const auto partial_reduction_execution_range =
      nd_range{partial_reduction_global_range, range(work_group_size)};

  // copy points to device memory
  // consider the first k points as the initial centroids
  q.copy(points.data(), dev_points, points.size());
  q.copy(points.data(), dev_centroids, num_centroids);
  *converged = false;
  q.wait();

  // main cycle: assign points to clusters, calculate new centroids, check for
  // convergence

  for (iter = 0; iter < max_iter; iter++) {
    // Step 1: Assign points to clusters
    // For each point, calculate the distance to each centroid and assign the point to the closest
    // one

    {
      auto kernel = assign_points_to_clusters_kernel_v2{.points        = dev_points,
                                                        .centroids     = dev_centroids,
                                                        .num_centroids = num_centroids,
                                                        .associations  = dev_associations};
      q.parallel_for(num_points, kernel).wait();
    }

    // Step 2: calculate new centroids by averaging the points in each cluster
    // HAHA: SE NON CASTIAMO A DOUBLE, IL RISULTATO E' SBAGLIATO
    q.fill<double>(partial_sums_x, 0.0, num_work_groups * num_centroids);
    q.fill<double>(partial_sums_y, 0.0, num_work_groups * num_centroids);
    q.fill<size_t>(partial_counts, 0, num_work_groups * num_centroids);
    q.wait();

    // Step 2.1: Parallel reduction over points
    {
      auto kernel = partial_reduction_kernel_v2{.num_centroids  = num_centroids,
                                                .num_points     = num_points,
                                                .points         = dev_points,
                                                .associations   = dev_associations,
                                                .partial_sums_x = partial_sums_x,
                                                .partial_sums_y = partial_sums_y,
                                                .partial_counts = partial_counts};
      q.parallel_for(partial_reduction_execution_range, kernel).wait();
    }

    // Step 2.2: Final reduction and compute centroids
    {
      auto kernel = final_reduction_kernel_v2{.num_work_groups = num_work_groups,
                                              .num_centroids   = num_centroids,
                                              .partial_sums_x  = partial_sums_x,
                                              .partial_sums_y  = partial_sums_y,
                                              .partial_counts  = partial_counts,
                                              .centroids       = dev_centroids,
                                              .new_centroids   = dev_new_centroids};
      q.parallel_for(num_centroids, kernel).wait();
    }

    // Step 3: Check for convergence
    {
      auto kernel = check_convergence_kernel_v2{.num_centroids = num_centroids,
                                                .tol           = tol,
                                                .centroids     = dev_centroids,
                                                .new_centroids = dev_new_centroids,
                                                .converged     = converged};
      q.single_task(kernel).wait();
    }

    q.copy(dev_new_centroids, dev_centroids, num_centroids).wait();

    if (*converged) {
      break;
    }
  }

  auto centroids_h    = std::vector<point_t>(num_centroids);
  auto associations_h = std::vector<size_t>(points.size());
  q.memcpy(centroids_h.data(), dev_centroids, num_centroids * sizeof(point_t));
  q.memcpy(associations_h.data(), dev_associations, points.size() * sizeof(size_t));
  q.wait();

  auto clusters_h = std::vector<std::vector<point_t>>{centroids_h.size()};
  for (auto &i : clusters_h) {
    i = std::vector<point_t>{};
  }

  for (size_t i = 0; i < points.size(); i++) {
    clusters_h[associations_h[i]].push_back(points[i]);
  }

  // free memory
  sycl::free(dev_points, q);
  sycl::free(dev_centroids, q);
  sycl::free(dev_new_centroids, q);
  sycl::free(dev_associations, q);
  sycl::free(converged, q);
  sycl::free(partial_sums_x, q);
  sycl::free(partial_sums_y, q);
  sycl::free(partial_counts, q);

  return kmeans_cluster_t{
      .centroids = centroids_h,
      .clusters  = clusters_h,
  };
}
