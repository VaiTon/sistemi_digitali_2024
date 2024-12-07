#include <cstddef>
#include <stdexcept>
#include <vector>

#include "kmeans.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

class UsmAssignPointsKernel;
class UsmNewCentroidsKernel;
class UsmConvergedKernel;
class UsmUpdateCentroidsKernel;
class UsmPartialReductionKernel;
class UsmFinalReductionKernel;

kmeans_cluster_t kmeans_sycl(queue q, const size_t k, const std::vector<Point> &points, const size_t max_iter,
                             float tol) {
  if (k > points.size()) {
    throw std::invalid_argument("Number of clusters (" + std::to_string(k) +
                                ") must be less than the number of points (" + std::to_string(points.size()) + ")");
  }

  // tolerance must be squared
  tol *= tol;

  const auto points_n = points.size();

  // device memory
  const auto points_d        = malloc_device<Point>(points.size(), q); // Points
  const auto centroids_d     = malloc_device<Point>(k, q);             // Centroids
  const auto new_centroids_d = malloc_device<Point>(k, q);             // Updated centroids after each iteration

  const auto assoc_d   = malloc_device<size_t>(points.size(), q); // Association of each point to a cluster
  const auto converged = malloc_host<bool>(1, q);                 // Convergence flag

  const size_t work_group_size = q.get_device().get_info<info::device::max_work_group_size>();
  const size_t num_work_groups = (points_n + work_group_size - 1) / work_group_size;

  const auto partial_sums_x = malloc_shared<double>(num_work_groups * k, q);
  const auto partial_sums_y = malloc_shared<double>(num_work_groups * k, q);
  const auto partial_counts = malloc_shared<size_t>(num_work_groups * k, q);

  assert(points_d != nullptr);
  assert(centroids_d != nullptr);
  assert(new_centroids_d != nullptr);
  assert(assoc_d != nullptr);
  assert(converged != nullptr);
  assert(partial_sums_x != nullptr);
  assert(partial_sums_y != nullptr);
  assert(partial_counts != nullptr);

  // copy points to device memory
  // consider the first k points as the initial centroids
  q.copy(points.data(), points_d, points.size());
  q.copy(points.data(), centroids_d, k);
  *converged = false;
  q.wait();

  // main cycle: assign points to clusters, calculate new centroids, check for
  // convergence

  auto iter = size_t{0}; // Iteration counter
  while (!*converged && iter < max_iter) {

    // Step 1: Assign points to clusters
    // For each point, calculate the distance to each centroid and assign the point to the closest one

    q.parallel_for<UsmAssignPointsKernel>(points_n, [=](const size_t point_idx) {
       auto min_val = std::numeric_limits<double>::max();
       auto min_idx = size_t{0};

       for (size_t centroid_idx = 0; centroid_idx < k; centroid_idx++) {
         const auto dist = squared_distance(points_d[point_idx], centroids_d[centroid_idx]);
         if (dist < min_val) {
           min_val = dist;
           min_idx = centroid_idx;
         }
       }

       assoc_d[point_idx] = min_idx;
     }).wait();

    // Step 2: calculate new centroids by averaging the points in each cluster

    q.fill(partial_sums_x, 0.0, num_work_groups * k);
    q.fill(partial_sums_y, 0.0, num_work_groups * k);
    q.fill(partial_counts, 0, num_work_groups * k);
    q.wait();

    // Step 2.1: Parallel reduction over points
    auto global_range = range{points_n};
    if (points_n % work_group_size != 0) { // the global range must be a multiple of the work group size
      global_range = range{(points_n / work_group_size + 1) * work_group_size};
    }

    const auto execution_range = nd_range{global_range, range(work_group_size)};
    q.parallel_for<UsmPartialReductionKernel>(execution_range, [=](const nd_item<> &item) {
       const auto p_idx = item.get_global_id(0);
       if (p_idx >= points_n)
         return; // Out-of-bounds guard

       const auto wg_idx = item.get_group(0); // Workgroup index
       const auto k_idx  = assoc_d[p_idx];    // Cluster index

       // Use atomic operations to update partial results
       const auto x_atomic =
           atomic_ref<double, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(
               partial_sums_x[wg_idx * k + k_idx]);
       const auto y_atomic =
           atomic_ref<double, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(
               partial_sums_y[wg_idx * k + k_idx]);
       const auto count_atomic =
           atomic_ref<size_t, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(
               partial_counts[wg_idx * k + k_idx]);

       x_atomic += static_cast<double>(points_d[p_idx].x);
       y_atomic += static_cast<double>(points_d[p_idx].y);
       count_atomic += size_t{1};
     }).wait();

    // Step 2.2: Final reduction and compute centroids

    q.parallel_for<UsmFinalReductionKernel>(range(k), [=](const id<> k_idx) {
       auto final_x     = double{0.0};
       auto final_y     = double{0.0};
       auto final_count = size_t{0};

       // Accumulate results from all workgroups
       for (auto wg = size_t{0}; wg < num_work_groups; ++wg) {
         final_x += partial_sums_x[wg * k + k_idx];
         final_y += partial_sums_y[wg * k + k_idx];
         final_count += partial_counts[wg * k + k_idx];
       }

       // Compute the final centroid
       if (final_count > 0) {
         final_x /= static_cast<double>(final_count);
         final_y /= static_cast<double>(final_count);
         new_centroids_d[k_idx] = Point{static_cast<float>(final_x), static_cast<float>(final_y)};
       } else {
         // No points in cluster, centroid remains unchanged
         new_centroids_d[k_idx] = centroids_d[k_idx];
       }
     }).wait();

    // check for convergence
    q.single_task<UsmConvergedKernel>([=] {
       *converged = true;
       for (size_t i = 0; i == 0 || (i < k && !*converged); i++) {
         *converged &= squared_distance(centroids_d[i], new_centroids_d[i]) <= tol;
       }
     }).wait();

    // if not converged, update the centroids
    q.copy(new_centroids_d, centroids_d, k).wait();
    iter++;
  }

  auto centroids_h    = std::vector<Point>(k);
  auto associations_h = std::vector<size_t>(points.size());
  q.memcpy(centroids_h.data(), centroids_d, k * sizeof(Point));
  q.memcpy(associations_h.data(), assoc_d, points.size() * sizeof(size_t));
  q.wait();

  // free memory
  sycl::free(points_d, q);
  sycl::free(centroids_d, q);
  sycl::free(new_centroids_d, q);
  sycl::free(assoc_d, q);
  sycl::free(converged, q);
  sycl::free(partial_sums_x, q);
  sycl::free(partial_sums_y, q);
  sycl::free(partial_counts, q);

  auto clusters_h = std::vector<std::vector<Point>>{centroids_h.size()};
  for (size_t i = 0; i < k; i++) {
    clusters_h[i] = std::vector<Point>{};
  }

  for (size_t i = 0; i < points.size(); i++) {
    clusters_h[associations_h[i]].push_back(points[i]);
  }

  return kmeans_cluster_t{
      .centroids = centroids_h,
      .clusters  = clusters_h,
  };
}
