#include "kmeans_usm.hpp"
#include "point.hpp"
#include "sycl_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace sycl;

// to support FPGA devices (32-bit required for atomic operations)
typedef uint32_t custom_size_t;

namespace kernels::v1 {

class assign_points_to_clusters {
  const point_t *points;
  const point_t *centroids;
  const size_t   num_centroids;
  size_t        *associations;

public:
  assign_points_to_clusters(const point_t *points, const point_t *centroids,
                            const size_t num_centroids, size_t *associations)
      : points(points), centroids(centroids), num_centroids(num_centroids),
        associations(associations) {}

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

class partial_reduction {

  // input
  const size_t   num_points;
  const point_t *points;
  const size_t  *associations;

  // output
  point_t       *new_centroids;
  custom_size_t *new_clusters_size;

public:
  partial_reduction(const size_t num_points, const point_t *points, const size_t *associations,
                    point_t *new_centroids, custom_size_t *new_clusters_size)
      : num_points(num_points), points(points), associations(associations),
        new_centroids(new_centroids), new_clusters_size(new_clusters_size) {}

  void operator()(const size_t p_idx) const {
    if (p_idx >= num_points) {
      return; // Out-of-bounds guard
    }

    const auto cluster_idx = associations[p_idx]; // Cluster index

    const auto custom_atomic_ref = []<typename T>(T &ptr) {
      constexpr auto mem_order = memory_order::relaxed;
      constexpr auto mem_scope = memory_scope::device;
      constexpr auto mem_space = access::address_space::global_space;

      return atomic_ref<T, mem_order, mem_scope, mem_space>{ptr};
    };

    // Use atomic operations to update partial results
    const auto atom_x = custom_atomic_ref(new_centroids[cluster_idx].x);
    const auto atom_y = custom_atomic_ref(new_centroids[cluster_idx].y);
    const auto atom_c = custom_atomic_ref(new_clusters_size[cluster_idx]);

    atom_x += points[p_idx].x;
    atom_y += points[p_idx].y;
    atom_c += custom_size_t{1};
  }
};

class final_reduction {
  // input
  const point_t       *centroids;
  const custom_size_t *new_clusters_size;

  // input-output
  point_t *new_centroids;

public:
  final_reduction(const point_t *centroids, const custom_size_t *new_clusters_size,
                  point_t *new_centroids)
      : centroids(centroids), new_clusters_size(new_clusters_size), new_centroids(new_centroids) {}

  void operator()(const size_t cluster_idx) const {

    const auto cluster_points_count = new_clusters_size[cluster_idx];

    if (cluster_points_count <= 0) {
      // No points in cluster, centroid remains unchanged
      new_centroids[cluster_idx] = centroids[cluster_idx];
      return;
    }

    // Compute the final centroid
    new_centroids[cluster_idx].x /= static_cast<float>(cluster_points_count);
    new_centroids[cluster_idx].y /= static_cast<float>(cluster_points_count);
  }
};

class check_convergence {

  // input
  const size_t   num_centroids;
  const double   tol;
  const point_t *centroids;
  const point_t *new_centroids;

  // output
  bool *converged;

public:
  check_convergence(const size_t num_centroids, const double tol, const point_t *centroids,
                    const point_t *new_centroids, bool *converged)
      : num_centroids(num_centroids), tol(tol), centroids(centroids), new_centroids(new_centroids),
        converged(converged) {}

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
} // namespace kernels::v1

kmeans_cluster_t kmeans_usm_v1::cluster(const size_t max_iter, const double tol) {
  if (points.size() > UINT32_MAX) {
    throw std::runtime_error("Only up to UINT32_MAX points are supported on this backend");
  }

  const size_t num_points    = points.size();
  const size_t num_centroids = this->num_centroids;

  // Points
  const auto dev_points            = required_ptr(malloc_device<point_t>(points.size(), q));
  // Centroids
  const auto dev_centroids         = required_ptr(malloc_device<point_t>(num_centroids, q));
  // Updated centroids after each iteration
  const auto dev_new_centroids     = required_ptr(malloc_device<point_t>(num_centroids, q));
  const auto dev_new_clusters_size = required_ptr(malloc_device<custom_size_t>(num_centroids, q));
  // Association of each point to a cluster
  const auto dev_associations      = required_ptr(malloc_device<size_t>(points.size(), q));
  const auto converged             = required_ptr(malloc_host<bool>(1, q)); // Convergence flag

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
      auto kernel = kernels::v1::assign_points_to_clusters{
          dev_points,
          dev_centroids,
          num_centroids,
          dev_associations,
      };
      q.parallel_for(num_points, kernel).wait();
    }

    // Step 2: calculate new centroids by averaging the points in each cluster
    // Step 2.0: Initialize partial sums and counts
    q.fill(dev_new_centroids, point_t{0.0, 0.0}, num_centroids);
    q.fill(dev_new_clusters_size, custom_size_t{0}, num_centroids);
    q.wait();

    // Step 2.1: Parallel reduction over points
    {
      auto kernel = kernels::v1::partial_reduction{
          num_points, dev_points, dev_associations, dev_new_centroids, dev_new_clusters_size,
      };
      q.parallel_for(num_points, kernel).wait();
    }

    // Step 2.2: Final reduction and compute centroids
    {
      auto kernel = kernels::v1::final_reduction{
          dev_centroids,
          dev_new_clusters_size,
          dev_new_centroids,
      };
      q.parallel_for(num_centroids, kernel).wait();
    }

    // Step 3: Check for convergence
    {
      auto kernel = kernels::v1::check_convergence{
          num_centroids, tol, dev_centroids, dev_new_centroids, converged,
      };
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
  sycl::free(dev_new_clusters_size, q);

  return kmeans_cluster_t{centroids_h, clusters_h};
}
