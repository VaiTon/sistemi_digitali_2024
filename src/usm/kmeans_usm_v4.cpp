#include "kmeans_usm.hpp"
#include "sycl_utils.hpp"

#include <cstddef>
#include <vector>

using namespace sycl;

namespace kernels::v4 {

class assign_points_to_clusters {
  point_t const *const points;
  size_t const         num_points;

  point_t const *const centroids;
  size_t const         num_centroids;

  point_t *const assoc_mtx; // associations matrix
  size_t *const  assoc_arr; // associations array

public:
  assign_points_to_clusters(point_t const *centroids, size_t const num_centroids,
                            point_t const *points, size_t const num_points,
                            point_t *const assoc_mtx, size_t *const assoc_arr)
      : points(points), num_points(num_points), centroids(centroids), num_centroids(num_centroids),
        assoc_mtx(assoc_mtx), assoc_arr(assoc_arr) {}

  void operator()(size_t const p_idx) const {
    auto const point = points[p_idx];

    auto   min_val = std::numeric_limits<double>::max();
    size_t min_idx = 0;

    for (size_t c_idx = 0; c_idx < num_centroids; c_idx++) {
      auto const dist = squared_distance(point, centroids[c_idx]);

      if (dist < min_val) {
        min_val = dist;
        min_idx = c_idx;
      }
    }

    // assoc_mtx is a matrix of size num_centroids x num_points
    // every row represents a centroid
    // every column represents a point
    // row major order is used

    auto const cluster_idx = min_idx;
    auto const assoc_idx   = cluster_idx * num_points + p_idx;

    assoc_mtx[assoc_idx] = point;
    assoc_arr[p_idx]     = cluster_idx;
  }
};

class reduce_to_centroids {
  size_t const         groups_per_centroid_;
  point_t const *const associations;
  size_t const         num_points;

  local_accessor<double, 1> const local_sums_x;
  local_accessor<double, 1> const local_sums_y;
  local_accessor<size_t, 1> const local_counts;

  // output
  point_t *const global_centroids;
  size_t *const  global_counts;

  static constexpr auto global_mem_order = memory_order::relaxed;
  static constexpr auto global_mem_scope = memory_scope::device;
  static constexpr auto global_mem_space = access::address_space::global_space;

  template <typename T>
  using global_atomic_ref = atomic_ref<T, global_mem_order, global_mem_scope, global_mem_space>;

public:
  reduce_to_centroids(size_t const groups_per_centroid, size_t const num_points,
                      point_t const *const associations, point_t *const global_centroids,
                      size_t *const global_counts, local_accessor<double, 1> const &local_sums_x,
                      local_accessor<double, 1> const &local_sums_y,
                      local_accessor<size_t, 1> const &local_counts)
      : groups_per_centroid_(groups_per_centroid), associations(associations),
        num_points(num_points), local_sums_x(local_sums_x), local_sums_y(local_sums_y),
        local_counts(local_counts), global_centroids(global_centroids),
        global_counts(global_counts) {}

  void operator()(nd_item<> const &item) const {

    size_t const group_idx  = item.get_group(0);       // Workgroup index
    size_t const group_size = item.get_local_range(0); // Workgroup size
    size_t const item_idx   = item.get_local_id(0);    // Local index

    // to which centroid does this workgroup belong? (matrix row)
    auto const cluster_idx         = group_idx / groups_per_centroid_;
    auto const threads_per_cluster = groups_per_centroid_ * group_size;

    // which point does this thread process? (matrix column)
    auto const p_idx            = group_idx % threads_per_cluster + item_idx;
    // associations is a matrix of size num_centroids x num_points
    // every row represents a centroid
    // every column represents a point
    // row major order is used
    auto const global_assoc_idx = cluster_idx * num_points + p_idx; // Global association index

    if (p_idx >= num_points) {
      local_sums_x[item_idx] = 0.0;
      local_sums_y[item_idx] = 0.0;
      local_counts[item_idx] = 0;
    } else {
      // grazie ChatGPT
      // copy every point to local memory
      auto const assoc       = associations[global_assoc_idx];
      local_sums_x[item_idx] = assoc.x;
      local_sums_y[item_idx] = assoc.y;
      local_counts[item_idx] = assoc.is_zero() ? 0 : 1;
    }

    // reduce via stride halving
    for (size_t stride = group_size / 2; stride > 0; stride /= 2) {
      item.barrier(access::fence_space::local_space);

      if (item_idx < stride) {
        auto const other_idx = item_idx + stride;

        local_sums_x[item_idx] += local_sums_x[other_idx];
        local_sums_y[item_idx] += local_sums_y[other_idx];
        local_counts[item_idx] += local_counts[other_idx];
      }
    }

    // write the result to global memory
    if (item_idx == 0) {
      global_atomic_ref<float> const  global_sums_x_ref{global_centroids[cluster_idx].x};
      global_atomic_ref<float> const  global_sums_y_ref{global_centroids[cluster_idx].y};
      global_atomic_ref<size_t> const global_counts_ref{global_counts[cluster_idx]};

      auto const x = local_sums_x[0];
      auto const y = local_sums_y[0];
      auto const c = local_counts[0];

      global_sums_x_ref += static_cast<float>(x);
      global_sums_y_ref += static_cast<float>(y);
      global_counts_ref += c;
    }
  }
};

class final_reduction {
  size_t const *const  new_clusters_size;
  point_t const *const centroids;

  point_t *const new_centroids;

public:
  final_reduction(point_t *const new_centroids, size_t const *const new_clusters_size,
                  point_t const *const centroids)
      : new_clusters_size(new_clusters_size), centroids(centroids), new_centroids(new_centroids) {}

  void operator()(size_t const k_idx) const {

    auto const new_cluster_size = new_clusters_size[k_idx];

    if (new_cluster_size == 0) {
      // keep the old centroid
      new_centroids[k_idx] = centroids[k_idx];
      return;
    }

    assert(!new_centroids[k_idx].is_zero());

    // normalize the centroid
    new_centroids[k_idx].x /= static_cast<float>(new_cluster_size);
    new_centroids[k_idx].y /= static_cast<float>(new_cluster_size);
  }
};

class check_convergence {
  size_t const         num_centroids;
  double const         tol;
  point_t const *const centroids;
  point_t const *const new_centroids;

  bool *const converged;

public:
  check_convergence(point_t const *const centroids, size_t const num_centroids,
                    point_t const *const new_centroids, double const tol, bool *const converged)
      : num_centroids(num_centroids), tol(tol), centroids(centroids), new_centroids(new_centroids),
        converged(converged) {}

  void operator()() const {

    // tolerance must be squared
    auto const tol = this->tol * this->tol;

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

} // namespace kernels::v4

kmeans_cluster_t kmeans_usm_v4::cluster(size_t const max_iter, double const tol) {
  auto const num_points    = points.size();
  auto const num_centroids = this->num_centroids;

  // Points
  auto const dev_points    = required_ptr(malloc_device<point_t>(points.size(), q));
  auto const dev_centroids = required_ptr(malloc_device<point_t>(num_centroids, q));

  // Associations
  // every row represents a centroid
  // every column represents a point
  // row major order is used
  // the matrix should be sparse

  auto const assoc_matrix_size = num_centroids * points.size();
  auto const dev_assoc_matrix  = required_ptr(malloc_device<point_t>(assoc_matrix_size, q));
  auto const dev_assoc_array   = required_ptr(malloc_device<size_t>(assoc_matrix_size, q));

  auto const dev_new_centroids     = required_ptr(malloc_device<point_t>(num_centroids, q));
  auto const dev_new_clusters_size = required_ptr(malloc_device<size_t>(num_centroids, q));
  auto const converged             = required_ptr(malloc_host<bool>(1, q));

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

    // Step 2: calculate new centroids by averaging the points in each cluster
    // q.fill(dev_assoc_matrix, point_t{0.0f, 0.0f}, assoc_matrix_size).wait();
    // xPres: .fill è lentissimo
    q.memset(dev_assoc_array, 0, assoc_matrix_size * sizeof(point_t)).wait();

    q.submit([&](handler &h) {
       kernels::v4::assign_points_to_clusters const kernel{
           dev_centroids, num_centroids, dev_points, num_points, dev_assoc_matrix, dev_assoc_array,
       };

       h.parallel_for(num_points, kernel);
     }).wait();

    // q.fill(dev_new_centroids, point_t{0.0f, 0.0f}, num_centroids);
    q.memset(dev_new_clusters_size, 0, num_centroids * sizeof(point_t));
    // q.fill(dev_new_clusters_size, size_t{0}, num_centroids);
    q.memset(dev_new_centroids, 0, num_centroids * sizeof(point_t));
    q.wait();

    // Step 2.1: Parallel reduction over points
    q.submit([&](handler &cgh) {
       // every workgroup will reduce up to 1024 points
       size_t const local_size = q.get_device().get_info<info::device::max_work_group_size>() / 8;
       size_t const groups_per_centroid = (num_points + local_size - 1) / local_size;

       local_accessor<double, 1> const local_sums_x{local_size, cgh};
       local_accessor<double, 1> const local_sums_y{local_size, cgh};
       local_accessor<size_t, 1> const local_counts{local_size, cgh};

       kernels::v4::reduce_to_centroids const kernel{
           groups_per_centroid,   num_points,   dev_assoc_matrix, dev_new_centroids,
           dev_new_clusters_size, local_sums_x, local_sums_y,     local_counts,
       };

       auto const total_groups = num_centroids * groups_per_centroid;

       auto const global_size = total_groups * local_size;
       auto const exec_range  = nd_range{range{global_size}, range{local_size}};

       cgh.parallel_for(exec_range, kernel);
     }).wait();

    q.wait();

    // Step 2.2: Final reduction and compute centroids
    q.submit([&](handler &cgh) {
       kernels::v4::final_reduction const kernel{
           dev_new_centroids,
           dev_new_clusters_size,
           dev_centroids,
       };

       cgh.parallel_for(num_centroids, kernel);
     }).wait();

    // Step 3: Check for convergence
    q.submit([&](handler &cgh) {
       kernels::v4::check_convergence const kernel{
           dev_centroids, num_centroids, dev_new_centroids, tol, converged,
       };
       cgh.single_task(kernel);
     }).wait();

    q.copy(dev_new_centroids, dev_centroids, num_centroids).wait();

    if (*converged) {
      break;
    }
  }

  auto host_centroids    = std::vector<point_t>(num_centroids);
  auto host_associations = std::vector<size_t>(points.size());

  q.copy<point_t>(dev_centroids, host_centroids.data(), num_centroids);
  q.copy<size_t>(dev_assoc_array, host_associations.data(), points.size());
  q.wait();

  auto host_clusters = std::vector<std::vector<point_t>>{host_centroids.size()};
  for (size_t cluster_idx = 0; cluster_idx < host_centroids.size(); cluster_idx++) {
    host_clusters[cluster_idx] = std::vector<point_t>{};
  }

  for (size_t point_idx = 0; point_idx < host_associations.size(); point_idx++) {
    auto const cluster_idx = host_associations[point_idx];
    host_clusters[cluster_idx].push_back(points[point_idx]);
  }

  // free memory
  sycl::free(dev_points, q);
  sycl::free(dev_centroids, q);
  sycl::free(dev_new_centroids, q);
  sycl::free(dev_assoc_matrix, q);
  sycl::free(converged, q);
  sycl::free(dev_new_clusters_size, q);

  return kmeans_cluster_t{host_centroids, host_clusters};
}
