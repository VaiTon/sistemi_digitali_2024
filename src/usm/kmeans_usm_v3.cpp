#include "kmeans_usm.hpp"
#include "sycl_utils.hpp"

#include <cstddef>
#include <vector>

using namespace sycl;

namespace kernels::v3 {

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

    if (new_centroids[k_idx].is_zero()) {
      return;
    }

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
    auto const tol_sq = this->tol * this->tol;

    bool conv = true;
    for (size_t i = 0; i == 0 || i < num_centroids; i++) {
      conv &= squared_distance(centroids[i], new_centroids[i]) < tol_sq;

      if (!conv) {
        break;
      }
    }

    *converged = conv; // access pointer one time
  }
};

} // namespace kernels::v3

kmeans_cluster_t kmeans_usm_v3::cluster(size_t const max_iter, double const tol) {
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

  auto const assoc_matrix_size = num_centroids * num_points;
  auto const assoc_array_size  = num_points;

  auto const dev_assoc_matrix = required_ptr(malloc_device<point_t>(assoc_matrix_size, q));
  auto const dev_assoc_array  = required_ptr(malloc_device<size_t>(assoc_array_size, q));

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
    q.memset(dev_assoc_matrix, 0, assoc_matrix_size * sizeof(point_t));
    q.memset(dev_assoc_array, 0, assoc_array_size * sizeof(point_t));
    q.wait();

    q.submit([&](handler &h) {
       kernels::v3::assign_points_to_clusters const kernel{
           dev_centroids, num_centroids, dev_points, num_points, dev_assoc_matrix, dev_assoc_array,
       };

       h.parallel_for(num_points, kernel);
     }).wait();

    // q.fill(dev_new_centroids, point_t{0.0f, 0.0f}, num_centroids);
    q.memset(dev_new_clusters_size, 0, num_centroids * sizeof(point_t));
    // q.fill(dev_new_clusters_size, size_t{0}, num_centroids);
    q.memset(dev_new_centroids, 0, num_centroids * sizeof(point_t));
    q.wait();

    // Step 2.1: For each centroid, submit a SYCL reduction kernel
    for (size_t c_idx = 0; c_idx < num_centroids; c_idx++) {
      q.submit([&](handler &cgh) {
        size_t const assoc_cluster_offset = c_idx * num_points;

        auto const centroid_red = reduction<point_t>(dev_new_centroids + c_idx, point_t{0.0f, 0.0f},
                                                     sycl::plus<point_t>{});
        auto const count_red =
            reduction<size_t>(dev_new_clusters_size + c_idx, size_t{0}, sycl::plus<size_t>{});

        auto reduction_func = [=](id<> const &p_idx, auto &centroid, auto &count) {
          auto point = dev_assoc_matrix[assoc_cluster_offset + p_idx];
          centroid += point;

          if (!point.is_zero()) {
            count += 1;
          }
        };

        cgh.parallel_for(range{num_points}, centroid_red, count_red, reduction_func);
      });
    }

    q.wait();

    // Step 2.2: Final reduction and compute centroids
    q.submit([&](handler &cgh) {
       kernels::v3::final_reduction const kernel{
           dev_new_centroids,
           dev_new_clusters_size,
           dev_centroids,
       };

       cgh.parallel_for(num_centroids, kernel);
     }).wait();

    // Step 3: Check for convergence
    q.submit([&](handler &cgh) {
       kernels::v3::check_convergence const kernel{
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
