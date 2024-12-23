#include "kmeans_ocv.hpp"

#include <cstdint>
#include <limits>
#include <opencv2/core/hal/intrin.hpp>

using namespace cv;
typedef uint32_t size_type;

/// \param cluster_idx cluster index
/// \param points_x x coordinates of the points
/// \param points_y y coordinates of the points
/// \param num_points number of points
/// \param assoc cluster assignments
/// \param sum_x output sum of x coordinates
/// \param sum_y output sum of y coordinates
/// \param count output count of points in the cluster
inline void accumulate_points_ocv(const size_type                     cluster_idx,
                                  const std::shared_ptr<float[]>     &points_x,
                                  const std::shared_ptr<float[]>     &points_y,
                                  const size_type                     num_points,
                                  const std::shared_ptr<size_type[]> &assoc, v_float32x4 &sum_x,
                                  v_float32x4 &sum_y, v_uint32x4 &count) {

  for (size_type i = 0; i < num_points; i += 4) {
    // Load 4 cluster assignments
    v_uint32x4 cluster_ids = v_load(&assoc[i]);

    // Create a mask for points belonging to the current cluster
    v_uint32x4 mask = cluster_ids == v_setall_u32(cluster_idx);

    v_float32x4 px = v_load(&points_x[i]);
    v_float32x4 py = v_load(&points_y[i]);

    // Mask the x and y values and add them to the sum
    px = px & v_reinterpret_as_f32(mask);
    py = py & v_reinterpret_as_f32(mask);

    sum_x = sum_x + px;
    sum_y = sum_y + py;

    // Increment the count for each cluster
    count = count + (mask & v_setall_u32(1));
  }
}

inline v_float32x4 squared_distance_ocv(const v_float32x4 &px, const v_float32x4 &py,
                                        const v_float32x4 &cx, const v_float32x4 &cy) {
  const auto dx = px - cx;  // dx = px - cx
  const auto dy = py - cy;  // dy = py - cy
  return dx * dx + dy * dy; // dx^2 + dy^2
}

inline float horizontal_sum_ocv(const v_float32x4 &vec) {
  float buf[4];
  v_store(buf, vec);
  return buf[0] + buf[1] + buf[2] + buf[3];
}

inline uint32_t horizontal_sum_ocv(const v_uint32x4 &vec) {
  uint32_t buf[4];
  v_store(buf, vec);
  return buf[0] + buf[1] + buf[2] + buf[3];
}

void check_arguments(const size_t &num_points, const size_t &num_centroids) {
  if (num_points > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument("Number of points exceeds the maximum supported value for SIMD");
  }
  if (num_centroids > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument("Number of clusters exceeds the maximum supported value for SIMD");
  }
}

template <typename T> std::shared_ptr<T[]> cv_shared_ptr(T *ptr) {
  auto deleter = [](T *p) { fastFree(p); };
  return std::shared_ptr<T[]>(ptr, deleter);
}

template <typename T> std::shared_ptr<T[]> cv_malloc_shared(const size_t n) {
  T *ptr = static_cast<T *>(fastMalloc(n * sizeof(T)));
  return cv_shared_ptr<T>(ptr);
}

kmeans_cluster_t kmeans_ocv::cluster(const size_t max_iter, double tol) {

  check_arguments(this->points.size(), this->num_centroids);

  const auto num_points    = static_cast<size_type>(this->points.size());
  const auto num_centroids = static_cast<size_type>(this->num_centroids);

  tol *= tol; // we compare squared distances to avoid sqrt

  auto points_x = cv_malloc_shared<float>(num_points);
  auto points_y = cv_malloc_shared<float>(num_points);

  for (size_type i = 0; i < num_points; i++) {
    points_x[i] = points[i].x;
    points_y[i] = points[i].y;
  }

  const auto centroids_x = cv_malloc_shared<float>(num_centroids);
  const auto centroids_y = cv_malloc_shared<float>(num_centroids);

  for (size_type i = 0; i < num_centroids; i++) {
    centroids_x[i] = points[i].x;
    centroids_y[i] = points[i].y;
  }

  auto assoc = cv_malloc_shared<uint32_t>(num_points);

  auto new_centroids_x = cv_malloc_shared<float>(num_centroids);
  auto new_centroids_y = cv_malloc_shared<float>(num_centroids);

  bool converged = false;

  for (iter = 0; iter < max_iter; iter++) {

    // STEP 1: Assign points to the closest centroid

    // take points in batches of 4
    for (size_type p_idx = 0; p_idx < num_points; p_idx += 4) {

      // Load 4 x-coordinates and 4 y-coordinates of points into SIMD registers
      const v_float32x4 px = v_load(&points_x[p_idx]);
      const v_float32x4 py = v_load(&points_y[p_idx]);

      v_float32x4 min_dist = v_setall_f32(std::numeric_limits<float>::max());
      v_uint32x4  min_idx  = v_setall_u32(-1);

      // Iterate through all centroids
      for (size_type c_idx = 0; c_idx < num_centroids; c_idx++) {
        // Load the x and y coordinates of the current centroid into SIMD registers
        const auto cx = v_setall_f32(centroids_x[c_idx]);
        const auto cy = v_setall_f32(centroids_y[c_idx]);

        // Compute the squared Euclidean distance between the current points and the centroid
        auto dist = squared_distance_ocv(px, py, cx, cy);

        // Create a mask indicating where the distance is smaller than the current minimum distance
        v_uint32x4 mask = v_reinterpret_as_u32(dist < min_dist);

        // Update the minimum distance with the smaller distance
        min_dist = v_min(dist, min_dist);

        // Update the index of the closest centroid based on the mask
        // Create a vector with the current centroid index
        auto v_idx = v_setall_u32(c_idx);
        min_idx    = v_select(mask, v_idx, min_idx);
      }

      // Store the indices of the closest centroids for these 4 points
      v_store(&assoc[p_idx], min_idx);
    }

    for (size_type c_idx = 0; c_idx < num_centroids; c_idx++) {
      auto sum_x = v_setzero_f32();
      auto sum_y = v_setzero_f32();
      auto count = v_setzero_u32();

      accumulate_points_ocv(c_idx, points_x, points_y, num_points, assoc, sum_x, sum_y, count);

      const auto centroid_x  = horizontal_sum_ocv(sum_x);
      const auto centroid_y  = horizontal_sum_ocv(sum_y);
      const auto total_count = horizontal_sum_ocv(count);

      if (total_count > 0) {
        const auto total_count_f = static_cast<float>(total_count);
        new_centroids_x[c_idx]   = centroid_x / total_count_f;
        new_centroids_y[c_idx]   = centroid_y / total_count_f;
      }
    }

    converged = true;
    for (size_type i = 0; i < num_centroids && converged; i += 4) {
      const auto c_x = v_load(&centroids_x[i]);
      const auto c_y = v_load(&centroids_y[i]);
      const auto n_x = v_load(&new_centroids_x[i]);
      const auto n_y = v_load(&new_centroids_y[i]);

      auto dist = squared_distance_ocv(c_x, c_y, n_x, n_y);

      const auto v_tol = tol * 4; // we compare the entire SIMD vector at once;
      converged &= v_reduce_sum(dist) < v_tol;
    }

    if (converged) {
      break;
    }

    for (size_type i = 0; i < num_centroids; i++) {
      centroids_x[i] = new_centroids_x[i];
      centroids_y[i] = new_centroids_y[i];
    }
  }

  // Collect the final clusters
  auto clusters = std::vector<std::vector<point_t>>{static_cast<size_t>(num_centroids)};
  for (size_type i = 0; i < num_centroids; i++) {
    clusters[i] = std::vector<point_t>{};
  }

  for (size_type i = 0; i < num_points; i++) {
    clusters[assoc[i]].push_back(point_t{points_x[i], points_y[i]});
  }

  // Dealigned copy of the centroids
  auto dealigned_centroids = std::vector<point_t>{num_centroids};
  for (size_type i = 0; i < num_centroids; i++) {
    dealigned_centroids[i] = point_t{centroids_x[i], centroids_y[i]};
  }

  return kmeans_cluster_t{dealigned_centroids, clusters};
}
