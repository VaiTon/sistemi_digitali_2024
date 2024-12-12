#include "kmeans_simd.hpp"

#include <smmintrin.h>

template <typename T, size_t Alignment> T *aligned_malloc(const size_t size) {
  void *ptr = _mm_malloc(size * sizeof(T), Alignment);
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return static_cast<T *>(ptr);
}

template <typename T> void aligned_free(T *ptr) { _mm_free(ptr); }

/// Load 4 points (x0, y0, x1, y1, x2, y2, x3, y3) into two registers
inline void deinterleave_sse(const point_t *points, __m128 &x, __m128 &y) {
  const __m128 xy0 = _mm_load_ps(&points[0].x); // Load [x0, y0, x1, y1]
  const __m128 xy1 = _mm_load_ps(&points[2].x); // Load [x2, y2, x3, y3]

  static constexpr auto mask_x = _MM_SHUFFLE(2, 0, 2, 0);
  static constexpr auto mask_y = _MM_SHUFFLE(3, 1, 3, 1);

  // Deinterleave the values into two registers: one for x values and one for y values
  x = _mm_shuffle_ps(xy0, xy1, mask_x); // Extract x's: [x0, x1, x2, x3]
  y = _mm_shuffle_ps(xy0, xy1, mask_y); // Extract y's: [y0, y1, y2, y3]
}

/// \param c_idx cluster index
/// \param points_x x coordinates of the points
/// \param points_y y coordinates of the points
/// \param points_n number of points
/// \param assoc cluster assignments
/// \param sum_x output sum of x coordinates
/// \param sum_y output sum of y coordinates
/// \param count output count of points in the cluster
void accumulate_points_sse(const int c_idx, const float *points_x, const float *points_y,
                           const size_t points_n, const int *assoc, __m128 &sum_x, __m128 &sum_y,
                           __m128i &count) {

  for (size_t i = 0; i < points_n; i += 4) {

    // Load 4 cluster assignments
    const auto    assoc_p     = reinterpret_cast<const __m128i *>(&assoc[i]);
    const __m128i cluster_ids = _mm_load_si128(assoc_p);

    // Create a mask for points belonging to the current cluster
    const __m128 mask = _mm_castsi128_ps(_mm_cmpeq_epi32(cluster_ids, _mm_set1_epi32(c_idx)));

    __m128 px = _mm_load_ps(&points_x[i]); // Load x coordinates (4 points)
    __m128 py = _mm_load_ps(&points_y[i]); // Load y coordinates (4 points)

    // Mask the x and y values and add them to the sum
    px    = _mm_and_ps(px, mask);
    sum_x = _mm_add_ps(sum_x, px);

    py    = _mm_and_ps(py, mask);
    sum_y = _mm_add_ps(sum_y, py);

    // Increment the count for each cluster
    const __m128i mask_count = _mm_castps_si128(mask);
    auto          b          = _mm_and_si128(mask_count, _mm_set1_epi32(1));
    count                    = _mm_add_epi32(count, b);
  }
}

inline float squared_distance_comp(const float px, const float py, const float cx, const float cy) {
  const auto dx = px - cx;
  const auto dy = py - cy;
  return dx * dx + dy * dy;
}

inline float horizontal_sum_sse(const __m128 vec) {
  __m128 shuf = _mm_movehdup_ps(vec); // High halves of vec
  __m128 sums = _mm_add_ps(vec, shuf);
  shuf        = _mm_movehl_ps(shuf, sums); // High halves of the results
  sums        = _mm_add_ss(sums, shuf);    // Final sum
  return _mm_cvtss_f32(sums);
}

inline int horizontal_sum_sse(const __m128i vec) {
  const __m128i sum1 = _mm_hadd_epi32(vec, vec);
  const __m128i sum2 = _mm_hadd_epi32(sum1, sum1);
  return _mm_cvtsi128_si32(sum2);
}

inline __m128 squared_distance_sse(const __m128 px, const __m128 py, const __m128 cx,
                                   const __m128 cy) {
  const __m128 dx = _mm_sub_ps(px, cx);                      // dx = px - cx
  const __m128 dy = _mm_sub_ps(py, cy);                      // dy = py - cy
  return _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)); // dx^2 + dy^2
}

kmeans_cluster_t kmeans_simd::cluster(const size_t max_iter, double tol) {
  if (this->points_.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("Number of points exceeds the maximum supported value");
  }
  if (this->k_ > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("Number of clusters exceeds the maximum supported value");
  }

  // Copy member variables to local variables
  const auto points_n = static_cast<uint32_t>(points_.size());
  const auto k        = static_cast<uint32_t>(k_);

  if (k > points_n) {
    throw std::invalid_argument("Number of clusters must be less than or equal "
                                "to the number of points");
  }

  tol *= tol; // Use squared distance for convergence check

  constexpr auto alignment = 16;

  // Aligned copies
  const auto points_x = aligned_malloc<float, alignment>(points_n);
  const auto points_y = aligned_malloc<float, alignment>(points_n);

  // Copy the points to aligned arrays
  for (size_t i = 0; i < points_n; i++) {
    points_x[i] = points_[i].x;
    points_y[i] = points_[i].y;
  }

  auto centroids_x = aligned_malloc<float, alignment>(k);
  auto centroids_y = aligned_malloc<float, alignment>(k);

  // Copy the first k points as initial centroids
  for (size_t i = 0; i < k; i++) {
    centroids_x[i] = points_[i].x;
    centroids_y[i] = points_[i].y;
  }

  const auto assoc = aligned_malloc<int, alignment>(points_n);

  auto new_centroids_x = aligned_malloc<float, alignment>(k);
  auto new_centroids_y = aligned_malloc<float, alignment>(k);

  bool converged = false; // Convergence flag

  for (iter = 0; iter < max_iter; iter++) {

    // Assign each point to the closest centroid
    // Parallelize the loop for 4 points at a time
    for (size_t p_idx = 0; p_idx < points_n; p_idx += 4) {

      const __m128 px = _mm_load_ps(&points_x[p_idx]); // load x coordinates (4 points)
      const __m128 py = _mm_load_ps(&points_y[p_idx]); // load y coordinates (4 points)

      __m128  min_dist = _mm_set1_ps(std::numeric_limits<float>::max());
      __m128i min_idx  = _mm_set1_epi32(-1);

      // Iterate over all centroids
      for (size_t c_idx = 0; c_idx < k; c_idx++) {
        const __m128 cx = _mm_set1_ps(centroids_x[c_idx]);
        const __m128 cy = _mm_set1_ps(centroids_y[c_idx]);

        const __m128 dist = squared_distance_sse(px, py, cx, cy);

        const __m128 mask = _mm_cmplt_ps(dist, min_dist); // Update the closest centroid
        min_dist          = _mm_min_ps(min_dist, dist);
        min_idx           = _mm_blendv_epi8(min_idx, _mm_set1_epi32(c_idx), _mm_castps_si128(mask));
      }

      _mm_store_si128(reinterpret_cast<__m128i *>(&assoc[p_idx]), min_idx);
    }

    // Step 2: Calculate new centroids
    for (size_t c_idx = 0; c_idx < k; c_idx++) {
      __m128  sum_x = _mm_setzero_ps();
      __m128  sum_y = _mm_setzero_ps();
      __m128i count = _mm_setzero_si128();
      accumulate_points_sse(c_idx, points_x, points_y, points_n, assoc, sum_x, sum_y, count);

      const auto centroid_x  = horizontal_sum_sse(sum_x);
      const auto centroid_y  = horizontal_sum_sse(sum_y);
      const auto total_count = static_cast<float>(horizontal_sum_sse(count));

      // Update the centroid if there are points in the cluster
      if (total_count > 0) {
        new_centroids_x[c_idx] = centroid_x / total_count;
        new_centroids_y[c_idx] = centroid_y / total_count;
      }
    }

    converged = true;
    for (size_t i = 0; i < k && converged; ++i) {
      converged &= squared_distance_comp(new_centroids_x[i], new_centroids_y[i], //
                                         centroids_x[i], centroids_y[i]) < tol;
    }

    if (converged) {
      break;
    }

    // copy new centroids to centroids
    for (size_t i = 0; i < k; i++) {
      centroids_x[i] = new_centroids_x[i];
      centroids_y[i] = new_centroids_y[i];
    }
  }

  // Collect the final clusters
  auto clusters = std::vector<std::vector<point_t>>{k};
  for (size_t i = 0; i < k; i++) {
    clusters[i] = std::vector<point_t>{};
  }

  for (uint32_t i = 0; i < points_n; i++) {
    clusters[assoc[i]].push_back(point_t{points_x[i], points_y[i]});
  }

  // Dealigned copy of the centroids
  auto dealigned_centroids = std::vector<point_t>{k};
  for (size_t i = 0; i < k; i++) {
    dealigned_centroids[i] = point_t{centroids_x[i], centroids_y[i]};
  }

  aligned_free(points_x);
  aligned_free(points_y);
  aligned_free(assoc);
  aligned_free(centroids_x);
  aligned_free(centroids_y);
  aligned_free(new_centroids_x);
  aligned_free(new_centroids_y);

  return kmeans_cluster_t{dealigned_centroids, clusters};
}
