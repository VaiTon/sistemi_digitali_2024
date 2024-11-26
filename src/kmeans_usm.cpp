#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <vector>

#include "kmeans.hpp"
#include "kmeans_sycl.hpp"

using namespace sycl;

kmeans_cluster_t kmeans_usm(queue q, const size_t k, const vector<Point> &points, const size_t max_iter, float tol) {
  if (k > points.size()) {
    throw std::invalid_argument("Number of clusters (" + std::to_string(k) +
                                ") must be less than the number of points (" + std::to_string(points.size()) + ")");
  }

  // tolerance must be squared
  tol *= tol;

  const auto points_n = points.size();

  // device memory
  const auto points_d        = malloc_device<Point>(points.size(), q);  // Points
  const auto centroids_d     = malloc_device<Point>(k, q);              // Centroids
  const auto new_centroids_d = malloc_device<Point>(k, q);              // Updated centroids after each iteration
  const auto assoc_d         = malloc_device<size_t>(points.size(), q); // Association of each point to a cluster
  const auto converged       = malloc_host<bool>(1, q);                 // Convergence flag

  assert(points_d != nullptr);
  assert(centroids_d != nullptr);
  assert(new_centroids_d != nullptr);
  assert(assoc_d != nullptr);
  assert(converged != nullptr);

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
    // for each point, calculate the distance to each centroid and assign the
    // point to the closest one
    q.parallel_for(points_n, [=](const size_t point_idx) {
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

    // calculate new centroids by averaging the points in each cluster
    q.parallel_for(k, [=](const size_t k_idx) {
       auto x = 0.0;
       auto y = 0.0;

       auto count = size_t{0};
       for (size_t p_idx = 0; p_idx < points_n; p_idx++) {
         if (assoc_d[p_idx] == k_idx) {
           x += points_d[p_idx].x;
           y += points_d[p_idx].y;
           count++;
         }
       }

       // If there are no points in the cluster, keep the centroid in the same position
       if (count > 0) {
         x /= static_cast<double>(count);
         y /= static_cast<double>(count);

         new_centroids_d[k_idx] = Point{static_cast<float>(x), static_cast<float>(y)};
       }
     }).wait();

    // check for convergence
    q.single_task([=] {
       *converged = true;
       for (size_t i = 0; i == 0 || (i < k && !*converged); i++) {
         *converged &= squared_distance(centroids_d[i], new_centroids_d[i]) <= tol;
       }
     }).wait();

    // if not converged, update the centroids
    if (!*converged) {
      q.parallel_for(k, [=](auto item) { centroids_d[item] = new_centroids_d[item]; }).wait();
    }

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
