#include "cuda_utils.cuh"
#include "kmeans.hpp"
#include "kmeans_cuda.hpp"
#include "point.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

typedef unsigned long long count_t; // type for partial counts, supported by atomicAdd

__host__ __device__ double dev_squared_distance(const point_t &lhs, const point_t &rhs) {
  const double dx = lhs.x - rhs.x;
  const double dy = lhs.y - rhs.y;

  return dx * dx + dy * dy;
}

__global__ void cuda_assign_points_to_clusters(const point_t *points, const size_t num_points,
                                               const point_t *centroids, const size_t num_centroids,
                                               size_t *associations) {

  const auto p_idx = blockIdx.x * blockDim.x + threadIdx.x; // Point index

  if (p_idx >= num_points) {
    return; // Out-of-bounds guard
  }

  double min_val = INFINITY;
  size_t min_idx = 0;

  for (size_t c_idx = 0; c_idx < num_centroids; c_idx++) {
    double dist = dev_squared_distance(points[p_idx], centroids[c_idx]);

    if (dist < min_val) {
      min_val = dist;
      min_idx = c_idx;
    }
  }

  // xPres: Molto divertente che se non metti -arch=sm_35 compila tranquillamente ma poi non
  // lancia i kernel neanche a pagare, e ovviamente NON DA NESSUN TIPO DI ERRORE
  associations[p_idx] = min_idx;
}

__global__ void cuda_prep_new_centroids(const size_t num_centroids, point_t *new_centroids,
                                        count_t *new_counts) {
  const auto k_idx = blockIdx.x * blockDim.x + threadIdx.x; // Cluster index

  if (k_idx >= num_centroids) {
    return; // Out-of-bounds guard
  }

  new_centroids[k_idx] = point_t{0.0, 0.0};
  new_counts[k_idx]    = 0;
}

__global__ void cuda_partial_reduction(const point_t *points, const size_t num_points,
                                       const size_t *associations, point_t *new_centroids,
                                       count_t *new_counts) {

  const auto p_idx = blockIdx.x * blockDim.x + threadIdx.x; // Point index (1 dim)

  if (p_idx >= num_points) {
    return; // Out-of-bounds guard
  }

  const auto k_idx = associations[p_idx]; // Cluster index

  // Use atomic operations to update partial results
  atomicAdd(&new_centroids[k_idx].x, static_cast<float>(points[p_idx].x));
  atomicAdd(&new_centroids[k_idx].y, static_cast<float>(points[p_idx].y));
  atomicAdd(&new_counts[k_idx], count_t{1});
}

__global__ void cuda_final_reduction(const point_t *centroids, point_t *new_centroids,
                                     const count_t *new_counts) {

  const auto k_idx = blockIdx.x * blockDim.x + threadIdx.x; // Cluster index

  const auto count = new_counts[k_idx];

  if (count <= 0) {
    // No points in cluster, centroid remains unchanged
    new_centroids[k_idx] = centroids[k_idx];
    return;
  }

  new_centroids[k_idx].x /= static_cast<float>(count);
  new_centroids[k_idx].y /= static_cast<float>(count);
}

__global__ void cuda_check_convergence(const size_t num_centroids, double tol,
                                       const point_t *centroids, const point_t *new_centroids,
                                       bool *converged) {

  // tolerance must be squared
  tol = tol * tol;

  bool conv = true;
  for (size_t i = 0; i < num_centroids; i++) {
    conv &= dev_squared_distance(centroids[i], new_centroids[i]) < tol;

    if (!conv) {
      break;
    }
  }

  *converged = conv; // access pointer one time
}

__global__ void cuda_print_point(const point_t *data) {
  printf("Point: (%f, %f)\n", data->x, data->y);
}

kmeans_cluster_t kmeans_cuda::cluster(const size_t max_iter, const double tol) {
  const size_t num_points    = points.size();
  const size_t num_centroids = this->num_centroids;

  cuda_check(cudaSetDevice(0));

  // Points
  auto dev_points        = cuda_require_dev_malloc<point_t>(num_points);
  // Centroids
  auto dev_centroids     = cuda_require_dev_malloc<point_t>(num_centroids);
  // Updated centroids after each iteration
  auto dev_new_centroids = cuda_require_dev_malloc<point_t>(num_centroids);
  auto dev_new_counts    = cuda_require_dev_malloc<count_t>(num_centroids);
  // Association of each point to a cluster
  auto dev_associations  = cuda_require_dev_malloc<size_t>(num_points);
  auto dev_converged     = cuda_require_dev_malloc<bool>(1);

  // copy points to device memory
  cuda_check(cuda_memcopy(dev_points, points.data(), num_points, cudaMemcpyHostToDevice));
  {
    cuda_check(cudaDeviceSynchronize());
    ;
    auto points_h = std::vector<point_t>(num_points);
    cuda_check(cuda_memcopy(points_h.data(), dev_points, num_points, cudaMemcpyDeviceToHost));
    cuda_check(cudaDeviceSynchronize());
    ;

    std::clog << "First 10 points:\n";
    for (size_t i = 0; i < 10 && i < points_h.size(); i++) {
      std::clog << "Point " << i << ": (" << points_h[i].x << ", " << points_h[i].y << ")\n";
    }
  }

  // consider the first k points as the initial centroids
  cuda_check(cuda_memcopy(dev_centroids, points.data(), num_centroids, cudaMemcpyHostToDevice));
  cuda_check(cudaMemset(dev_converged, false, sizeof(bool)));

  cuda_check(cudaDeviceSynchronize());
  ;

  // main cycle: assign points to clusters, calculate new centroids, check for
  // convergence

  for (iter = 0; iter < max_iter; iter++) {
    // Step 1: Assign points to clusters
    // For each point, calculate the distance to each centroid and assign the point to the closest
    // one

    {
      uint32_t block_size = 256;
      uint32_t grid_size  = (num_points + block_size - 1) / block_size;
      std::clog << "cuda_assign_points_to_clusters<<<" << block_size << ", " << grid_size << ">>>("
                << dev_points << ", " << num_points << ", " << dev_centroids << ", "
                << num_centroids << ", " << dev_associations << ")\n";
      cuda_assign_points_to_clusters<<<block_size, grid_size>>>(
          dev_points, num_points, dev_centroids, num_centroids, dev_associations);

      cuda_check(cudaDeviceSynchronize());
    }

    // Step 2: calculate new centroids by averaging the points in each cluster

    // Step 2.0: Prepare new centroids
    {
      uint32_t block_size = (num_centroids < 256) ? num_centroids : 256;
      uint32_t grid_size  = (num_centroids + block_size - 1) / block_size;

      std::clog << "running cuda_prep_new_centroids with block_size=" << block_size
                << ", grid_size=" << grid_size << "\n";
      cuda_prep_new_centroids<<<block_size, grid_size>>>(num_centroids, dev_new_centroids,
                                                         dev_new_counts);

      cuda_check(cudaDeviceSynchronize());
      ;
    }

    // Step 2.1: Parallel reduction over points
    {
      uint32_t block_size = 256;
      uint32_t grid_size  = (num_points + block_size - 1) / block_size;

      std::clog << "running cuda_partial_reduction with block_size=" << block_size
                << ", grid_size=" << grid_size << "\n";
      cuda_partial_reduction<<<block_size, grid_size>>>(dev_points, num_points, dev_associations,
                                                        dev_new_centroids, dev_new_counts);
      cuda_check(cudaDeviceSynchronize());
      ;
    }

    // Step 2.2: Final reduction and compute centroids
    {
      uint32_t block_size = (num_centroids < 256) ? num_centroids : 256;
      uint32_t grid_size  = (num_centroids + block_size - 1) / block_size;

      std::clog << "running cuda_final_reduction with block_size=" << block_size
                << ", grid_size=" << grid_size << "\n";
      cuda_final_reduction<<<block_size, grid_size>>>(dev_centroids, dev_new_centroids,
                                                      dev_new_counts);

      cuda_check(cudaDeviceSynchronize());
      ;
    }

    // Step 3: Check for convergence
    {
      std::clog << "running cuda_check_convergence\n";
      cuda_check_convergence<<<1, 1>>>(num_centroids, tol, dev_centroids, dev_new_centroids,
                                       dev_converged);
      cuda_check(cudaDeviceSynchronize());
    }

    // Step 4: Update centroids
    cuda_check(
        cuda_memcopy(dev_centroids, dev_new_centroids, num_centroids, cudaMemcpyDeviceToDevice));
    cuda_check(cudaDeviceSynchronize());

    bool converged_h;
    cuda_check(cuda_memcopy(&converged_h, dev_converged, 1, cudaMemcpyDeviceToHost));
    cuda_check(cudaDeviceSynchronize());

    if (converged_h) {
      break;
    }
  }

  auto centroids_h    = std::vector<point_t>(num_centroids);
  auto associations_h = std::vector<size_t>(points.size());

  cuda_check(
      cuda_memcopy(centroids_h.data(), dev_centroids, num_centroids, cudaMemcpyDeviceToHost));
  cuda_check(
      cuda_memcopy(associations_h.data(), dev_associations, points.size(), cudaMemcpyDeviceToHost));

  auto clusters_h = std::vector<std::vector<point_t>>{centroids_h.size()};
  for (auto &cluster : clusters_h) {
    cluster = std::vector<point_t>{};
  }

  for (size_t i = 0; i < points.size(); i++) {
    clusters_h[associations_h[i]].push_back(points[i]);
  }

  // free memory
  cuda_check(cuda_dev_free(dev_points));
  cuda_check(cuda_dev_free(dev_centroids));
  cuda_check(cuda_dev_free(dev_new_centroids));
  cuda_check(cuda_dev_free(dev_associations));
  cuda_check(cuda_dev_free(dev_new_counts));
  cuda_check(cuda_dev_free(dev_converged));

  return kmeans_cluster_t{centroids_h, clusters_h};
}
