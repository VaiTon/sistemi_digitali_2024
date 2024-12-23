#ifndef KMEANS_GPU_H
#define KMEANS_GPU_H

#include "kmeans.hpp"

class kmeans_omp final : public kmeans {
public:
  kmeans_omp(const size_t k, const std::vector<point_t> &data) : kmeans(k, data) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};

#endif // KMEANS_GPU_H
