#pragma once

#include <kmeans.hpp>

/// Naive CPU-based implementation of the k-means algorithm.
/// Creates lists of associated points for each cluster for each iteration.
class kmeans_cuda final : public kmeans {
public:
  kmeans_cuda(const size_t k, const std::vector<point_t> &points) : kmeans(k, points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};
