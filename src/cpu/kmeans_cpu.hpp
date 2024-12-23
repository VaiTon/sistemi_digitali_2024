#ifndef KMEANS_CPU_HPP
#define KMEANS_CPU_HPP

#include <kmeans.hpp>

/// Naive CPU-based implementation of the k-means algorithm.
/// Creates lists of associated points for each cluster for each iteration.
class kmeans_cpu_v1 final : public kmeans {
public:
  kmeans_cpu_v1(const size_t k, const std::vector<point_t> &points) : kmeans(k, points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};

/// CPU-based implementation of the k-means algorithm.
/// Uses an associations list to keep track of the points associated with each cluster.
class kmeans_cpu_v2 final : public kmeans {
public:
  kmeans_cpu_v2(const size_t k, const std::vector<point_t> &points) : kmeans(k, points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};



#endif // KMEANS_CPU_HPP
