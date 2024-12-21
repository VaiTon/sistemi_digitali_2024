#ifndef KMEANS_CPU_HPP
#define KMEANS_CPU_HPP

#include <kmeans.hpp>

class kmeans_omp final : public kmeans {
public:
  kmeans_omp(const size_t k, const std::vector<point_t> &points) : kmeans(k, points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};

#endif // KMEANS_CPU_HPP
