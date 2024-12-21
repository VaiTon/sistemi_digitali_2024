#ifndef KMEANS_SIMD_HPP
#define KMEANS_SIMD_HPP

#include <kmeans.hpp>
#include <vector>

class kmeans_simd final : public kmeans {
public:
  kmeans_simd(const size_t k, const std::vector<point_t> &points) : kmeans(k, points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};

#endif // KMEANS_SIMD_HPP