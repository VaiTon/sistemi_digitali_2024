#ifndef KMEANS_SIMD_HPP
#define KMEANS_SIMD_HPP

#include <kmeans.hpp>
#include <vector>

class kmeans_simd final : kmeans {
public:
  kmeans_simd(const size_t k, const std::vector<point_t> &points) : k_(k), points_(points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
  size_t           get_iters() override { return iter; }

private:
  const size_t               k_;
  const std::vector<point_t> points_;
  size_t                     iter = 0;
};

#endif // KMEANS_SIMD_HPP