#ifndef KMEANS_CPU_HPP
#define KMEANS_CPU_HPP

#include <kmeans.hpp>

class kmeans_cpu final : kmeans {
public:
  kmeans_cpu(const size_t k, const std::vector<point_t> &points) : k(k), points(points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
  size_t           get_iters() override { return iter; }

private:
  const size_t               k;
  const std::vector<point_t> points;
  size_t                     iter = 0;
};

#endif // KMEANS_CPU_HPP
