#ifndef KMEANS_USM_H
#define KMEANS_USM_H
#include <kmeans.hpp>
#include <sycl/sycl.hpp>

class kmeans_usm final : kmeans {
public:
  kmeans_usm(const sycl::queue &q, const size_t k, const std::vector<point_t> &points)
      : q(q), k(k), points(points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
  size_t           get_iters() override { return iter; }

private:
  sycl::queue                q;
  const size_t               k;
  const std::vector<point_t> points;
  size_t                     iter = 0;
};

#endif // KMEANS_USM_H
