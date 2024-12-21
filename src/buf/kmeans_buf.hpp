#ifndef KMEANS_BUF_HPP
#define KMEANS_BUF_HPP

#include "kmeans.hpp"
#include <sycl/sycl.hpp>

class kmeans_buf final : public kmeans {
public:
  kmeans_buf(const sycl::queue &q, const size_t k, const std::vector<point_t> &points)
      : kmeans(k, points), q(q) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;

private:
  sycl::queue q;
};

#endif // KMEANS_BUF_HPP
