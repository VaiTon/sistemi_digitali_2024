#ifndef KMEANS_USM_H
#define KMEANS_USM_H

#include <kmeans.hpp>
#include <sycl/sycl.hpp>

class kmeans_usm final : public kmeans {
public:
  kmeans_usm(const sycl::queue &q, const size_t k, const std::vector<point_t> &points)
      : kmeans(k, points), q(q) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;

private:
  sycl::queue q;
};

#endif // KMEANS_USM_H
