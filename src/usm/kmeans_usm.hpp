#ifndef KMEANS_USM_H
#define KMEANS_USM_H

#include <kmeans.hpp>
#include <sycl/sycl.hpp>

class kmeans_usm : public kmeans {
public:
  kmeans_usm(const sycl::queue &q, const size_t k, const std::vector<point_t> &points)
      : kmeans(k, points), q(q) {}

protected:
  sycl::queue q;
};

class kmeans_usm_v1 final : public kmeans_usm {
public:
  kmeans_usm_v1(const sycl::queue &q, const size_t k, const std::vector<point_t> &points)
      : kmeans_usm(q, k, points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};

class kmeans_usm_v2 final : public kmeans_usm {
public:
  kmeans_usm_v2(const sycl::queue &q, const size_t k, const std::vector<point_t> &points)
      : kmeans_usm(q, k, points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};

class kmeans_usm_v3 final : public kmeans_usm {
public:
  kmeans_usm_v3(const sycl::queue &q, const size_t k, const std::vector<point_t> &points)
      : kmeans_usm(q, k, points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};

#endif // KMEANS_USM_H
