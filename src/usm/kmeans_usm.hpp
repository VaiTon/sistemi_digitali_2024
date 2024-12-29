#ifndef KMEANS_USM_H
#define KMEANS_USM_H

#include <kmeans.hpp>
#include <sycl/sycl.hpp>

class kmeans_usm : public kmeans {
public:
  kmeans_usm(sycl::queue const &q, size_t const k, std::vector<point_t> const &points)
      : kmeans(k, points), q(q) {}

protected:
  sycl::queue q;
};

class kmeans_usm_v1 final : public kmeans_usm {
public:
  kmeans_usm_v1(sycl::queue const &q, size_t const k, std::vector<point_t> const &points)
      : kmeans_usm(q, k, points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};

class kmeans_usm_v2 final : public kmeans_usm {
public:
  kmeans_usm_v2(sycl::queue const &q, size_t const k, std::vector<point_t> const &points)
      : kmeans_usm(q, k, points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};

class kmeans_usm_v3 final : public kmeans_usm {
public:
  kmeans_usm_v3(sycl::queue const &q, size_t const k, std::vector<point_t> const &points)
      : kmeans_usm(q, k, points) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};

#endif // KMEANS_USM_H
