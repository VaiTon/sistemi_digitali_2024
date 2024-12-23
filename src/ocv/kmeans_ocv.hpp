#ifndef KMEANS_OCV_HPP
#define KMEANS_OCV_HPP

#include "kmeans.hpp"

class kmeans_ocv final : public kmeans {
public:
  kmeans_ocv(const size_t k, const std::vector<point_t> &data) : kmeans(k, data) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};

#endif // KMEANS_OCV_HPP
