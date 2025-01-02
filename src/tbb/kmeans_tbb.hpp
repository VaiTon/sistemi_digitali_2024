#pragma once
#include "kmeans.hpp"

class kmeans_tbb final : public kmeans {
public:
  kmeans_tbb(size_t const k, std::vector<point_t> const &data) : kmeans(k, data) {}

  kmeans_cluster_t cluster(size_t max_iter, double tol) override;
};