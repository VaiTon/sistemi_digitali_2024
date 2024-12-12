#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <cstddef>
#include <vector>

#include "point.hpp"

struct kmeans_cluster_t {
  std::vector<point_t>              centroids;
  std::vector<std::vector<point_t>> clusters;
};

class kmeans {
public:
  virtual kmeans_cluster_t cluster(size_t max_iter, double tol) = 0;
  virtual size_t           get_iters()                          = 0;
  virtual ~kmeans()                                             = default;
};

inline double squared_distance(const point_t &lhs, const point_t &rhs) {
  const double dx = lhs.x - rhs.x;
  const double dy = lhs.y - rhs.y;

  return dx * dx + dy * dy;
}

#endif // KMEANS_HPP
