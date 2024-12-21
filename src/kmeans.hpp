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
  kmeans(const size_t k, const std::vector<point_t> &points) : k(k), points(points) {
    if (k == 0) {
      throw std::invalid_argument("Number of clusters must be greater than 0");
    }
    if (k > points.size()) {
      throw std::invalid_argument("Number of clusters must be less than or equal "
                                  "to the number of points");
    }
  }

  size_t get_iters() const { return iter; }

  virtual kmeans_cluster_t cluster(size_t max_iter, double tol) = 0;
  virtual ~kmeans()                                             = default;

protected:
  const size_t               k;
  const std::vector<point_t> points;
  size_t                     iter = 0;
};

inline double squared_distance(const point_t &lhs, const point_t &rhs) {
  const double dx = lhs.x - rhs.x;
  const double dy = lhs.y - rhs.y;

  return dx * dx + dy * dy;
}

#endif // KMEANS_HPP
