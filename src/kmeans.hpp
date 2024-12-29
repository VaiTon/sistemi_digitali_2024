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
  kmeans(size_t k, std::vector<point_t> const &points);

  [[nodiscard]] size_t get_iters() const { return iter; }

  virtual kmeans_cluster_t cluster(size_t max_iter, double tol) = 0;
  virtual ~kmeans()                                             = default;

protected:
  size_t const                num_centroids;
  std::vector<point_t> const &points;
  size_t                      iter = 0;
};

inline kmeans::kmeans(size_t const k, std::vector<point_t> const &points)
    : num_centroids(k), points(points) {
  if (k == 0) {
    throw std::invalid_argument("Number of clusters must be greater than 0");
  }
  if (k > points.size()) {
    throw std::invalid_argument("Number of clusters must be less than or equal "
                                "to the number of points");
  }
}

#endif // KMEANS_HPP
