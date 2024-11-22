#pragma once

#include <cmath>
#include <cstddef>
#include <ostream>
#include <vector>

using std::ostream;
using std::vector;

struct Point {
  float x, y;
};

struct kmeans_cluster_t {
  vector<Point>         centroids;
  vector<vector<Point>> clusters;
};

kmeans_cluster_t kmeans(size_t k, const vector<Point> &points);

inline bool operator==(const Point &lhs, const Point &rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }

inline std::ostream &operator<<(std::ostream &os, const Point &p) {
  os << "[" << p.x << ", " << p.y << "]";
  return os;
}

inline double squared_distance(const Point &lhs, const Point &rhs) {
  const double dx = lhs.x - rhs.x;
  const double dy = lhs.y - rhs.y;

  return dx * dx + dy * dy;
}
