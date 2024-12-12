#ifndef KMEANS_POINT_HPP
#define KMEANS_POINT_HPP

#include <ostream>

struct point_t {
  float x, y;
};

inline bool operator==(const point_t &lhs, const point_t &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

inline std::ostream &operator<<(std::ostream &os, const point_t &p) {
  os << "[" << p.x << ", " << p.y << "]";
  return os;
}

#endif // KMEANS_POINT_HPP
