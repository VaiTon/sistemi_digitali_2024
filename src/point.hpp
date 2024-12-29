#ifndef KMEANS_POINT_HPP
#define KMEANS_POINT_HPP

#include <ostream>

struct point_t {
  float x, y;

  [[nodiscard]] bool  is_zero() const;
  [[nodiscard]] float squared_distance(point_t const &rhs) const;
  point_t             operator+(point_t const &rhs) const;
  point_t            &operator+=(point_t const &rhs);
};

inline bool    point_t::is_zero() const { return x == 0.0f && y == 0.0f; }
inline point_t point_t::operator+(point_t const &rhs) const { return {x + rhs.x, y + rhs.y}; }

inline point_t &point_t::operator+=(point_t const &rhs) {
  x += rhs.x;
  y += rhs.y;
  return *this;
}

inline bool operator==(point_t const &lhs, point_t const &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

inline std::ostream &operator<<(std::ostream &os, point_t const &p) {
  os << "[" << p.x << ", " << p.y << "]";
  return os;
}

inline float squared_distance(point_t const &lhs, point_t const &rhs) {
  float const dx = lhs.x - rhs.x;
  float const dy = lhs.y - rhs.y;

  return dx * dx + dy * dy;
}

inline float point_t::squared_distance(point_t const &rhs) const {
  return ::squared_distance(*this, rhs);
}

#endif // KMEANS_POINT_HPP
