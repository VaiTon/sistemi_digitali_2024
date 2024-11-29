#pragma once

#include <cstddef>
#include <fstream>
#include <ostream>
#include <sstream>
#include <sycl/sycl.hpp>
#include <vector>

struct Point {
  float x, y;
};

struct kmeans_cluster_t {
  std::vector<Point>              centroids;
  std::vector<std::vector<Point>> clusters;
};

inline bool operator==(const Point &lhs, const Point &rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }

inline std::ostream &operator<<(std::ostream &os, const Point &p) {
  os << "[" << p.x << ", " << p.y << "]";
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const std::vector<Point> &p) {
  for (size_t i = 0; i < p.size(); i++) {
    os << p[i];
    if (i != p.size() - 1) {
      os << ", ";
    }
  }

  return os;
}

inline double squared_distance(const Point &lhs, const Point &rhs) {
  const double dx = lhs.x - rhs.x;
  const double dy = lhs.y - rhs.y;

  return dx * dx + dy * dy;
}

kmeans_cluster_t kmeans_cpu_seq(size_t k, const std::vector<Point> &points, size_t max_iter, float tol);
kmeans_cluster_t kmeans_sycl_buf(sycl::queue q, size_t k, const std::vector<Point> &points, size_t max_iter, float tol);
kmeans_cluster_t kmeans_sycl_usm(sycl::queue q, size_t k, const std::vector<Point> &points, size_t max_iter, float tol);

inline std::vector<Point> get_data(const std::string &filename) {
  auto data       = std::vector<Point>{};
  auto input_file = std::ifstream{filename};

  if (!input_file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::string line;
  while (std::getline(input_file, line)) {
    auto        ss = std::istringstream{line};
    std::string token;
    std::getline(ss, token, ',');
    float x = std::stof(token);
    std::getline(ss, token, ',');
    float y = std::stof(token);

    data.push_back(Point{x, y});
  }

  return data;
}