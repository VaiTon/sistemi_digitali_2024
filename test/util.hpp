#ifndef UTIL_HPP
#define UTIL_HPP

#include <fstream>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <ostream>
#include <sstream>
#include <vector>
#include <fmt/format.h>

#include "kmeans.hpp"

inline std::vector<point_t> get_data(std::string const &filename) {
  auto data       = std::vector<point_t>{};
  auto input_file = std::ifstream{filename};

  if (!input_file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::string line;
  while (std::getline(input_file, line)) {
    auto ss = std::istringstream{line};

    std::string token;
    std::getline(ss, token, ',');
    float x = std::stof(token);

    std::getline(ss, token, ',');
    float y = std::stof(token);

    data.push_back(point_t{x, y});
  }

  return data;
}

class logger {
public:
  static logger &get() {
    static logger l{std::cerr};
    return l;
  }

  static std::ostream &info() { return get()._output_stream << "-- "; }
  static std::ostream &warn() { return get()._output_stream << "!!: "; }
  static std::ostream &error() { return get()._output_stream << "ERROR: "; }
  static std::ostream &debug() { return get()._output_stream << "DEBUG: "; }
  static std::ostream &trace() { return get()._output_stream << "TRACE: "; }
  static std::ostream &raw() { return get()._output_stream; }

private:
  explicit logger(std::ostream &os) : _output_stream{os} {}
  std::ostream &_output_stream;
};

template <typename T> std::ostream &operator<<(std::ostream &os, std::vector<T> const &p) {
  os << "[";
  for (size_t i = 0; i < p.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << p[i];
  }
  os << "]";
  return os;
}

#endif // UTIL_HPP
