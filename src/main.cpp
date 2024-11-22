#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "kmeans.hpp"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <data.csv> <k> [output_file]"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::ostream *output = &std::cout;

  std::string input_filename{argv[1]};
  size_t k = std::stoi(argv[2]);


  std::ofstream output_steam; // declared outside the block to get destroyed at main end
  if (argc == 4) {
    std::string output_filename{argv[3]};
    output_steam.open(output_filename);
    output = &output_steam;
  }

  // read data from csv file
  std::vector<Point> data;
  std::ifstream input_file{input_filename};

  if (!input_file.is_open()) {
    std::cerr << "Failed to open file: " << input_filename << std::endl;
    return EXIT_FAILURE;
  }

  std::string line;
  while (std::getline(input_file, line)) {
    std::istringstream ss{line};
    std::string token;
    std::getline(ss, token, ',');
    float x = std::stof(token);
    std::getline(ss, token, ',');
    float y = std::stof(token);

    data.push_back(Point{x, y});
  }

  std::clog << "Data size: " << data.size() << std::endl;

  auto [centroids, clusters] = kmeans(k, data);

  // print a json struct [{c: [x,y], a: [[x,y], ...
  *output << "[";
  for (size_t i = 0; i < k; ++i) {
    *output << "{\"c\": [" << centroids[i].x << ", " << centroids[i].y
              << "], \"a\": [";
    for (size_t j = 0; j < clusters[i].size(); ++j) {
      *output << "[" << clusters[i][j].x << ", " << clusters[i][j].y << "]";
      if (j != clusters[i].size() - 1) {
        *output << ", ";
      }
    }
    *output << "]}";
    if (i != k - 1) {
      *output << ", ";
    }
  }
  *output << "]" << std::endl;

  return EXIT_SUCCESS;
}
