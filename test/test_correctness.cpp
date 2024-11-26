
#include <fstream>
#include <iostream>
int main(const int argc, char **argv) {
  auto         *output = &std::cout;
  std::ofstream output_steam; // declared outside the block to get destroyed at main end
  if (argc == 4) {
    const std::string output_filename{argv[3]};
    output_steam.open(output_filename);
    output = &output_steam;
  }

  *output << "[";
  for (size_t i = 0; i < k; ++i) {
    *output << "{\"c\": [" << centroids[i].x << ", " << centroids[i].y << "], \"a\": [";
    for (size_t j = 0; j < clusters[i].size(); ++j) {
      *output << "[" << clusters[i][j].x << ", " << clusters[i][j].y << "]";
      if (j != clusters[i].size() - 1) {
        *output << ", ";
      }
    }

    *output << (i == k - 1 ? "]}" : "]}, ") << std::endl;
  }
  *output << "]" << std::endl;

  return EXIT_SUCCESS;

  return 0;
}