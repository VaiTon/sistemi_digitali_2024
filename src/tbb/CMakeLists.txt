add_library(kmeans-tbb kmeans_tbb.hpp
        kmeans_tbb.cpp)

target_link_libraries(kmeans-tbb PRIVATE tbb)
target_link_libraries(kmeans-tbb PUBLIC kmeans)