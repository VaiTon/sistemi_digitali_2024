add_library(kmeans-omp kmeans_omp.hpp
        kmeans_omp.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(kmeans-omp PUBLIC kmeans)
target_link_libraries(kmeans-omp PUBLIC OpenMP::OpenMP_CXX)
