add_library(kmeans INTERFACE kmeans.hpp point.hpp)

add_subdirectory(cpu)
add_subdirectory(omp)
add_subdirectory(usm)
add_subdirectory(buf)
add_subdirectory(simd)