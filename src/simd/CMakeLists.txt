add_library(kmeans-simd kmeans_simd.hpp
        kmeans_simd.cpp)

target_compile_options(kmeans-simd PRIVATE -msse4.2)
target_link_libraries(kmeans-simd PUBLIC kmeans)