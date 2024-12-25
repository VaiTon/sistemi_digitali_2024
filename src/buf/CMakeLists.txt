add_library(kmeans-buf kmeans_buf.hpp
        kmeans_buf.cpp)

add_sycl_to_target(TARGET kmeans-buf)
target_link_libraries(kmeans-buf PUBLIC OpenMP::OpenMP_CXX)