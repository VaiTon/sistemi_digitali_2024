add_library(kmeans-buf kmeans_buf.hpp
        kmeans_buf.cpp)

add_sycl_to_target(TARGET kmeans-buf)