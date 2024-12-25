add_library(kmeans-usm kmeans_usm.hpp
        kmeans_usm_v2.cpp
        kmeans_usm_v3.cpp)

add_sycl_to_target(TARGET kmeans-usm)
target_link_libraries(kmeans-usm PUBLIC OpenMP::OpenMP_CXX)