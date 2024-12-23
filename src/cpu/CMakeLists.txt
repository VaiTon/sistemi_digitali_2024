add_library(kmeans-cpu kmeans_cpu.hpp
        kmeans_cpu_v1.cpp
        kmeans_cpu_v2.cpp)

target_link_libraries(kmeans-cpu PUBLIC kmeans)