add_executable(kmeans-cpu main.cpp
        kmeans_cpu.cpp)

add_executable(kmeans-gpu main.cpp
        kmeans_sycl.cpp)
add_sycl_to_target(TARGET kmeans-gpu SOURCES kmeans_sycl.cpp)
