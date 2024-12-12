cmake_minimum_required(VERSION 3.20)
project(kmeans CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_ONEAPI "Use Intel SYCL implementation" OFF)
option(KMEANS_OMP "Use OpenMP for KMeans" ON)
option(KMEANS_BUF "Add buffer implementation" ON)
set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp targets")

if (KMEANS_OMP)
    find_package(OpenMP REQUIRED)
    add_compile_definitions(KMEANS_OMP)
    link_libraries(${OpenMP_CXX_LIBRARIES})
endif ()

if (USE_ONEAPI)
    find_package(IntelSYCL REQUIRED)
    set(CMAKE_CXX_COMPILER icpx)
    set(CMAKE_C_COMPILER icx)
    add_compile_options(-fsycl)
else ()
    find_package(AdaptiveCpp REQUIRED)

    if (ACPP_TARGETS MATCHES "^omp.+")
        add_compile_options(-fopenmp)
    endif ()

    message(STATUS "Using AdaptiveCpp")
    message(STATUS "AdaptiveCpp targets: ${ACPP_TARGETS}")
    set(ENV{ACPP_TARGETS} ${ACPP_TARGETS})
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

if (SYCL_TYPE STREQUAL "BUF")
    add_definitions(-DUSE_BUFFERS)
endif ()

# Ninja colorize output
if (CMAKE_GENERATOR STREQUAL "Ninja")
    add_definitions(-fdiagnostics-color=always)
endif ()

add_compile_options(-msse4.1)

set(SOURCES
        src/point.hpp
        src/cpu/kmeans_cpu.hpp src/cpu/kmeans_cpu.cpp
        src/usm/kmeans_usm.hpp src/usm/kmeans_usm.cpp
        src/buf/kmeans_buf.hpp src/buf/kmeans_buf.cpp
        src/simd/kmeans_simd.hpp src/simd/kmeans_simd.cpp
)


add_compile_definitions(SYCL_TYPE="${SYCL_TYPE}")

include_directories(src)

add_executable(test-devices ${SOURCES} test/test_devices.cpp test/util.hpp)
add_sycl_to_target(TARGET test-devices src/buf/kmeans_buf.cpp src/usm/kmeans_usm.cpp)

add_executable(test-correctness ${SOURCES} test/test_correctness.cpp test/util.hpp)
add_sycl_to_target(TARGET test-correctness src/buf/kmeans_buf.cpp src/usm/kmeans_usm.cpp)

add_executable(test-results ${SOURCES} test/test_results.cpp test/util.hpp)
add_sycl_to_target(TARGET test-results src/buf/kmeans_buf.cpp src/usm/kmeans_usm.cpp)
