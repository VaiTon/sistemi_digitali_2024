cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(kmeans CXX)

option(USE_ONEAPI "Use Intel SYCL implementation" OFF)
option(CPU_OPENMP "Disable OpenMP in CPU implementation" OFF)

set(SYCL_TYPE "USM" CACHE STRING "Use USM or Buffers")
set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp targets")

if (USE_ONEAPI)
    find_package(IntelSYCL REQUIRED)
    set(CMAKE_CXX_COMPILER icpx)
    set(CMAKE_C_COMPILER icx)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
else ()
    find_package(AdaptiveCpp REQUIRED)

    if (ACPP_TARGETS MATCHES "^omp.+")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif ()

    message(STATUS "Using AdaptiveCpp")
    message(STATUS "AdaptiveCpp targets: ${ACPP_TARGETS}")
    set(ENV{ACPP_TARGETS} ${ACPP_TARGETS})
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# OpenMP
if (CPU_OPENMP)
    if (USE_ONEAPI)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
    else ()
        find_package(OpenMP REQUIRED)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_compile_definitions(USE_OPENMP)
    endif ()
endif ()

# Ninja colorize output
if (CMAKE_GENERATOR STREQUAL "Ninja")
    add_definitions(-fdiagnostics-color=always)
endif ()


if (SYCL_TYPE STREQUAL "USM")
    set(SOURCES src/kmeans.hpp src/kmeans_cpu.cpp src/kmeans_usm.cpp)
elseif (SYCL_TYPE STREQUAL "BUF")
    set(SOURCES src/kmeans.hpp src/kmeans_cpu.cpp src/kmeans_buf.cpp)
else ()
    message(FATAL_ERROR "No SYCL_TYPE: ${SYCL_TYPE}. Available: USM, BUF")
endif ()

add_compile_definitions(SYCL_TYPE="${SYCL_TYPE}")

include_directories(src)

add_executable(test-devices ${SOURCES} test/test_devices.cpp)
add_sycl_to_target(TARGET test-devices src/kmeans_buf.cpp src/kmeans_usm.cpp)

add_executable(test-correctness ${SOURCES} test/test_correctness.cpp)
add_sycl_to_target(TARGET test-correctness src/kmeans_buf.cpp src/kmeans_usm.cpp)

add_executable(test-results ${SOURCES} test/test_results.cpp)
add_sycl_to_target(TARGET test-results src/kmeans_buf.cpp src/kmeans_usm.cpp)
