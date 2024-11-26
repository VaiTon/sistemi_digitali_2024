cmake_minimum_required(VERSION 3.26.5)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(kmeans CXX)

option(USE_ONEAPI "Use Intel SYCL implementation" OFF)
set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp targets")

option(SD_SYCL_REDUCTION "Use SYCL reduction" OFF)
if (SD_SYCL_REDUCTION)
    add_definitions(-DSD_SYCL_REDUCTION)
endif ()

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

add_executable(test-devices
        src/kmeans.hpp
        src/kmeans_sycl.hpp
        src/kmeans_cpu.cpp
        src/kmeans_buf.cpp
        src/kmeans_usm.cpp
        test/test_devices.cpp
)
add_sycl_to_target(TARGET test-devices)

include_directories(src)
add_executable(test-correctness
        src/kmeans.hpp
        src/kmeans_sycl.hpp
        src/kmeans_cpu.cpp
        src/kmeans_buf.cpp
        src/kmeans_usm.cpp
        test/test_correctness.cpp
)
add_sycl_to_target(TARGET test-correctness)
