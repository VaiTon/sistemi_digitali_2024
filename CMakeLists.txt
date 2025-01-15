cmake_minimum_required(VERSION 3.20)
project(kmeans CXX)

set(CMAKE_CXX_STANDARD 17) # on Colab we're stuck with GCC 11
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(KMEANS_ONEAPI "Use Intel SYCL implementation" OFF)

option(KMEANS_CUDA "Use CUDA implementation" OFF)
set(KMEANS_CUDA_ARCH 75 CACHE STRING "CUDA architecture to use")

option(KMEANS_USM "Use SYCL USM implementation" ON)
option(KMEANS_BUFFER "Use SYCL buffer implementation" ON)
option(KMEANS_OCV "Use OpenCV implementation" ON)
option(KMEANS_SIMD "Use x86 SIMD implementation" ON)
option(KMEANS_TBB "Use TBB implementation" ON)


if (KMEANS_CUDA)
    add_compile_definitions(USE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES ${KMEANS_CUDA_ARCH})
    enable_language(CUDA)
    message(STATUS "Using CUDA")
endif ()
if (KMEANS_TBB)
    add_compile_definitions(USE_TBB)
    find_package(TBB REQUIRED)
    message(STATUS "Found TBB: ${TBB_VERSION}")
endif ()


if (KMEANS_ONEAPI)
    find_package(IntelSYCL REQUIRED)
    add_compile_options(-fsycl)
    add_compile_options(-qopenmp)

    message(STATUS "Using Intel SYCL")
else ()
    find_package(AdaptiveCpp REQUIRED)

    if (ACPP_TARGETS MATCHES "^omp.+")
        add_compile_options(-fopenmp)
    endif ()

    message(STATUS "Using AdaptiveCpp with targets: ${ACPP_TARGETS}")
    set(ENV{ACPP_TARGETS} ${ACPP_TARGETS})
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wsign-compare")

# Ninja colorize output
if (CMAKE_GENERATOR STREQUAL "Ninja")
    add_definitions(-fdiagnostics-color=always)
endif ()

add_compile_definitions(SYCL_TYPE="${SYCL_TYPE}")

find_package(OpenMP REQUIRED)
find_package(fmt REQUIRED)

include_directories(src)
add_subdirectory(src)

## Tests

add_executable(test-devices test/test_devices.cpp test/util.hpp)

target_link_libraries(test-devices PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(test-devices PRIVATE fmt::fmt)
target_link_libraries(test-devices PRIVATE ${BACKENDS})
add_sycl_to_target(TARGET test-devices)

add_executable(test-results test/test_results.cpp test/util.hpp)
target_link_libraries(test-results PRIVATE fmt::fmt)
target_link_libraries(test-results PRIVATE ${BACKENDS})
add_sycl_to_target(TARGET test-results)
