cmake_minimum_required(VERSION 3.20)
project(kmeans CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(USE_ONEAPI "Use Intel SYCL implementation" OFF)

set(KMEANS_TEST_NUM_POINTS 1000000 CACHE STRING "Number of points to use in tests")
set(KMEANS_TEST_NUM_CLUSTERS 128 CACHE STRING "Number of clusters to use in tests")


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

# Ninja colorize output
if (CMAKE_GENERATOR STREQUAL "Ninja")
    add_definitions(-fdiagnostics-color=always)
endif ()

add_compile_definitions(SYCL_TYPE="${SYCL_TYPE}")

include_directories(src)

add_library(kmeans INTERFACE src/kmeans.hpp)

find_package(OpenMP REQUIRED)
add_library(kmeans-cpu src/cpu/kmeans_cpu.hpp
        src/cpu/kmeans_cpu_v1.cpp
        src/cpu/kmeans_cpu_v2.cpp
        src/cpu/kmeans_cpu_v3.cpp)
target_link_libraries(kmeans-cpu PUBLIC kmeans)
target_link_libraries(kmeans-cpu PUBLIC OpenMP::OpenMP_CXX)


add_library(kmeans-usm src/usm/kmeans_usm.hpp
        src/usm/kmeans_usm.hpp
        src/usm/kmeans_usm_v2.cpp
        src/usm/kmeans_usm_v3.cpp)
add_sycl_to_target(TARGET kmeans-usm src/usm/kmeans_usm.cpp)
target_link_libraries(kmeans-usm PUBLIC kmeans)

add_library(kmeans-buf src/buf/kmeans_buf.hpp src/buf/kmeans_buf.cpp)
add_sycl_to_target(TARGET kmeans-buf src/buf/kmeans_buf.cpp)
target_link_libraries(kmeans-buf PUBLIC kmeans)

add_library(kmeans-simd src/simd/kmeans_simd.hpp src/simd/kmeans_simd.cpp)
add_sycl_to_target(TARGET kmeans-simd src/simd/kmeans_simd.cpp)
target_link_libraries(kmeans-simd PUBLIC kmeans)
target_compile_options(kmeans-simd PRIVATE -msse4.1)

## Tests

set(LIBRARIES kmeans-cpu kmeans-usm kmeans-buf kmeans-simd)

add_executable(test-devices test/test_devices.cpp test/util.hpp)
target_link_libraries(test-devices PRIVATE ${LIBRARIES})
add_sycl_to_target(TARGET test-devices)

add_executable(test-correctness test/test_correctness.cpp test/util.hpp)
target_link_libraries(test-correctness PRIVATE ${LIBRARIES})
add_sycl_to_target(TARGET test-correctness)

add_executable(test-results test/test_results.cpp test/util.hpp)
target_link_libraries(test-results PRIVATE ${LIBRARIES})
add_sycl_to_target(TARGET test-results)


set(TEST_DATA_FILE "${CMAKE_BINARY_DIR}/data.txt")
add_custom_command(OUTPUT ${TEST_DATA_FILE}
        COMMAND sh -c 'python3 ${CMAKE_SOURCE_DIR}/scripts/rand_data.py ${KMEANS_TEST_NUM_POINTS} > ${TEST_DATA_FILE}'
)
add_custom_target(generate_data DEPENDS ${TEST_DATA_FILE})

add_custom_target(runtest_results
        COMMAND mkdir -p ${CMAKE_BINARY_DIR}/results
        COMMAND ${CMAKE_BINARY_DIR}/test-results ${TEST_DATA_FILE} ${KMEANS_TEST_NUM_CLUSTERS} ${CMAKE_BINARY_DIR}/results
        DEPENDS generate_data test-results
        USES_TERMINAL
)

add_custom_target(runtest_correctness
        COMMAND ${CMAKE_BINARY_DIR}/test-correctness ${TEST_DATA_FILE} ${KMEANS_TEST_NUM_CLUSTERS}
        DEPENDS generate_data test-correctness
        USES_TERMINAL
)

add_custom_target(runtest_devices
        COMMAND ${CMAKE_BINARY_DIR}/test-devices ${TEST_DATA_FILE} ${KMEANS_TEST_NUM_CLUSTERS}
        DEPENDS test-devices generate_data
        USES_TERMINAL
)
