cmake_minimum_required(VERSION 3.20)
project(kmeans CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_ONEAPI "Use Intel SYCL implementation" OFF)
option(KMEANS_OMP "Use OpenMP for KMeans" ON)
set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp targets")

set(KMEANS_TEST_NUM_POINTS 1000000 CACHE STRING "Number of points to use in tests")
set(KMEANS_TEST_NUM_CLUSTERS 128 CACHE STRING "Number of clusters to use in tests")

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

# Ninja colorize output
if (CMAKE_GENERATOR STREQUAL "Ninja")
    add_definitions(-fdiagnostics-color=always)
endif ()

add_compile_options(-msse4.1)
add_compile_definitions(SYCL_TYPE="${SYCL_TYPE}")

include_directories(src)

add_library(kmeans INTERFACE src/kmeans.hpp)

add_library(kmeans-cpu src/cpu/kmeans_cpu.hpp src/cpu/kmeans_cpu.cpp)
target_link_libraries(kmeans-cpu PUBLIC kmeans)
set_property(TARGET kmeans-cpu PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(kmeans-usm src/usm/kmeans_usm.hpp src/usm/kmeans_usm.cpp)
add_sycl_to_target(TARGET kmeans-usm src/usm/kmeans_usm.cpp)
target_link_libraries(kmeans-usm PUBLIC kmeans)
set_property(TARGET kmeans-cpu PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(kmeans-buf src/buf/kmeans_buf.hpp src/buf/kmeans_buf.cpp)
add_sycl_to_target(TARGET kmeans-buf src/buf/kmeans_buf.cpp)
target_link_libraries(kmeans-buf PUBLIC kmeans)
set_property(TARGET kmeans-cpu PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(kmeans-simd src/simd/kmeans_simd.hpp src/simd/kmeans_simd.cpp)
add_sycl_to_target(TARGET kmeans-simd src/simd/kmeans_simd.cpp)
target_link_libraries(kmeans-simd PUBLIC kmeans)
set_property(TARGET kmeans-cpu PROPERTY POSITION_INDEPENDENT_CODE ON)

## Tests

add_executable(test-devices ${SOURCES} test/test_devices.cpp test/util.hpp)
target_link_libraries(test-devices PRIVATE kmeans-cpu kmeans-usm kmeans-buf kmeans-simd)
add_sycl_to_target(TARGET test-devices src/buf/kmeans_buf.cpp src/usm/kmeans_usm.cpp)

add_executable(test-correctness ${SOURCES} test/test_correctness.cpp test/util.hpp)
target_link_libraries(test-correctness PRIVATE kmeans-cpu kmeans-usm kmeans-buf kmeans-simd)
add_sycl_to_target(TARGET test-correctness src/buf/kmeans_buf.cpp src/usm/kmeans_usm.cpp)

add_executable(test-results ${SOURCES} test/test_results.cpp test/util.hpp)
target_link_libraries(test-results PRIVATE kmeans-cpu kmeans-usm kmeans-buf kmeans-simd)
add_sycl_to_target(TARGET test-results src/buf/kmeans_buf.cpp src/usm/kmeans_usm.cpp)


set(TEST_DATA_FILE "${CMAKE_CACHEFILE_DIR}/data.txt")
add_custom_command(OUTPUT ${TEST_DATA_FILE}
        COMMAND sh -c 'python3 ${CMAKE_SOURCE_DIR}/scripts/rand_data.py ${KMEANS_TEST_NUM_POINTS} > ${TEST_DATA_FILE}'
)
add_custom_target(generate_data DEPENDS ${TEST_DATA_FILE})

add_custom_target(runtest_results
        COMMAND ${CMAKE_BINARY_DIR}/test-results ${TEST_DATA_FILE} ${KMEANS_TEST_NUM_CLUSTERS} ${CMAKE_CACHEFILE_DIR}/output
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
