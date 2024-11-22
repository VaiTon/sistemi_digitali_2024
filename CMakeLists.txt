cmake_minimum_required(VERSION 3.27.5)

set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(kmeans CXX)

option(USE_ONEAPI "Use Intel SYCL implementation" OFF)
set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp targets")

if (USE_ONEAPI)
    find_package(IntelSYCL REQUIRED)
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
add_subdirectory(src)
