add_library(kmeans INTERFACE kmeans.hpp point.hpp)

add_subdirectory(cpu)
add_subdirectory(omp)
add_subdirectory(usm)
add_subdirectory(buf)
add_subdirectory(simd)
add_subdirectory(ocv)

set(LIBRARIES kmeans-cpu kmeans-usm kmeans-buf kmeans-simd kmeans-omp kmeans-ocv)

if (USE_CUDA)
    add_subdirectory(cuda)
endif()

message(STATUS "Libraries: ${LIBRARIES}")
set(LIBRARIES ${LIBRARIES} PARENT_SCOPE)
