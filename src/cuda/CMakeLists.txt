add_library(kmeans-cuda kmeans_cuda.hpp kmeans_cuda.cu cuda_utils.cuh)
target_link_libraries(kmeans-cuda ${CUDA_LIBRARIES})
target_compile_options(kmeans-cuda PRIVATE -Wall -Wextra -Wconversion)
set_target_properties(kmeans-cuda PROPERTIES CUDA_ARCHITECTURES 35)

list(APPEND LIBRARIES kmeans-cuda)
set(LIBRARIES ${LIBRARIES} PARENT_SCOPE)
