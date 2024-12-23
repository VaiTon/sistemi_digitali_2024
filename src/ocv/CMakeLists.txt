add_library(kmeans-ocv kmeans_ocv.hpp
        kmeans_ocv.cpp)

target_link_libraries(kmeans-ocv PUBLIC kmeans)
find_package(OpenCV REQUIRED)
target_include_directories(kmeans-ocv PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(kmeans-ocv PRIVATE ${OpenCV_LIBS})
