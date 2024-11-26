#pragma once
#include "kmeans.hpp"

#include <sycl/sycl.hpp>

kmeans_cluster_t kmeans_buf(sycl::queue q, size_t k, const vector<Point> &points, size_t max_iter, float tol);
kmeans_cluster_t kmeans_usm(sycl::queue q, size_t k, const vector<Point> &points, size_t max_iter, float tol);
