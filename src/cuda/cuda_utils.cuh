#pragma once

#include <stdexcept>

inline void cuda_check(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
  }
}

template <typename T> cudaError_t cuda_dev_malloc(T &ptr, size_t n) {
  return cudaMalloc(&ptr, n * sizeof(T));
}

template <typename T> T *cuda_require_dev_malloc(size_t n) {
  T *ptr;
  cuda_check(cudaMalloc(&ptr, n * sizeof(T)));
  return ptr;
}

template <typename T> cudaError_t cuda_dev_free(T *ptr) { return cudaFree(ptr); }

template <typename T> cudaError_t cuda_memcopy(T *dst, const T *src, size_t n, cudaMemcpyKind kind) {
  return cudaMemcpy(dst, src, n * sizeof(T), kind);
}
