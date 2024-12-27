#pragma once
#include <new>

template <typename T> auto required_ptr(T *ptr) -> T * {
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}
