// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
// v57 (Windows build check wincheck57c): aligned host allocation for both toolchains. MSVC's C library has no
// std::aligned_alloc; it has _aligned_malloc / _aligned_free (which must free what it allocates).
#pragma once
#include <cstddef>
#include <cstdlib>
#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace pelias::os {
// n bytes at an `align` boundary (align a power of two; n a multiple of align, as std::aligned_alloc asks); nullptr on failure
inline void* aligned_alloc_bytes(std::size_t align, std::size_t n) {
#if defined(_MSC_VER)
  return _aligned_malloc(n, align);
#else
  return std::aligned_alloc(align, n);
#endif
}
inline void aligned_free_bytes(void* p) {
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  std::free(p);
#endif
}
}  // namespace pelias::os
