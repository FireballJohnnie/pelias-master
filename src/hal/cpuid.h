// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
#pragma once
// cpuid.h - header-only CPU feature checks and the per-function AVX2 target, the same on gcc, clang and MSVC (the
// Windows build, pre-build list item 4). MSVC compiles AVX2 intrinsics in any function, so its target is empty and the
// check reads cpuid / xgetbv itself; gcc and clang take a target attribute and their builtin.
#if defined(__x86_64__) || defined(_M_X64)
#define PELIAS_HAL_X86 1
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define PELIAS_TARGET_AVX2_FN
#else
#define PELIAS_TARGET_AVX2_FN __attribute__((target("avx2")))
#endif
#endif

namespace pelias::hal {

// AVX2 usable: the CPU has it and the operating system saves the YMM registers.
inline bool cpu_has_avx2() {
#if defined(PELIAS_HAL_X86)
#if defined(_MSC_VER) && !defined(__clang__)
  int r[4];
  __cpuid(r, 0);
  if (r[0] < 7) return false;
  __cpuid(r, 1);
  if (!((r[2] >> 27) & 1) || !((r[2] >> 28) & 1) || (_xgetbv(0) & 6) != 6) return false;  // OSXSAVE, AVX, YMM saved
  __cpuidex(r, 7, 0);
  return (r[1] >> 5) & 1;
#else
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2");
#endif
#else
  return false;
#endif
}

}  // namespace pelias::hal
