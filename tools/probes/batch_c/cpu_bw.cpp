// cpu_bw.cpp - AVX2 streaming read, compiled on its own with AVX2 enabled
// (g++ -mavx2 -O3 / cl /arch:AVX2). The rest of the program needs no AVX2.
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define PELIAS_X86 1
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

bool cpu_has_avx2() {
#if defined(PELIAS_X86)
#if defined(_MSC_VER)
  int info[4] = {0, 0, 0, 0};
  __cpuidex(info, 7, 0);
  return (info[1] & (1 << 5)) != 0;
#else
  return __builtin_cpu_supports("avx2") != 0;
#endif
#else
  return false;
#endif
}

#if defined(PELIAS_X86)
// 256 bytes per iteration: eight 32-byte loads folded into four accumulators.
uint64_t cpu_read_sum_avx2(const void* p, size_t bytes) {
  const auto* q = static_cast<const __m256i*>(p);
  const size_t n = bytes / 256;
  __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
  for (size_t i = 0; i < n; ++i) {
    a0 = _mm256_add_epi64(a0, _mm256_add_epi64(_mm256_load_si256(q), _mm256_load_si256(q + 4)));
    a1 = _mm256_add_epi64(a1, _mm256_add_epi64(_mm256_load_si256(q + 1), _mm256_load_si256(q + 5)));
    a2 = _mm256_add_epi64(a2, _mm256_add_epi64(_mm256_load_si256(q + 2), _mm256_load_si256(q + 6)));
    a3 = _mm256_add_epi64(a3, _mm256_add_epi64(_mm256_load_si256(q + 3), _mm256_load_si256(q + 7)));
    q += 8;
  }
  const __m256i s = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
  alignas(32) uint64_t out[4];
  _mm256_store_si256(reinterpret_cast<__m256i*>(out), s);
  return out[0] + out[1] + out[2] + out[3];
}
#else
uint64_t cpu_read_sum_avx2(const void* p, size_t bytes) {
  const auto* w = static_cast<const uint64_t*>(p);
  uint64_t s = 0;
  for (size_t i = 0; i < bytes / 8; ++i) s += w[i];
  return s;
}
#endif
