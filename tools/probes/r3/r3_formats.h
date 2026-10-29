// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// r3_formats.h - the weight formats the R3 streaming kernel reads, decoded one 32-value sub-block at a time, shared by the
// GPU kernel (r3.cu) and the host reference. Checked value for value against Pelias's own dequantizers
// (tools/probes/r3/test_r3_formats.cpp) before any GPU time. Device layouts (the engine repacks at load):
//   Q7ZT  base: 224 bytes per 256 (value i at bits 7i..7i+6, b = (q + 127) >> 1); hdr: u64 per 256 = fp16 d | eight 6-bit
//         levels << 16; mask: u8 per 256 (two gated 32-blocks); hb: two u32 per 256 (H of the gated blocks, in order).
//         Value (fold route) = (2b + h - 127) * d * lv_j / 63, h = 0 on ungated blocks.
//   Q4TG2 the GGUF block as stored (144 bytes, 16-aligned): fp16 d, eight 6-bit levels, eight tag bytes (2 bits per 8
//         values), 128 bytes of 4-bit codes. Value = d * lv_j / 63 * grid_tag[code] (the v2 grids, Pelias id 204).
//   Q5K   the GGUF block as stored (176 bytes, 16-aligned): fp16 d, fp16 dmin, 12 scale bytes, qh[32], qs[128].
//   Q6K   repacked: 208 bytes per 256 (ql[128], qh[64], int8 scales[16]) + fp16 d in its own array.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define R3_HD __host__ __device__ __forceinline__
#else
#define R3_HD inline
#endif

namespace r3 {

enum Fmt : int { kQ7ZT = 0, kQ4TG2 = 1, kQ5K = 2, kQ6K = 3 };

inline float half_to_float(uint16_t h) {
  const uint32_t s = static_cast<uint32_t>(h >> 15) << 31, e = (h >> 10) & 31u, m = h & 1023u;
  uint32_t bits;
  if (e == 0) {
    if (m == 0) {
      bits = s;
    } else {  // subnormal
      int ee = -1;
      uint32_t mm = m;
      do { ++ee; mm <<= 1; } while ((mm & 1024u) == 0);
      bits = s | static_cast<uint32_t>(127 - 15 - ee) << 23 | (mm & 1023u) << 13;
    }
  } else if (e == 31) {
    bits = s | 0x7f800000u | m << 13;
  } else {
    bits = s | (e + 112u) << 23 | m << 13;
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

R3_HD uint16_t rd16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

// ---- Q4_TG2 grids (quant_cpu.cpp, Q4TGGrids::h): grid 0 = (i - 7) / 8; grid 1 = the 1.3 log grid with level 8 = 0;
// grid 2 = the 1.45 log grid; grid 3 = 0.6 x grid 0. Built in float exactly as the quantizer does.
inline float tg2_grid(int t, int i) {
  if (t == 0) return (static_cast<float>(i) - 7.0f) / 8.0f;
  if (t == 3) return 0.6f * ((static_cast<float>(i) - 7.0f) / 8.0f);
  const double r = t == 1 ? 1.3 : 1.45;
  const double top = std::pow(r, 7.0);
  if (t == 1 && i == 8) return 0.0f;
  const int m = i >= 8 ? i - 8 : 7 - i;
  const float v = static_cast<float>(std::pow(r, static_cast<double>(m)) / top);
  return i >= 8 ? v : -v;
}

// ---- one 32-value sub-block j (0..7) of 256-block B of a row, as floats
inline void q7zt_decode32(const uint8_t* base, const uint64_t* hdr, const uint8_t* mask, const uint32_t* hb, int64_t rb, int j, float* y) {
  const uint64_t h = hdr[rb];
  const float d = half_to_float(static_cast<uint16_t>(h & 0xffffu));
  const float s = (d * static_cast<float>((h >> (16 + 6 * j)) & 63u)) / 63.0f;
  const unsigned m = mask[rb];
  uint32_t H = 0;
  if ((m >> j) & 1u) {
    int slot = 0;
    for (int t = 0; t < j; ++t) slot += (m >> t) & 1u;
    H = hb[rb * 2 + slot];
  }
  const uint8_t* b = base + rb * 224;
  for (int i = 0; i < 32; ++i) {
    const int idx = 32 * j + i, bit = 7 * idx;
    const uint32_t lo = b[bit >> 3], hi = (bit >> 3) + 1 < 224 ? b[(bit >> 3) + 1] : 0u;
    const int bb = static_cast<int>(((lo | hi << 8) >> (bit & 7)) & 127u);
    const int q = ((m >> j) & 1u) ? 2 * bb + static_cast<int>((H >> i) & 1u) - 127 : 2 * bb - 127;
    y[i] = static_cast<float>(q) * s;
  }
}

inline void q4tg2_decode32(const uint8_t* blk, int j, float* y) {
  const float d = half_to_float(rd16(blk));
  uint64_t packed = 0;
  for (int b = 0; b < 6; ++b) packed |= static_cast<uint64_t>(blk[2 + b]) << (8 * b);
  const float s = (d * static_cast<float>((packed >> (6 * j)) & 63u)) / 63.0f;
  const unsigned tb = blk[8 + j];
  for (int i = 0; i < 32; ++i) {
    const int idx = 32 * j + i;
    const int code = (blk[16 + idx / 2] >> (4 * (idx & 1))) & 15;
    y[i] = s * tg2_grid(static_cast<int>((tb >> (2 * (i / 8))) & 3u), code);
  }
}

R3_HD void k4_scale_min(int j, const uint8_t* q, int* sc, int* mn) {
  if (j < 4) {
    *sc = q[j] & 63;
    *mn = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

inline void q5k_decode32(const uint8_t* blk, int j, float* y) {
  const float d = half_to_float(rd16(blk)), dmin = half_to_float(rd16(blk + 2));
  int sc, mn;
  k4_scale_min(j, blk + 4, &sc, &mn);
  const float d1 = d * static_cast<float>(sc), m1 = dmin * static_cast<float>(mn);
  const uint8_t* qh = blk + 16;
  const uint8_t* ql = blk + 48 + 32 * (j / 2);
  for (int l = 0; l < 32; ++l) {
    const int lo = (j & 1) ? (ql[l] >> 4) : (ql[l] & 0xF);
    const int q = lo + (((qh[l] >> j) & 1) ? 16 : 0);
    y[l] = d1 * static_cast<float>(q) - m1;
  }
}

inline void q6k_decode32(const uint8_t* rp, uint16_t dh, int j, float* y) {
  const float d = half_to_float(dh);
  const int half = j / 4, part = j % 4;
  const uint8_t* ql = rp + 64 * half;
  const uint8_t* qh = rp + 128 + 32 * half;
  const int8_t* sc = reinterpret_cast<const int8_t*>(rp + 192 + 8 * half);
  for (int l = 0; l < 32; ++l) {
    int q;
    if (part == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
    else if (part == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
    else if (part == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
    else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
    y[l] = d * static_cast<float>(sc[2 * part + l / 16]) * static_cast<float>(q - 32);
  }
}

// GGUF Q6_K block (210 bytes: ql, qh, scales, d) -> repacked 208 bytes + d
inline void q6k_repack(const uint8_t* in, uint8_t* out208, uint16_t* d) {
  std::memcpy(out208, in, 208);
  *d = rd16(in + 208);
}

// GGUF Q7_ZT block (241 bytes) -> the device arrays for one 256-block
inline void q7zt_repack(const uint8_t* in, uint8_t* base224, uint64_t* hdr, uint8_t* mask, uint32_t* hb2) {
  uint64_t h = rd16(in);
  for (int b = 0; b < 6; ++b) h |= static_cast<uint64_t>(in[2 + b]) << (16 + 8 * b);
  *hdr = h;
  *mask = in[8];
  std::memcpy(base224, in + 9, 224);
  for (int s = 0; s < 2; ++s) hb2[s] = static_cast<uint32_t>(in[233 + 4 * s]) | static_cast<uint32_t>(in[234 + 4 * s]) << 8 |
                                       static_cast<uint32_t>(in[235 + 4 * s]) << 16 | static_cast<uint32_t>(in[236 + 4 * s]) << 24;
}

}  // namespace r3
