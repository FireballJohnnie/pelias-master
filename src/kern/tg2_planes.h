// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
#pragma once
// kern/tg2_planes.h - v54 (endgame, author 23:14): Q4_TG2 rows held as bit planes, so the R3 expert stage can read 2, 3 or
// all 4 bits of every code (the sliding precision of the user's slider) and fetch only those bytes. The same 144 bytes per
// 256 values, rearranged per row of nB blocks:
//   [16 nB]  the blocks' 16-byte headers, block after block (fp16 d, six level bytes, eight tag bytes: as stored)
//   [64 nB]  plane A: the top two bits of every code; per block, per half (128 values) 32 bytes: value v of the half at
//            bits 2 (v & 15) of 32-bit word v >> 4
//   [32 nB]  plane B: the third bit; per block, per half 16 bytes: value v at bit v & 31 of word v >> 5
//   [32 nB]  plane C: the fourth (lowest) bit, as plane B
// A row's first 80 nB bytes are the 2-bit base, the first 112 nB the 3-bit base, all 144 nB the stored code. Reading 4
// bits gives the stored code exactly (the same values, the same bits as the stored layout). Reading fewer bits gives the
// code with the missing low bits zero, and the base grid replaces each of the 16 levels by the mean of the levels sharing
// its top bits (the v27 rule), so a 2-bit code reads the mean of its 4 levels, a 3-bit code the mean of its 2.
#include <cstdint>
#include <cstring>

#include "kern/gpu_decode.h"

namespace pelias::kern::gpu {

constexpr std::uint32_t g_q4_tg2p = 1009;  // GPU-path id (not a file type): Q4_TG2 rows in the plane layout above
// v54c (author 00:22-00:25: the hot experts are Q5_K; "more experts on the vram"): Q5_K rows (176 bytes per 256 values) in
// the same plane layout, its 5-bit code q = low nibble | high bit << 4 split top bits first:
//   [16 nB] headers (fp16 d, fp16 dmin, 12 scale bytes, as stored) | [64 nB] A (bits 4-3) | [32 nB] B (bit 2) |
//   [32 nB] C (bit 1) | [32 nB] D (bit 0); value v of a half (v = 32 j' + l for the half's sub-block j' and value l)
// A row's first 16 nB + 32 nB x b bytes hold b bits; 5 bits = the stored code exactly. Fewer bits read the code with its
// missing low bits at the middle of their range (the expected value of a uniform remainder).
constexpr std::uint32_t g_q5_kp = 1010;

// the bytes of a plane row's first `bits` bits, for its type (rb = the row's full bytes); other types: rb
PG_HD std::uint64_t plane_prefix(std::uint32_t type, std::uint64_t rb, int bits) {
  if (type == g_q4_tg2p) {
    const std::uint64_t nB = rb / 144;
    return 16 * nB + 32 * nB * static_cast<std::uint64_t>(bits < 4 ? bits : 4);
  }
  if (type == g_q5_kp) {
    const std::uint64_t nB = rb / 176;
    return 16 * nB + 32 * nB * static_cast<std::uint64_t>(bits < 5 ? bits : 5);
  }
  return rb;
}

// the bytes of a row's first `bits` bits (2, 3 or 4)
PG_HD std::uint64_t tg2p_prefix(std::uint64_t nB, int bits) { return bits >= 4 ? 144 * nB : bits == 3 ? 112 * nB : 80 * nB; }

// eight 2-bit fields (bits 2k, 2k + 1 of a16) to bits 4k, 4k + 1
PG_HD std::uint32_t tg2p_spread2(std::uint32_t a16) {
  std::uint32_t t = a16 & 0xffffu;
  t = (t | (t << 8)) & 0x00ff00ffu;
  t = (t | (t << 4)) & 0x0f0f0f0fu;
  t = (t | (t << 2)) & 0x33333333u;
  return t;
}
// eight bits (bit k of b8) to bit 4k
PG_HD std::uint32_t tg2p_spread1(std::uint32_t b8) {
  std::uint32_t u = b8 & 0xffu;
  u = (u | (u << 12)) & 0x000f000fu;
  u = (u | (u << 6)) & 0x03030303u;
  u = (u | (u << 3)) & 0x11111111u;
  return u;
}
// the 8 codes (nibble k = value k) of 8 consecutive values from their plane bits (a16: 8 x 2 bits, b8 / c8: 8 x 1 bit);
// with fewer bits read, pass zero for the missing planes
PG_HD std::uint32_t tg2p_codes8(std::uint32_t a16, std::uint32_t b8, std::uint32_t c8) {
  return (tg2p_spread2(a16) << 2) | (tg2p_spread1(b8) << 1) | tg2p_spread1(c8);
}

// host: one row of nB stored Q4_TG2 blocks (144 nB bytes) into the plane layout (out: 144 nB bytes, not overlapping in)
inline void tg2p_plane_row(const std::uint8_t* in, std::uint8_t* out, int nB) {
  std::memset(out, 0, static_cast<std::size_t>(144) * static_cast<std::size_t>(nB));
  for (int b = 0; b < nB; ++b) {
    const std::uint8_t* blk = in + 144 * b;
    std::memcpy(out + 16 * b, blk, 16);
    for (int h = 0; h < 2; ++h)
      for (int v = 0; v < 128; ++v) {
        const int idx = 128 * h + v;
        const int code = (blk[16 + idx / 2] >> (4 * (idx & 1))) & 15;
        out[16 * nB + 64 * b + 32 * h + (v >> 2)] |= static_cast<std::uint8_t>((code >> 2) << (2 * (v & 3)));
        out[80 * nB + 32 * b + 16 * h + (v >> 3)] |= static_cast<std::uint8_t>(((code >> 1) & 1) << (v & 7));
        out[112 * nB + 32 * b + 16 * h + (v >> 3)] |= static_cast<std::uint8_t>((code & 1) << (v & 7));
      }
  }
}
// host: back to the stored layout, from the first `bits` bits only (the missing low bits zero); for the tests
inline void tg2p_unplane_row(const std::uint8_t* in, std::uint8_t* out, int nB, int bits) {
  for (int b = 0; b < nB; ++b) {
    std::uint8_t* blk = out + 144 * b;
    std::memcpy(blk, in + 16 * b, 16);
    std::memset(blk + 16, 0, 128);
    for (int h = 0; h < 2; ++h)
      for (int v = 0; v < 128; ++v) {
        const int idx = 128 * h + v;
        int code = ((in[16 * nB + 64 * b + 32 * h + (v >> 2)] >> (2 * (v & 3))) & 3) << 2;
        if (bits >= 3) code |= ((in[80 * nB + 32 * b + 16 * h + (v >> 3)] >> (v & 7)) & 1) << 1;
        if (bits >= 4) code |= (in[112 * nB + 32 * b + 16 * h + (v >> 3)] >> (v & 7)) & 1;
        blk[16 + idx / 2] |= static_cast<std::uint8_t>(code << (4 * (idx & 1)));
      }
  }
}
// one half (128 values) of a Q5_K plane block back to the stored layout's bytes for that half: qlb (64 bytes: the low
// nibbles of sub-blocks 4 h .. 4 h + 3) and the high bits j = 4 h .. 4 h + 3 of qh[32] (other bits zero). Plane words as
// loaded (little-endian): a[8] (A), b[4], c[4], d[4] (B, C, D; zero where not read). The GPU's reads (R3 stage) and the
// host test share this code
PG_HD void q5p_unpack_half(const std::uint32_t* a, const std::uint32_t* b, const std::uint32_t* c, const std::uint32_t* d, int h,
                           std::uint8_t* qh, std::uint8_t* qlb) {
  for (int i = 0; i < 32; ++i) qh[i] = 0;
  for (int i = 0; i < 64; ++i) qlb[i] = 0;
  for (int jj = 0; jj < 4; ++jj)
    for (int l = 0; l < 32; ++l) {
      const std::uint32_t q = (((a[2 * jj + (l >> 4)] >> (2 * (l & 15))) & 3u) << 3) | (((b[jj] >> l) & 1u) << 2) | (((c[jj] >> l) & 1u) << 1) |
                              ((d[jj] >> l) & 1u);
      qlb[32 * (jj >> 1) + l] = static_cast<std::uint8_t>(qlb[32 * (jj >> 1) + l] | ((q & 15u) << (4 * (jj & 1))));
      qh[l] = static_cast<std::uint8_t>(qh[l] | (((q >> 4) & 1u) << (4 * h + jj)));
    }
}
// host: one row of nB stored Q5_K blocks (176 nB bytes) into the plane layout (g_q5_kp)
inline void q5p_plane_row(const std::uint8_t* in, std::uint8_t* out, int nB) {
  std::memset(out, 0, static_cast<std::size_t>(176) * static_cast<std::size_t>(nB));
  for (int b = 0; b < nB; ++b) {
    const std::uint8_t* blk = in + 176 * b;
    std::memcpy(out + 16 * b, blk, 16);
    const std::uint8_t* qh = blk + 16;
    const std::uint8_t* qs = blk + 48;
    for (int j = 0; j < 8; ++j)
      for (int l = 0; l < 32; ++l) {
        const int q = ((qs[32 * (j / 2) + l] >> (4 * (j & 1))) & 15) | (((qh[l] >> j) & 1) << 4);
        const int h = j >> 2, v = 32 * (j & 3) + l;
        out[16 * nB + 64 * b + 32 * h + (v >> 2)] |= static_cast<std::uint8_t>((q >> 3) << (2 * (v & 3)));
        out[80 * nB + 32 * b + 16 * h + (v >> 3)] |= static_cast<std::uint8_t>(((q >> 2) & 1) << (v & 7));
        out[112 * nB + 32 * b + 16 * h + (v >> 3)] |= static_cast<std::uint8_t>(((q >> 1) & 1) << (v & 7));
        out[144 * nB + 32 * b + 16 * h + (v >> 3)] |= static_cast<std::uint8_t>((q & 1) << (v & 7));
      }
  }
}
// host: back to stored Q5_K from the first `bits` bits (missing low bits zero); for the tests
inline void q5p_unplane_row(const std::uint8_t* in, std::uint8_t* out, int nB, int bits) {
  for (int b = 0; b < nB; ++b) {
    std::uint8_t* blk = out + 176 * b;
    std::memcpy(blk, in + 16 * b, 16);
    std::memset(blk + 16, 0, 160);
    for (int j = 0; j < 8; ++j)
      for (int l = 0; l < 32; ++l) {
        const int h = j >> 2, v = 32 * (j & 3) + l;
        int q = ((in[16 * nB + 64 * b + 32 * h + (v >> 2)] >> (2 * (v & 3))) & 3) << 3;
        if (bits >= 3) q |= ((in[80 * nB + 32 * b + 16 * h + (v >> 3)] >> (v & 7)) & 1) << 2;
        if (bits >= 4) q |= ((in[112 * nB + 32 * b + 16 * h + (v >> 3)] >> (v & 7)) & 1) << 1;
        if (bits >= 5) q |= (in[144 * nB + 32 * b + 16 * h + (v >> 3)] >> (v & 7)) & 1;
        blk[48 + 32 * (j / 2) + l] |= static_cast<std::uint8_t>((q & 15) << (4 * (j & 1)));
        blk[16 + l] |= static_cast<std::uint8_t>(((q >> 4) & 1) << j);
      }
  }
}
// host: a stored row of either planed type into its plane layout (by the stored quant type id: 204 Q4_TG2, 13 Q5_K)
inline void plane_row(std::uint32_t stored_type, const std::uint8_t* in, std::uint8_t* out, std::uint64_t rb) {
  if (stored_type == 204) tg2p_plane_row(in, out, static_cast<int>(rb / 144));
  else q5p_plane_row(in, out, static_cast<int>(rb / 176));
}

// host: the base grid of one tag's 16 levels for `bits` bits: each level replaced by the mean (in double, rounded once to
// float) of the 2^(4 - bits) levels sharing its top bits; 4 bits: the levels themselves
inline void tg2p_base_grid(const float* g, int bits, float* out) {
  const int span = bits >= 4 ? 1 : bits == 3 ? 2 : 4;
  for (int c = 0; c < 16; ++c) {
    const int c0 = c / span * span;
    double s = 0.0;
    for (int i = 0; i < span; ++i) s += static_cast<double>(g[c0 + i]);
    out[c] = static_cast<float>(s / span);
  }
}

}  // namespace pelias::kern::gpu
