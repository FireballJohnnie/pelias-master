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
