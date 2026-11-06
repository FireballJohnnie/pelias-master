// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
#pragma once
// kern/tg2_planes.h - v54 (endgame, author 23:14): Q4_TG2 rows held as bit planes, so the R3 expert stage can read 2, 3 or
// all 4 bits of every code (the sliding precision of the user's slider) and fetch only those bytes. The same 144 bytes per
// 256 values, rearranged per row of nB blocks:
//   [16 nB]  the blocks' 16-byte headers, block after block (fp16 d, six level bytes, eight tag bytes: as stored)
//   [64 nB]  plane A: the top two bits of every code, as two 1-bit planes (v55f, author: the planes are read as 1-bit
//            layers): per block, per half (128 values) 32 bytes = 8 words; for sub-block j' (values 32 j' .. 32 j' + 31
//            of the half) word 2 j' holds the top bit (code bit 3) and word 2 j' + 1 the next (code bit 2), value l of the
//            sub-block at bit l (v55e and before: 2-bit fields, value v at bits 2 (v & 15) of word v >> 4)
//   [32 nB]  plane B: the third bit; per block, per half 16 bytes: value v at bit v & 31 of word v >> 5
//   [32 nB]  plane C: the fourth (lowest) bit, as plane B
// A row's first 80 nB bytes are the 2-bit base, the first 112 nB the 3-bit base, all 144 nB the stored code. Reading 4
// bits gives the stored code exactly (the same values, the same bits as the stored layout). Reading fewer bits gives the
// code with the missing low bits zero, and the base grid replaces each of the 16 levels by the mean of the levels sharing
// its top bits (the v27 rule), so a 2-bit code reads the mean of its 4 levels, a 3-bit code the mean of its 2.
#include <bit>
#include <cstdint>
#include <cstring>

#include "kern/gpu_decode.h"

namespace pelias::kern::gpu {

constexpr std::uint32_t g_q4_tg2p = 1009;  // GPU-path id (not a file type): Q4_TG2 rows in the plane layout above
// v54c (author 00:22-00:25: the hot experts are Q5_K; "more experts on the vram"): Q5_K rows (176 bytes per 256 values) in
// the same plane layout, its 5-bit code q = low nibble | high bit << 4 split top bits first:
//   [16 nB] headers (fp16 d, fp16 dmin, 12 scale bytes, as stored) | [64 nB] A (bits 4 and 3 as two 1-bit planes, as
//   Q4_TG2's A since v55f: word 2 j' bit 4, word 2 j' + 1 bit 3) | [32 nB] B (bit 2) | [32 nB] C (bit 1) | [32 nB] D
//   (bit 0); value v of a half (v = 32 j' + l for the half's sub-block j' and value l) at bit l of its sub-block's word
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

// eight bits (bit k of b8) to bit 4k
PG_HD std::uint32_t tg2p_spread1(std::uint32_t b8) {
  std::uint32_t u = b8 & 0xffu;
  u = (u | (u << 12)) & 0x000f000fu;
  u = (u | (u << 6)) & 0x03030303u;
  u = (u | (u << 3)) & 0x11111111u;
  return u;
}
// the 8 codes (nibble k = value k) of 8 consecutive values from their plane bits (v55f: ah8 / al8 the two bits of plane
// A, b8 / c8 planes B / C; 8 x 1 bit each); with fewer bits read, pass zero for the missing planes
PG_HD std::uint32_t tg2p_codes8(std::uint32_t ah8, std::uint32_t al8, std::uint32_t b8, std::uint32_t c8) {
  return (tg2p_spread1(ah8) << 3) | (tg2p_spread1(al8) << 2) | (tg2p_spread1(b8) << 1) | tg2p_spread1(c8);
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
        const int jp = v >> 5, l = v & 31;  // the half's sub-block, the value in it
        out[16 * nB + 64 * b + 32 * h + 8 * jp + (l >> 3)] |= static_cast<std::uint8_t>(((code >> 3) & 1) << (l & 7));
        out[16 * nB + 64 * b + 32 * h + 8 * jp + 4 + (l >> 3)] |= static_cast<std::uint8_t>(((code >> 2) & 1) << (l & 7));
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
        const int jp = v >> 5, l = v & 31;
        int code = (((in[16 * nB + 64 * b + 32 * h + 8 * jp + (l >> 3)] >> (l & 7)) & 1) << 3) |
                   (((in[16 * nB + 64 * b + 32 * h + 8 * jp + 4 + (l >> 3)] >> (l & 7)) & 1) << 2);
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
      const std::uint32_t q = (((a[2 * jj] >> l) & 1u) << 4) | (((a[2 * jj + 1] >> l) & 1u) << 3) | (((b[jj] >> l) & 1u) << 2) |
                              (((c[jj] >> l) & 1u) << 1) | ((d[jj] >> l) & 1u);
      qlb[32 * (jj >> 1) + l] = static_cast<std::uint8_t>(qlb[32 * (jj >> 1) + l] | ((q & 15u) << (4 * (jj & 1))));
      qh[l] = static_cast<std::uint8_t>(qh[l] | (((q >> 4) & 1u) << (4 * h + jj)));
    }
}
// v55: four consecutive values' codes straight from the plane bits, one per byte (byte k = 16 A4 + 8 A3 + 4 B + 2 C + D
// of value k), ready for dp4a: ah4 / al4 = their 4 bits of plane A's two bits (v55f), b4 / c4 / d4 = their 4 bits of
// planes B, C, D (zero where not read). The dp4a route; the 1-bit route (q5p_sq_bits) reads the planes without this
PG_HD std::uint32_t q5p_bytes4(std::uint32_t ah4, std::uint32_t al4, std::uint32_t b4, std::uint32_t c4, std::uint32_t d4) {
  constexpr std::uint32_t m = 0x00204081u;  // bit k -> byte k (the four shifted copies never overlap, so no carries)
  return ((((ah4 & 15u) * m) & 0x01010101u) << 4) | ((((al4 & 15u) * m) & 0x01010101u) << 3) | ((((b4 & 15u) * m) & 0x01010101u) << 2) |
         ((((c4 & 15u) * m) & 0x01010101u) << 1) | (((d4 & 15u) * m) & 0x01010101u);
}
// v55: the word for values 4 g .. 4 g + 3 of sub-block jj of a half, from the half's plane words as loaded (a[8], b/c/d[4])
PG_HD std::uint32_t q5p_word(const std::uint32_t* a, const std::uint32_t* b, const std::uint32_t* c, const std::uint32_t* d, int jj, int g) {
  return q5p_bytes4(a[2 * jj] >> (4 * g), a[2 * jj + 1] >> (4 * g), b[jj] >> (4 * g), c[jj] >> (4 * g), d[jj] >> (4 * g));
}
// v55f (author: the 1-bit decoder, "AND and count the ones"): an activation sub-block (32 int8 values x) as its 8 bit planes
// of u = x + 128 (0..255): plane k bit l = bit k of u_l. Built once per token and sub-block, shared by every row
PG_HD void act_bits32(const std::int8_t* x, std::uint32_t* u8) {
  for (int k = 0; k < 8; ++k) u8[k] = 0u;
  for (int l = 0; l < 32; ++l) {
    const std::uint32_t u = static_cast<std::uint32_t>(static_cast<int>(x[l]) + 128);
    for (int k = 0; k < 8; ++k) u8[k] |= ((u >> k) & 1u) << l;
  }
}
// v55f: sum over the 32 values of w_l x_l for one weight bit plane w (bit l = 0 / 1) by AND + popcount against the
// activation's bit planes: sum w u = sum_k 2^k popc(w & U_k), sum w x = that - 128 popc(w). Exact integers
PG_HD int plane_dot32(std::uint32_t w, const std::uint32_t* u8) {
#if defined(__CUDA_ARCH__)
  int s = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) s += __popc(w & u8[k]) << k;
  return s - (__popc(w) << 7);
#else
  int s = 0;
  for (int k = 0; k < 8; ++k) s += std::popcount(w & u8[k]) << k;  // v56: std::popcount (MSVC has no __builtin_popcount)
  return s - (std::popcount(w) << 7);
#endif
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
        out[16 * nB + 64 * b + 32 * h + 8 * (j & 3) + (l >> 3)] |= static_cast<std::uint8_t>(((q >> 4) & 1) << (l & 7));
        out[16 * nB + 64 * b + 32 * h + 8 * (j & 3) + 4 + (l >> 3)] |= static_cast<std::uint8_t>(((q >> 3) & 1) << (l & 7));
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
        int q = (((in[16 * nB + 64 * b + 32 * h + 8 * (j & 3) + (l >> 3)] >> (l & 7)) & 1) << 4) |
                (((in[16 * nB + 64 * b + 32 * h + 8 * (j & 3) + 4 + (l >> 3)] >> (l & 7)) & 1) << 3);
        if (bits >= 3) q |= ((in[80 * nB + 32 * b + 16 * h + (v >> 3)] >> (v & 7)) & 1) << 2;
        if (bits >= 4) q |= ((in[112 * nB + 32 * b + 16 * h + (v >> 3)] >> (v & 7)) & 1) << 1;
        if (bits >= 5) q |= (in[144 * nB + 32 * b + 16 * h + (v >> 3)] >> (v & 7)) & 1;
        blk[48 + 32 * (j / 2) + l] |= static_cast<std::uint8_t>((q & 15) << (4 * (j & 1)));
        blk[16 + l] |= static_cast<std::uint8_t>(((q >> 4) & 1) << j);
      }
  }
}
// v60 E (the head sketch for MTP drafts; docs/PELIAS_GAME_PLAN_V59_FORWARD.md item 2b): a Q6_K matrix's top two code planes
// with its stored sub-block scales, so the draft's head product can read ~38 % of the head's bytes by the 1-bit route and
// re-score its top rows exactly. The 6-bit code q (0..63, value d x sc x (q - 32)) keeps bits 5 and 4 as two 1-bit planes;
// read alone they give the code with its low four bits at the middle of their range (the v27 rule: q - 32 -> 16 c - 24.5
// for the 2-bit code c), the same rule the Q4_TG2 / Q5_K planes use. Row layout for k values (10 k / 32 bytes; k = 2048:
// 640): [k / 32 sub-blocks x 8 bytes: word 2 s = bit 5 of sub-block s's 32 codes (value l at bit l), word 2 s + 1 = bit 4]
// [k / 16 scale bytes: the blocks' 16 int8 scales as stored, scale 2 s + hf for values 16 hf .. 16 hf + 15 of sub-block s].
// The block's fp16 d stays in R3's d6 array (one per 256). Sub-block s of the row = block s / 8, half (s / 4) & 1, p = s & 3
// in the R3 kernel's own numbering (q6_sub: ql bytes 64 h + 32 (p & 1) + i at nibble p >> 1, qh bytes 32 h + i bits 2 p).
constexpr std::uint32_t g_q6_ksk = 1011;  // GPU-path id (not a file type): the sketch layout
PG_HD std::uint64_t q6k_sketch_row_bytes(std::int64_t k) { return static_cast<std::uint64_t>(k) * 10 / 32; }
// one 256-value block (its 208 bytes in R3's layout: ql[128] | qh[64] | scales[16]) -> its 16 plane words and 16 scale bytes
PG_HD void q6k_sketch_block(const std::uint8_t* b208, std::uint32_t* planes16, std::uint8_t* sc16) {
  for (int h = 0; h < 2; ++h)
    for (int p = 0; p < 4; ++p) {
      std::uint32_t hi = 0, nx = 0;
      for (int i = 0; i < 32; ++i) {
        const int q = ((b208[64 * h + 32 * (p & 1) + i] >> (4 * (p >> 1))) & 15) | (((b208[128 + 32 * h + i] >> (2 * p)) & 3) << 4);
        hi |= static_cast<std::uint32_t>((q >> 5) & 1) << i;
        nx |= static_cast<std::uint32_t>((q >> 4) & 1) << i;
      }
      planes16[2 * (4 * h + p)] = hi;
      planes16[2 * (4 * h + p) + 1] = nx;
    }
  for (int i = 0; i < 16; ++i) sc16[i] = b208[192 + i];
}
// twice the sketch's integer sum over one 16-value half (hf 0: values 0-15, 1: 16-31) of a sub-block: 2 sum (16 c - 24.5) x
// = 32 (2 D5 + D4) - 49 X with D5 / D4 the plane dots (plane_dot32 on the half's bits) and X = the half's sum of x (the
// caller has it from plane_dot32(mask, u8), once per token and sub-block). Exact integers; the GPU kernel and the host test
// run this same function
PG_HD int q6k_sketch_half2(std::uint32_t hi, std::uint32_t nx, const std::uint32_t* u8, int hf, int xsum16) {
  const std::uint32_t m = hf ? 0xffff0000u : 0x0000ffffu;
  return 32 * (2 * plane_dot32(hi & m, u8) + plane_dot32(nx & m, u8)) - 49 * xsum16;
}
// v60 E3 (v60j MEASURED: the popcount form read the 159 MB sketch at 318 GB/s - popc issues at a quarter of the INT rate on
// sm_89, ~36 per sub-block): the same integers formed the way the Q5_KP path forms its codes - the four codes of values 4 g ..
// 4 g + 3 of a sub-block spread to one byte each (c = 2 bit5 + bit4, 0..3; the q5p_bytes4 multiply: bit k -> byte k) and dp4a'd
// against the four int8 activations as one word. Exact integers, identical to q6k_sketch_half2 (the host test asserts it)
PG_HD int dp4a_i32(int a, int b, int c) {
#if defined(__CUDA_ARCH__)
  return __dp4a(a, b, c);
#else
  for (int k = 0; k < 4; ++k) {
    const int ab = static_cast<int>(static_cast<std::int8_t>((static_cast<unsigned>(a) >> (8 * k)) & 0xffu));
    const int bb = static_cast<int>(static_cast<std::int8_t>((static_cast<unsigned>(b) >> (8 * k)) & 0xffu));
    c += ab * bb;
  }
  return c;
#endif
}
PG_HD std::uint32_t q6k_sketch_codes4(std::uint32_t hi4, std::uint32_t nx4) {
  constexpr std::uint32_t m = 0x00204081u;
  return ((((hi4 & 15u) * m) & 0x01010101u) << 1) | (((nx4 & 15u) * m) & 0x01010101u);
}
// twice the half's sketch sum, dp4a form: xw = the sub-block's 32 int8 activations as 8 words (xw[4 hf .. 4 hf + 3] = the half)
PG_HD int q6k_sketch_half2_dp(std::uint32_t hi, std::uint32_t nx, const std::int32_t* xw, int hf, int xsum16) {
  int cx = 0;
  for (int g = 0; g < 4; ++g) {
    const int sh = 16 * hf + 4 * g;
    cx = dp4a_i32(static_cast<int>(q6k_sketch_codes4(hi >> sh, nx >> sh)), xw[4 * hf + g], cx);
  }
  return 32 * cx - 49 * xsum16;
}
// one sub-block's sketch term as the R3 kernel forms the exact one (acc += d * xs * float(sc0 (a0 - 32 x0) + sc1 (a1 - 32 x1))):
// d * xs * (sc0 T0 + sc1 T1) / 2 with T the doubled half sums above
PG_HD float q6k_sketch_sub(float d, float xs, int sc0, int sc1, int t0, int t1) { return d * xs * (0.5f * static_cast<float>(sc0 * t0 + sc1 * t1)); }

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
