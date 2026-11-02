// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// The GPU decoders (kern/gpu_decode.h, GP3-10 GPU) compiled for the host: every supported type, every sub-block of a
// quantized row, gives exactly the floats of quant::dequantize_row (bit for bit, NaN-free). The GPU build compiles the same
// functions with mul / add mapped to __fmul_rn / __fadd_rn, which the device never fuses, so this test pins the device
// values too (the rung-1 GPU test checks them on the card).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gpu_test_rows.h"
#include "kern/gpu_decode.h"
#include "kern/tg2_planes.h"
#include "pelias_test.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"

namespace q = pelias::quant;
namespace g = pelias::kern::gpu;

namespace {
std::vector<float> wave(std::int64_t n, std::uint32_t seed, float scale) {
  std::vector<float> x(static_cast<std::size_t>(n));
  for (float& v : x) {
    seed = seed * 1664525u + 1013904223u;
    const float u = (static_cast<float>(seed >> 8) / 16777216.0f) - 0.5f;
    v = scale * u / (0.25f + std::fabs(u));
  }
  return x;
}

}  // namespace

TEST(gpu_decoders_equal_dequantize_row_bit_for_bit) {
  namespace tg = pelias::test_gpu;
  const std::vector<q::Type> types = tg::readable_types();
  const g::Tables tb = tg::host_tables();
  const std::int64_t n = 2048;
  int checked = 0;
  for (const q::Type t : types) {
    const auto id = static_cast<std::uint32_t>(t);
    CHECK(g::type_supported(id));
    CHECK(g::block_bytes(id) == q::type_info(id).block_bytes);
    CHECK(g::block_values(id) == q::type_info(id).block_elements);
    for (std::uint32_t seed : {1u, 77u, 9001u}) {
      const std::vector<float> x = wave(n, seed + id, seed == 77u ? 3.0f : 0.05f);
      const std::vector<std::uint8_t> row = tg::make_row(t, x, seed * 31u + id);
      std::vector<float> ref(static_cast<std::size_t>(n)), dev(static_cast<std::size_t>(n));
      CHECK(q::dequantize_row(t, row.data(), ref.data(), n).ok());
      for (std::int64_t j = 0; j < n / 32; ++j) g::decode32(id, row.data(), j, tb, dev.data() + 32 * j);
      bool same = true, finite = true;
      for (std::int64_t i = 0; i < n; ++i) {
        std::uint32_t a, b;
        std::memcpy(&a, &ref[static_cast<std::size_t>(i)], 4);
        std::memcpy(&b, &dev[static_cast<std::size_t>(i)], 4);
        same = same && a == b;
        finite = finite && std::isfinite(dev[static_cast<std::size_t>(i)]);
      }
      if (!same) std::printf("    %s seed %u: decode32 differs from dequantize_row\n", q::type_info(id).name, seed);
      CHECK(same);
      CHECK(finite);
      ++checked;
    }
  }
  CHECK(types.size() == 32);  // every type dequantize_row reads
  CHECK(checked == 3 * static_cast<int>(types.size()));
  CHECK(!g::type_supported(static_cast<std::uint32_t>(q::Type::q8_1)));  // an activation type: never a weight
  CHECK(!g::type_supported(static_cast<std::uint32_t>(q::Type::q8_k)));
  std::printf("    %zu types x 3 rows of 2048: every sub-block equal to dequantize_row\n", types.size());
}

TEST(gpu_decoder_half_float_is_exact_for_every_half) {
  bool same = true;
  for (std::uint32_t h = 0; h < 65536; ++h) {
    const float a = q::fp16_to_fp32(static_cast<std::uint16_t>(h)), b = g::half_float(static_cast<std::uint16_t>(h));
    std::uint32_t ua, ub;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    if (std::isnan(a)) same = same && std::isnan(b);
    else same = same && ua == ub;
  }
  CHECK(same);
}

// GP3-14: Q8_0 in R3's repacked layout (g_q8_0r, kern/gpu_stream's upload: per 256 values qs[256] | d[8]) decodes to the
// same values as the GGUF row, bit for bit (the batched kernels read the copy R3 decodes from)
TEST(gpu_decoder_q8_0_r3_layout_equals_q8_0) {
  const std::int64_t k = 1024;
  std::vector<float> x(static_cast<std::size_t>(k));
  for (std::int64_t i = 0; i < k; ++i) x[static_cast<std::size_t>(i)] = std::sin(0.37f * static_cast<float>(i)) * (1.0f + static_cast<float>(i % 7));
  const std::uint64_t rb = q::row_bytes(static_cast<std::uint32_t>(q::Type::q8_0), static_cast<std::uint64_t>(k));
  std::vector<std::uint8_t> row(rb), rr(static_cast<std::size_t>(k / 256 * 272));
  CHECK(q::quantize_row(q::Type::q8_0, x.data(), row.data(), k).ok());
  for (std::int64_t b = 0; b < k / 256; ++b)
    for (int i = 0; i < 8; ++i) {
      const std::uint8_t* p = row.data() + 272 * b + 34 * i;
      std::memcpy(rr.data() + 272 * b + 32 * i, p + 2, 32);
      std::memcpy(rr.data() + 272 * b + 256 + 2 * i, p, 2);
    }
  CHECK(g::block_values(g::g_q8_0r) == 256 && g::block_bytes(g::g_q8_0r) == 272);
  const g::Tables tb{};
  int same = 0;
  for (std::int64_t j = 0; j < k / 32; ++j) {
    float a[32], b2[32];
    g::decode32(g::g_q8_0, row.data(), j, tb, a);
    g::decode32(g::g_q8_0r, rr.data(), j, tb, b2);
    same += std::memcmp(a, b2, sizeof a) == 0 ? 1 : 0;
  }
  CHECK(same == k / 32);
}

// v54: Q4_TG2 bit planes (kern/tg2_planes.h). The spreads exhaustively; rows planed and back: 4 bits give the stored row
// exactly, 3 / 2 bits the stored row with the low code bits zero; the 8-code words the GPU builds from the plane words
// (the R3 stage's reads) equal the stored nibble words; the base grids are the means of the levels sharing the top bits.
TEST(tg2_planes_layout_and_reads) {
  bool sp = true;
  for (std::uint32_t a = 0; a < 65536u; ++a) {
    std::uint32_t r = 0;
    for (int k = 0; k < 8; ++k) r |= ((a >> (2 * k)) & 3u) << (4 * k);
    sp = sp && g::tg2p_spread2(a) == r;
  }
  for (std::uint32_t b = 0; b < 256u; ++b) {
    std::uint32_t r = 0;
    for (int k = 0; k < 8; ++k) r |= ((b >> k) & 1u) << (4 * k);
    sp = sp && g::tg2p_spread1(b) == r;
  }
  CHECK(sp);
  std::uint32_t seed = 12345u;
  auto rnd = [&]() {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<std::uint8_t>(seed >> 24);
  };
  int rows = 0;
  bool round = true, masked = true, words = true;
  for (const int nB : {1, 2, 8}) {
    for (int rep = 0; rep < 20; ++rep, ++rows) {
      std::vector<std::uint8_t> row(static_cast<std::size_t>(144 * nB)), pl(row.size()), back(row.size());
      for (auto& x : row) x = rnd();
      g::tg2p_plane_row(row.data(), pl.data(), nB);
      g::tg2p_unplane_row(pl.data(), back.data(), nB, 4);
      round = round && back == row;
      for (const int bits : {2, 3}) {
        g::tg2p_unplane_row(pl.data(), back.data(), nB, bits);
        const std::uint8_t keep = bits == 3 ? 0xEE : 0xCC;  // both nibbles of a byte, low bits cleared
        bool m = true;
        for (int b = 0; b < nB; ++b) {
          m = m && std::memcmp(back.data() + 144 * b, row.data() + 144 * b, 16) == 0;
          for (int i = 16; i < 144; ++i) m = m && back[static_cast<std::size_t>(144 * b + i)] == (row[static_cast<std::size_t>(144 * b + i)] & keep);
        }
        masked = masked && m;
      }
      // the GPU's reads: per block B, half h, sub-block jj, group g: plane words as little-endian 32-bit loads
      auto u32 = [&](std::size_t off) {
        std::uint32_t v;
        std::memcpy(&v, pl.data() + off, 4);
        return v;
      };
      for (int b = 0; b < nB; ++b)
        for (int h = 0; h < 2; ++h)
          for (int jj = 0; jj < 4; ++jj)
            for (int gq = 0; gq < 4; ++gq) {
              const std::size_t ab = static_cast<std::size_t>(16 * nB + 64 * b + 32 * h), bb = static_cast<std::size_t>(80 * nB + 32 * b + 16 * h),
                                cb = static_cast<std::size_t>(112 * nB + 32 * b + 16 * h);
              const std::uint32_t a16 = (u32(ab + 4 * static_cast<std::size_t>(2 * jj + (gq >> 1))) >> (16 * (gq & 1))) & 0xffffu;
              const std::uint32_t b8 = (u32(bb + 4 * static_cast<std::size_t>(jj)) >> (8 * gq)) & 0xffu;
              const std::uint32_t c8 = (u32(cb + 4 * static_cast<std::size_t>(jj)) >> (8 * gq)) & 0xffu;
              std::uint32_t want;
              std::memcpy(&want, row.data() + 144 * b + 16 + 64 * h + 4 * (4 * jj + gq), 4);
              words = words && g::tg2p_codes8(a16, b8, c8) == want;
              words = words && g::tg2p_codes8(a16, b8, 0) == (want & 0xEEEEEEEEu) && g::tg2p_codes8(a16, 0, 0) == (want & 0xCCCCCCCCu);
            }
    }
  }
  CHECK(round);
  CHECK(masked);
  CHECK(words);
  bool grids = true;
  for (int t = 0; t < 4; ++t) {
    const float* lv = q::q4_tg_grid(t, 2);
    float o2[16], o3[16], o4[16];
    g::tg2p_base_grid(lv, 2, o2);
    g::tg2p_base_grid(lv, 3, o3);
    g::tg2p_base_grid(lv, 4, o4);
    for (int c = 0; c < 16; ++c) {
      const int q2 = c & ~3, q3 = c & ~1;
      grids = grids && o4[c] == lv[c];
      grids = grids && o3[c] == static_cast<float>((static_cast<double>(lv[q3]) + lv[q3 + 1]) / 2.0);
      grids = grids && o2[c] == static_cast<float>((static_cast<double>(lv[q2]) + lv[q2 + 1] + lv[q2 + 2] + lv[q2 + 3]) / 4.0);
    }
  }
  CHECK(grids);
  std::printf("    spreads exhaustive %s; %d rows (1 / 2 / 8 blocks): 4 bits round trip %s, 3 / 2 bits = stored with low bits zero %s; "
              "GPU word reads %s; base grids %s\n",
              sp ? "ok" : "WRONG", rows, round ? "exact" : "WRONG", masked ? "yes" : "NO", words ? "= stored words" : "WRONG", grids ? "ok" : "WRONG");
}

// v54c: Q5_K bit planes: rows planed and back (5 bits = stored; fewer = stored with low code bits zero), the per-half
// unpack the GPU runs gives the stored half bytes, and plane_prefix's byte counts
TEST(q5_planes_layout_and_reads) {
  std::uint32_t seed = 777u;
  auto rnd = [&]() {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<std::uint8_t>(seed >> 24);
  };
  int rows = 0;
  bool round = true, masked = true, halves = true, words = true;
  for (const int nB : {1, 2, 8}) {
    for (int rep = 0; rep < 20; ++rep, ++rows) {
      std::vector<std::uint8_t> row(static_cast<std::size_t>(176 * nB)), pl(row.size()), back(row.size()), want(row.size());
      for (auto& x : row) x = rnd();
      g::q5p_plane_row(row.data(), pl.data(), nB);
      g::q5p_unplane_row(pl.data(), back.data(), nB, 5);
      round = round && back == row;
      for (const int bits : {2, 3, 4}) {
        g::q5p_unplane_row(pl.data(), back.data(), nB, bits);
        // expected: every code with its low 5 - bits bits cleared
        want = row;
        for (int b = 0; b < nB; ++b) {
          std::uint8_t* blk = want.data() + 176 * b;
          for (int j = 0; j < 8; ++j)
            for (int l = 0; l < 32; ++l) {
              int q = ((blk[48 + 32 * (j / 2) + l] >> (4 * (j & 1))) & 15) | (((blk[16 + l] >> j) & 1) << 4);
              q &= ~((1 << (5 - bits)) - 1);
              blk[48 + 32 * (j / 2) + l] = static_cast<std::uint8_t>((blk[48 + 32 * (j / 2) + l] & ~(15 << (4 * (j & 1)))) | ((q & 15) << (4 * (j & 1))));
              blk[16 + l] = static_cast<std::uint8_t>((blk[16 + l] & ~(1 << j)) | (((q >> 4) & 1) << j));
            }
        }
        masked = masked && back == want;
      }
      auto u32 = [&](std::size_t off) {
        std::uint32_t v;
        std::memcpy(&v, pl.data() + off, 4);
        return v;
      };
      for (int b = 0; b < nB; ++b)
        for (int h = 0; h < 2; ++h) {
          std::uint32_t a[8], bb[4], c[4], d[4];
          for (int i = 0; i < 8; ++i) a[i] = u32(static_cast<std::size_t>(16 * nB + 64 * b + 32 * h + 4 * i));
          for (int i = 0; i < 4; ++i) {
            bb[i] = u32(static_cast<std::size_t>(80 * nB + 32 * b + 16 * h + 4 * i));
            c[i] = u32(static_cast<std::size_t>(112 * nB + 32 * b + 16 * h + 4 * i));
            d[i] = u32(static_cast<std::size_t>(144 * nB + 32 * b + 16 * h + 4 * i));
          }
          std::uint8_t qh[32], qlb[64];
          g::q5p_unpack_half(a, bb, c, d, h, qh, qlb);
          const std::uint8_t* blk = row.data() + 176 * b;
          bool ok = std::memcmp(qlb, blk + 48 + 64 * h, 64) == 0;
          for (int l = 0; l < 32; ++l) ok = ok && qh[l] == (blk[16 + l] & (0x0f << (4 * h)));
          halves = halves && ok;
          // v55: the dp4a words straight from the planes, at 5 / 4 / 3 / 2 bits (planes not read passed as zero)
          for (int bits = 2; bits <= 5; ++bits) {
            std::uint32_t b2[4], c2[4], d2[4];
            for (int i = 0; i < 4; ++i) {
              b2[i] = bits >= 3 ? bb[i] : 0u;
              c2[i] = bits >= 4 ? c[i] : 0u;
              d2[i] = bits >= 5 ? d[i] : 0u;
            }
            for (int jj = 0; jj < 4; ++jj)
              for (int g8 = 0; g8 < 8; ++g8) {
                const std::uint32_t w = g::q5p_word(a, b2, c2, d2, jj, g8);
                for (int k = 0; k < 4; ++k) {
                  const int j = 4 * h + jj, l = 4 * g8 + k;
                  int q = ((blk[48 + 32 * (j / 2) + l] >> (4 * (j & 1))) & 15) | (((blk[16 + l] >> j) & 1) << 4);
                  q &= ~((1 << (5 - bits)) - 1);
                  words = words && static_cast<int>((w >> (8 * k)) & 0xffu) == q;
                }
              }
          }
        }
    }
  }
  const bool pre = g::plane_prefix(g::g_q5_kp, 176 * 8, 2) == 640 && g::plane_prefix(g::g_q5_kp, 176 * 8, 5) == 1408 &&
                   g::plane_prefix(g::g_q5_kp, 176 * 8, 9) == 1408 && g::plane_prefix(g::g_q4_tg2p, 144 * 2, 3) == 224 &&
                   g::plane_prefix(g::g_q4_tg2p, 144 * 2, 5) == 288 && g::plane_prefix(13, 1408, 2) == 1408;
  CHECK(round);
  CHECK(masked);
  CHECK(halves);
  CHECK(words);
  CHECK(pre);
  std::printf("    Q5_K planes: %d rows: 5 bits round trip %s; 2 / 3 / 4 bits = stored with low code bits zero %s; GPU half unpack %s; "
              "dp4a words at 5/4/3/2 bits %s; prefixes %s\n",
              rows, round ? "exact" : "WRONG", masked ? "yes" : "NO", halves ? "= stored" : "WRONG", words ? "= stored codes" : "WRONG",
              pre ? "ok" : "WRONG");
}
