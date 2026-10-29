// Sandbox check of r3_formats.h against Pelias's own dequantizers (quant_cpu): every value of every block must match.
// Build (from the repo root): g++ -O2 -std=c++20 -Iinclude -Isrc -Ibuild/host/generated tools/probes/r3/test_r3_formats.cpp \
//        build/host/libpelias_quant.a build/host/libpelias_os.a -lpthread
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "quant/block_types.h"
#include "quant/quant_cpu.h"
#include "r3_formats.h"
namespace q = pelias::quant;
int main() {
  const int n = 256 * 64;
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> x(n);
  for (int i = 0; i < n; ++i) x[i] = nd(rng) * std::exp(0.7f * nd(rng)) * 0.02f;
  int fails = 0;
  const struct { q::Type t; int bytes; const char* name; } T[] = {{q::Type::q7_zt, 241, "Q7_ZT"}, {q::Type::q4_tg2, 144, "Q4_TG2"},
                                                                   {q::Type::q5_k, 176, "Q5_K"}, {q::Type::q6_k, 210, "Q6_K"}};
  for (const auto& t : T) {
    std::vector<uint8_t> b(static_cast<size_t>(n / 256 * t.bytes));
    std::vector<float> ref(n), got(n);
    if (!q::quantize_row(t.t, x.data(), b.data(), n).ok() || !q::dequantize_row(t.t, b.data(), ref.data(), n).ok()) { std::printf("%s: pelias failed\n", t.name); return 2; }
    for (int B = 0; B < n / 256; ++B) {
      const uint8_t* blk = b.data() + static_cast<size_t>(B) * t.bytes;
      for (int j = 0; j < 8; ++j) {
        float* y = got.data() + B * 256 + 32 * j;
        if (t.t == q::Type::q7_zt) {
          uint8_t base[224]; uint64_t hdr; uint8_t mask; uint32_t hb[2];
          r3::q7zt_repack(blk, base, &hdr, &mask, hb);
          r3::q7zt_decode32(base, &hdr, &mask, hb, 0, j, y);
        } else if (t.t == q::Type::q4_tg2) {
          r3::q4tg2_decode32(blk, j, y);
        } else if (t.t == q::Type::q5_k) {
          r3::q5k_decode32(blk, j, y);
        } else {
          uint8_t rp[208]; uint16_t d;
          r3::q6k_repack(blk, rp, &d);
          r3::q6k_decode32(rp, d, j, y);
        }
      }
    }
    int bad = 0;
    for (int i = 0; i < n; ++i) if (got[i] != ref[i]) ++bad;
    std::printf("%-7s %d of %d values differ from Pelias's dequantizer\n", t.name, bad, n);
    fails += bad != 0;
  }
  std::printf("%s\n", fails ? "FORMAT MISMATCH" : "all formats match value for value");
  return fails;
}
