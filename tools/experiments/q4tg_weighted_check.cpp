// Run E diagnosis (2026-10-28 23:5x): Q4_TG vs Q4_K with the importance matrix on real blk.20 experts, plain and
// importance-weighted error; Q4_TG with sqrt (as built for E) and full importance. Args: slice ne0 rows imp.f32
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include "quant/quant_cpu.h"
#include "quant/block_types.h"
namespace q = pelias::quant;
int main(int, char** argv) {
  const int ne0 = std::atoi(argv[2]), rows = std::atoi(argv[3]);
  std::FILE* f = std::fopen(argv[1], "rb"); std::vector<std::uint16_t> raw(static_cast<size_t>(4) * rows * ne0); if (std::fread(raw.data(), 2, raw.size(), f) != raw.size()) return 1;
  std::FILE* g = std::fopen(argv[4], "rb"); std::vector<float> imp(4 * ne0); if (std::fread(imp.data(), 4, imp.size(), g) != imp.size()) return 2;
  double e[5] = {0, 0, 0, 0, 0}, w[5] = {0, 0, 0, 0, 0}, ref = 0, refw = 0; std::vector<float> ones(ne0, 1.0f);
  std::vector<float> x(ne0), y(ne0); std::vector<std::uint8_t> b(q::row_bytes(203, ne0));
  for (int ex = 0; ex < 4; ++ex) for (int rr = 0; rr < rows; rr += 4) {
    for (int i = 0; i < ne0; ++i) { std::uint32_t u = static_cast<std::uint32_t>(raw[(static_cast<size_t>(ex) * rows + rr) * ne0 + i]) << 16; std::memcpy(&x[i], &u, 4); }
    const float* im = imp.data() + ex * ne0;
    struct { q::Type t; const float* m; } cfg[4] = {{q::Type::q4_k, im}, {q::Type::q4_tg, im}, {q::Type::q4_k, nullptr}, {q::Type::q4_tg, nullptr}};
    for (int c = 0; c < 4; ++c) {
      (void)q::quantize_rows(cfg[c].t, x.data(), b.data(), 1, ne0, cfg[c].m); (void)q::dequantize_row(cfg[c].t, b.data(), y.data(), ne0);
      for (int i = 0; i < ne0; ++i) { double d = (y[i] - x[i]) * (y[i] - x[i]); e[c] += d; w[c] += d * im[i]; }
    }
    for (int i = 0; i < ne0 / 256; ++i) q::q4_tg_block(x.data() + 256 * i, b.data() + 144 * i, im + 256 * i);
    (void)q::dequantize_row(q::Type::q4_tg, b.data(), y.data(), ne0);
    for (int i = 0; i < ne0; ++i) { double d = (y[i] - x[i]) * (y[i] - x[i]); e[4] += d; w[4] += d * im[i]; }
    for (int i = 0; i < ne0; ++i) { ref += x[i] * x[i]; refw += x[i] * x[i] * im[i]; }
  }
  std::printf("  Q4_TG full imatrix: plain %.6f weighted %.6f\n", e[4]/ref, w[4]/refw);
  std::printf("%s\n  plain error:     Q4_K+imatrix %.6f  Q4_TG+sqrt(imatrix) %.6f | Q4_K plain %.6f  Q4_TG plain %.6f\n  weighted error:  Q4_K+imatrix %.6f  Q4_TG+sqrt(imatrix) %.6f | Q4_K plain %.6f  Q4_TG plain %.6f\n", argv[1],
              e[0]/ref, e[1]/ref, e[2]/ref, e[3]/ref, w[0]/refw, w[1]/refw, w[2]/refw, w[3]/refw);
}
