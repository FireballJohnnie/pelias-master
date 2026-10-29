// Q4_TG2 file check before any machine time (author 01:49; the run-E lesson): the C++ Q4_TG2 path as the quantizer will
// call it (quantize_rows with the importance matrix) vs Q4_K with the same matrix and vs Q4_TG (run E's sqrt form), on
// real blk.20 experts, importance-weighted relative error. Args: slice.bin ne0 rows imp.f32
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "quant/block_types.h"
#include "quant/quant_cpu.h"
namespace q = pelias::quant;
int main(int, char** argv) {
  const int ne0 = std::atoi(argv[2]), rows = std::atoi(argv[3]);
  std::FILE* f = std::fopen(argv[1], "rb");
  std::vector<std::uint16_t> raw(static_cast<size_t>(4) * rows * ne0);
  if (!f || std::fread(raw.data(), 2, raw.size(), f) != raw.size()) return 1;
  std::FILE* g = std::fopen(argv[4], "rb");
  std::vector<float> imp(static_cast<size_t>(4) * ne0);
  if (!g || std::fread(imp.data(), 4, imp.size(), g) != imp.size()) return 2;
  const q::Type types[3] = {q::Type::q4_k, q::Type::q4_tg, q::Type::q4_tg2};
  double e[3] = {0, 0, 0}, w[3] = {0, 0, 0}, ref = 0, refw = 0;
  std::vector<float> x(static_cast<size_t>(ne0)), y(static_cast<size_t>(ne0));
  std::vector<std::uint8_t> b(static_cast<size_t>(q::row_bytes(12, ne0)));
  for (int ex = 0; ex < 4; ++ex)
    for (int rr = 0; rr < rows; rr += 4) {
      for (int i = 0; i < ne0; ++i) {
        std::uint32_t u = static_cast<std::uint32_t>(raw[(static_cast<size_t>(ex) * rows + rr) * ne0 + i]) << 16;
        std::memcpy(&x[size_t(i)], &u, 4);
      }
      const float* im = imp.data() + static_cast<size_t>(ex) * ne0;
      for (int c = 0; c < 3; ++c) {
        if (!q::quantize_rows(types[c], x.data(), b.data(), 1, ne0, im).ok() || !q::dequantize_row(types[c], b.data(), y.data(), ne0).ok()) return 3;
        for (int i = 0; i < ne0; ++i) {
          const double d = (double(y[size_t(i)]) - x[size_t(i)]) * (double(y[size_t(i)]) - x[size_t(i)]);
          e[c] += d; w[c] += d * im[i];
        }
      }
      for (int i = 0; i < ne0; ++i) { ref += double(x[size_t(i)]) * x[size_t(i)]; refw += double(x[size_t(i)]) * x[size_t(i)] * im[i]; }
    }
  const char* nm[3] = {"Q4_K + imatrix", "Q4_TG (run E, sqrt)", "Q4_TG2 (v2, full)"};
  for (int c = 0; c < 3; ++c)
    std::printf("  %-22s plain %.6f  weighted %.6f  (%+.1f %% vs Q4_K weighted)\n", nm[c], e[c] / ref, w[c] / refw, 100.0 * (w[c] / w[0] - 1.0));
  return 0;
}
