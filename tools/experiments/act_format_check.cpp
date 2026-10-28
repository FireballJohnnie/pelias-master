// provenance: original. Build: g++ -std=c++20 -O2 -Isrc -Iinclude tools/experiments/act_format_check.cpp -Wl,--start-group build/host/*.a -Wl,--end-group -lpthread
// paper check: eight fp32 scales per 256 (gpu256x8) vs q8_1 (gpu path, the measured a8all run), and the noise of
// gpu256 / gpu256e / gpu256x8 relative to q8_1, on rows shaped like real activations (heavy tails, a few outliers)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "model/arch_qwen35moe.h"
#include "quant/quant_cpu.h"
namespace q = pelias::quant;
int main() {
  std::mt19937 rng(42);
  std::student_t_distribution<float> heavy(3.0f);
  const int n = 2048, rows = 2000;
  double max_rel_diff = 0, sum_sq_diff = 0, sum_sq_x = 0, e1 = 0, e256 = 0, ee = 0, ex8 = 0;
  long differ = 0, total = 0;
  for (int r = 0; r < rows; ++r) {
    std::vector<float> x(n);
    for (auto& v : x) v = heavy(rng);
    for (int k = 0; k < 4; ++k) x[rng() % n] *= 20.0f;  // a few outlier channels, as in real activations
    std::vector<float> a = x, b = x, c = x, d = x;
    std::vector<std::uint8_t> blk(36);
    for (int i = 0; i < n; i += 32) {  // q8_1 as act_round_plain does it
      q::quantize_q8_1(a.data() + i, blk.data(), 32);
      std::uint16_t dh; std::memcpy(&dh, blk.data(), 2);
      const float dd = q::fp16_to_fp32(dh);
      for (int j = 0; j < 32; ++j) a[i + j] = dd * float(int8_t(blk[4 + j]));
    }
    pelias::model::act_round_q8_256x8(b.data(), n);
    pelias::model::act_round_q8_256(c.data(), n);
    pelias::model::act_round_q8_256e(d.data(), n);
    for (int i = 0; i < n; ++i) {
      const double df = double(b[i]) - a[i];
      if (df != 0) ++differ;
      ++total;
      sum_sq_diff += df * df;
      sum_sq_x += double(x[i]) * x[i];
      if (a[i] != 0) max_rel_diff = std::max(max_rel_diff, std::fabs(df / a[i]));
      e1 += std::pow(double(a[i]) - x[i], 2); ex8 += std::pow(double(b[i]) - x[i], 2);
      e256 += std::pow(double(c[i]) - x[i], 2); ee += std::pow(double(d[i]) - x[i], 2);
    }
  }
  std::printf("gpu256x8 vs q8_1: values that differ %.4f %% ; largest relative difference %.2e ; rms difference / rms signal %.2e\n",
              100.0 * differ / total, max_rel_diff, std::sqrt(sum_sq_diff / sum_sq_x));
  std::printf("q8_1 own rounding noise: rms error / rms signal %.2e\n", std::sqrt(e1 / sum_sq_x));
  std::printf("rounding noise (squared error) relative to q8_1: gpu256x8 %.4f  gpu256e %.3f  gpu256 %.3f\n", ex8 / e1, ee / e1, e256 / e1);
}
