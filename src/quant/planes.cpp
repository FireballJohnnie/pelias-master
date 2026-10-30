// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
#include "quant/planes.h"

#include <cmath>
#include <cstring>

#include "quant/quant_cpu.h"

namespace pelias::quant {

namespace {
// kSign8[b][j] = +1 when bit j of b is set, else -1
struct Sign8 {
  alignas(32) float v[256][8];
  constexpr Sign8() : v{} {
    for (int b = 0; b < 256; ++b)
      for (int j = 0; j < 8; ++j) v[b][j] = ((b >> j) & 1) ? 1.0f : -1.0f;
  }
};
constexpr Sign8 kSign8Table{};
constexpr const float (&kSign8)[256][8] = kSign8Table.v;

float round_f16(double v) { return fp16_to_fp32(fp32_to_fp16(static_cast<float>(v))); }

// weighted error of one block for signs sg[k][i] (+1/-1) and scales s[k]
double block_err(const float* r, const float* w, const float s[kPlaneMaxK], const signed char sg[kPlaneMaxK][kPlaneBlock], int k) {
  double e = 0.0;
  for (int i = 0; i < kPlaneBlock; ++i) {
    double v = r[i];
    for (int p = 0; p < k; ++p) v -= static_cast<double>(s[p]) * sg[p][i];
    e += static_cast<double>(w[i]) * v * v;
  }
  return e;
}

// solves the k x k system a x = b in place (Gaussian elimination with partial pivoting); false if singular
bool solve(double a[kPlaneMaxK][kPlaneMaxK], double b[kPlaneMaxK], int k) {
  for (int c = 0; c < k; ++c) {
    int piv = c;
    for (int r = c + 1; r < k; ++r)
      if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
    if (std::fabs(a[piv][c]) < 1e-300) return false;
    if (piv != c) {
      for (int j = 0; j < k; ++j) std::swap(a[c][j], a[piv][j]);
      std::swap(b[c], b[piv]);
    }
    for (int r = c + 1; r < k; ++r) {
      const double f = a[r][c] / a[c][c];
      for (int j = c; j < k; ++j) a[r][j] -= f * a[c][j];
      b[r] -= f * b[c];
    }
  }
  for (int c = k - 1; c >= 0; --c) {
    double v = b[c];
    for (int j = c + 1; j < k; ++j) v -= a[c][j] * b[j];
    b[c] = v / a[c][c];
  }
  return true;
}
}  // namespace

double plane_fit_row(const float* r, const float* imp, std::int64_t n, int k, int refit_rounds, std::uint8_t* out) {
  if (k < 1) k = 1;
  if (k > kPlaneMaxK) k = kPlaneMaxK;
  double total = 0.0;
  float ones[kPlaneBlock];
  for (float& v : ones) v = 1.0f;
  for (std::int64_t b0 = 0; b0 < n; b0 += kPlaneBlock) {
    const float* rb = r + b0;
    const float* wb = imp ? imp + b0 : ones;
    double wsum = 0.0;
    for (int i = 0; i < kPlaneBlock; ++i) wsum += wb[i];
    if (!(wsum > 0.0)) wb = ones, wsum = kPlaneBlock;  // no importance on this block: weigh every value the same
    float s[kPlaneMaxK] = {};
    signed char sg[kPlaneMaxK][kPlaneBlock];
    // plane by plane
    float rr[kPlaneBlock];
    std::memcpy(rr, rb, sizeof rr);
    for (int p = 0; p < k; ++p) {
      double acc = 0.0;
      for (int i = 0; i < kPlaneBlock; ++i) {
        sg[p][i] = rr[i] >= 0.0f ? 1 : -1;
        acc += static_cast<double>(wb[i]) * std::fabs(static_cast<double>(rr[i]));
      }
      s[p] = round_f16(acc / wsum);
      for (int i = 0; i < kPlaneBlock; ++i) rr[i] -= s[p] * static_cast<float>(sg[p][i]);
    }
    double err = block_err(rb, wb, s, sg, k);
    // the planes refitted together
    for (int round = 0; k >= 2 && round < refit_rounds; ++round) {
      double a[kPlaneMaxK][kPlaneMaxK] = {}, y[kPlaneMaxK] = {};
      for (int i = 0; i < kPlaneBlock; ++i)
        for (int p = 0; p < k; ++p) {
          y[p] += static_cast<double>(wb[i]) * sg[p][i] * rb[i];
          for (int q = 0; q < k; ++q) a[p][q] += static_cast<double>(wb[i]) * sg[p][i] * sg[q][i];
        }
      if (!solve(a, y, k)) break;
      float s2[kPlaneMaxK];
      for (int p = 0; p < k; ++p) s2[p] = round_f16(y[p]);
      signed char g2[kPlaneMaxK][kPlaneBlock];
      for (int i = 0; i < kPlaneBlock; ++i) {
        double best = 0.0;
        int bp = 0;
        for (int pat = 0; pat < (1 << k); ++pat) {
          double v = rb[i];
          for (int p = 0; p < k; ++p) v -= static_cast<double>(s2[p]) * (((pat >> p) & 1) ? 1.0 : -1.0);
          const double d = std::fabs(v);
          if (pat == 0 || d < best) best = d, bp = pat;
        }
        for (int p = 0; p < k; ++p) g2[p][i] = ((bp >> p) & 1) ? 1 : -1;
      }
      const double e2 = block_err(rb, wb, s2, g2, k);
      if (!(e2 < err)) break;
      err = e2;
      std::memcpy(s, s2, sizeof s);
      std::memcpy(sg, g2, sizeof sg);
    }
    total += err;  // weighted by the importance (by 1 on a block with none)
    std::uint8_t* o = out + (b0 / kPlaneBlock) * 34 * k;
    for (int p = 0; p < k; ++p) {
      const std::uint16_t h = fp32_to_fp16(s[p]);
      o[2 * p] = static_cast<std::uint8_t>(h & 0xff);
      o[2 * p + 1] = static_cast<std::uint8_t>(h >> 8);
    }
    std::uint8_t* bits = o + 2 * k;
    std::memset(bits, 0, static_cast<std::size_t>(32 * k));
    for (int p = 0; p < k; ++p)
      for (int i = 0; i < kPlaneBlock; ++i)
        if (sg[p][i] > 0) bits[32 * p + i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));
  }
  return total;
}

void plane_add_row(const std::uint8_t* in, std::int64_t n, int k_stored, int k_use, float* y) {
  if (k_use > k_stored) k_use = k_stored;
  for (std::int64_t b0 = 0; b0 < n; b0 += kPlaneBlock) {
    const std::uint8_t* o = in + (b0 / kPlaneBlock) * 34 * k_stored;
    const std::uint8_t* bits = o + 2 * k_stored;
    float* yb = y + b0;
    for (int p = 0; p < k_use; ++p) {
      const float s = fp16_to_fp32(static_cast<std::uint16_t>(o[2 * p] | (o[2 * p + 1] << 8)));
      const std::uint8_t* bp = bits + 32 * p;
      // 8 values per sign byte from the +/-1 table: no branch per value (random signs mispredict half the time), and the
      // same sums as value by value (s x +/-1 is exact, added in the same order)
      for (int k = 0; k < kPlaneBlock / 8; ++k) {
        const float* t = kSign8[bp[k]];
        float* yk = yb + 8 * k;
        for (int j = 0; j < 8; ++j) yk[j] += s * t[j];
      }
    }
  }
}

void plane_add_row_sparse(const std::uint8_t* in, const std::uint8_t* mask, std::int64_t n, int k_stored, int k_use, float* y) {
  const std::int64_t bytes = 34 * static_cast<std::int64_t>(k_stored);
  for (std::int64_t b = 0; b < n / kPlaneBlock; ++b) {
    if (!((mask[b / 8] >> (b % 8)) & 1)) continue;
    plane_add_row(in, kPlaneBlock, k_stored, k_use, y + b * kPlaneBlock);  // one block: the same layout as a 256-value row
    in += bytes;
  }
}

}  // namespace pelias::quant
