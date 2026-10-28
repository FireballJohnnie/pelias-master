// provenance: re-expressed (not copied) from ggml ggml/src/ggml-quants.c (make_q3_quants, make_qkx3_quants,
//   make_qp_quants, quantize_row_q{2,3,5}_K_ref, quantize_row_q{2,3,4,5,6}_K_impl, quantize_q{2,3,4,5,6}_K,
//   quantize_row_iq4_nl_impl, quantize_iq4_nl, quantize_iq4_xs, best_index_int8) at llama.cpp commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a (MIT; THIRD_PARTY_NOTICES.md), read as a specification so the bytes match
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// The quantizers that take an importance matrix (one weight per column of the row: how much that column's activations
// matter), and the K-quant and IQ4 quantizers that had no Pelias version yet. Each type follows the reference's own
// split: without weights the deterministic "ref" routine, with weights the search routine. Same operand order and
// integer-to-float conversions as the reference; this module is compiled with contraction off (no fused multiply-add).
#include <algorithm>
#include <cmath>
#include <cstring>

#include "quant/quant_cpu.h"
#include "quant/quant_internal.h"

namespace pelias::quant {

namespace {

using detail::get_scale_min_k4;
using detail::make_qkx2_quants;
using detail::make_qx_quants;
using detail::put_u16;

constexpr int kQKK = 256;
constexpr float kGroupMaxEps = 1e-15f;
constexpr std::int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline float h2f(std::uint16_t v) { return fp16_to_fp32(v); }

// ---- helpers

float make_q3_quants(int n, int nmax, const float* x, std::int8_t* L, bool do_rmse) {
  float max = 0, amax = 0;
  for (int i = 0; i < n; ++i) {
    const float ax = std::fabs(x[i]);
    if (ax > amax) {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < kGroupMaxEps) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    return 0.f;
  }
  const float iscale = static_cast<float>(-nmax) / max;
  if (do_rmse) {
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      l = std::max(-nmax, std::min(nmax - 1, l));
      L[i] = static_cast<std::int8_t>(l);
      const float w = x[i] * x[i];
      sumlx += w * x[i] * static_cast<float>(l);
      suml2 += w * static_cast<float>(l) * static_cast<float>(l);
    }
    for (int itry = 0; itry < 5; ++itry) {
      int n_changed = 0;
      for (int i = 0; i < n; ++i) {
        const float w = x[i] * x[i];
        float slx = sumlx - w * x[i] * static_cast<float>(L[i]);
        if (slx > 0) {
          float sl2 = suml2 - w * static_cast<float>(L[i]) * static_cast<float>(L[i]);
          int new_l = nearest_int(x[i] * sl2 / slx);
          new_l = std::max(-nmax, std::min(nmax - 1, new_l));
          if (new_l != L[i]) {
            slx += w * x[i] * static_cast<float>(new_l);
            sl2 += w * static_cast<float>(new_l) * static_cast<float>(new_l);
            if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
              L[i] = static_cast<std::int8_t>(new_l);
              sumlx = slx;
              suml2 = sl2;
              ++n_changed;
            }
          }
        }
      }
      if (!n_changed) break;
    }
    for (int i = 0; i < n; ++i) L[i] = static_cast<std::int8_t>(L[i] + nmax);
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
  }
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * x[i]);
    l = std::max(-nmax, std::min(nmax - 1, l));
    L[i] = static_cast<std::int8_t>(l + nmax);
  }
  return 1 / iscale;
}

float make_qkx3_quants(int n, int nmax, const float* x, const float* weights, std::uint8_t* L, float* the_min, std::uint8_t* Laux,
                       float rmin, float rdelta, int nstep, bool use_mad) {
  auto wt = [&](int i) { return weights ? weights[i] : x[i] * x[i]; };
  float min = x[0], max = x[0];
  float sum_w = wt(0);
  float sum_x = sum_w * x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
    const float w = wt(i);
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0) min = 0;
  if (max <= min) {
    std::memset(L, 0, static_cast<std::size_t>(n));
    *the_min = -min;
    return 0.f;
  }
  float iscale = static_cast<float>(nmax) / (max - min);
  float scale = 1 / iscale;
  float best_mad = 0;
  for (int i = 0; i < n; ++i) {
    const int l = nearest_int(iscale * (x[i] - min));
    L[i] = static_cast<std::uint8_t>(std::max(0, std::min(nmax, l)));
    float diff = scale * static_cast<float>(L[i]) + min - x[i];
    diff = use_mad ? std::fabs(diff) : diff * diff;
    best_mad += wt(i) * diff;
  }
  if (nstep < 1) {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * static_cast<float>(is) + static_cast<float>(nmax)) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      // the best fit so far may put min at or near max, so v may be inf, nan or out of range
      const float v = iscale * (x[i] - min);
      const int l = v > 0 ? nearest_int(v < static_cast<float>(nmax) ? v : static_cast<float>(nmax)) : 0;
      Laux[i] = static_cast<std::uint8_t>(l);
      const float w = wt(i);
      const float lf = static_cast<float>(l);
      sum_l += w * lf;
      sum_l2 += w * lf * lf;
      sum_xl += w * lf * x[i];
    }
    const float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0) {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * static_cast<float>(Laux[i]) + this_min - x[i];
        diff = use_mad ? std::fabs(diff) : diff * diff;
        mad += wt(i) * diff;
      }
      if (mad < best_mad) {
        for (int i = 0; i < n; ++i) L[i] = Laux[i];
        best_mad = mad;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

float make_qp_quants(int n, int nmax, const float* x, std::uint8_t* L, const float* quant_weights) {
  float max = 0;
  for (int i = 0; i < n; ++i) max = std::max(max, x[i]);
  if (max < kGroupMaxEps) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    return 0.f;
  }
  float iscale = static_cast<float>(nmax) / max;
  for (int i = 0; i < n; ++i) L[i] = static_cast<std::uint8_t>(nearest_int(iscale * x[i]));
  const float scale = 1 / iscale;
  float best_mse = 0;
  for (int i = 0; i < n; ++i) {
    const float diff = x[i] - scale * static_cast<float>(L[i]);
    const float w = quant_weights[i];
    best_mse += w * diff * diff;
  }
  for (int is = -4; is <= 4; ++is) {
    if (is == 0) continue;
    const float iscale_is = (0.1f * static_cast<float>(is) + static_cast<float>(nmax)) / max;
    const float scale_is = 1 / iscale_is;
    float mse = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale_is * x[i]);
      l = std::min(nmax, l);
      const float diff = x[i] - scale_is * static_cast<float>(l);
      const float w = quant_weights[i];
      mse += w * diff * diff;
    }
    if (mse < best_mse) {
      best_mse = mse;
      iscale = iscale_is;
    }
  }
  float sumlx = 0, suml2 = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * x[i]);
    l = std::min(nmax, l);
    L[i] = static_cast<std::uint8_t>(l);
    const float w = quant_weights[i];
    sumlx += w * x[i] * static_cast<float>(l);
    suml2 += w * static_cast<float>(l) * static_cast<float>(l);
  }
  for (int itry = 0; itry < 5; ++itry) {
    int n_changed = 0;
    for (int i = 0; i < n; ++i) {
      const float w = quant_weights[i];
      float slx = sumlx - w * x[i] * static_cast<float>(L[i]);
      float sl2 = suml2 - w * static_cast<float>(L[i]) * static_cast<float>(L[i]);
      if (slx > 0 && sl2 > 0) {
        int new_l = nearest_int(x[i] * sl2 / slx);
        new_l = std::min(nmax, new_l);
        if (new_l != L[i]) {
          slx += w * x[i] * static_cast<float>(new_l);
          sl2 += w * static_cast<float>(new_l) * static_cast<float>(new_l);
          if (slx * slx * suml2 > sumlx * sumlx * sl2) {
            L[i] = static_cast<std::uint8_t>(new_l);
            sumlx = slx;
            suml2 = sl2;
            ++n_changed;
          }
        }
      }
    }
    if (!n_changed) break;
  }
  return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

int best_index_int8(int n, const std::int8_t* val, float x) {
  if (x <= static_cast<float>(val[0])) return 0;
  if (x >= static_cast<float>(val[n - 1])) return n - 1;
  int ml = 0, mu = n - 1;
  while (mu - ml > 1) {
    const int mav = (ml + mu) / 2;
    if (x < static_cast<float>(val[mav])) mu = mav;
    else ml = mav;
  }
  return x - static_cast<float>(val[mu - 1]) < static_cast<float>(val[mu]) - x ? mu - 1 : mu;
}

// pack 256 two-bit levels as the reference does (Q2_K, Q3_K low bits)
void pack2(const std::uint8_t* L, std::uint8_t* qs) {
  for (int j = 0; j < kQKK; j += 128)
    for (int l = 0; l < 32; ++l)
      qs[j / 4 + l] = static_cast<std::uint8_t>(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
}

// ---- Q2_K: scales[16] | qs[64] | d | dmin (84 bytes)

void q2_k_ref_block(const float* x, std::uint8_t* y) {
  std::uint8_t L[kQKK], Laux[16];
  float weights[16], mins[kQKK / 16], scales[kQKK / 16];
  const float q4scale = 15.f;
  std::uint8_t* sc = y;
  float max_scale = 0, max_min = 0;
  for (int j = 0; j < kQKK / 16; ++j) {
    for (int l = 0; l < 16; ++l) weights[l] = std::fabs(x[16 * j + l]);
    scales[j] = make_qkx2_quants(16, 3, x + 16 * j, weights, L + 16 * j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
    if (scales[j] > max_scale) max_scale = scales[j];
    if (mins[j] > max_min) max_min = mins[j];
  }
  std::uint16_t d16, dmin16;
  if (max_scale > 0) {
    const float iscale = q4scale / max_scale;
    for (int j = 0; j < kQKK / 16; ++j) sc[j] = static_cast<std::uint8_t>(nearest_int(iscale * scales[j]));
    d16 = fp32_to_fp16(max_scale / q4scale);
  } else {
    for (int j = 0; j < kQKK / 16; ++j) sc[j] = 0;
    d16 = fp32_to_fp16(0.f);
  }
  if (max_min > 0) {
    const float iscale = q4scale / max_min;
    for (int j = 0; j < kQKK / 16; ++j) sc[j] = static_cast<std::uint8_t>(sc[j] | (nearest_int(iscale * mins[j]) << 4));
    dmin16 = fp32_to_fp16(max_min / q4scale);
  } else {
    dmin16 = fp32_to_fp16(0.f);
  }
  for (int j = 0; j < kQKK / 16; ++j) {
    const float d = h2f(d16) * static_cast<float>(sc[j] & 0xF);
    if (d == 0.0f) continue;
    const float dm = h2f(dmin16) * static_cast<float>(sc[j] >> 4);
    for (int ii = 0; ii < 16; ++ii) {
      int l = nearest_int((x[16 * j + ii] + dm) / d);
      l = std::max(0, std::min(3, l));
      L[16 * j + ii] = static_cast<std::uint8_t>(l);
    }
  }
  pack2(L, y + 16);
  put_u16(y + 80, d16);
  put_u16(y + 82, dmin16);
}

void q2_k_impl_block(const float* x, std::uint8_t* y, const float* qwb) {
  std::uint8_t L[kQKK], Laux[16], Ls[kQKK / 16], Lm[kQKK / 16];
  float mins[kQKK / 16], scales[kQKK / 16], sw[kQKK / 16], weight[16];
  std::memset(sw, 0, sizeof sw);
  float sumx2 = 0;
  for (int j = 0; j < kQKK; ++j) sumx2 += x[j] * x[j];
  const float sigma2 = sumx2 / kQKK;
  for (int j = 0; j < kQKK / 16; ++j) {
    const float* qw = qwb + 16 * j;
    for (int l = 0; l < 16; ++l) weight[l] = qw[l] * std::sqrt(sigma2 + x[16 * j + l] * x[16 * j + l]);
    for (int l = 0; l < kQKK / 16; ++l) sw[j] += weight[l];
    scales[j] = make_qkx3_quants(16, 3, x + 16 * j, weight, L + 16 * j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
  }
  float dm = make_qp_quants(kQKK / 16, 15, scales, Ls, sw);
  float mm = make_qp_quants(kQKK / 16, 15, mins, Lm, sw);
  const std::uint16_t d16 = fp32_to_fp16(dm), dmin16 = fp32_to_fp16(mm);
  dm = h2f(d16);
  mm = h2f(dmin16);
  std::uint8_t* sc = y;
  for (int j = 0; j < kQKK / 16; ++j) sc[j] = static_cast<std::uint8_t>(Ls[j] | (Lm[j] << 4));
  for (int j = 0; j < kQKK / 16; ++j) {  // requantize with the stored scales
    const float d = dm * static_cast<float>(sc[j] & 0xF);
    if (d == 0.0f) continue;
    const float m = mm * static_cast<float>(sc[j] >> 4);
    for (int ii = 0; ii < 16; ++ii) {
      int l = nearest_int((x[16 * j + ii] + m) / d);
      l = std::max(0, std::min(3, l));
      L[16 * j + ii] = static_cast<std::uint8_t>(l);
    }
  }
  pack2(L, y + 16);
  put_u16(y + 80, d16);
  put_u16(y + 82, dmin16);
}

// ---- Q3_K: hmask[32] | qs[64] | scales[12] | d (110 bytes)

void q3_k_finish(const float* x, std::uint8_t* y, std::int8_t* L, std::uint16_t d16) {
  std::uint8_t* sc = y + 96;
  for (int j = 0; j < kQKK / 16; ++j) {
    std::int8_t s = static_cast<std::int8_t>(j < 8 ? sc[j] & 0xF : sc[j - 8] >> 4);
    s = static_cast<std::int8_t>((s | (((sc[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32);
    const float d = h2f(d16) * static_cast<float>(s);
    if (d == 0.0f) continue;
    for (int ii = 0; ii < 16; ++ii) {
      int l = nearest_int(x[16 * j + ii] / d);
      l = std::max(-4, std::min(3, l));
      L[16 * j + ii] = static_cast<std::int8_t>(l + 4);
    }
  }
  std::uint8_t* hmask = y;
  std::memset(hmask, 0, kQKK / 8);
  int m = 0;
  std::uint8_t hm = 1;
  for (int j = 0; j < kQKK; ++j) {
    if (L[j] > 3) {
      hmask[m] = static_cast<std::uint8_t>(hmask[m] | hm);
      L[j] = static_cast<std::int8_t>(L[j] - 4);
    }
    if (++m == kQKK / 8) {
      m = 0;
      hm = static_cast<std::uint8_t>(hm << 1);
    }
  }
  std::uint8_t Lu[kQKK];
  for (int j = 0; j < kQKK; ++j) Lu[j] = static_cast<std::uint8_t>(L[j]);
  pack2(Lu, y + 32);
  put_u16(y + 108, d16);
}

void q3_k_put_scale(std::uint8_t* sc, int j, int l) {
  if (j < 8) sc[j] = static_cast<std::uint8_t>(l & 0xF);
  else sc[j - 8] = static_cast<std::uint8_t>(sc[j - 8] | ((l & 0xF) << 4));
  l >>= 4;
  sc[j % 4 + 8] = static_cast<std::uint8_t>(sc[j % 4 + 8] | (l << (2 * (j / 4))));
}

void q3_k_ref_block(const float* x, std::uint8_t* y) {
  std::int8_t L[kQKK];
  float scales[kQKK / 16];
  float max_scale = 0, amax = 0;
  for (int j = 0; j < kQKK / 16; ++j) {
    scales[j] = make_q3_quants(16, 4, x + 16 * j, L + 16 * j, true);
    const float scale = std::fabs(scales[j]);
    if (scale > amax) {
      amax = scale;
      max_scale = scales[j];
    }
  }
  std::uint8_t* sc = y + 96;
  std::memset(sc, 0, 12);
  std::uint16_t d16;
  if (max_scale != 0.0f) {
    const float iscale = -32.f / max_scale;
    for (int j = 0; j < kQKK / 16; ++j) {
      std::int8_t l = static_cast<std::int8_t>(nearest_int(iscale * scales[j]));
      l = static_cast<std::int8_t>(std::max(-32, std::min(31, static_cast<int>(l))) + 32);
      q3_k_put_scale(sc, j, l);
    }
    d16 = fp32_to_fp16(1 / iscale);
  } else {
    d16 = fp32_to_fp16(0.f);
  }
  q3_k_finish(x, y, L, d16);
}

void q3_k_impl_block(const float* x, std::uint8_t* y, const float* qwb) {
  std::int8_t L[kQKK], Ls[kQKK / 16];
  float scales[kQKK / 16], weight[16], sw[kQKK / 16];
  float sumx2 = 0;
  for (int j = 0; j < kQKK; ++j) sumx2 += x[j] * x[j];
  const float sigma2 = 2 * sumx2 / kQKK;
  for (int j = 0; j < kQKK / 16; ++j) {
    if (qwb) {
      const float* qw = qwb + 16 * j;
      for (int l = 0; l < 16; ++l) weight[l] = qw[l] * std::sqrt(sigma2 + x[16 * j + l] * x[16 * j + l]);
    } else {
      for (int l = 0; l < 16; ++l) weight[l] = x[16 * j + l] * x[16 * j + l];
    }
    float sumw = 0;
    for (int l = 0; l < 16; ++l) sumw += weight[l];
    sw[j] = sumw;
    scales[j] = make_qx_quants(16, 4, x + 16 * j, L + 16 * j, 1, weight);
  }
  std::uint8_t* sc = y + 96;
  std::memset(sc, 0, 12);
  const float d_block = make_qx_quants(kQKK / 16, 32, scales, Ls, 1, sw);
  for (int j = 0; j < kQKK / 16; ++j) q3_k_put_scale(sc, j, Ls[j]);
  q3_k_finish(x, y, L, fp32_to_fp16(d_block));
}

// ---- Q4_K weighted: d | dmin | scales[12] | qs[128] (144 bytes);  Q5_K: d | dmin | scales[12] | qh[32] | qs[128] (176)

void k_put_scales(std::uint8_t* sc, const std::uint8_t* Ls, const std::uint8_t* Lm) {
  for (int j = 0; j < kQKK / 32; ++j) {
    const std::uint8_t ls = Ls[j], lm = Lm[j];
    if (j < 4) {
      sc[j] = ls;
      sc[j + 4] = lm;
    } else {
      sc[j + 4] = static_cast<std::uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
      sc[j - 4] = static_cast<std::uint8_t>(sc[j - 4] | ((ls >> 4) << 6));
      sc[j - 0] = static_cast<std::uint8_t>(sc[j - 0] | ((lm >> 4) << 6));
    }
  }
}

void k_requant(const float* x, const std::uint8_t* sc, std::uint16_t d16, std::uint16_t dmin16, int nmax, std::uint8_t* L) {
  std::uint8_t s, m;
  for (int j = 0; j < kQKK / 32; ++j) {
    get_scale_min_k4(j, sc, &s, &m);
    const float d = h2f(d16) * static_cast<float>(s);
    if (d == 0.0f) continue;
    const float dm = h2f(dmin16) * static_cast<float>(m);
    for (int ii = 0; ii < 32; ++ii) {
      int l = nearest_int((x[32 * j + ii] + dm) / d);
      l = std::max(0, std::min(nmax, l));
      L[32 * j + ii] = static_cast<std::uint8_t>(l);
    }
  }
}

// the weighted search shared by Q4_K (nmax 15) and Q5_K (nmax 31); fills scales, d, dmin and L
void k_impl_scales(const float* x, const float* qwb, int nmax, std::uint8_t* y, std::uint8_t* L, std::uint16_t* d16, std::uint16_t* dmin16) {
  std::uint8_t Laux[32], Ls[kQKK / 32], Lm[kQKK / 32];
  float weights[32], sw[kQKK / 32], mins[kQKK / 32], scales[kQKK / 32];
  float sum_x2 = 0;
  for (int l = 0; l < kQKK; ++l) sum_x2 += x[l] * x[l];
  const float sigma2 = 2 * sum_x2 / kQKK;
  const float av_x = std::sqrt(sigma2);
  for (int j = 0; j < kQKK / 32; ++j) {
    if (qwb) {
      const float* qw = qwb + 32 * j;
      for (int l = 0; l < 32; ++l) weights[l] = qw[l] * std::sqrt(sigma2 + x[32 * j + l] * x[32 * j + l]);
    } else {
      for (int l = 0; l < 32; ++l) weights[l] = av_x + std::fabs(x[32 * j + l]);
    }
    float sumw = 0;
    for (int l = 0; l < 32; ++l) sumw += weights[l];
    sw[j] = sumw;
    scales[j] = make_qkx3_quants(32, nmax, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
  }
  const float d_block = make_qp_quants(kQKK / 32, 63, scales, Ls, sw);
  const float m_block = make_qp_quants(kQKK / 32, 63, mins, Lm, sw);
  if (nmax == 31)
    for (int j = 0; j < kQKK / 32; ++j) {  // the Q5_K routine clamps; Q4_K's does not (make_qp_quants keeps them <= 63)
      Ls[j] = std::min<std::uint8_t>(63, Ls[j]);
      Lm[j] = std::min<std::uint8_t>(63, Lm[j]);
    }
  k_put_scales(y + 4, Ls, Lm);
  *d16 = fp32_to_fp16(d_block);
  *dmin16 = fp32_to_fp16(m_block);
  k_requant(x, y + 4, *d16, *dmin16, nmax, L);
}

void q4_k_impl_block(const float* x, std::uint8_t* y, const float* qwb) {
  std::uint8_t L[kQKK];
  std::uint16_t d16, dmin16;
  std::memset(y + 4, 0, 12);
  k_impl_scales(x, qwb, 15, y, L, &d16, &dmin16);
  put_u16(y, d16);
  put_u16(y + 2, dmin16);
  std::uint8_t* q = y + 16;
  for (int j = 0; j < kQKK; j += 64) {
    for (int l = 0; l < 32; ++l) q[l] = static_cast<std::uint8_t>(L[j + l] | (L[j + l + 32] << 4));
    q += 32;
  }
}

void q5_k_pack(const std::uint8_t* L, std::uint8_t* y) {
  std::uint8_t* qh = y + 16;
  std::uint8_t* ql = y + 48;
  std::memset(qh, 0, kQKK / 8);
  std::uint8_t m1 = 1, m2 = 2;
  for (int n = 0; n < kQKK; n += 64) {
    for (int j = 0; j < 32; ++j) {
      int l1 = L[n + j];
      if (l1 > 15) {
        l1 -= 16;
        qh[j] = static_cast<std::uint8_t>(qh[j] | m1);
      }
      int l2 = L[n + j + 32];
      if (l2 > 15) {
        l2 -= 16;
        qh[j] = static_cast<std::uint8_t>(qh[j] | m2);
      }
      ql[j] = static_cast<std::uint8_t>(l1 | (l2 << 4));
    }
    m1 = static_cast<std::uint8_t>(m1 << 2);
    m2 = static_cast<std::uint8_t>(m2 << 2);
    ql += 32;
  }
}

void q5_k_ref_block(const float* x, std::uint8_t* y) {
  std::uint8_t L[kQKK], Laux[32];
  float mins[kQKK / 32], scales[kQKK / 32], weights[32];
  float max_scale = 0, max_min = 0;
  for (int j = 0; j < kQKK / 32; ++j) {
    float sum_x2 = 0;
    for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
    const float av_x = std::sqrt(sum_x2 / 32);
    for (int l = 0; l < 32; ++l) weights[l] = av_x + std::fabs(x[32 * j + l]);
    scales[j] = make_qkx2_quants(32, 31, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -0.5f, 0.1f, 15, false);
    if (scales[j] > max_scale) max_scale = scales[j];
    if (mins[j] > max_min) max_min = mins[j];
  }
  const float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
  const float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
  std::uint8_t Ls[kQKK / 32], Lm[kQKK / 32];
  for (int j = 0; j < kQKK / 32; ++j) {
    std::uint8_t ls = static_cast<std::uint8_t>(nearest_int(inv_scale * scales[j]));
    std::uint8_t lm = static_cast<std::uint8_t>(nearest_int(inv_min * mins[j]));
    Ls[j] = std::min<std::uint8_t>(63, ls);
    Lm[j] = std::min<std::uint8_t>(63, lm);
  }
  std::memset(y + 4, 0, 12);
  k_put_scales(y + 4, Ls, Lm);
  const std::uint16_t d16 = fp32_to_fp16(max_scale / 63.f), dmin16 = fp32_to_fp16(max_min / 63.f);
  k_requant(x, y + 4, d16, dmin16, 31, L);
  q5_k_pack(L, y);
  put_u16(y, d16);
  put_u16(y + 2, dmin16);
}

void q5_k_impl_block(const float* x, std::uint8_t* y, const float* qwb) {
  std::uint8_t L[kQKK];
  std::uint16_t d16, dmin16;
  std::memset(y + 4, 0, 12);
  k_impl_scales(x, qwb, 31, y, L, &d16, &dmin16);
  q5_k_pack(L, y);
  put_u16(y, d16);
  put_u16(y + 2, dmin16);
}

// ---- Q6_K weighted: ql[128] | qh[64] | scales[16] | d (210 bytes)

void q6_k_impl_block(const float* x, std::uint8_t* y, const float* qwb) {
  std::int8_t L[kQKK];
  float scales[kQKK / 16];
  float max_scale = 0, max_abs_scale = 0;
  for (int ib = 0; ib < kQKK / 16; ++ib) {
    const float scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, qwb ? qwb + 16 * ib : nullptr);
    scales[ib] = scale;
    const float abs_scale = std::fabs(scale);
    if (abs_scale > max_abs_scale) {
      max_abs_scale = abs_scale;
      max_scale = scale;
    }
  }
  if (max_abs_scale < kGroupMaxEps) {
    std::memset(y, 0, 210);
    put_u16(y + 208, fp32_to_fp16(0.f));
    return;
  }
  const float iscale = -128.f / max_scale;
  const std::uint16_t d16 = fp32_to_fp16(1 / iscale);
  std::int8_t* sc = reinterpret_cast<std::int8_t*>(y + 192);
  for (int ib = 0; ib < kQKK / 16; ++ib) sc[ib] = static_cast<std::int8_t>(std::min(127, nearest_int(iscale * scales[ib])));
  for (int j = 0; j < kQKK / 16; ++j) {
    const float d = h2f(d16) * static_cast<float>(sc[j]);
    if (d == 0.0f) continue;
    for (int ii = 0; ii < 16; ++ii) {
      int l = nearest_int(x[16 * j + ii] / d);
      l = std::max(-32, std::min(31, l));
      L[16 * j + ii] = static_cast<std::int8_t>(l + 32);
    }
  }
  std::uint8_t* ql = y;
  std::uint8_t* qh = y + 128;
  for (int j = 0; j < kQKK; j += 128) {
    for (int l = 0; l < 32; ++l) {
      const std::uint8_t q1 = L[j + l + 0] & 0xF, q2 = L[j + l + 32] & 0xF, q3 = L[j + l + 64] & 0xF, q4 = L[j + l + 96] & 0xF;
      ql[l + 0] = static_cast<std::uint8_t>(q1 | (q3 << 4));
      ql[l + 32] = static_cast<std::uint8_t>(q2 | (q4 << 4));
      qh[l] = static_cast<std::uint8_t>((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6));
    }
    ql += 64;
    qh += 32;
  }
  put_u16(y + 208, d16);
}

// ---- IQ4_NL (d | qs[16], 18 bytes per 32) and IQ4_XS (d | scales_h | scales_l[4] | qs[128], 136 bytes per 256)

void iq4_impl(int super_block_size, const float* x, std::uint16_t* dh, std::uint8_t* q4, std::uint16_t* scales_h, std::uint8_t* scales_l,
              const float* quant_weights, int ntry) {
  constexpr int block_size = 32;
  std::uint8_t L[kQKK];
  float scales[kQKK / 32], weight[32];
  float sigma2 = 0;
  for (int j = 0; j < super_block_size; ++j) sigma2 += x[j] * x[j];
  sigma2 *= 2.f / static_cast<float>(super_block_size);
  std::memset(q4, 0, static_cast<std::size_t>(super_block_size / 2));
  *dh = fp32_to_fp16(0.f);
  float max_scale = 0, amax_scale = 0;
  const std::int8_t* values = kIq4nl;
  for (int ib = 0; ib < super_block_size / block_size; ++ib) {
    const float* xb = x + ib * block_size;
    std::uint8_t* Lb = L + ib * block_size;
    if (quant_weights) {
      const float* qw = quant_weights + ib * block_size;
      for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * std::sqrt(sigma2 + xb[j] * xb[j]);
    } else {
      for (int j = 0; j < block_size; ++j) weight[j] = xb[j] * xb[j];
    }
    float amax = 0, max = 0;
    for (int j = 0; j < block_size; ++j) {
      const float ax = std::fabs(xb[j]);
      if (ax > amax) {
        amax = ax;
        max = xb[j];
      }
    }
    if (amax < kGroupMaxEps) {
      scales[ib] = 0;
      continue;
    }
    float d = ntry > 0 ? -max / static_cast<float>(values[0]) : max / static_cast<float>(values[0]);
    float id = 1 / d;
    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < block_size; ++j) {
      const float al = id * xb[j];
      const int l = best_index_int8(16, values, al);
      Lb[j] = static_cast<std::uint8_t>(l);
      const float q = static_cast<float>(values[l]);
      const float w = weight[j];
      sumqx += w * q * xb[j];
      sumq2 += w * q * q;
    }
    d = sumq2 > 0 ? sumqx / sumq2 : 0.f;
    float best = d * sumqx;
    for (int itry = -ntry; itry <= ntry; ++itry) {
      id = static_cast<float>(itry + values[0]) / max;
      sumqx = sumq2 = 0;
      for (int j = 0; j < block_size; ++j) {
        const float al = id * xb[j];
        const int l = best_index_int8(16, values, al);
        const float q = static_cast<float>(values[l]);
        const float w = weight[j];
        sumqx += w * q * xb[j];
        sumq2 += w * q * q;
      }
      if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
        d = sumqx / sumq2;
        best = d * sumqx;
      }
    }
    scales[ib] = d;
    const float abs_d = std::fabs(d);
    if (abs_d > amax_scale) {
      amax_scale = abs_d;
      max_scale = d;
    }
  }
  if (super_block_size / block_size > 1) {
    const int nb = super_block_size / block_size;
    std::memset(scales_h, 0, static_cast<std::size_t>((nb + 7) / 8) * sizeof(std::uint16_t));
    const float d = -max_scale / 32;
    *dh = fp32_to_fp16(d);
    const float id = d != 0.0f ? 1 / d : 0.f;
    for (int ib = 0; ib < nb; ++ib) {
      int l = nearest_int(id * scales[ib]);
      l = std::max(-32, std::min(31, l));
      const float dl = d * static_cast<float>(l);
      const float idl = dl != 0.0f ? 1 / dl : 0.f;
      std::uint8_t* Lb = L + ib * block_size;
      const float* xb = x + ib * block_size;
      for (int j = 0; j < block_size; ++j) Lb[j] = static_cast<std::uint8_t>(best_index_int8(16, values, idl * xb[j]));
      l += 32;
      const std::uint8_t l_l = static_cast<std::uint8_t>(l & 0xf);
      const std::uint8_t l_h = static_cast<std::uint8_t>(l >> 4);
      if (ib % 2 == 0) scales_l[ib / 2] = l_l;
      else scales_l[ib / 2] = static_cast<std::uint8_t>(scales_l[ib / 2] | (l_l << 4));
      scales_h[ib / 8] = static_cast<std::uint16_t>(scales_h[ib / 8] | (l_h << 2 * (ib % 8)));
    }
  } else {
    *dh = fp32_to_fp16(scales[0]);
    if (ntry > 0) {
      const float id = scales[0] != 0.0f ? 1 / scales[0] : 0;
      for (int j = 0; j < super_block_size; ++j) L[j] = static_cast<std::uint8_t>(best_index_int8(16, values, id * x[j]));
    }
  }
  for (int i = 0; i < super_block_size / 32; ++i)
    for (int j = 0; j < 16; ++j) q4[16 * i + j] = static_cast<std::uint8_t>(L[32 * i + j] | (L[32 * i + 16 + j] << 4));
}

void iq4_nl_block(const float* x, std::uint8_t* y, const float* qw) {
  std::uint16_t dh, unused_h = 0;
  iq4_impl(32, x, &dh, y + 2, &unused_h, nullptr, qw, 7);
  put_u16(y, dh);
}

void iq4_xs_block(const float* x, std::uint8_t* y, const float* qw) {
  std::uint16_t dh, scales_h = 0;
  iq4_impl(kQKK, x, &dh, y + 8, &scales_h, y + 4, qw, 7);
  put_u16(y, dh);
  put_u16(y + 2, scales_h);
}

}  // namespace

// ---- entry points

void quantize_q2_k(const float* x, std::uint8_t* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n / kQKK; ++i) q2_k_ref_block(x + i * kQKK, out + i * 84);
}
void quantize_q3_k(const float* x, std::uint8_t* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n / kQKK; ++i) q3_k_ref_block(x + i * kQKK, out + i * 110);
}
void quantize_q5_k(const float* x, std::uint8_t* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n / kQKK; ++i) q5_k_ref_block(x + i * kQKK, out + i * 176);
}
void quantize_iq4_nl(const float* x, std::uint8_t* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n / 32; ++i) iq4_nl_block(x + i * 32, out + i * 18, nullptr);
}
void quantize_iq4_xs(const float* x, std::uint8_t* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n / kQKK; ++i) iq4_xs_block(x + i * kQKK, out + i * 136, nullptr);
}

bool uses_importance(Type t) {
  switch (t) {
    case Type::q2_k: case Type::q3_k: case Type::q4_k: case Type::q5_k: case Type::q6_k: case Type::iq4_nl: case Type::iq4_xs:
    case Type::q7_zt: case Type::q4_tg:
      return true;
    default:
      return false;
  }
}

Status quantize_rows(Type t, const float* x, std::uint8_t* out, std::int64_t nrow, std::int64_t n_per_row, const float* imatrix) {
  if (!imatrix || !uses_importance(t)) {
    // the reference quantizes whole rows the same way with or without a matrix for every other type
    for (std::int64_t r = 0; r < nrow; ++r) {
      const Status s = quantize_row(t, x + r * n_per_row, out + static_cast<std::uint64_t>(r) * row_bytes(static_cast<std::uint32_t>(t), static_cast<std::uint64_t>(n_per_row)), n_per_row);
      if (!s) return s;
    }
    return Status::good();
  }
  const TypeInfo ti = type_info(t);
  if (n_per_row <= 0 || n_per_row % ti.block_elements != 0)
    return Status::fail(Code::invalid_argument, "row of " + std::to_string(n_per_row) + " elements is not a whole number of " + ti.name + " blocks");
  const std::uint64_t rb = row_bytes(static_cast<std::uint32_t>(t), static_cast<std::uint64_t>(n_per_row));
  for (std::int64_t r = 0; r < nrow; ++r) {
    const float* xr = x + r * n_per_row;
    std::uint8_t* o = out + static_cast<std::uint64_t>(r) * rb;
    switch (t) {
      case Type::q2_k: for (std::int64_t i = 0; i < n_per_row / kQKK; ++i) q2_k_impl_block(xr + i * kQKK, o + i * 84, imatrix + i * kQKK); break;
      case Type::q3_k: for (std::int64_t i = 0; i < n_per_row / kQKK; ++i) q3_k_impl_block(xr + i * kQKK, o + i * 110, imatrix + i * kQKK); break;
      case Type::q4_k: for (std::int64_t i = 0; i < n_per_row / kQKK; ++i) q4_k_impl_block(xr + i * kQKK, o + i * 144, imatrix + i * kQKK); break;
      case Type::q5_k: for (std::int64_t i = 0; i < n_per_row / kQKK; ++i) q5_k_impl_block(xr + i * kQKK, o + i * 176, imatrix + i * kQKK); break;
      case Type::q6_k: for (std::int64_t i = 0; i < n_per_row / kQKK; ++i) q6_k_impl_block(xr + i * kQKK, o + i * 210, imatrix + i * kQKK); break;
      case Type::iq4_nl: for (std::int64_t i = 0; i < n_per_row / 32; ++i) iq4_nl_block(xr + i * 32, o + i * 18, imatrix + i * 32); break;
      case Type::iq4_xs: for (std::int64_t i = 0; i < n_per_row / kQKK; ++i) iq4_xs_block(xr + i * kQKK, o + i * 136, imatrix + i * kQKK); break;
      case Type::q7_zt: for (std::int64_t i = 0; i < n_per_row / 256; ++i) q7_zt_block(xr + i * 256, o + i * 241, imatrix + i * 256); break;
      case Type::q4_tg: {  // weights sqrt(importance): the run-E choice, after urm_ZT's over-fit with the full importance
        float tw[256];
        for (std::int64_t i = 0; i < n_per_row / 256; ++i) {
          for (int k = 0; k < 256; ++k) tw[k] = std::sqrt(std::max(0.0f, imatrix[i * 256 + k]));
          q4_tg_block(xr + i * 256, o + i * 144, tw);
        }
        break;
      }
      default: return Status::fail(Code::internal, "weighted quantize dispatch");
    }
  }
  return Status::good();
}

}  // namespace pelias::quant
