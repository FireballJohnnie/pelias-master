// provenance: re-expressed (not copied) from ggml ggml/src/ggml-quants.c at llama.cpp commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a (MIT; THIRD_PARTY_NOTICES.md), read as a specification
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// quant_internal.h - the scale-search helpers the K-quant quantizers share (quant_cpu.cpp defines them;
// quant_weighted.cpp uses them too). Internal to the quant module.
#pragma once

#include <cstdint>

namespace pelias::quant::detail {

void put_u16(std::uint8_t* p, std::uint16_t v);
std::uint16_t get_u16(const std::uint8_t* p);
// asymmetric (scale, min) search over nstep + 1 candidate scales; weights required
float make_qkx2_quants(int n, int nmax, const float* x, const float* weights, std::uint8_t* L, float* the_min, std::uint8_t* Laux,
                       float rmin, float rdelta, int nstep, bool use_mad);
// symmetric scale search; levels stored offset by nmax; qw (importance weights) may be null
float make_qx_quants(int n, int nmax, const float* x, std::int8_t* L, int rmse_type, const float* qw);
// the 12-byte scale and min field of Q4_K / Q5_K
void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m);

}  // namespace pelias::quant::detail
