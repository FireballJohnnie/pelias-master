// provenance: original (sandbox check). Build: g++ -std=c++20 -O2 -Isrc -Iinclude tools/experiments/q8z_real_check.cpp -Wl,--start-group build/host/*.a -Wl,--end-group -lpthread; run with the blk.20 attn_qkv bf16 slice
// Copyright (c) 2026 John Lipka and the Pelias contributors
// Q8_Z vs Q8_0 on the real blk.20 attn_qkv slice (bf16, 2048 x 8192), every row, rotated per 32 and plain
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include "quant/quant_cpu.h"
namespace q = pelias::quant;
int main(int, char** v) {
  std::ifstream f(v[1], std::ios::binary);
  std::vector<std::uint16_t> raw(2048ull * 8192);
  f.read(reinterpret_cast<char*>(raw.data()), raw.size() * 2);
  for (int rot = 1; rot >= 0; --rot) {
    double ez = 0, e0 = 0, ex = 0;
    std::vector<float> row(2048), bz(2048), b0(2048);
    std::vector<std::uint8_t> z(q::row_bytes(200, 2048)), z0(q::row_bytes(8, 2048));
    for (int r = 0; r < 8192; ++r) {
      for (int i = 0; i < 2048; ++i) row[i] = q::bf16_to_fp32(raw[size_t(r) * 2048 + i]);
      if (rot) q::hadamard32(row.data(), 2048);
      q::quantize_row(q::Type::q8_z, row.data(), z.data(), 2048); q::dequantize_row(q::Type::q8_z, z.data(), bz.data(), 2048);
      q::quantize_row(q::Type::q8_0, row.data(), z0.data(), 2048); q::dequantize_row(q::Type::q8_0, z0.data(), b0.data(), 2048);
      for (int i = 0; i < 2048; ++i) { ez += std::pow(bz[i] - row[i], 2); e0 += std::pow(b0[i] - row[i], 2); ex += std::pow(row[i], 2); }
    }
    std::printf("%s: Q8_Z (8.25 bpw) %.6f   Q8_0 (8.5 bpw) %.6f   ratio %.3f\n", rot ? "rotated" : "plain  ", ez / ex, e0 / ex, ez / e0);
  }
}
