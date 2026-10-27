// provenance: original (test-only oracle harness; calls the reference library built unchanged, never linked into the
//   product; masterwork 3.3, gate P-E6 on synthetic rows)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// quant_oracle - compares Pelias's quantizers with the reference library's, byte for byte:
//   1. fp32 -> fp16 and fp32 -> bf16 on all 2^32 bit patterns; fp16 -> fp32 and bf16 -> fp32 on all 2^16
//   2. Q8_0, Q4_K, Q6_K, F16, BF16 through the reference's quantize-chunk entry (no importance matrix, as a stock
//      quantization uses), and the activation quantizers Q8_1 and Q8_K, on many generated rows of several shapes
//   3. dequantization of every weight type Pelias reads, on random blocks
// Exit 0 only when every byte (and every float bit pattern) matches. Usage: quant_oracle [--blocks N] [--seed S]
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-quants.h"
#include "io/sha256.h"
#include "quant/quant_cpu.h"
#include "unit/quant_golden.h"

namespace q = pelias::quant;

namespace {

int g_fail = 0;

void report(const char* what, std::uint64_t checked, std::uint64_t bad, const std::string& first) {
  std::printf("%-44s %12llu checked  %llu differ%s%s\n", what, static_cast<unsigned long long>(checked), static_cast<unsigned long long>(bad),
              bad ? "  first: " : "", first.c_str());
  if (bad) ++g_fail;
}

std::uint32_t bits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
float from_bits(std::uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

void check_scalars() {
  std::uint64_t bad16 = 0, badb = 0;
  std::string f16, fb;
  for (std::uint64_t u = 0; u <= 0xFFFFFFFFull; ++u) {
    const float f = from_bits(static_cast<std::uint32_t>(u));
    const std::uint16_t a = q::fp32_to_fp16(f);
    const std::uint16_t b = ggml_fp32_to_fp16(f);
    if (a != b && bad16++ == 0) f16 = "bits " + std::to_string(u) + " ours " + std::to_string(a) + " ref " + std::to_string(b);
    const std::uint16_t c = q::fp32_to_bf16(f);
    const std::uint16_t d = ggml_fp32_to_bf16(f).bits;
    if (c != d && badb++ == 0) fb = "bits " + std::to_string(u) + " ours " + std::to_string(c) + " ref " + std::to_string(d);
  }
  report("fp32 -> fp16, every bit pattern", 1ull << 32, bad16, f16);
  report("fp32 -> bf16, every bit pattern", 1ull << 32, badb, fb);
  std::uint64_t bad_h = 0, bad_bh = 0;
  std::string fh, fbh;
  for (std::uint32_t h = 0; h <= 0xFFFF; ++h) {
    const auto hh = static_cast<std::uint16_t>(h);
    if (bits(q::fp16_to_fp32(hh)) != bits(ggml_fp16_to_fp32(hh)) && bad_h++ == 0) fh = "half " + std::to_string(h);
    ggml_bf16_t bb;
    bb.bits = hh;
    if (bits(q::bf16_to_fp32(hh)) != bits(ggml_bf16_to_fp32(bb)) && bad_bh++ == 0) fbh = "bf16 " + std::to_string(h);
  }
  report("fp16 -> fp32, every half", 65536, bad_h, fh);
  report("bf16 -> fp32, every bf16", 65536, bad_bh, fbh);
}

// ---- row generators: each fills `n` floats; all values finite and |x| well inside the rounding helper's range
using Gen = void (*)(std::mt19937_64&, float*, std::size_t);

float to_bf16_value(float f) { return q::bf16_to_fp32(q::fp32_to_bf16(f)); }

void gen_weights(std::mt19937_64& r, float* x, std::size_t n) {  // weight-like, rounded to bf16 as real inputs are
  std::normal_distribution<float> nd(0.0f, 0.02f);
  for (std::size_t i = 0; i < n; ++i) x[i] = to_bf16_value(nd(r));
}
void gen_normal(std::mt19937_64& r, float* x, std::size_t n) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (std::size_t i = 0; i < n; ++i) x[i] = nd(r);
}
void gen_heavy(std::mt19937_64& r, float* x, std::size_t n) {  // heavy tails and outliers
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> u(0.05f, 1.0f);
  for (std::size_t i = 0; i < n; ++i) x[i] = nd(r) / u(r) * 0.01f;
  for (std::size_t i = 0; i < n; i += 97) x[i] *= 300.0f;
}
void gen_positive(std::mt19937_64& r, float* x, std::size_t n) {  // one-signed rows (norm-like)
  std::uniform_real_distribution<float> u(0.5f, 2.0f);
  for (std::size_t i = 0; i < n; ++i) x[i] = u(r);
}
void gen_negative(std::mt19937_64& r, float* x, std::size_t n) {
  std::uniform_real_distribution<float> u(-3.0f, -0.001f);
  for (std::size_t i = 0; i < n; ++i) x[i] = u(r);
}
void gen_ties(std::mt19937_64& r, float* x, std::size_t n) {  // small integers and halves: rounding ties everywhere
  std::uniform_int_distribution<int> u(-64, 64);
  for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<float>(u(r)) * 0.5f;
}
void gen_sparse(std::mt19937_64& r, float* x, std::size_t n) {  // mostly zero, some blocks entirely zero, spikes
  std::uniform_int_distribution<int> u(0, 99);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (std::size_t i = 0; i < n; ++i) x[i] = u(r) < 3 ? nd(r) : 0.0f;
  for (std::size_t i = 0; i + 512 <= n; i += 2048) std::fill(x + i, x + i + 512, 0.0f);
}
void gen_constant(std::mt19937_64& r, float* x, std::size_t n) {  // constant blocks, both signs
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (std::size_t i = 0; i < n; i += 256) {
    const float c = nd(r);
    for (std::size_t k = i; k < i + 256 && k < n; ++k) x[k] = c;
  }
}
void gen_tiny(std::mt19937_64& r, float* x, std::size_t n) {  // near and below fp16's range, including subnormals
  std::uniform_real_distribution<float> e(-44.0f, -10.0f);
  std::uniform_int_distribution<int> s(0, 1);
  for (std::size_t i = 0; i < n; ++i) x[i] = (s(r) ? 1.0f : -1.0f) * std::exp2(e(r));
}
void gen_large(std::mt19937_64& r, float* x, std::size_t n) {  // large but finite after quantization
  std::uniform_real_distribution<float> u(-30000.0f, 30000.0f);
  for (std::size_t i = 0; i < n; ++i) x[i] = u(r);
}
void gen_wide(std::mt19937_64& r, float* x, std::size_t n) {  // magnitudes spread over many octaves within a block
  std::uniform_real_distribution<float> e(-20.0f, 10.0f);
  std::uniform_int_distribution<int> s(0, 1);
  for (std::size_t i = 0; i < n; ++i) x[i] = (s(r) ? 1.0f : -1.0f) * std::exp2(e(r));
}

struct Case {
  const char* name;
  Gen gen;
};
const Case kCases[] = {{"weights (bf16 values)", gen_weights}, {"normal", gen_normal},     {"heavy tails", gen_heavy},
                       {"positive", gen_positive},             {"negative", gen_negative}, {"ties", gen_ties},
                       {"sparse / zero blocks", gen_sparse},   {"constant blocks", gen_constant},
                       {"tiny / subnormal", gen_tiny},         {"large", gen_large},       {"wide range", gen_wide}};

std::string first_diff(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b, std::size_t block_bytes) {
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) return "byte " + std::to_string(i) + " (block " + std::to_string(i / block_bytes) + ", offset " + std::to_string(i % block_bytes) + ")";
  return {};
}

void check_quantizers(std::uint64_t blocks256, std::uint64_t seed) {
  struct T {
    q::Type ours;
    ggml_type ref;
    const char* name;
  };
  const T types[] = {{q::Type::q8_0, GGML_TYPE_Q8_0, "Q8_0"}, {q::Type::q4_k, GGML_TYPE_Q4_K, "Q4_K"}, {q::Type::q6_k, GGML_TYPE_Q6_K, "Q6_K"},
                     {q::Type::f16, GGML_TYPE_F16, "F16"},    {q::Type::bf16, GGML_TYPE_BF16, "BF16"}};
  const std::int64_t shapes[][2] = {{1, 256}, {3, 512}, {7, 2048}, {2, 4096}};  // rows x elements per row
  for (const T& t : types) {
    for (const Case& c : kCases) {
      std::mt19937_64 rng(seed);
      std::uint64_t checked = 0, bad = 0;
      std::string first;
      std::uint64_t done = 0;
      int shape = 0;
      while (done < blocks256) {
        const std::int64_t rows = shapes[shape % 4][0], per_row = shapes[shape % 4][1];
        ++shape;
        const std::size_t n = static_cast<std::size_t>(rows * per_row);
        std::vector<float> x(n);
        c.gen(rng, x.data(), n);
        const std::size_t bytes = ggml_row_size(t.ref, per_row) * static_cast<std::size_t>(rows);
        std::vector<std::uint8_t> a(bytes, 0), b(bytes, 0);
        for (std::int64_t row = 0; row < rows; ++row) {
          const pelias::Status s = q::quantize_row(t.ours, x.data() + row * per_row, a.data() + static_cast<std::size_t>(row) * ggml_row_size(t.ref, per_row), per_row);
          if (!s) {
            std::printf("quantize_row failed: %s\n", s.text().c_str());
            ++g_fail;
            return;
          }
        }
        ggml_quantize_chunk(t.ref, x.data(), b.data(), 0, rows, per_row, nullptr);
        const std::size_t block_bytes = ggml_type_size(t.ref);
        checked += bytes / block_bytes;
        if (a != b) {
          for (std::size_t k = 0; k < bytes; k += block_bytes)
            if (std::memcmp(a.data() + k, b.data() + k, block_bytes) != 0) ++bad;
          if (first.empty()) first = std::string(c.name) + ": " + first_diff(a, b, block_bytes);
        }
        done += n / 256;
      }
      report((std::string(t.name) + " / " + c.name).c_str(), checked, bad, first);
    }
  }
  // activation quantizers: called on their reference entries directly
  for (int which = 0; which < 2; ++which) {
    const char* name = which == 0 ? "Q8_1" : "Q8_K";
    const std::size_t block_bytes = which == 0 ? 36 : 292;
    const std::size_t block_elems = which == 0 ? 32 : 256;
    for (const Case& c : kCases) {
      std::mt19937_64 rng(seed + 1);
      const std::size_t n = static_cast<std::size_t>(blocks256) * 256;
      std::vector<float> x(n);
      c.gen(rng, x.data(), n);
      const std::size_t bytes = n / block_elems * block_bytes;
      std::vector<std::uint8_t> a(bytes, 0), b(bytes, 0);  // zeroed: the reference leaves Q8_K sums of an all-zero block unwritten
      if (which == 0) {
        q::quantize_q8_1(x.data(), a.data(), static_cast<std::int64_t>(n));
        quantize_row_q8_1_ref(x.data(), reinterpret_cast<block_q8_1*>(b.data()), static_cast<std::int64_t>(n));
      } else {
        q::quantize_q8_k(x.data(), a.data(), static_cast<std::int64_t>(n));
        quantize_row_q8_K_ref(x.data(), reinterpret_cast<block_q8_K*>(b.data()), static_cast<std::int64_t>(n));
      }
      std::uint64_t bad = 0;
      for (std::size_t k = 0; k < bytes; k += block_bytes)
        if (std::memcmp(a.data() + k, b.data() + k, block_bytes) != 0) ++bad;
      report((std::string(name) + " / " + c.name).c_str(), bytes / block_bytes, bad, bad ? std::string(c.name) + ": " + first_diff(a, b, block_bytes) : std::string());
    }
  }
}

void check_dequantizers(std::uint64_t blocks, std::uint64_t seed) {
  struct T {
    q::Type ours;
    ggml_type ref;
    const char* name;
    int d_at, dmin_at;  // byte offsets of the block's fp16 scale and fp16 min scale (-1: none)
  };
  const T types[] = {{q::Type::q8_0, GGML_TYPE_Q8_0, "Q8_0", 0, -1},     {q::Type::q4_k, GGML_TYPE_Q4_K, "Q4_K", 0, 2},
                     {q::Type::q6_k, GGML_TYPE_Q6_K, "Q6_K", 208, -1},   {q::Type::q2_k, GGML_TYPE_Q2_K, "Q2_K", 80, 82},
                     {q::Type::q3_k, GGML_TYPE_Q3_K, "Q3_K", 108, -1},   {q::Type::q5_k, GGML_TYPE_Q5_K, "Q5_K", 0, 2},
                     {q::Type::iq4_xs, GGML_TYPE_IQ4_XS, "IQ4_XS", 0, -1}, {q::Type::q4_0, GGML_TYPE_Q4_0, "Q4_0", 0, -1},
                     {q::Type::q4_1, GGML_TYPE_Q4_1, "Q4_1", 0, 2},      {q::Type::q5_0, GGML_TYPE_Q5_0, "Q5_0", 0, -1},
                     {q::Type::q5_1, GGML_TYPE_Q5_1, "Q5_1", 0, 2},      {q::Type::iq4_nl, GGML_TYPE_IQ4_NL, "IQ4_NL", 0, -1},
                     {q::Type::mxfp4, GGML_TYPE_MXFP4, "MXFP4", -1, -1}, {q::Type::tq1_0, GGML_TYPE_TQ1_0, "TQ1_0", 52, -1},
                     {q::Type::tq2_0, GGML_TYPE_TQ2_0, "TQ2_0", 64, -1}, {q::Type::iq2_xxs, GGML_TYPE_IQ2_XXS, "IQ2_XXS", 0, -1},
                     {q::Type::iq2_xs, GGML_TYPE_IQ2_XS, "IQ2_XS", 0, -1}, {q::Type::iq2_s, GGML_TYPE_IQ2_S, "IQ2_S", 0, -1},
                     {q::Type::iq3_xxs, GGML_TYPE_IQ3_XXS, "IQ3_XXS", 0, -1}, {q::Type::iq3_s, GGML_TYPE_IQ3_S, "IQ3_S", 0, -1},
                     {q::Type::iq1_s, GGML_TYPE_IQ1_S, "IQ1_S", 0, -1},  {q::Type::iq1_m, GGML_TYPE_IQ1_M, "IQ1_M", -1, -1}};
  for (const T& t : types) {
    std::mt19937_64 rng(seed + 7);
    const std::size_t block_bytes = ggml_type_size(t.ref);
    const auto block_elems = static_cast<std::size_t>(ggml_blck_size(t.ref));
    std::vector<std::uint8_t> in(blocks * block_bytes);
    for (auto& v : in) v = static_cast<std::uint8_t>(rng());
    // keep each block's fp16 scales finite (a real file has no inf or NaN scales)
    std::uniform_real_distribution<float> sc(1e-6f, 0.5f);
    for (std::size_t k = 0; k < blocks; ++k) {
      std::uint8_t* blk = in.data() + k * block_bytes;
      const std::uint16_t d = q::fp32_to_fp16(sc(rng)), m = q::fp32_to_fp16(sc(rng));
      if (t.d_at >= 0) std::memcpy(blk + t.d_at, &d, 2);  // otherwise random (MXFP4 exponent byte, IQ1_M scale nibbles)
      if (t.dmin_at >= 0) std::memcpy(blk + t.dmin_at, &m, 2);
    }
    const std::size_t n = blocks * block_elems;
    std::vector<float> a(n), b(n);
    const pelias::Status s = q::dequantize_row(t.ours, in.data(), a.data(), static_cast<std::int64_t>(n));
    if (!s) {
      std::printf("dequantize_row failed: %s\n", s.text().c_str());
      ++g_fail;
      return;
    }
    ggml_get_type_traits(t.ref)->to_float(in.data(), b.data(), static_cast<std::int64_t>(n));
    std::uint64_t bad = 0;
    std::string first;
    for (std::size_t i = 0; i < n; ++i)
      if (bits(a[i]) != bits(b[i]) && bad++ == 0) first = "element " + std::to_string(i);
    report((std::string("dequantize ") + t.name).c_str(), n, bad, first);
  }
}


// The golden hashes of tests/unit/quant_golden.h: computed here from the REFERENCE library's output on the fixed input,
// and compared with Pelias's output and with the constants in the header (PENDING until filled from this run).
void check_golden() {
  const std::vector<float> x = ptest::golden_input();
  const auto n = static_cast<std::int64_t>(x.size());
  struct T {
    const char* name;
    ggml_type ref;
    q::Type ours;
  };
  const T types[] = {{"Q8_0", GGML_TYPE_Q8_0, q::Type::q8_0}, {"Q4_K", GGML_TYPE_Q4_K, q::Type::q4_k}, {"Q6_K", GGML_TYPE_Q6_K, q::Type::q6_k},
                     {"F16", GGML_TYPE_F16, q::Type::f16},    {"BF16", GGML_TYPE_BF16, q::Type::bf16}, {"Q8_1", GGML_TYPE_Q8_1, q::Type::q8_1},
                     {"Q8_K", GGML_TYPE_Q8_K, q::Type::q8_k}};
  for (const T& t : types) {
    const std::size_t bytes = ggml_row_size(t.ref, n);
    std::vector<std::uint8_t> ref(bytes, 0), ours(bytes, 0);
    if (t.ref == GGML_TYPE_Q8_1) quantize_row_q8_1_ref(x.data(), reinterpret_cast<block_q8_1*>(ref.data()), n);
    else if (t.ref == GGML_TYPE_Q8_K) quantize_row_q8_K_ref(x.data(), reinterpret_cast<block_q8_K*>(ref.data()), n);
    else ggml_quantize_chunk(t.ref, x.data(), ref.data(), 0, 1, n, nullptr);
    (void)q::quantize_row(t.ours, x.data(), ours.data(), n);
    const std::string hr = pelias::io::Sha256::hex_of(ref.data(), ref.size());
    const std::string ho = pelias::io::Sha256::hex_of(ours.data(), ours.size());
    std::string in_header = "?";
    for (const auto& g : ptest::kGoldenHashes)
      if (std::string(g.type) == t.name) in_header = g.sha256;
    std::printf("golden %-5s reference %s  ours %s  header %s\n", t.name, hr.c_str(), ho == hr ? "same" : "DIFFERENT", in_header == hr ? "same" : in_header.c_str());
    if (ho != hr) ++g_fail;
  }
  // the read-only types: the reference's dequantized floats on the fixed blocks
  for (const auto& g : ptest::kDequantGolden) {
    const std::vector<std::uint8_t> in = ptest::dequant_golden_input(g);
    const auto ref_type = static_cast<ggml_type>(g.type_id);
    const auto nd = static_cast<std::int64_t>(64 * ggml_blck_size(ref_type));
    std::vector<float> ref(static_cast<std::size_t>(nd)), ours(static_cast<std::size_t>(nd));
    ggml_get_type_traits(ref_type)->to_float(in.data(), ref.data(), nd);
    (void)q::dequantize_row(static_cast<q::Type>(g.type_id), in.data(), ours.data(), nd);
    const std::string hr = pelias::io::Sha256::hex_of(reinterpret_cast<const std::uint8_t*>(ref.data()), ref.size() * 4);
    const std::string ho = pelias::io::Sha256::hex_of(reinterpret_cast<const std::uint8_t*>(ours.data()), ours.size() * 4);
    std::printf("golden %-6s dequantized: reference %s  ours %s  header %s\n", g.type, hr.c_str(), ho == hr ? "same" : "DIFFERENT",
                std::string(g.sha256) == hr ? "same" : g.sha256);
    if (ho != hr) ++g_fail;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::uint64_t blocks = 20000, seed = 20261026;
  bool scalars = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--blocks" && i + 1 < argc) blocks = std::stoull(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
    else if (a == "--no-scalars") scalars = false;
  }
  std::printf("quant_oracle: %llu blocks of 256 per case, seed %llu\n", static_cast<unsigned long long>(blocks), static_cast<unsigned long long>(seed));
  check_golden();
  if (scalars) check_scalars();
  check_quantizers(blocks, seed);
  check_dequantizers(blocks, seed);
  std::printf("%s: %d check(s) differ\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
