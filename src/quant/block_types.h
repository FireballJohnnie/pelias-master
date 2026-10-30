// provenance: original (type ids and block sizes READ-SRC from llama.cpp ggml/include/ggml.h and ggml/src/ggml-common.h at
//   commit 9bc6578a574d1dda88610ad8532247d74779bd9e, the size static_asserts; masterwork 5.3)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// block_types.h - every GGUF tensor type: its numeric id in the file, the number of elements per block and the bytes per
// block. Row bytes = ne0 / block_elements * block_bytes. Types the product does not compute on yet are still listed, so
// a file can be inspected and sized in full; whether a type is *supported* for compute is a separate question (5.3).
#pragma once

#include <cstdint>

namespace pelias::quant {

enum class Type : std::uint32_t {
  f32 = 0, f16 = 1, q4_0 = 2, q4_1 = 3, q5_0 = 6, q5_1 = 7, q8_0 = 8, q8_1 = 9,
  q2_k = 10, q3_k = 11, q4_k = 12, q5_k = 13, q6_k = 14, q8_k = 15,
  iq2_xxs = 16, iq2_xs = 17, iq3_xxs = 18, iq1_s = 19, iq4_nl = 20, iq3_s = 21, iq2_s = 22, iq4_xs = 23,
  i8 = 24, i16 = 25, i32 = 26, i64 = 27, f64 = 28, iq1_m = 29, bf16 = 30,
  tq1_0 = 34, tq2_0 = 35, i2_s = 36, mxfp4 = 39, nvfp4 = 40, q1_0 = 41, q2_0 = 42,
  // Pelias-only ids, far above the reference library's range so it refuses such a file instead of misreading it
  q8_z = 200,
  q7_z = 201,
  q4_tg = 203,
  q4_tg2 = 204,  // Q4_TG2: Q4_TG's layout with the v2 grids (an exact 0 in two of them; the author's 0-weight idea) and full importance  // Q4_TG: 4-bit values with a 2-bit grid tag per 8 values (author's bit-tag matrix, C4; run E 2026-10-28)
  q7_zt = 202,  // Q7_ZT: 7-bit base + the 8th bit gated on 2 of 8 32-blocks per 256 (game plan, Q7_ZH and the binary gate)  // Q7_Z: Q8_Z's scales with tight 7-bit values (game plan, 7pH ladder, Q7_P rung, 2026-10-28)  // Q8_Z: the author's "zip code" scales with 8-bit values (game plan, Q7_PT review, 2026-10-28)
};

struct TypeInfo {
  const char* name;
  std::uint32_t block_elements;  // elements per block
  std::uint32_t block_bytes;     // bytes per block
  bool known;                    // false for ids the file format reserves or removed
};

// Looks up a type id as it appears in a GGUF file. Unknown ids return {"?", 0, 0, false}.
TypeInfo type_info(std::uint32_t id);
inline TypeInfo type_info(Type t) { return type_info(static_cast<std::uint32_t>(t)); }

// Bytes of one row of ne0 elements, or 0 if the type is unknown or ne0 is not a whole number of blocks.
std::uint64_t row_bytes(std::uint32_t id, std::uint64_t ne0);

// Bits per weight as a number, for display.
double bits_per_weight(std::uint32_t id);

}  // namespace pelias::quant
