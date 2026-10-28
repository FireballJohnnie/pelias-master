// provenance: original (sizes READ-SRC, see block_types.h)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "quant/block_types.h"

namespace pelias::quant {

TypeInfo type_info(std::uint32_t id) {
  // {name, elements per block, bytes per block}. Sizes: ggml_half = 2 bytes; QK_K = 256; K_SCALE_SIZE = 12;
  // IQ3S_N_SCALE = QK_K/64 = 4.
  switch (id) {
    case 0: return {"F32", 1, 4, true};
    case 1: return {"F16", 1, 2, true};
    case 2: return {"Q4_0", 32, 18, true};      // 2 + 32/2
    case 3: return {"Q4_1", 32, 20, true};      // 2*2 + 32/2
    case 6: return {"Q5_0", 32, 22, true};      // 2 + 4 + 32/2
    case 7: return {"Q5_1", 32, 24, true};      // 2*2 + 4 + 32/2
    case 8: return {"Q8_0", 32, 34, true};      // 2 + 32
    case 9: return {"Q8_1", 32, 36, true};      // 2*2 + 32
    case 10: return {"Q2_K", 256, 84, true};    // 2*2 + 256/16 + 256/4
    case 11: return {"Q3_K", 256, 110, true};   // 2 + 256/4 + 256/8 + 12
    case 12: return {"Q4_K", 256, 144, true};   // 2*2 + 12 + 256/2
    case 13: return {"Q5_K", 256, 176, true};   // 2*2 + 12 + 256/2 + 256/8
    case 14: return {"Q6_K", 256, 210, true};   // 2 + 256/16 + 3*256/4
    case 15: return {"Q8_K", 256, 292, true};   // 4 + 256 + 256/16*2
    case 16: return {"IQ2_XXS", 256, 66, true}; // 2 + 256/8*2
    case 17: return {"IQ2_XS", 256, 74, true};  // 2 + 256/8*2 + 256/32
    case 18: return {"IQ3_XXS", 256, 98, true}; // 2 + 3*256/8
    case 19: return {"IQ1_S", 256, 50, true};   // 2 + 256/8 + 256/16
    case 20: return {"IQ4_NL", 32, 18, true};   // 2 + 32/2
    case 21: return {"IQ3_S", 256, 110, true};  // 2 + 13*256/32 + 4
    case 22: return {"IQ2_S", 256, 82, true};   // 2 + 256/4 + 256/16
    case 23: return {"IQ4_XS", 256, 136, true}; // 2 + 2 + 256/64 + 256/2
    case 24: return {"I8", 1, 1, true};
    case 25: return {"I16", 1, 2, true};
    case 26: return {"I32", 1, 4, true};
    case 27: return {"I64", 1, 8, true};
    case 28: return {"F64", 1, 8, true};
    case 29: return {"IQ1_M", 256, 56, true};   // 256/8 + 256/16 + 256/32
    case 30: return {"BF16", 1, 2, true};
    case 34: return {"TQ1_0", 256, 54, true};   // 2 + 256/64 + (256 - 4*256/64)/5
    case 35: return {"TQ2_0", 256, 66, true};   // 2 + 256/4
    case 39: return {"MXFP4", 32, 17, true};    // 1 + 32/2
    case 40: return {"NVFP4", 64, 36, true};    // 64/16 + 64/2
    case 41: return {"Q1_0", 128, 18, true};    // 2 + 128/8
    case 42: return {"Q2_0", 64, 18, true};     // 2 + 64/4
    case 200: return {"Q8_Z", 256, 264, true};
    case 201: return {"Q7_Z", 256, 232, true};  // Pelias-only: 2 (fp16 d) + 6 (levels) + 224 (7-bit values)  // Pelias-only: 2 (fp16 d) + 6 (eight 6-bit levels) + 256 (int8)
    default: return {"?", 0, 0, false};
  }
}

std::uint64_t row_bytes(std::uint32_t id, std::uint64_t ne0) {
  const TypeInfo t = type_info(id);
  if (!t.known || t.block_elements == 0 || ne0 % t.block_elements != 0) return 0;
  return ne0 / t.block_elements * t.block_bytes;
}

double bits_per_weight(std::uint32_t id) {
  const TypeInfo t = type_info(id);
  if (!t.known || t.block_elements == 0) return 0.0;
  return static_cast<double>(t.block_bytes) * 8.0 / static_cast<double>(t.block_elements);
}

}  // namespace pelias::quant
