// provenance: original (expected sizes from masterwork 5.3, READ-SRC)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "pelias_test.h"
#include "quant/block_types.h"

using namespace pelias::quant;

TEST(block_types_sizes_match_the_reference_table) {
  struct Row { std::uint32_t id; const char* name; std::uint32_t elems; std::uint32_t bytes; };
  const Row rows[] = {{0, "F32", 1, 4}, {1, "F16", 1, 2}, {2, "Q4_0", 32, 18}, {3, "Q4_1", 32, 20}, {6, "Q5_0", 32, 22}, {7, "Q5_1", 32, 24}, {8, "Q8_0", 32, 34},
                      {9, "Q8_1", 32, 36}, {10, "Q2_K", 256, 84}, {11, "Q3_K", 256, 110}, {12, "Q4_K", 256, 144}, {13, "Q5_K", 256, 176}, {14, "Q6_K", 256, 210},
                      {15, "Q8_K", 256, 292}, {16, "IQ2_XXS", 256, 66}, {17, "IQ2_XS", 256, 74}, {18, "IQ3_XXS", 256, 98}, {19, "IQ1_S", 256, 50}, {20, "IQ4_NL", 32, 18},
                      {21, "IQ3_S", 256, 110}, {22, "IQ2_S", 256, 82}, {23, "IQ4_XS", 256, 136}, {29, "IQ1_M", 256, 56}, {30, "BF16", 1, 2}, {34, "TQ1_0", 256, 54},
                      {35, "TQ2_0", 256, 66}, {39, "MXFP4", 32, 17}};
  for (const Row& r : rows) {
    const TypeInfo t = pelias::quant::type_info(r.id);
    CHECK(t.known);
    CHECK_EQ(std::string(t.name), std::string(r.name));
    CHECK_EQ(t.block_elements, r.elems);
    CHECK_EQ(t.block_bytes, r.bytes);
  }
  CHECK(!pelias::quant::type_info(4).known);   // removed ids
  CHECK(!pelias::quant::type_info(31).known);
  CHECK(!pelias::quant::type_info(43).known);
  CHECK_EQ(row_bytes(12, 2048), 2048u / 256 * 144);
  CHECK_EQ(row_bytes(12, 100), 0u);  // not a whole number of blocks
  CHECK(bits_per_weight(12) > 4.49 && bits_per_weight(12) < 4.51);
  CHECK(bits_per_weight(14) > 6.56 && bits_per_weight(14) < 6.57);
}
