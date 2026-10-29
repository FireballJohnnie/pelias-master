// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <cstring>

#include "io/gguf_reader.h"
#include "os/file_map.h"
#include "pelias_test.h"
#include "unit/gguf_builder.h"

using namespace pelias;
using namespace pelias::io;

namespace {
ptest::GgufBuilder sample() {
  ptest::GgufBuilder b;
  b.kv_string("general.architecture", "qwen35moe").kv_u32("qwen35moe.block_count", 2).kv_f32("qwen35moe.rope.freq_base", 1e7f).kv_bool("tokenizer.ggml.add_bos_token", false)
      .kv_string_array("tokenizer.ggml.tokens", {"<a>", "b", "ccc"}).kv_i32_array("qwen35moe.rope.dimension_sections", {1, 2, 3, 4});
  b.tensor("token_embd.weight", {256, 8}, 12).tensor("blk.0.attn_norm.weight", {256}, 0).tensor("blk.0.ffn_gate_exps.weight", {256, 4, 2}, 12)
      .tensor("blk.0.ffn_down_exps.weight", {256, 4, 2}, 14).tensor("output.weight", {256, 8}, 14);
  return b;
}
}  // namespace

TEST(gguf_reads_a_synthetic_file) {
  std::uint64_t ds = 0;
  const auto bytes = sample().build(&ds);
  os::FileMap m;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/synthetic.gguf", bytes, m));
  GgufReader r;
  const Status s = r.parse(m);
  CHECK(s.ok());
  if (!s) std::printf("      %s\n", s.text().c_str());
  CHECK_EQ(r.version(), 3u);
  CHECK_EQ(r.alignment(), 32u);
  CHECK_EQ(r.data_start(), ds);
  CHECK_EQ(r.kv().size(), 6u);
  CHECK_EQ(r.tensors().size(), 5u);
  CHECK_EQ(r.get_string("general.architecture"), std::string("qwen35moe"));
  CHECK_EQ(r.get_uint("qwen35moe.block_count"), 2u);
  CHECK(r.get_float("qwen35moe.rope.freq_base") == 1e7f);
  bool found = false;
  CHECK_EQ(r.get_uint("tokenizer.ggml.add_bos_token", &found), 0u);
  CHECK(found);
  const GgufValue* toks = r.find("tokenizer.ggml.tokens");
  CHECK(toks && toks->type == GgufValueType::array && toks->array_count == 3);
  CHECK_EQ(r.array_string(*toks, 2), std::string("ccc"));
  CHECK_EQ(r.array_string(*toks, 3, &found), std::string());
  CHECK(!found);
  const GgufTensor* t = r.find_tensor("blk.0.ffn_down_exps.weight");
  CHECK(t && t->type == 14 && t->n_dims == 3 && t->bytes == 256 * 4 * 2 / 256 * 210);
  CHECK_EQ(r.tensors()[0].file_offset, ds);
  CHECK_EQ(r.tensors()[1].data_offset, (256ull * 8 / 256 * 144 + 31) / 32 * 32);
  CHECK_EQ(r.data_start() + r.data_bytes(), m.size());
}

TEST(gguf_refuses_damage_with_name_and_offset) {
  const auto good = sample().build();
  os::FileMap m;
  GgufReader r;
  // 1. wrong magic
  auto b = good;
  b[1] = 'X';
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.gguf", b, m));
  CHECK(r.parse(m).code == Code::corrupt);
  // 2. version 1 is unsupported; version 9 too
  b = good;
  b[4] = 1;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.gguf", b, m));
  CHECK(r.parse(m).code == Code::unsupported);
  b[4] = 9;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.gguf", b, m));
  CHECK(r.parse(m).code == Code::unsupported);
  // 3. truncated inside the tensor table: the error names the tensor and the byte
  b = good;
  b.resize(b.size() / 2);
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.gguf", b, m));
  Status s = r.parse(m);
  CHECK(s.code == Code::corrupt);
  CHECK(s.offset != 0);
  // 4. tensor count claims more than the file holds
  b = good;
  const std::int64_t huge = 1000000000;
  std::memcpy(&b[8], &huge, 8);
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.gguf", b, m));
  s = r.parse(m);
  CHECK(s.code == Code::corrupt);
  // 5. data truncated: the last tensor runs past the end, named
  b = good;
  b.resize(b.size() - 100);
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.gguf", b, m));
  s = r.parse(m);
  CHECK(s.code == Code::corrupt);
  CHECK_EQ(s.name, std::string("output.weight"));
  CHECK(r.parse(m, false).ok());  // header-only reading still works
  // 6. a duplicate tensor name
  ptest::GgufBuilder dup = sample();
  dup.tensor("output.weight", {256}, 0);
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.gguf", dup.build(), m));
  s = r.parse(m);
  CHECK(s.code == Code::corrupt);
  CHECK_EQ(s.name, std::string("output.weight"));
  // 7. a dimension that is not a multiple of the block
  ptest::GgufBuilder odd = sample();
  odd.tensor("blk.1.x", {100}, 12);
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.gguf", odd.build(), m));
  s = r.parse(m);
  CHECK(s.code == Code::corrupt);
  CHECK_EQ(s.name, std::string("blk.1.x"));
  // 8. empty file
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.gguf", {}, m));
  CHECK(r.parse(m).code == Code::corrupt);
  // 9. alignment that is not a power of two
  ptest::GgufBuilder al = sample();
  al.alignment(48);
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.gguf", al.build(), m));
  s = r.parse(m);
  CHECK(s.code == Code::corrupt);
  CHECK_EQ(s.name, std::string("general.alignment"));
  // 10. alignment 64 is fine
  ptest::GgufBuilder al64 = sample();
  al64.alignment(64);
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.gguf", al64.build(), m));
  CHECK(r.parse(m).ok());
  CHECK_EQ(r.alignment(), 64u);
}

TEST(gguf_reads_the_real_header_when_present) {
  os::FileMap m;
  const Status o = m.open(ptest::data_dir() + "/unsloth_header_16MiB.bin");
  if (!o) {
    ptest::skip("unsloth_header_16MiB.bin not in the data directory (kept on the author's drive)");
    return;
  }
  GgufReader r;
  Status s = r.parse(m);  // the data is cut off, so the full check must refuse and name the first tensor
  CHECK(s.code == Code::corrupt);
  CHECK_EQ(s.name, std::string("output.weight"));
  s = r.parse(m, false);
  CHECK(s.ok());
  if (!s) std::printf("      %s\n", s.text().c_str());
  // ground truth read independently (python, 2026-10-26 17:54): version 3, 733 tensors, 54 kv, data start 10,990,048
  CHECK_EQ(r.version(), 3u);
  CHECK_EQ(r.tensors().size(), 733u);
  CHECK_EQ(r.kv().size(), 54u);
  CHECK_EQ(r.data_start(), 10990048u);
  CHECK_EQ(r.get_string("general.architecture"), std::string("qwen35moe"));
  CHECK_EQ(r.get_uint("qwen35moe.expert_count"), 256u);
  const GgufValue* toks = r.find("tokenizer.ggml.tokens");
  CHECK(toks && toks->array_count == 248320);
  const GgufTensor* last = r.find_tensor("blk.39.post_attention_norm.weight");
  CHECK(last && last->data_offset == 22123530752ull && last->type == 0);
  // the file is 22,134,528,992 bytes (P-M4): data start + padded data must equal it
  CHECK_EQ(r.data_start() + r.data_bytes(), 22134528992ull);
  int q4 = 0, q6 = 0, q5 = 0, q8 = 0, f32 = 0;
  for (const auto& t : r.tensors()) {
    if (t.type == 12) ++q4; else if (t.type == 14) ++q6; else if (t.type == 13) ++q5; else if (t.type == 8) ++q8; else if (t.type == 0) ++f32;
  }
  CHECK_EQ(q4, 80); CHECK_EQ(q6, 4); CHECK_EQ(q5, 37); CHECK_EQ(q8, 251); CHECK_EQ(f32, 361);
}

// os::FileMap::load (pelias generate --no-mmap): the same bytes as the map, in private memory; a missing file is refused
TEST(file_map_load_equals_the_map) {
  const std::vector<std::uint8_t> bytes = ptest::GgufBuilder().kv_string("general.architecture", "x").tensor("a", {64, 3}, 0).build();
  pelias::os::FileMap mapped;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/file_map_load.gguf", bytes, mapped));
  pelias::os::FileMap loaded;
  std::uint64_t last = 0;
  static std::uint64_t seen;
  seen = 0;
  CHECK(loaded.load(ptest::data_dir() + "/file_map_load.gguf", [](std::uint64_t d, std::uint64_t) { seen = d; }).ok());
  last = seen;
  CHECK(loaded.is_open() && loaded.size() == mapped.size() && last == mapped.size());
  CHECK(std::memcmp(loaded.data(), mapped.data(), static_cast<std::size_t>(mapped.size())) == 0);
  pelias::io::GgufReader r;
  CHECK(r.parse(loaded).ok());
  pelias::os::FileMap moved = std::move(loaded);
  CHECK(moved.size() == mapped.size() && !loaded.is_open());
  moved.close();
  CHECK(!moved.is_open());
  pelias::os::FileMap missing;
  const pelias::Status st = missing.load(ptest::data_dir() + "/no_such_file.gguf");
  CHECK(!st.ok() && st.code == pelias::Code::not_found);
}
