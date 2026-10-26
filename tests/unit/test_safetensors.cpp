// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <cstring>
#include <string>
#include <vector>

#include "io/safetensors.h"
#include "os/file_map.h"
#include "pelias_test.h"
#include "unit/gguf_builder.h"

using namespace pelias;
using namespace pelias::io;

namespace {
std::vector<std::uint8_t> st_file(const std::string& header, std::uint64_t data_bytes) {
  std::vector<std::uint8_t> v;
  const std::uint64_t n = header.size();
  const auto* p = reinterpret_cast<const std::uint8_t*>(&n);
  v.insert(v.end(), p, p + 8);
  v.insert(v.end(), header.begin(), header.end());
  v.resize(v.size() + static_cast<std::size_t>(data_bytes), 7);
  return v;
}
}  // namespace

TEST(safetensors_reads_a_shard_header) {
  const std::string h = R"({"__metadata__":{"format":"pt"},"a.weight":{"dtype":"BF16","shape":[4,8],"data_offsets":[0,64]},"b.bias":{"dtype":"F32","shape":[3],"data_offsets":[64,76]}})";
  os::FileMap m;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/t.safetensors", st_file(h, 76), m));
  SafetensorsReader r;
  const Status s = r.parse(m);
  CHECK(s.ok());
  if (!s) std::printf("      %s\n", s.text().c_str());
  CHECK_EQ(r.tensors().size(), 2u);
  CHECK_EQ(r.tensors()[0].name, std::string("a.weight"));
  CHECK_EQ(r.tensors()[0].bytes(), 64u);
  CHECK_EQ(r.tensors()[1].file_offset, 8 + h.size() + 64);
  CHECK(r.metadata().get("format") && r.metadata().get("format")->as_string() == "pt");
}

TEST(safetensors_refuses_damage) {
  os::FileMap m;
  SafetensorsReader r;
  // header length past the end
  std::vector<std::uint8_t> v(8, 0);
  v[0] = 200;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.safetensors", v, m));
  CHECK(r.parse(m).code == Code::corrupt);
  // shape and offsets disagree
  std::string h = R"({"a":{"dtype":"BF16","shape":[4,8],"data_offsets":[0,60]}})";
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.safetensors", st_file(h, 60), m));
  Status s = r.parse(m);
  CHECK(s.code == Code::corrupt);
  CHECK_EQ(s.name, std::string("a"));
  // a gap between tensors
  h = R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}})";
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.safetensors", st_file(h, 24), m));
  s = r.parse(m);
  CHECK(s.code == Code::corrupt);
  CHECK_EQ(s.name, std::string("b"));
  // unknown dtype
  h = R"({"a":{"dtype":"X9","shape":[2],"data_offsets":[0,8]}})";
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.safetensors", st_file(h, 8), m));
  CHECK(r.parse(m).code == Code::unsupported);
  // data runs past the file
  h = R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})";
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.safetensors", st_file(h, 8), m));
  CHECK(r.parse(m).code == Code::corrupt);
  // malformed JSON names the byte
  h = R"({"a":{"dtype":"F32",)";
  CHECK(ptest::write_and_map(ptest::data_dir() + "/bad.safetensors", st_file(h, 0), m));
  s = r.parse(m);
  CHECK(s.code == Code::corrupt);
  CHECK(s.offset >= 8);
}
