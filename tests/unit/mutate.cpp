// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// mutate.cpp - the corrupt-input gate (PG2) for the GGUF and safetensors readers: takes a valid file, applies
// --count random mutations (byte flips, truncations, insertions, count and offset edits) one at a time, and checks
// that the reader either accepts the file (a mutation can be harmless) or refuses it with a Status, never crashes
// and never claims a tensor outside the file. Deterministic from --seed. Exit 1 on any accepted file whose tensors
// are out of bounds.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "io/gguf_reader.h"
#include "io/safetensors.h"
#include "os/file_map.h"
#include "unit/gguf_builder.h"

using namespace pelias;

namespace {
std::uint64_t rng_state = 0x9E3779B97F4A7C15ull;
std::uint64_t next_u64() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}
std::size_t pick(std::size_t n) { return n ? static_cast<std::size_t>(next_u64() % n) : 0; }

std::vector<std::uint8_t> mutate(const std::vector<std::uint8_t>& src, std::string& what) {
  std::vector<std::uint8_t> v = src;
  switch (pick(7)) {
    case 0: { const std::size_t i = pick(v.size()); v[i] ^= static_cast<std::uint8_t>(1u << pick(8)); what = "bit flip"; break; }
    case 1: { const std::size_t i = pick(v.size()); v[i] = static_cast<std::uint8_t>(next_u64()); what = "byte set"; break; }
    case 2: { v.resize(pick(v.size())); what = "truncate"; break; }
    case 3: { const std::size_t i = pick(v.size()); v.insert(v.begin() + static_cast<std::ptrdiff_t>(i), static_cast<std::uint8_t>(next_u64())); what = "insert"; break; }
    case 4: { const std::size_t i = pick(v.size()); v.erase(v.begin() + static_cast<std::ptrdiff_t>(i)); what = "delete"; break; }
    case 5: {  // overwrite 8 bytes in the first 4 KiB (counts, lengths, offsets live there) with an extreme value
      const std::size_t i = pick(v.size() < 4096 ? v.size() : 4096);
      const std::uint64_t vals[] = {0, 1, 0x7FFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull, 0x80000000ull, 4096, 0x100000000ull};
      const std::uint64_t x = vals[pick(7)];
      for (int k = 0; k < 8 && i + static_cast<std::size_t>(k) < v.size(); ++k) v[i + static_cast<std::size_t>(k)] = static_cast<std::uint8_t>(x >> (8 * k));
      what = "extreme value";
      break;
    }
    default: {  // a burst of random bytes
      const std::size_t i = pick(v.size()), n = 1 + pick(64);
      for (std::size_t k = 0; k < n && i + k < v.size(); ++k) v[i + k] = static_cast<std::uint8_t>(next_u64());
      what = "burst";
    }
  }
  return v;
}

bool tensors_in_bounds(const io::GgufReader& r, std::uint64_t size) {
  for (const auto& t : r.tensors())
    if (t.file_offset > size || t.bytes > size - t.file_offset) return false;
  return r.data_start() + r.data_bytes() <= size;
}
}  // namespace

int main(int argc, char** argv) {
  int count = 1500;
  std::string data = "tests/data";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--count") count = std::atoi(argv[i + 1]);
    if (std::string(argv[i]) == "--data") data = argv[i + 1];
    if (std::string(argv[i]) == "--seed") rng_state = std::strtoull(argv[i + 1], nullptr, 10);
  }
  ptest::GgufBuilder b;
  b.kv_string("general.architecture", "qwen35moe").kv_u32("qwen35moe.block_count", 3).kv_string_array("tokenizer.ggml.tokens", {"a", "bb", "ccc", "dddd"})
      .kv_i32_array("qwen35moe.rope.dimension_sections", {16, 24, 24, 0}).kv_f32("x.f", 2.5f).kv_bool("x.b", true);
  for (int l = 0; l < 3; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    b.tensor(p + "attn_norm.weight", {256}, 0).tensor(p + "attn_q.weight", {256, 64}, 8).tensor(p + "ffn_gate_exps.weight", {256, 8, 4}, 12).tensor(p + "ffn_down_exps.weight", {256, 8, 4}, 14);
  }
  b.tensor("token_embd.weight", {256, 32}, 12).tensor("output.weight", {256, 32}, 14);
  const std::vector<std::uint8_t> good = b.build();
  const std::string st_h = R"({"__metadata__":{"format":"pt"},"a.weight":{"dtype":"BF16","shape":[4,8],"data_offsets":[0,64]},"b.bias":{"dtype":"F32","shape":[3],"data_offsets":[64,76]},"c":{"dtype":"I64","shape":[2,2],"data_offsets":[76,108]}})";
  std::vector<std::uint8_t> st_good;
  {
    const std::uint64_t n = st_h.size();
    const auto* p = reinterpret_cast<const std::uint8_t*>(&n);
    st_good.insert(st_good.end(), p, p + 8);
    st_good.insert(st_good.end(), st_h.begin(), st_h.end());
    st_good.resize(st_good.size() + 108, 3);
  }
  int accepted = 0, refused = 0, bad = 0;
  int by_code[16] = {};
  os::FileMap m;
  io::GgufReader r;
  io::SafetensorsReader sr;
  // sanity: the unmutated files parse
  if (!ptest::write_and_map(data + "/mut.gguf", good, m) || !r.parse(m).ok() || !tensors_in_bounds(r, m.size())) { std::puts("the unmutated GGUF does not parse"); return 1; }
  if (!ptest::write_and_map(data + "/mut.safetensors", st_good, m) || !sr.parse(m).ok()) { std::puts("the unmutated safetensors does not parse"); return 1; }
  for (int i = 0; i < count; ++i) {
    std::string what;
    const bool gguf = (i % 3) != 2;  // two thirds GGUF, one third safetensors
    const std::vector<std::uint8_t> v = mutate(gguf ? good : st_good, what);
    if (!ptest::write_and_map(data + (gguf ? "/mut.gguf" : "/mut.safetensors"), v, m)) { std::puts("cannot write the mutated file"); return 1; }
    Status s = gguf ? r.parse(m) : sr.parse(m);
    if (s.ok()) {
      ++accepted;
      bool in = true;
      if (gguf) in = tensors_in_bounds(r, m.size());
      else for (const auto& t : sr.tensors()) if (t.file_offset > m.size() || t.bytes() > m.size() - t.file_offset) in = false;
      if (!in) { ++bad; std::printf("  ACCEPTED OUT-OF-BOUNDS after %s (mutation %d)\n", what.c_str(), i); }
    } else {
      ++refused;
      ++by_code[static_cast<int>(s.code) & 15];
    }
  }
  std::printf("mutations %d: refused %d (corrupt %d, unsupported %d, other %d), accepted %d (all in bounds: %s)\n", count, refused, by_code[static_cast<int>(Code::corrupt)],
              by_code[static_cast<int>(Code::unsupported)], refused - by_code[static_cast<int>(Code::corrupt)] - by_code[static_cast<int>(Code::unsupported)], accepted, bad ? "NO" : "yes");
  return bad ? 1 : 0;
}
