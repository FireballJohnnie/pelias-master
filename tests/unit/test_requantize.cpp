// provenance: original (the expected hashes MEASURED in the sandbox 2026-10-26: the reference llama-quantize at commit
//   2ca15f5, built unchanged, run on the same synthetic file with Q4_K_M)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <filesystem>
#include <string>

#include "io/gguf_compare.h"
#include "io/gguf_reader.h"
#include "io/requantize.h"
#include "io/sha256.h"
#include "os/file_map.h"
#include "pelias_test.h"
#include "unit/synth_model.h"

using namespace pelias;

namespace {
// SHA-256 of the synthetic input, and of the reference quantizer's Q4_K_M output on it (whole files)
const char* kSynthInputSha = "ea28cb7f08d0d69cad4ee6ebc057ed1d0b87bdd0b981c9f98215e164cbbd7392";
const char* kReferenceQ4KMSha = "5c87a4085a5d6ea73d35f8f5bbbea5f93a14d0c67940c79259939a24a33f37a2";

std::string file_sha(const std::string& path) {
  os::FileMap m;
  if (!m.open(path)) return "unreadable";
  return io::Sha256::hex_of(m.data(), static_cast<std::size_t>(m.size()));
}
}  // namespace

TEST(quantize_matches_the_reference_end_to_end) {
  const std::string header = ptest::data_dir() + "/unsloth_header_16MiB.bin";
  if (!std::filesystem::exists(header)) {
    ptest::skip("unsloth_header_16MiB.bin not in the data directory (kept on the author's drive)");
    return;
  }
  const std::string in_path = ptest::data_dir() + "/synth_qwen35moe.gguf";
  const std::string out_path = ptest::data_dir() + "/synth_qwen35moe_Q4_K_M.pelias.gguf";
  std::error_code ec;
  std::filesystem::remove(in_path, ec);  // this test's own outputs from an earlier run
  std::filesystem::remove(out_path, ec);
  CHECK(ptest::write_synthetic_qwen35moe(header, in_path).ok());
  const std::string in_sha = file_sha(in_path);
  CHECK_EQ(in_sha, std::string(kSynthInputSha));
  if (in_sha != kSynthInputSha) std::printf("      input sha %s\n", in_sha.c_str());

  os::FileMap m;
  CHECK(m.open(in_path).ok());
  io::GgufReader r;
  CHECK(r.parse(m).ok());
  io::RequantOptions opt;
  opt.ftype = quant::FileType::q4_k_m;
  opt.threads = 3;
  const auto rep = io::requantize(m, r, out_path, opt);
  CHECK(rep.ok());
  if (!rep.ok()) {
    std::printf("      %s\n", rep.status.text().c_str());
    return;
  }
  CHECK_EQ(rep.value.tensors, 753u);
  CHECK_EQ(rep.value.by_type[14], 64u);   // Q6_K
  CHECK_EQ(rep.value.by_type[12], 379u);  // Q4_K
  CHECK_EQ(rep.value.by_type[0], 310u);   // F32
  const std::string out_sha = file_sha(out_path);
  CHECK_EQ(out_sha, std::string(kReferenceQ4KMSha));
  if (out_sha != kReferenceQ4KMSha) std::printf("      output sha %s\n", out_sha.c_str());

  // the same output with one thread: the result does not depend on the thread count
  const std::string out1 = ptest::data_dir() + "/synth_qwen35moe_Q4_K_M.t1.gguf";
  std::filesystem::remove(out1, ec);
  opt.threads = 1;
  CHECK(io::requantize(m, r, out1, opt).ok());
  CHECK_EQ(file_sha(out1), out_sha);
  // nothing is overwritten
  CHECK(io::requantize(m, r, out1, opt).status.code == Code::invalid_argument);
}

TEST(rot32_selects_product_weights_only) {
  using pelias::io::rot32_selected;
  const std::vector<std::string> dense{"attn", "ssm_", "shexp", "output"}, exps{"_exps"}, all{"all"};
  CHECK(rot32_selected("blk.0.attn_qkv.weight", dense));
  CHECK(rot32_selected("blk.3.attn_output.weight", dense));
  CHECK(rot32_selected("blk.0.ssm_out.weight", dense));
  CHECK(rot32_selected("blk.0.ssm_alpha.weight", dense));
  CHECK(rot32_selected("blk.7.ffn_down_shexp.weight", dense));
  CHECK(rot32_selected("output.weight", dense));
  CHECK(!rot32_selected("blk.0.ssm_conv1d.weight", dense));      // a convolution, not a product
  CHECK(!rot32_selected("blk.0.ssm_norm.weight", dense));
  CHECK(!rot32_selected("output_norm.weight", dense));
  CHECK(!rot32_selected("blk.0.attn_norm.weight", dense));
  CHECK(!rot32_selected("blk.0.ffn_gate_exps.weight", dense));
  CHECK(rot32_selected("blk.0.ffn_gate_exps.weight", exps));
  CHECK(rot32_selected("blk.39.ffn_down_exps.weight", exps));
  CHECK(!rot32_selected("blk.0.ffn_gate_inp.weight", all));      // the router stays as it is
  CHECK(!rot32_selected("token_embd.weight", all));              // a lookup table
  CHECK(!rot32_selected("blk.0.ffn_gate_inp_shexp.weight", all));
  CHECK(rot32_selected("blk.0.attn_gate.weight", all));
  CHECK(!rot32_selected("blk.0.attn_gate.weight", {}));
}
