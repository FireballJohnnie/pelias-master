// provenance: original (the expected hashes MEASURED in the sandbox 2026-10-26: the reference llama-quantize at commit
//   2ca15f5, built unchanged, run on the same synthetic file with Q4_K_M)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "io/gguf_compare.h"
#include "io/gguf_reader.h"
#include "io/imatrix_weights.h"
#include "io/requantize.h"
#include "io/sha256.h"
#include "model/imatrix.h"
#include "os/file_map.h"
#include "pelias_test.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"
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
  const std::vector<std::string> dense_mtp{"attn", "ssm_", "shexp", "output", "eh_proj"};
  CHECK(rot32_selected("blk.40.nextn.eh_proj.weight", dense_mtp));  // the MTP piece: rotated (author 2026-10-29)
  CHECK(!rot32_selected("blk.40.nextn.eh_proj.weight", dense));     // only when the group is named
  CHECK(!rot32_selected("blk.40.nextn.enorm.weight", all));         // its norms stay as they are
  CHECK(!rot32_selected("token_embd.weight", dense_mtp));           // the token table stays unrotated (-0.9 % only)
}

namespace {
// Writes `text` to `path` (test files only).
bool write_text(const std::string& path, const std::string& text) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
  return std::fclose(f) == 0 && ok;
}
// SHA-256 of the reference quantizer's output (llama-quantize at 2ca15f5, built unchanged, run in the data directory as
// `llama-quantize --imatrix synth_imatrix.gguf --tensor-type-file synth_mix.txt synth_qwen35moe.gguf OUT Q4_K_M`, the
// pattern file made by `tools/recipe.py llama-file` from synth_mix.tsv), MEASURED in the sandbox 2026-10-28
const char* kReferenceRecipeImatrixSha = "bbab87c816b9fd1f3c6fd92c8314368f5b05902312df229b12bedcd3ae9caf4d";
}  // namespace

TEST(recipe_file_is_read_and_checked) {
  const std::string d = ptest::data_dir();
  CHECK(write_text(d + "/recipe_ok.tsv", "a.weight\tq8_0\r\n\nb.weight\tIQ4_XS\n"));
  const auto ok = io::read_recipe(d + "/recipe_ok.tsv");
  CHECK(ok.ok() && ok.value.size() == 2 && ok.value.at("a.weight") == 8u && ok.value.at("b.weight") == 23u);
  CHECK(write_text(d + "/recipe_dup.tsv", "a.weight\tQ8_0\na.weight\tQ4_K\n"));
  CHECK(!io::read_recipe(d + "/recipe_dup.tsv").ok());
  CHECK(write_text(d + "/recipe_type.tsv", "a.weight\tQ9_X\n"));
  CHECK(!io::read_recipe(d + "/recipe_type.tsv").ok());
  CHECK(write_text(d + "/recipe_form.tsv", "a.weight Q8_0\n"));
  CHECK(!io::read_recipe(d + "/recipe_form.tsv").ok());  CHECK(write_text(d + "/recipe_pelias.tsv", "a.weight\tQ8_Z\n"));  // a Pelias-only type id (200)
  const auto pz = io::read_recipe(d + "/recipe_pelias.tsv");
  CHECK(pz.ok() && pz.value.at("a.weight") == 200u);
}

TEST(quantize_with_recipe_and_imatrix_matches_the_reference) {
  const std::string d = ptest::data_dir();
  const std::string header = d + "/unsloth_header_16MiB.bin";
  if (!std::filesystem::exists(header)) {
    ptest::skip("unsloth_header_16MiB.bin not in the data directory (kept on the author's drive)");
    return;
  }
  const std::string in_path = d + "/synth_qwen35moe.gguf";
  std::error_code ec;
  if (!std::filesystem::exists(in_path)) CHECK(ptest::write_synthetic_qwen35moe(header, in_path).ok());
  CHECK_EQ(file_sha(in_path), std::string(kSynthInputSha));
  os::FileMap m;
  CHECK(m.open(in_path).ok());
  io::GgufReader r;
  CHECK(r.parse(m).ok());

  // the stock types, then a mixed recipe from them (every tensor named)
  const std::string stock = d + "/synth_stock_for_recipe.gguf";
  std::filesystem::remove(stock, ec);
  io::RequantOptions o0;
  o0.threads = 3;
  CHECK(io::requantize(m, r, stock, o0).ok());
  os::FileMap ms;
  CHECK(ms.open(stock).ok());
  io::GgufReader rs;
  CHECK(rs.parse(ms).ok());
  std::string tsv;
  for (const io::GgufTensor& t : rs.tensors()) {
    std::string ty = quant::type_info(t.type).name;
    if (t.name.find("ffn_gate_exps") != std::string::npos || t.name.find("ffn_up_exps") != std::string::npos) ty = "IQ4_XS";
    else if (t.name.find("ffn_down_exps") != std::string::npos) ty = "Q5_K";
    else if (t.name.find("attn_qkv.weight") != std::string::npos) ty = "Q8_0";
    else if (t.name.find("ssm_out.weight") != std::string::npos) ty = "Q6_K";
    tsv += t.name + "\t" + ty + "\n";
  }
  CHECK(write_text(d + "/synth_mix.tsv", tsv));

  // a deterministic importance matrix over the experts and some dense weights; expert 1 of each stack sees no token
  model::ImatrixCollector col;
  for (const io::GgufTensor& t : r.tensors()) {
    const bool exps = t.name.find("_exps.weight") != std::string::npos;
    const bool dense = t.name.find("attn_qkv.weight") != std::string::npos || t.name.find("ssm_out.weight") != std::string::npos ||
                       t.name.find("attn_output.weight") != std::string::npos;
    if (!exps && !dense) continue;
    const auto cols = static_cast<std::int64_t>(t.ne[0]), mats = exps ? static_cast<std::int64_t>(t.ne[2]) : 1;
    model::ImatrixEntry& e = col.entry(t.name, cols, mats);
    std::vector<float> row(static_cast<std::size_t>(cols));
    for (std::int64_t k = 0; k < mats; ++k) {
      if (exps && k == 1) continue;
      for (int rep = 0; rep < 3; ++rep) {
        for (std::int64_t j = 0; j < cols; ++j)
          row[static_cast<std::size_t>(j)] = 0.05f + static_cast<float>((j * 37 + k * 11 + rep * 5 + static_cast<std::int64_t>(t.name.size())) % 97) / 16.0f;
        model::ImatrixCollector::add_row(e, k, row.data());
      }
    }
  }
  col.finish_chunk();
  const std::string imat = d + "/synth_imatrix.gguf";
  std::filesystem::remove(imat, ec);
  CHECK(col.write(imat, {"synthetic"}, 512).ok());
  const auto iw = io::read_imatrix_weights(imat);
  CHECK(iw.ok());
  if (!iw.ok()) return;
  // weights are sum2 / count; a matrix with no token weighs every column 1
  for (const auto& [name, e] : col.entries()) {
    const std::vector<float>& w = iw.value.weights.at(name);
    CHECK(w.size() == e.sum2.size());
    if (e.counts.size() > 1) CHECK(w[static_cast<std::size_t>(e.columns)] == 1.0f);
    CHECK(w[0] == e.sum2[0] / static_cast<float>(e.counts[0]));
    break;
  }

  const auto rc = io::read_recipe(d + "/synth_mix.tsv");
  CHECK(rc.ok());
  io::RequantOptions opt;
  opt.threads = 3;
  opt.recipe = rc.value;
  opt.imatrix = &iw.value;
  opt.imatrix_name = "synth_imatrix.gguf";
  const std::string out = d + "/synth_mix_imatrix.pelias.gguf";
  std::filesystem::remove(out, ec);
  const auto rep = io::requantize(m, r, out, opt);
  CHECK(rep.ok());
  if (!rep.ok()) {
    std::printf("      %s\n", rep.status.text().c_str());
    return;
  }
  CHECK(rep.value.with_imatrix > 0);
  const std::string sha = file_sha(out);
  CHECK_EQ(sha, std::string(kReferenceRecipeImatrixSha));
  if (sha != kReferenceRecipeImatrixSha) std::printf("      output sha %s\n", sha.c_str());

  // a tensor the reference never quantizes (a norm, F32) may change float width but not become a block format
  std::string norm;
  for (const io::GgufTensor& t : rs.tensors())
    if (t.name.find("attn_norm.weight") != std::string::npos) { norm = t.name; break; }
  CHECK(!norm.empty());
  // a Q8_Z recipe goes through the whole quantizer: attn_qkv at Q8_Z, read back as type 200 and dequantizable
  {
    std::string tsv3;
    for (const io::GgufTensor& t : rs.tensors())
      tsv3 += t.name + "\t" + (t.name.find("attn_qkv.weight") != std::string::npos ? std::string("Q8_Z") : std::string(quant::type_info(t.type).name)) + "\n";
    CHECK(write_text(d + "/synth_q8z.tsv", tsv3));
    io::RequantOptions o3;
    o3.threads = 3;
    o3.recipe = io::read_recipe(d + "/synth_q8z.tsv").value;
    const std::string out3 = d + "/synth_q8z.gguf";
    std::filesystem::remove(out3, ec);
    const auto rep3 = io::requantize(m, r, out3, o3);
    CHECK(rep3.ok());
    if (!rep3.ok()) std::printf("      %s\n", rep3.status.text().c_str());
    os::FileMap m3;
    CHECK(m3.open(out3).ok());
    io::GgufReader r3;
    CHECK(r3.parse(m3).ok());
    int seen = 0;
    for (const io::GgufTensor& t : r3.tensors())
      if (t.name.find("attn_qkv.weight") != std::string::npos) {
        CHECK(t.type == 200u);
        ++seen;
        std::vector<float> row(static_cast<std::size_t>(t.ne[0]));
        CHECK(quant::dequantize_row(quant::Type::q8_z, m3.data() + t.file_offset, row.data(), static_cast<std::int64_t>(t.ne[0])).ok());
      }
    CHECK(seen > 0);
  }
  for (const char* want : {"BF16", "Q8_0"}) {
    std::string tsv2;
    for (const io::GgufTensor& t : rs.tensors()) tsv2 += t.name + "\t" + (t.name == norm ? std::string(want) : std::string(quant::type_info(t.type).name)) + "\n";
    CHECK(write_text(d + "/synth_norm_" + want + ".tsv", tsv2));
    io::RequantOptions o2;
    o2.threads = 3;
    o2.recipe = io::read_recipe(d + "/synth_norm_" + want + ".tsv").value;
    const std::string out2 = d + "/synth_norm_" + want + ".gguf";
    std::filesystem::remove(out2, ec);
    const auto rep2 = io::requantize(m, r, out2, o2);
    if (std::string(want) == "Q8_0") {
      CHECK(!rep2.ok());
      continue;
    }
    CHECK(rep2.ok());
    os::FileMap m2;
    CHECK(m2.open(out2).ok());
    io::GgufReader r2;
    CHECK(r2.parse(m2).ok());
    for (const io::GgufTensor& t : r2.tensors())
      if (t.name == norm) CHECK(t.type == static_cast<std::uint32_t>(quant::Type::bf16));
  }
}
