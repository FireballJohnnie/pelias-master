// provenance: original (the conversion READ-SRC from llama.cpp convert_hf_to_gguf.py, conversion/base.py and
//   conversion/qwen.py at commit 2ca15f5404760548c39e7b92bd43116a09414a1a, used as a specification; no code copied)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// stage_a.h - Stage A (masterwork 9.1, D33): a Hugging Face model folder (config.json, safetensors shards, tokenizer
// files, model card) to a GGUF file.
//
//   --type BF16    the file the reference converter writes with --outtype bf16: the same metadata in the same order,
//                  the same tensors in the same order with the same shapes, types and bytes (P-X6)
//   --type Q4_K_M  (and the other stock mixes) the file the reference quantizer writes from that BF16 file, made in one
//                  pass without writing the BF16 file first (P-E6 on top of P-X6)
//
// One step cannot be reproduced bit for bit: the reference computes ssm_a = -exp(A_log) with the exponential of the
// PyTorch build it runs (MKL's, MEASURED to differ from the correctly rounded value by one unit in the last place on
// 13 of the reference model's 960 values). Pelias computes the correctly rounded value, the same on every platform.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "io/gguf_writer.h"
#include "io/json.h"
#include "io/requantize.h"
#include "io/safetensors.h"
#include "pelias/status.h"
#include "quant/type_rules.h"

namespace pelias::ingest {

// How one output tensor is made from one source tensor, row by row (a row = the last array dimension).
struct TensorPlan {
  std::string name;                  // the GGUF name
  std::vector<std::uint64_t> shape;  // the array shape the reference writes, outermost first
  std::uint32_t type = 0;            // quant::Type: BF16 or F32 (the reference converter's choice for --outtype bf16)
  std::size_t shard = 0;             // source shard and tensor
  std::size_t source = 0;
  // rows: same, or part of an expert-stacked source (gate or up half), or V heads moved from grouped to tiled order
  enum class Rows { same, expert_part, heads } rows = Rows::same;
  std::uint64_t part_rows = 0, part_offset = 0;  // expert_part: output rows per expert, first source row of the part
  std::uint64_t head_start = 0, nk = 0, nvpk = 0, hd = 0;  // heads: first index moved, K heads, V per K, head size
  bool head_columns = false;         // heads move along the row (columns) instead of across rows
  enum class Op { none, plus_one, neg_exp } op = Op::none;
};

// What an architecture plug-in produces: its metadata entries (between the general.* ones and the quantization
// version), the tensors in the reference's write order, and the facts the stock type rule needs.
struct ArchPlan {
  std::vector<io::GgufKvBlob> kv;
  std::vector<TensorPlan> tensors;
  quant::ModelFacts facts;
  std::string arch;  // general.architecture
};

// A source tensor as seen after the reference's renames (vision tensors dropped, "language_model." removed, the MTP
// block numbered after the last layer), in the reference's reading order (shards sorted, names sorted in a shard).
struct SourceRef {
  std::string name;  // after the renames
  std::size_t shard = 0, index = 0;
  const io::SafetensorsTensor* t = nullptr;
};

// Merged hyperparameters: config.json with text_config's entries on top, as the reference reads them.
Result<ArchPlan> plan_qwen35moe(const io::Json& hparams, const std::vector<SourceRef>& sources, std::uint32_t converter_file_type);

struct ConvertOptions {
  quant::FileType ftype = quant::FileType::bf16;
  int threads = 0;
};

// The whole conversion. The output is written under "<out>.partial" and renamed when complete; an existing output is
// never overwritten.
Result<io::RequantReport> convert_folder(const std::string& folder, const std::string& out, const ConvertOptions& opt,
                                         const io::RequantProgress& progress = {});

// An opened folder ready to convert: the metadata entries of the BF16 file, the planned tensors in its order, and a
// reader of any planned tensor's rows in its BF16-file type (BF16 bytes, or F32 for the tensors the converter widens).
class StageASource {
 public:
  static Result<std::unique_ptr<StageASource>> open(const std::string& folder);
  ~StageASource();
  [[nodiscard]] const std::vector<io::GgufKvBlob>& kv() const;
  [[nodiscard]] const std::vector<TensorPlan>& tensors() const;
  [[nodiscard]] const quant::ModelFacts& facts() const;
  Status read_rows(std::size_t i, std::uint64_t row0, std::uint64_t n, std::uint8_t* out) const;
  // Rows of tensor i (the last dimension is one row) and bytes per row.
  [[nodiscard]] std::uint64_t rows(std::size_t i) const;
  [[nodiscard]] std::uint64_t row_bytes(std::size_t i) const;

 private:
  StageASource() = default;
  struct Impl;
  Impl* impl_ = nullptr;
};

// The metadata the conversion would write, without the tensors' data (tests and the vocabulary check).
struct ConvertPreview {
  std::vector<io::GgufKvBlob> kv;
  std::vector<io::GgufOutTensor> tensors;  // as the BF16 file lists them
};
Result<ConvertPreview> preview_folder(const std::string& folder);

// The folder name the reference uses for the model (pathlib's Path(folder).name).
std::string folder_name(const std::string& folder);

}  // namespace pelias::ingest
