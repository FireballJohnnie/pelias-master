// provenance: original (the computation READ-SRC from llama.cpp src/models/qwen35moe.cpp, src/llama-graph.cpp
//   build_moe_ffn / build_ffn / build_norm, src/models/models.h build_gdn_l2_norm and ggml/src/ggml-cpu/ops.cpp
//   (rms_norm, rope multi / IMRoPE, ssm_conv, gated_delta_net, softplus) at commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a, read as a specification; no code copied)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// arch_qwen35moe.h - the qwen35moe trunk, written once in plain fp32 as the reference every faster path is checked
// against (Milestone 1: CPU reference forward pass). Weights stay in the file in whatever type it stores them and are
// dequantized row by row as they are used; activations are fp32 throughout by default (no 8-bit activation rounding, no
// fp16 KV cache), so this measures what the weights do and nothing else. ForwardOptions::act_int8 adds the reference's
// 8-bit activation rounding on chosen weights, to measure that effect on its own.
//
// One call runs one sequence from position 0 with fresh state (zero DeltaNet and conv state, empty attention cache),
// the way a perplexity or KL-divergence chunk is evaluated. The MTP block is loaded by the file but not run here.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "io/gguf_reader.h"
#include "model/imatrix.h"
#include "model/spec.h"
#include "pelias/status.h"

namespace pelias::model {

// A matrix (or a stack of matrices) inside the mapped file: rows of ne[0] elements, ne[1] rows per matrix, ne[2]
// matrices (the experts). Never owns memory.
struct WeightView {
  const std::uint8_t* data = nullptr;
  std::uint32_t type = 0;
  std::uint64_t ne[3] = {1, 1, 1};
  std::uint64_t row_bytes = 0;
  std::string name;
  [[nodiscard]] bool present() const { return data != nullptr; }
};

struct Qwen35MoeLayer {
  bool recurrent = false;
  std::vector<float> attn_norm, post_norm;
  // full attention
  WeightView wq, wk, wv, wo;  // wq gives [query | gate] per head
  std::vector<float> q_norm, k_norm;
  // gated DeltaNet
  WeightView wqkv, wgate, wbeta, walpha, ssm_out;
  std::vector<float> conv1d;  // [channel][tap]
  std::vector<float> ssm_dt, ssm_a, ssm_norm;
  // routed experts and the gated shared expert
  WeightView gate_inp, up_exps, gate_exps, down_exps, up_shexp, gate_shexp, down_shexp;
  std::vector<float> gate_inp_shexp;
};

struct Qwen35MoeWeights {
  WeightView tok_embd, output;
  std::vector<float> output_norm;
  std::vector<Qwen35MoeLayer> layers;  // the trunk only (spec.n_layer)
};

// Binds every trunk tensor by name, checks its shape against the spec and its type against what the reference can
// dequantize, and converts the small vectors (norms, conv taps, dt, A) to fp32. `file_base` is the start of the mapped
// file the reader parsed.
Result<Qwen35MoeWeights> bind_qwen35moe(const io::GgufReader& r, const std::uint8_t* file_base, const Qwen35MoeSpec& s);

struct ForwardOptions {
  int threads = 0;  // 0: every hardware thread
  // Optional routing statistics, spec.n_layer x spec.n_expert entries each (layer-major), added to on every call:
  // how many times each expert was among a token's chosen experts, and the sum of the routing weights it got.
  std::uint64_t* expert_picks = nullptr;
  double* expert_weight = nullptr;
  // Optional importance-matrix collection (model/imatrix.h); the caller calls finish_chunk() after each call.
  ImatrixCollector* imatrix = nullptr;
  // Optional 8-bit activation rounding, to measure what a serving or training kernel that rounds its inputs would add:
  // the inputs of every quantized weight whose name contains one of these strings ("all": every quantized weight)
  // are rounded to the reference's 8-bit activation format and back before the product. Block 256 = Q8_K (the
  // reference's CPU path), block 32 = q8_1 (its GPU path).
  std::vector<std::string> act_int8;
  int act_int8_block = 256;
};

// Runs tokens[0..n_tokens) and writes the logits of positions [out_begin, out_end) to `logits`
// ((out_end - out_begin) rows of spec.n_vocab floats). With `logits` null the output layer is skipped.
Status forward_reference(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, const std::int32_t* tokens, int n_tokens, int out_begin,
                         int out_end, float* logits, const ForwardOptions& opt = {});

}  // namespace pelias::model
