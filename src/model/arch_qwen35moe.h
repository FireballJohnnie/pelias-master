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
// One position's routing at one layer (game plan D-T2, `pelias kld --route-dump`): the experts picked, best first, with
// their renormalized weights; the router probabilities of the last picked and of the first unpicked expert (the
// margin); and the experts the NEXT layer's router picks when it is fed this layer's output through the next layer's
// post-attention norm (the pre-gating prediction; -1 at the last layer). For at most 8 experts per token.
struct RouteRecord {
  std::int16_t sel[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  float w[8] = {};
  float p_last = 0.0f, p_next = 0.0f;
  std::int16_t pred[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
};

struct WeightView {
  const std::uint8_t* data = nullptr;
  std::uint32_t type = 0;
  std::uint64_t ne[3] = {1, 1, 1};
  std::uint64_t row_bytes = 0;
  std::string name;
  // The file stores this weight rotated (pelias.rot32.tensors): every row was multiplied by the 32-value Walsh-Hadamard
  // transform before quantization, so each input row is rotated the same way before the product.
  bool rot32 = false;
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
  // the inputs of every quantized weight whose name contains one of these strings ("all": every quantized weight,
  // the output layer included) are rounded to the reference's 8-bit activation format and back before the product.
  // Path "cpu": the format llama.cpp's CPU path pairs with each weight type (READ-SRC ggml-cpu.c type_traits_cpu
  // vec_dot_type: Q8_K for the K-quants, IQ2/IQ3/IQ1/IQ4_XS and TQ types; Q8_0 for Q4_0, Q5_0, Q8_0, IQ4_NL, MXFP4;
  // Q8_1 for Q4_1, Q5_1). Path "gpu": q8_1 (32 values, fp16 scale) for every type, as its MMVQ kernels use; its MMQ
  // kernels keep the q8_1 scale in fp32 for some types, which this does not model (a second-order difference).
  std::vector<std::string> act_int8;
  bool act_int8_gpu = false;
  // With act_int8_gpu: the Pelias int8 kernel's own format (tools/probes/batch_c v7-v12, q8k_warp): one fp32 scale per 256
  // values, scale = amax / 127 (1 when amax is 0), q = round-half-even(x * (1 / scale)) clamped to -127..127, for every
  // weight type ("--act-int8-path gpu256"). q8_1 (one fp16 scale per 32) when false.
  bool act_gpu256 = false;
  // With act_gpu256, the per-32 refinements of the 256-value layout (candidates for the int8 kernel, game plan run list 1):
  // 1 = eight fp32 scales per 256, one per 32 (amax32 / 127, same rounding as act_round_q8_256), "gpu256x8";
  // 2 = one fp32 scale S per 256 and a power-of-two exponent e in 0..3 per 32 (scale S / 2^e, the largest e that still
  //     covers the 32-block), "gpu256e". 0 = one scale per 256 (the v7-v12 kernels).
  int act_gpu256_sub = 0;
  // With act_int8: rotate every 32-value block of a rounded input with the orthonormal Walsh-Hadamard transform
  // before the rounding and rotate it back after (game plan Block J, R1). Exact in real arithmetic when the weights
  // are rotated the same way, so this measures only what the rotation does to the activation rounding error (a
  // single large value is spread over the block's 32 values, which shrinks the block's scale).
  bool act_rot32 = false;
  // Optional routing record (game plan D-T2): n_tokens x n_layer entries (position-major), filled by every layer.
  RouteRecord* route = nullptr;
  // Optional state codes (C6 accuracy-assisted miss map): n_tokens x n_layer entries (position-major) of
  // quant::state_code_bytes(n_embd) bytes each, the code of each layer's normed input (after attn_norm).
  std::uint8_t* state_codes = nullptr;
  // Optional q, k record (C1, the H gate on KV pages): for every full-attention layer in order, n_tokens x
  // (n_head + n_head_kv) x head_dim fp16 values: each token's query heads (per-head normed and roped; the gate half left
  // out) then its key heads (normed and roped), exactly what the attention scores use.
  std::uint16_t* qk_dump = nullptr;
  // Speed ideas measured for their accuracy cost (game plan Block M). Each is off by default.
  // act_bits 4: with act_int8 on the gpu path, round to 4 bits per 32 values (symmetric, -7..7, fp16 scale) instead of
  // q8_1 (W4A4 with rotation, QuaRot-style).
  int act_bits = 8;
  // Lossy storage of the context state, to measure what a smaller cache or checkpoint costs (author 2026-10-29, the
  // DeltaNet-checkpoint trick applied to the active state). Each stored vector is rounded per 32 values and read back:
  // StateStore::q8 = 8-bit, fp16 scale per 32 (as q8_0); int4 = 4-bit (-7..7), fp16 scale per 32; int4r = int4 after a
  // fixed sign diagonal and a Walsh-Hadamard per 32 (rotated back after reading); int8row = 8-bit, fp32 scale per 128.
  // kv_store: every key and value head of the full-attention layers (keys after norm and rope), at every position.
  // state_store: each DeltaNet layer's recurrent state and its last conv inputs, once, just before position state_at
  // (a checkpoint taken there and restored); state_at < 0 means the middle of the chunk (the first scored position).
  enum class StateStore { exact, q8, int4, int4r, int8row };
  StateStore kv_store = StateStore::exact;
  StateStore state_store = StateStore::exact;
  std::int64_t state_at = -1;
  // neuron_skip f in (0, 1): in every routed expert, the fraction f of its hidden values (silu(gate) * up) smallest in
  // magnitude are set to zero for each token: the oracle upper bound of skipping neurons (their gate and up rows and
  // their down column would not be read).
  double neuron_skip = 0.0;
  // h_gate f in (0, 1]: on every Q8_Z weight, each token reads the 8th bit (H, the low bit of u = q + 127 = 2b + h) only on
  // the fraction f of each row's 32-blocks with the largest expected loss s^2 * sum(x^2) (the live gate, ZH: decided by
  // the token, not the calibration text); the other blocks use the base with H filled by 0.5. 0 = off (all H read).
  double h_gate = 0.0;
  // T-decode proof (run D; docs/PELIAS_7_T_H_PLAN.md section 6 step 6): Q7_ZT weights with the eight-scales-per-256
  // activation format (gpu256x8) are multiplied by the 1-bit route: the 7-bit base as integers, the 8th bit of each
  // gated 32-block added by AND + popcount against the activation's bit planes (quant/tdecode.h). Off: the dequantized
  // weights times the rounded activations (the same integers, summed in another order).
  bool t_decode = false;
  // With t_decode: false = the 1-bit route (AND + popcount); true = the 8th bit folded into the integers (q = 2b + h - 127)
  // in the same loop and order, the control that shows the two routes give bit-identical sums.
  bool t_decode_fold = false;
  // swap_margin m > 0 with hot (n_layer x n_expert, 1 = resident): when a token's last pick is not resident, the first
  // unpicked expert is, and their probabilities differ by less than m, the resident one is used instead.
  float swap_margin = 0.0f;
  std::vector<std::uint8_t> hot;
  // defer_experts d: each token's d lowest-weight routed experts are added to the residual stream one layer later
  // (after the next layer's MoE), as KTransformers' expert deferral; the last layer's are added before the output norm.
  int defer_experts = 0;
  // cold_gpu256 with hot (the engine's split: hot experts on the GPU, cold on the CPU): the routed experts NOT in the hot
  // set round their inputs to one scale per 256 (act_round_q8_256, the format the CPU cold kernel must use), the rest keep
  // the gpu-path format chosen above. Needs act_int8 on the gpu path.
  bool cold_gpu256 = false;
  // counters for the above (may be null, 6 entries): [0] swaps made, [1] token-layers whose last pick was not resident,
  // [2] hidden values zeroed, [3] hidden values seen, [4] expert rows (token, expert) computed cold, [5] all expert rows
  std::uint64_t* m_stats = nullptr;
};

// The orthonormal Walsh-Hadamard transform of every 32-value block of x[0..n) in place (n a multiple of 32). It is its
// own inverse: applying it twice gives x back (to rounding).
void act_rotate32(float* x, std::int64_t n);
// ForwardOptions::kv_store / state_store: round x (n values, a multiple of 32; 128 for int8row) as stored and read back.
void state_store_roundtrip(float* x, std::int64_t n, ForwardOptions::StateStore kind);
// one row to the Pelias int8 kernel's activation format and back (ForwardOptions::act_gpu256); n a multiple of 256
void act_round_q8_256(float* x, std::int64_t n);
// The live H gate (ForwardOptions::h_gate) on rows of a Q8_Z matrix: y[t][r] holds the full dot products on entry; on
// return, for each token t and row r, the 32-blocks outside the kept fraction have their 8th bit replaced by 0.5, i.e.
// y -= s_j * sum (h - 0.5) x over each dropped block. Blocks kept per row: round(f * blocks).
void h_gate_correct_q8z(const std::uint8_t* data, std::uint64_t row_bytes, std::int64_t n_out, std::int64_t k, const float* x,
                        std::int64_t n_tok, std::int64_t ldx, float* y, std::int64_t ldy, double f, int threads);
// T-decode (ForwardOptions::t_decode) on rows of a Q7_ZT matrix: x holds n_tok input rows (rotated if the weight is,
// not yet rounded); they are rounded to the per-32 int8 format here; y[t][r] = sum over 32-blocks of s_j a_j P_j with the
// gated blocks' 8th bit by AND + popcount.
void t_decode_q7zt(const std::uint8_t* data, std::uint64_t row_bytes, std::int64_t n_out, std::int64_t k, const float* x, std::int64_t n_tok,
                   std::int64_t ldx, float* y, std::int64_t ldy, int threads);
// the same with eight scales per 256 (ForwardOptions::act_gpu256_sub 1) or a power-of-two exponent per 32 (sub 2)
void act_round_q8_256x8(float* x, std::int64_t n);
void act_round_q8_256e(float* x, std::int64_t n);

// Runs tokens[0..n_tokens) and writes the logits of positions [out_begin, out_end) to `logits`
// ((out_end - out_begin) rows of spec.n_vocab floats). With `logits` null the output layer is skipped.
Status forward_reference(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, const std::int32_t* tokens, int n_tokens, int out_begin,
                         int out_end, float* logits, const ForwardOptions& opt = {});

}  // namespace pelias::model
