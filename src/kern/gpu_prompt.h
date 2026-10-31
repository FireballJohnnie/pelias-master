// provenance: original (moved from tools/probes/batch_c: v12x8 / k_moe_tile / k_moe_vec / the S3 work list, MEASURED
// 2026-10-30 21:05-21:40, gp/MOE; the arithmetic unchanged)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kern/gpu_prompt.h - the int8 prompt path (many tokens per call) for Q4_K weights: a dense GEMM and the locked MoE slice
// (DECISIONS_LOG 2026-10-30 21:44):
//   (1) activations rounded per 32 (gpu256x8) by rms_norm_q8x32 / act_q8x32 (kern/gpu_ops.h), plus one fp32 min factor per 32
//       (scale x sum) for the min term (act_digits);
//   (2) the expert work list built on the GPU from the router's picks (__match_any_sync + __popc counts; one warp with
//       __ballot_sync / __popc / scans builds the list; no host round trip, no padding);
//   (3) 64-row int8 tensor-core tiles (int8 x 4-bit mma, the 6-bit sub-block scale as an integer product, per-32
//       activation scale applied per sub-block, the min term in fp32 FMAs per
//       super-block (v10b; the bf16 two-digit mma of v10 measured 3.0-3.4e-06 against fp64)) for experts with more than 8 rows, the
//       column tile in blockIdx.x;
//   (4) a warp-vector dp4a path for experts with 1-8 rows;
//   (5) the down projection folded into the combine by fixed-point int32 global reductions (deterministic), then a
//       conversion to float. The int32 scale is the caller's (2^k); an overflow counter is kept (values above 2^31 / 8
//       per addend); fixed-point int64 and float reductions are the fallbacks.
// Q4_TG2 (locked 2026-10-30 23:22): the same tiles with each grid value as 24-bit fixed point in three int8 digits (three
// int8 mmas per 32-block), and the warp-vector path with the exact fp32 grid table. Q5_K (23:5x): Q4_K's tiles with the 5th
// bit from qh (values 0-31, exact int8). Q6_K (23:5x): values q - 32 (exact int8, no min term), two int8 k16 mmas per
// 32-block for its two scales per 32; its data must start 16-byte aligned. Other weight types use the exact paths of
// kern/gpu_ops.h (matmul / moe_matmul) until they get their own int8 step.
// Host API only, no CUDA types; stubs in gpu_ops_none.cpp.
#pragma once

#include <cstdint>

#include "kern/gpu_ops.h"

namespace pelias::kern::gpu {

// Q4_K, Q5_K, Q6_K or Q4_TG2 (rows x k) on the device in GGUF storage (not owned), with the prepared side tables (owned).
// Q4_K / Q5_K: one 16-byte record per 256 (d, eight 6-bit scales, dmin) and 8 fp32 min factors dmin x m_j per 256 (32
// bytes). Q6_K: rec = 16 int8 scales + fp32 d per 256 (32 bytes), no dig. Q4_TG2: rec = 8 fp32 sub-block scales per 256;
// dig = 512 bytes (64 three-byte digit words, 64 fp32 grid values). For an expert stack, rows = experts x
// n_out (expert-major, as GGUF stores ffn_*_exps).
struct PromptWeights {
  std::uint32_t type = 0;
  std::int64_t rows = 0, k = 0;
  const void* w = nullptr;
  void* rec = nullptr;
  void* dig = nullptr;
  // MoE stacks of a usage-tiered weight (experts of several types, each type its own tensor): emap (device, n_expert
  // ints, not owned) gives each expert's matrix index in this stack, -1 when the expert is in another stack; n_mat = the
  // stack's matrices (rows = n_mat x n_out). Unset (null / 0): every expert, matrix e = expert e. int8 types only.
  const int* emap = nullptr;
  int n_mat = 0;
};
// Q4_K, Q5_K, Q6_K or Q4_TG2 with k a multiple of 256
bool prompt_type_ok(std::uint32_t type, std::int64_t k);
// Types without an int8 step but with a GPU decoder (kern/gpu_decode.h) are accepted with no side tables: moe_prompt runs
// such a stack on the exact path (gpu_ops' moe_matmul, fp32), prompt_gemm refuses it (use gpu_ops' matmul on the rounded
// floats).
bool prompt_prepare(std::uint32_t type, const void* w_dev, std::int64_t rows, std::int64_t k, PromptWeights* out, Stream s = nullptr);
void prompt_release(PromptWeights* w);
const char* prompt_last_error();

// the min-term factors of per-32 activations: for each 32-block, scale x float(sum) in fp32, laid out [rows][k / 32]
// (the name is kept from v10, which stored two bf16 digits per 32 in the same 4 bytes)
bool act_digits(const float* scale, const int* sum, std::int64_t rows, std::int64_t k, float* dig, Stream s = nullptr);

// dense: y (rows x n_out, row stride n_out) = the activations (per-32 int8 q, scale, digits; rows of k) times W^T.
// q / scale / dig must be readable up to rows rounded up to 64 (tile reads; the extra rows are not stored).
bool prompt_gemm(const PromptWeights& w, std::int64_t n_out, const std::int8_t* q, const float* scale, const float* dig, std::int64_t rows, float* y,
                 Stream s = nullptr);

// Device scratch of one MoE slice, allocated once for up to max_tokens (no allocation in the step loop)
struct MoeScratch {
  std::int64_t max_tokens = 0, hidden = 0, ffn = 0;
  int n_used = 0, n_expert = 0;
  void* mem = nullptr;  // one allocation; the pointers below point into it
  int *cnt = nullptr, *off = nullptr, *cursor = nullptr, *meta = nullptr, *src = nullptr, *rtok = nullptr, *prow = nullptr;
  void *units_t = nullptr, *units_v = nullptr;
  float* rw = nullptr;
  std::int8_t* pq = nullptr;
  float* psc = nullptr;
  float* pdig = nullptr;
  float *hg = nullptr, *hu = nullptr;
  std::int8_t* q2 = nullptr;
  float* sc2 = nullptr;
  int* sum2 = nullptr;
  float* dig2 = nullptr;
  void* yfix = nullptr;  // int32 or int64 accumulators (tokens x hidden)
  unsigned* ovf = nullptr;
  // exact paths (allocated when moe_scratch_alloc's `exact` is set): token rows as floats (q x scale), the rounded silu x up
  // rows as floats, the down rows before the fold, the identity row map
  bool exact = false;
  float *xf = nullptr, *x2f = nullptr, *yd = nullptr;
  int* iota = nullptr;
  // one stack's own copy of the work list (PromptWeights::emap): its tile and vector units and their counts
  void *units_t2 = nullptr, *units_v2 = nullptr;
  int* meta2 = nullptr;
};
// exact: also allocate the exact paths' buffers (stacks of types without an int8 prompt step, run by gpu_ops' moe_matmul
// over the same GPU work list, folded into y by the same int32 / int64 / float reductions)
bool moe_scratch_alloc(std::int64_t max_tokens, int n_used, int n_expert, std::int64_t hidden, std::int64_t ffn, MoeScratch* out, bool exact = false);
void moe_scratch_free(MoeScratch* m);

// How the expert outputs are combined into the token rows
enum class MoeCombine : int { kFixed32 = 2, kFixed64 = 3, kFloat = 1 };

// One expert projection as one or more stacks (a usage-tiered weight: one stack per type, each with its emap); a plain
// weight is one stack with no emap.
struct MoeStacks {
  const PromptWeights* w = nullptr;
  int n = 0;
};
// One MoE slice for `tokens` rows: q / scale / dig are the tokens' per-32 activations (hidden wide, readable to tokens
// rounded up to 64); sel / wsel the router's picks (tokens x n_used, int32 expert ids and weights, as router_topk);
// gate / up / down Q4_K expert stacks (n_expert x ffn rows of hidden; n_expert x hidden rows of ffn). y (tokens x
// hidden) receives sum over picks of weight x down(round32(silu(gate) x up)). scale_log2: the fixed-point scale 2^k
// (kFixed32 / kFixed64). The overflow counter (int32) is left in m.ovf on the device for the caller to read when it syncs.
bool moe_prompt(const PromptWeights& gate, const PromptWeights& up, const PromptWeights& down, const std::int8_t* q, const float* scale, const float* dig,
                std::int64_t tokens, const std::int32_t* sel, const float* wsel, MoeScratch& m, MoeCombine combine, int scale_log2, float* y,
                Stream s = nullptr);
// The same with each projection as stacks (gate / up / down may each have 1-4 stacks; every stack covers its own
// experts; together they cover every routed expert).
bool moe_prompt_multi(MoeStacks gate, MoeStacks up, MoeStacks down, const std::int8_t* q, const float* scale, const float* dig, std::int64_t tokens,
                      const std::int32_t* sel, const float* wsel, MoeScratch& m, MoeCombine combine, int scale_log2, float* y, Stream s = nullptr);

}  // namespace pelias::kern::gpu
