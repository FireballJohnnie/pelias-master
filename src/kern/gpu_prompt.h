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
// Other weight types use the exact paths of kern/gpu_ops.h (matmul / moe_matmul) until they get their own int8 step.
// Host API only, no CUDA types; stubs in gpu_ops_none.cpp.
#pragma once

#include <cstdint>

#include "kern/gpu_ops.h"

namespace pelias::kern::gpu {

// Q4_K (rows x k) on the device in GGUF storage (not owned), with the prepared side tables (owned): one 16-byte record
// per 256 (d, eight 6-bit scales, dmin) and 8 fp32 min factors dmin x m_j per 256 (32 bytes). For an expert stack, rows = experts x
// n_out (expert-major, as GGUF stores ffn_*_exps).
struct PromptWeights {
  std::uint32_t type = 0;
  std::int64_t rows = 0, k = 0;
  const void* w = nullptr;
  void* rec = nullptr;
  void* dig = nullptr;
};
// Q4_K with k a multiple of 256
bool prompt_type_ok(std::uint32_t type, std::int64_t k);
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
};
bool moe_scratch_alloc(std::int64_t max_tokens, int n_used, int n_expert, std::int64_t hidden, std::int64_t ffn, MoeScratch* out);
void moe_scratch_free(MoeScratch* m);

// How the expert outputs are combined into the token rows
enum class MoeCombine : int { kFixed32 = 2, kFixed64 = 3, kFloat = 1 };

// One MoE slice for `tokens` rows: q / scale / dig are the tokens' per-32 activations (hidden wide, readable to tokens
// rounded up to 64); sel / wsel the router's picks (tokens x n_used, int32 expert ids and weights, as router_topk);
// gate / up / down Q4_K expert stacks (n_expert x ffn rows of hidden; n_expert x hidden rows of ffn). y (tokens x
// hidden) receives sum over picks of weight x down(round32(silu(gate) x up)). scale_log2: the fixed-point scale 2^k
// (kFixed32 / kFixed64). The overflow counter (int32) is left in m.ovf on the device for the caller to read when it syncs.
bool moe_prompt(const PromptWeights& gate, const PromptWeights& up, const PromptWeights& down, const std::int8_t* q, const float* scale, const float* dig,
                std::int64_t tokens, const std::int32_t* sel, const float* wsel, MoeScratch& m, MoeCombine combine, int scale_log2, float* y,
                Stream s = nullptr);

}  // namespace pelias::kern::gpu
