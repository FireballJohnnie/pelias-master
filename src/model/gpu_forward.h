// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// gpu_forward.h - GP3-12 rungs 2-3 (masterwork 16.1): the qwen35moe trunk on the GPU for a batch of tokens, one fresh
// sequence from position 0 as forward_reference runs it, built only from the rung-1 kernels (kern/gpu_ops.h,
// kern/gpu_prompt.h). Every quantized product rounds its input to the engine's per-32 int8 format (gpu256x8) first, as the
// reference does with matching_reference_options(); the product runs on kern/gpu_prompt's int8 tiles when its type and
// shape allow (Q4_K, Q5_K, Q6_K, Q4_TG2; outputs a multiple of 128) and on gpu_ops' exact kernels over the same rounded
// floats otherwise; float weights (F32 / F16 / BF16) take the unrounded input, as in the reference. Routed experts: the
// GPU work list, typed stacks for a usage-tiered weight, the float fold (the reference's sum order differs: rung
// tolerance). The weights stay in the mapped file: each layer's weights and side tables are copied into one device arena
// reused layer by layer (the 22.6 GB model does not fit a 12 GB card; GP3-14's resident bank replaces this for decode).
// Not run here: residual planes, the MTP block, the reference's records and statistics; the dense families and BitNet are
// refused (qwen35moe only in this first cut).
#pragma once

#include <cstdint>
#include <memory>

#include "model/arch_qwen35moe.h"
#include "pelias/status.h"

namespace pelias::model {

struct GpuForwardStats {
  double seconds = 0.0;             // the whole call
  double upload_seconds = 0.0;      // host -> device copies of the weights (the arena fills)
  std::uint64_t bytes_uploaded = 0;
};

class GpuForward {
 public:
  // max_tokens: the largest batch prefill() takes (sizes every buffer once; nothing is allocated per call)
  static Result<std::unique_ptr<GpuForward>> open(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, int max_tokens);
  ~GpuForward();
  GpuForward(const GpuForward&) = delete;
  GpuForward& operator=(const GpuForward&) = delete;
  // the forward_reference options whose roundings this path matches value for value: act_int8 "all" on the gpu path in
  // the per-32 format (gpu256x8), planes off
  static ForwardOptions matching_reference_options();
  // tokens[0..n) at positions 0..n-1 (fresh state); logits of positions [out_begin, out_end) as forward_reference writes
  // them ((out_end - out_begin) rows of n_vocab)
  Status prefill(const std::int32_t* tokens, int n, int out_begin, int out_end, float* logits, GpuForwardStats* st = nullptr);
  // device bytes held (arena, output layer, activations, scratch)
  [[nodiscard]] std::uint64_t device_bytes() const;

  struct Impl;

 private:
  explicit GpuForward(std::unique_ptr<Impl> p);
  std::unique_ptr<Impl> p_;
};

}  // namespace pelias::model
