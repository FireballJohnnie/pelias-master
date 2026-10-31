// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// gpu_engine.h - GP3-14 (masterwork step 10), stages 1-2: the qwen35moe engine for decode and chunked prefill on one
// GPU plus the CPU.
//   GPU, resident for the whole session: every dense weight of the trunk (attention, DeltaNet, router, shared expert,
//   norms) and the output layer; the hot-expert bank (whole experts: gate, up and down, as typed stacks with an expert
//   map); the attention K / V cache (max_ctx positions) and the DeltaNet state and conv tails of every layer.
//   CPU: every routed expert outside the bank (the cold experts), from the mapped file, with kern/cpu_dot's int8 path on
//   the same per-32 rounding (act_q8x32_quantize), on a persistent thread pool (game plan CPU item 1: no thread team per
//   product).
// One step runs n tokens (1 for decode, up to max_batch for a prefill chunk) at positions pos .. pos + n - 1. Per layer:
// the router runs on the GPU; its picks come to the host; the GPU starts the bank's experts and the shared expert while
// the CPU runs the cold picks; the cold sum goes back and is added. The roundings are the reference's
// (GpuForward::matching_reference_options); the sum orders differ (rung tolerance; real-model gate: gpu-check
// --float-ref, DECISIONS_LOG 2026-10-31 00:5x).
// Not yet: MTP drafting, the doorbell (the hand-off is a plain device sync per layer in this first cut), residual planes,
// the chunked DeltaNet kernel for long prompts, the KV page selector. qwen35moe only.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model/arch_qwen35moe.h"
#include "pelias/status.h"

namespace pelias::model {

struct EngineOptions {
  int max_ctx = 4096;   // positions the K / V cache holds
  int max_batch = 64;   // tokens per step (decode 1, prefill in chunks of up to this)
  // device bytes the engine may take; 0: the free memory at open minus vram_reserve
  std::uint64_t vram_budget = 0;
  std::uint64_t vram_reserve = std::uint64_t{512} << 20;
  // the hot bank: per layer, experts in priority order (most used first). The bank is filled across layers rank by rank
  // (every layer's first expert, then every layer's second, ...) until the budget or max_hot_per_layer is reached. Empty:
  // no bank, every routed expert runs on the CPU.
  std::vector<std::vector<int>> hot_order;
  int max_hot_per_layer = 256;
  int cpu_threads = 0;  // workers for the cold experts (0: hardware threads)
};

struct EngineStats {
  double seconds = 0.0;      // the whole step
  double cold_seconds = 0.0; // the CPU's cold-expert work (inside the step; overlaps the GPU's bank experts)
  double wait_seconds = 0.0; // host waiting on the device (router picks, the cold sum's hand-back)
  std::uint64_t hot_picks = 0, cold_picks = 0;
};

class GpuEngine {
 public:
  static Result<std::unique_ptr<GpuEngine>> open(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, const EngineOptions& o);
  ~GpuEngine();
  GpuEngine(const GpuEngine&) = delete;
  GpuEngine& operator=(const GpuEngine&) = delete;
  // back to position 0: empty cache, zero state
  Status reset();
  [[nodiscard]] std::int64_t pos() const;
  // tokens[0..n) at positions pos() .. pos() + n - 1 (n <= max_batch, pos() + n <= max_ctx); logits of every new token
  // (n rows of n_vocab) or, with last_only, of the last one (one row)
  Status step(const std::int32_t* tokens, int n, float* logits, bool last_only, EngineStats* st = nullptr);
  [[nodiscard]] std::uint64_t device_bytes() const;
  [[nodiscard]] int bank_experts() const;      // experts held on the GPU (all layers)
  [[nodiscard]] std::uint64_t bank_bytes() const;
  [[nodiscard]] std::string summary() const;   // one line: device bytes, dense, bank, cache, threads

  struct Impl;

 private:
  explicit GpuEngine(std::unique_ptr<Impl> p);
  std::unique_ptr<Impl> p_;
};

// The hot order from a routing record (pelias kld --route-dump, PELROUT1): per layer, experts by pick count, most picked
// first (ties: lower index); experts never picked come last in index order.
Result<std::vector<std::vector<int>>> hot_order_from_routes(const std::string& path, std::uint32_t n_layer, std::uint32_t n_expert);

}  // namespace pelias::model
