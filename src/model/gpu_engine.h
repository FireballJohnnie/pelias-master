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
// Hand-off (EngineOptions::doorbell): flags in pinned mapped memory; the router's picks and the normed rows are copied
// to pinned memory and the GPU rings; a CPU thread waiting on that flag runs the cold experts into pinned memory and rings
// back; the GPU waits in-stream and reads the cold sum straight from host memory. The MTP block uses the plain syncs.
// Not yet: the CUDA Graph per step (U8), BAR1 write-combining (Windows build), the MTP block's bank, residual planes, the
// chunked DeltaNet kernel for long prompts, the KV page selector. qwen35moe only.
#pragma once

#include <cstdint>
#include <functional>
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
  // GP3-14 stage 3 (MTP drafting): the largest verify batch (1 + drafts); 0 = no rollback buffers. Needs the DeltaNet
  // d_state a power of two (the checkpointing recurrence). mtp: run the file's MTP block on the GPU too (its experts on the
  // CPU path, its own attention cache).
  int max_verify = 0;
  bool mtp = false;
  // the doorbell (author 03:18): the trunk's CPU/GPU hand-offs through flags in pinned mapped memory, a CPU thread that
  // runs each layer's cold experts as its flag rings, the GPU waiting in-stream; a whole step is enqueued without a host
  // wait. false: one device sync per layer (the first cut). Falls back to the syncs if mapped memory is not available.
  bool doorbell = true;
  // the MTP block's bank (author 03:18, option 1: live pick counts, idle-time promotion): experts it can hold; -1 = as
  // many as the trunk's average per layer would get if the bank budget were spread over n_layer + 1 layers (DERIVED at
  // open); 0 = none (the MTP experts stay on the CPU path)
  int mtp_bank = -1;
  // R3 (kern/gpu_stream, the measured decode kernel; author 03:28-03:29): one-token steps run every dense product R3
  // reads (Q7_ZT, Q4_TG2, Q5_K, Q6_K, Q8_0, Q4_K, IQ4_XS; k 256-4096) on R3, grouped by shared input; batches keep the
  // batched kernels. One device copy per weight: Q8_0 in R3's repacked layout, read by the batched kernels as g_q8_0r;
  // Q4_K / Q5_K / Q4_TG2 shared as stored; Q6_K, IQ4_XS and Q7_ZT dense weights (none in the tiered file) keep a second,
  // GGUF copy for the batched kernels. false: no R3.
  bool r3 = true;
  // v19 (author 15:27): per-layer timeline of the doorbell steps: GPU timestamps (globaltimer) at the ring and around the
  // wait, CPU times of the wait and of each cold-path stage, worker wake-up and busy time. Off by default (timing costs).
  bool timeline = false;
};

struct EngineStats {
  double seconds = 0.0;      // the whole step
  double cold_seconds = 0.0; // the CPU's cold-expert work (inside the step; overlaps the GPU's bank experts)
  double wait_seconds = 0.0; // host waiting on the device (router picks, the cold sum's hand-back)
  std::uint64_t hot_picks = 0, cold_picks = 0;
};

// v19 timeline (EngineOptions::timeline), summed over the steps since timeline_reset(). Per layer, GPU clock: dense = from
// the previous layer's release to this layer's ring (norms, attention / DeltaNet, router; layer 0 not counted), hot = from
// the ring to the wait (the bank's experts and the shared expert), idle = the GPU waiting for the CPU's cold sum. CPU
// clock: wait = the cold-path thread waiting for the ring, cold = its cold-expert work. stage[0..4] = the cold path's
// input rounding, gate / up, silu + rounding, down, combine (wall time); busy[] = item time summed over every thread;
// wake = how long the first worker took to start after a gate / up or down job opened (summed over wake_n jobs).
struct EngineTimeline {
  int steps = 0, threads = 0;
  std::vector<double> gpu_dense, gpu_hot, gpu_idle, cpu_wait, cpu_cold;
  double stage[5] = {}, busy[5] = {};
  double wake = 0.0;
  int wake_n = 0;
  double cold_bytes = 0.0;
  std::uint64_t cold_picks = 0, hot_picks = 0;
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
  // hidden_out (optional, n x n_embd): every new token's post-norm hidden (output_norm of the trunk's last state), the MTP
  // block's seed, as ForwardOptions::hidden_out gives it
  Status step(const std::int32_t* tokens, int n, float* logits, bool last_only, EngineStats* st = nullptr, float* hidden_out = nullptr);
  // MTP verify: as step (every row's logits and hidden) with the DeltaNet state of every prefix kept, n <= max_verify;
  // then keep(k) keeps the first k tokens of that batch (1 <= k <= n): position pos0 + k, the state and the conv tails as
  // if only those k had run (bit for bit), the attention cache's later rows dropped.
  Status verify(const std::int32_t* tokens, int n, float* logits, float* hidden_out, EngineStats* st = nullptr);
  Status keep(int k);
  // The MTP block (EngineOptions::mtp): tokens[0..n) at MTP positions mtp_pos() .., token t paired with the hidden row
  // h_prev + t x n_embd (host); as model::mtp_forward. logits (n x n_vocab) and h_out (n x n_embd) may be null.
  Status mtp_step(const std::int32_t* tokens, const float* h_prev, int n, float* logits, float* h_out, EngineStats* st = nullptr);
  void mtp_truncate(std::int64_t pos);
  // idle-time promotion: the MTP block's bank is refilled with its most picked experts so far (picks counted on every MTP
  // step); nothing is uploaded when the set is unchanged. Returns the experts swapped in.
  Result<int> mtp_promote();
  [[nodiscard]] std::int64_t mtp_pos() const;
  [[nodiscard]] std::uint64_t device_bytes() const;
  [[nodiscard]] int max_batch() const;
  [[nodiscard]] int bank_experts() const;      // experts held on the GPU (all layers)
  [[nodiscard]] std::uint64_t bank_bytes() const;
  [[nodiscard]] std::string summary() const;   // one line: device bytes, dense, bank, cache, threads
  // v19 timeline (empty unless EngineOptions::timeline and the doorbell are on)
  [[nodiscard]] const EngineTimeline& timeline() const;
  void timeline_reset();

  struct Impl;

 private:
  explicit GpuEngine(std::unique_ptr<Impl> p);
  std::unique_ptr<Impl> p_;
};

// GP3-14 stage 3: greedy decoding with MTP drafting on the engine (model::generate_greedy_mtp's algorithm: draft up to
// n_draft tokens with the MTP block, verify them with the trunk as one batch, keep a draft while it equals the trunk's own
// greedy choice), the rejected drafts dropped by keep() instead of a re-run. The engine must be open with mtp and
// max_verify >= n_draft + 1; it is reset first. drafted / accepted as MtpStats.
Status engine_generate_greedy_mtp(GpuEngine& eng, const Qwen35MoeSpec& s, const std::vector<std::int32_t>& prompt, int n_predict, int n_draft,
                                  const std::vector<std::int32_t>& stops, std::vector<std::int32_t>& out, MtpStats* stats, EngineStats* es = nullptr,
                                  const std::function<void(std::int32_t)>& on_token = {});

// The hot order from a routing record (pelias kld --route-dump, PELROUT1): per layer, experts by pick count, most picked
// first (ties: lower index); experts never picked come last in index order.
Result<std::vector<std::vector<int>>> hot_order_from_routes(const std::string& path, std::uint32_t n_layer, std::uint32_t n_expert);

}  // namespace pelias::model
