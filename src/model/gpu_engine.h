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
  // v21: the per-operation timeline (implies timeline): a GPU stamp between the operations of each layer of a doorbell
  // step (one extra one-thread launch per stamp, so it slows the step a little; a diagnostic pass only)
  bool timeline_ops = false;
  // v24 tie rule (author 18:19): every router call orders experts by floor((logit - max) / router_tie, lower index first
  // within a band (router_topk's tie_band; the CPU reference's ForwardOptions::router_tie_band). 0: by probability.
  float router_tie = 0.0f;
  // v24 picked-expert reader: steps of 1-4 tokens (decode, the MTP verify) run every routed expert on the GPU, reading the
  // router's picks on the device; bank experts from VRAM, the rest from the model file's mapping registered for the GPU
  // (host_register). No doorbell, no CPU cold path in those steps. Larger steps keep the bank + doorbell path. Falls back to
  // off (said in summary()) if the mapping cannot be registered or an expert weight cannot be read by the kernel.
  bool cold_on_gpu = false;
  // v25 trace (author 19:15): for every reader step, per layer the picks, the experts read (once per expert) and their
  // bytes from VRAM / the file, and the reader's four launch times (GPU stamps), and per step the wall time; read with
  // trace_take(). Off by default (stamps cost launches).
  bool trace = false;
  // v26 (author 20:39 "don't hold back"): the reader v3 kernels (Q4_TG2 grid per lane in shared memory, two rows staged
  // per warp, half-warp down rows; bit for bit with v25's reader); false: v25's reader
  bool reader_v3 = true;
  // v26: VRAM cache of file experts for the reader, slots per layer (0: none). Its bytes come out of the bank's budget
  // before the bank is filled (the bank's least-used experts make room); misses are written into the least recently used
  // slot as the reader stages them.
  int reader_cache = 0;
  // v26: the shared expert as one more pick of the reader (its weights in VRAM; its sigmoid gate as the weight, added after
  // the routed picks). Needs reader_v3 and the shared expert's shape equal to a routed expert's.
  bool fold_shared = false;
  // v26 option B (author 20:20-20:39): steps of 2-4 tokens with the reader run every quantized product through R3 with the
  // weights read once for all the tokens and each token's arithmetic the one-token path's, so a verify batch's rows equal
  // the same tokens decoded one at a time bit for bit (no split) while sharing the weight reads. false: batched kernels.
  bool batch_identical = false;
  // v26: the layers of a reader step (1-4 tokens) recorded once as a CUDA graph per (tokens, DeltaNet tail parity, verify
  // checkpoints, op stamps) and replayed (one launch instead of ~1,200). The engine then runs on its own stream; the position
  // is read on the device (rope, attention, the K / V store), so a replay serves any position. The first step of each kind
  // runs plainly (it builds the plans), the second is captured. Falls back to plain launches if capture fails (summary()).
  bool graph = false;
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
  // v20: per step size (index n for n = 1 .. 8, 9 for larger batches): steps, wall time, GPU busy (dense + bank + shared,
  // summed over layers), GPU idle waiting for the CPU, CPU cold work, CPU waiting for the ring
  double by_n_steps[10] = {}, by_n_wall[10] = {}, by_n_gpu_busy[10] = {}, by_n_gpu_idle[10] = {}, by_n_cpu_cold[10] = {}, by_n_cpu_wait[10] = {};
  // v21 per-operation timeline (EngineOptions::timeline_ops), one-token steps only: GPU seconds per segment, summed over
  // steps, [0] DeltaNet layers, [1] attention layers; segment k = from stamp k - 1 to stamp k (kOpNames)
  static constexpr int kOps = 16;
  double ops[2][kOps] = {};
  int ops_steps = 0;
};
// the per-operation segments (v21), in layer order
inline const char* const kOpNames[EngineTimeline::kOps] = {
    "-", "attn/ssm norm", "input products (R3)", "small ops (gates, conv, l2 / q-k norm, rope)", "core (recurrence / attention)",
    "gated norm", "output product", "residual + post norm", "router + top-k", "picks to host + ring", "expert input rounding",
    "routed experts (bank / reader)", "shared expert + add", "wait for the CPU", "cold sum add", "residual"};

// v20: where an MTP greedy run spends its time (engine_generate_greedy_mtp): drafting (the MTP block, one token per
// draft), the trunk's verify by batch size (index 1 .. 8), keep(), the MTP block's catch-up on the kept tokens,
// promotion, and the host's argmax over the vocabulary
struct EngineMtpTiming {
  double draft_s = 0.0, keep_s = 0.0, catchup_s = 0.0, promote_s = 0.0, argmax_s = 0.0, prompt_s = 0.0;
  double verify_s[10] = {};
  int verify_n[10] = {};
  int drafts = 0;
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
  // v22 bank self-test: every bank expert of every layer through the GPU's bank kernels (1 token, and 2 tokens picking it)
  // against the CPU's cold-expert path on the same inputs, then `combos` random 8-pick mixes per layer (1 token). Call
  // between steps. report: a summary line and the worst cases; bad: results with a relative L2 gap above 1e-2.
  Status bank_selftest(int combos, std::string* report, int* bad);
  // v23 probe: arm a capture at absolute position pos (-1: off); the next step that covers it copies, for that row, the
  // residual stream after every layer and every layer's router picks. probe_get: the capture (false if none yet).
  void probe_set(std::int64_t pos);
  bool probe_get(std::vector<float>* x_layers, std::vector<std::int32_t>* picks, std::vector<float>* pick_w);
  // v25: the same capture's per-operation rows, per layer: the attention / DeltaNet block's output (before the residual
  // add, n_embd), the router logits (n_expert) and the MoE block's output (n_embd)
  bool probe_get_ops(std::vector<float>* block_out, std::vector<float>* router_logits, std::vector<float>* moe_out);
  // v25 trace text gathered since the last call (EngineOptions::trace), tab-separated lines
  std::string trace_take();
  // v26: the graph's state: "<note>; captures C, replays R"
  std::string graph_report() const;
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
                                  const std::function<void(std::int32_t)>& on_token = {}, EngineMtpTiming* tm = nullptr);

// The hot order from a routing record (pelias kld --route-dump, PELROUT1): per layer, experts by pick count, most picked
// first (ties: lower index); experts never picked come last in index order.
Result<std::vector<std::vector<int>>> hot_order_from_routes(const std::string& path, std::uint32_t n_layer, std::uint32_t n_expert);

}  // namespace pelias::model
