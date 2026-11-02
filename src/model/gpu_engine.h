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
  // v49 (accuracy tail step 2a): copy each reader step's router logits to pinned host memory and keep, per position, every
  // layer's router margins (logit 8th - 9th pick: the edge of the picks; logit 1st - 2nd); read with router_margins()
  bool router_margins = false;
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
  // v26: prefetch for the reader cache: after each layer, the next layer's router on this layer's output predicts its
  // picks; a second stream copies the predicted file experts into cache slots while the next layer's attention / DeltaNet
  // block runs; that layer's reader waits for it. Needs reader_cache > 0.
  bool prefetch = false;
  // v27 plane test (author 21:25-21:27): each token's lowest-ranked plane_tail routed picks read their Q4_TG2 rows as a 2-bit
  // base plane (each code's value replaced by the mean of the four sharing its top two bits). Lossy: for the accuracy gate
  // (the bytes read do not change yet). 0: off.
  int plane_tail = 0;
  // v28 (author 22:02-22:27, the author's split): with the reader, light experts (every matrix Q4_TG2, not in the bank) are
  // computed by the CPU from the unpinned file through the doorbell (cpu_threads workers), while the GPU reads the heavy
  // file experts and the bank; only the heavy experts are pinned. The CPU's sum is added after the reader.
  bool cpu_light = false;
  // v46a-1 (THE PLAN 46a, author 2026-11-01 17:55 "Get us to e"): with the reader, each layer's experts run on R3 (one
  // cooperative launch: gate and up of every pick, silu x up rounded to int8 per 32, down, with per-expert counters
  // between them) instead of the reader v3 kernels; the same grouping, cache and combine. Off: the reader v3 kernels.
  bool r3_experts = false;
  // v46a-2: how the stage loads expert rows: 0 each lane its own 16-byte pieces (v46a-1), 1 a warp copies its task's rows
  // contiguously into shared memory first for experts read from RAM, 2 the same for every expert (outputs equal in all three)
  int rx_load = 0;
  // v48 (author 20:13: the CPU only at 0 cost): with the expert reader on, step() runs a prompt of more than 4 tokens as
  // steps of up to 4 on the GPU-only path (reader / R3), so no expert pick of the prompt goes to the CPU; false = the
  // prompt's batches take the bank + CPU cold path as before (--cpu-prompt). v48b: the prompt's steps use R3 (weights once
  // for the step's tokens, each token in its one-token order) whether or not batch_identical is set
  bool gpu_prompt = true;  // v49e (author 21:09-21:13): the GPU prompt stays the default and is fixed forward; --cpu-prompt (temporary, for the diagnosis)
  // v49d: the GPU prompt's steps on R3 (true, v48b) or on the batched kernels (false, v48's plain path; --prompt-batched)
  bool prompt_r3 = true;
  // v50: the GPU prompt's R3 products take int16-level inputs as two int8 halves (hi + lo / 256, one load of the weights);
  // decode steps unchanged. MEASURED v49e: the int8 prompt's state cost W2's decode rows +65 % KL (--no-prompt-halves, temporary)
  bool prompt_halves = true;
  // v28: gate / up rows per warp in the reader v3 kernels (1 or 2; 2 = v26's shape)
  int reader_rows_per_warp = 2;
  // v28: pinned ranges are merged across gaps smaller than this (bytes; v27 used 8 MiB; 0 = exact ranges, one per matrix
  // run)
  std::uint64_t pin_gap = std::uint64_t{8} << 20;
  bool pin_small_pages = true;  // v34: madvise small pages on the ranges registered for the GPU (Linux; see hal/host_alloc_pages.h)
  // v40 (author 02:30, CPU turn of the split): gate/up rows per work item (64 = before; smaller tiles, less tail), the
  // two-row down kernel for one-token steps, and a software prefetch of the rows ahead. All exact (same per-row math).
  int cpu_up_rows = 64;
  bool cpu_down2 = false;
  bool cpu_prefetch = false;
  bool cpu_gu2 = false;
  bool cpu_tile = false;    // v45b: gate/up pairs and down row pairs as one call per block (grids loaded once), bit for bit
  bool spin_pause = false;  // v45b: a pause in every spin (Hyper-V pause-loop A/B); the counter padding is always on    // v45: gate row r and up row r in one call (one-token steps), bit for bit
  bool cpu_check = false;  // compare the two-row down against the one-row path, row by row (bit for bit); counts mismatches
  // v43: pin the CPU threads: pin_pool[0] = the doorbell thread (it works as the pool's first thread), pin_pool[1..] = the
  // pool workers in order, pin_host = the thread that drives the engine (-1: leave it). Empty: no pinning (as before).
  std::vector<int> pin_pool;
  int pin_host = -1;
  // v28b: the split's CPU side through v20's per-stage jobs (the pool sleeping between them) instead of the one-dispatch path
  // with the pool held hot; for comparison
  bool split_slow = false;
  // v30 (author 00:04-00:05, "open the door"): after start-up a background thread touches every page of the light experts
  // once (in place in the mapped file, no copy), so the CPU's split work stops paying the drive bridge's page faults
  bool preload_light = false;
};

struct EngineStats {
  double seconds = 0.0;      // the whole step
  double cold_seconds = 0.0; // the CPU's cold-expert work (inside the step; overlaps the GPU's bank experts)
  double wait_seconds = 0.0; // host waiting on the device (router picks, the cold sum's hand-back)
  std::uint64_t hot_picks = 0, cold_picks = 0;
  std::uint64_t cold_bytes = 0;  // v28: bytes of the cold experts the CPU computed (gate + up + down matrices, each pick)
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
  std::vector<double> cpu_picks, cpu_groups;  // v39: the split path's CPU picks and expert groups per layer (summed over steps)
  bool split_stamps = false;                  // v39: stage[] / busy[] came from the split path (gate/up incl. silu, down incl. combine)
  // v40: per pool thread (index by first use, up to 32): item time in the gate/up and down phases; the check's mismatches
  double thr_up[32] = {}, thr_dn[32] = {};
  int thr_n = 0;
  std::uint64_t down2_rows = 0, down2_mismatch = 0;
  std::uint64_t gu2_rows = 0, gu2_mismatch = 0, round_n = 0, round_mismatch = 0;  // v45 checks
  std::uint64_t gx_n = 0, gx_mismatch = 0;  // v45b: GPU rounded input vs the CPU's
  double lag_sum = 0.0, lag_max = 0.0;  // v45: each thread's first down item minus the end of gate/up (seconds)
  std::uint64_t lag_n = 0;
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
  // v49: per position 0 .. max_ctx-1, 2 x n_layer floats (layer il: [2 il] = 8th - 9th router logit, [2 il + 1] = 1st - 2nd);
  // have[pos] = 1 where a reader step filled it (EngineOptions::router_margins); false if off
  bool router_margins(std::vector<float>* m, std::vector<std::uint8_t>* have) const;
  // v26: the graph's state: "<note>; captures C, replays R"
  std::string graph_report() const;
  // v30: wait for the light experts' pre-load (if any); its seconds and bytes; false if none was started
  bool preload_wait(double* seconds, std::uint64_t* bytes);
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
