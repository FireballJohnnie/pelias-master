// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// gpu_engine.h - GP3-14 (masterwork step 10): the qwen35moe engine for decode and the GPU prompt on one GPU.
//   GPU, resident for the whole session: every dense weight of the trunk (attention, DeltaNet, router, shared expert,
//   norms) and the output layer, each held once in R3's layout; the hot-expert bank in bit planes (the slider's bits);
//   the reader cache (size pools); the attention K / V cache (max_ctx positions) and the DeltaNet state and conv tails.
//   Host RAM, pinned: the experts outside the bank, as bit planes (Q4_TG2 / Q5_K) or registered file ranges (the rest),
//   read by the GPU over PCIe into cache slots as they are picked.
// Every step is 1-4 tokens (decode 1, the MTP verify 2-4, a prompt as steps of 4): per layer the dense half (L3 for one
// token, R3 products otherwise), then the MoE half in one cooperative launch (post norm, router, picks, grouping, plan, the
// R3 expert stage with the shared expert folded in, combine, residual). Nothing runs on the CPU inside a step but the
// launches; the roundings are the reference's (GpuForward::matching_reference_options), the sum orders differ (rung
// tolerance; real-model gate: engine-check --float-ref). The step is recorded once per kind as a CUDA graph and replayed.
// v60 (2026-11-04): the paths the product never ran left the tree (the CPU cold path and its pool, the doorbell, the author's
// split, prefetch, the plane tail, the reader v3 kernels, the int8 bank stacks and tables for the batched kernels, the
// timelines); the A/B switches that were measured are locked. qwen35moe only.
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
  int max_batch = 64;   // tokens per step() call; a prompt runs as steps of up to 4 tokens on the GPU (v48)
  // device bytes the engine may take; 0: the free memory at open minus vram_reserve
  std::uint64_t vram_budget = 0;
  std::uint64_t vram_reserve = std::uint64_t{512} << 20;
  // the hot bank: per layer, experts in priority order (most used first). The bank is filled across layers rank by rank
  // (every layer's first expert, then every layer's second, ...) until the budget or max_hot_per_layer is reached.
  std::vector<std::vector<int>> hot_order;
  // v51: per layer, each expert's pick count from the routes file (empty: none); sizes the cache's size pools
  std::vector<std::vector<std::uint64_t>> hot_count;
  // v54b sliding bit planes (endgame; author 23:14, 23:54 "a slider ... 2, 3 or 4 bit", 23:56): plane_bits 2, 3 or 4 puts
  // the RAM experts' Q4_TG2 matrices in bit planes (kern/tg2_planes.h; pinned at open, their file bytes not registered) and
  // the R3 expert stage reads every pick with that many code bits; the cache's slots hold exactly those bits, so their size
  // follows the slider (2-bit -27 %, 3-bit -14 % of an average expert, DERIVED) and more experts fit. v54c (author 00:22-00:25:
  // "more experts on the vram"): Q5_K matrices too (5-bit codes), and the bank holds the slider's bits as well (a planed
  // bank: matrices alone, no batched-kernel tables). Each matrix reads min(plane_bits, its own bits); 5 = every code as
  // stored, bit for bit. v60: always on (2-5; the stacked int8 bank is gone). (--plane-bits)
  int plane_bits = 5;
  // v54b VRAM probe (author 23:55: any system, any MoE): after the dense weights, the free VRAM (minus vram_reserve) is split
  // between the reader cache (cache_share of it, as whole slots of the largest expert) and the bank (the rest, no per-layer
  // cap) instead of reader_cache / max_hot_per_layer. cache_share 0.435 = this machine's measured best (bank 4.43 GB,
  // cache 3.41 GB: bank 40 + cache 33, v46 step 1 / v51b). (--auto-vram [--cache-share F])
  bool auto_vram = false;
  double cache_share = 0.435;
  int max_hot_per_layer = 256;
  // v26: VRAM cache of file experts for the reader, slots per layer (0: none; auto_vram sets it). Its bytes come out of the
  // bank's budget before the bank is filled; misses are written into the least recently used slot as the reader stages them.
  int reader_cache = 0;
  // GP3-14 stage 3 (MTP drafting): the largest verify batch (1 + drafts); 0 = no rollback buffers. Needs the DeltaNet
  // d_state a power of two (the checkpointing recurrence). mtp: run the file's MTP block on the GPU too (v58: its MoE half on
  // the fused launch as the reader's row n_layer, its experts in bit planes / the file mapping; its own attention cache).
  int max_verify = 0;
  bool mtp = false;
  // v60 commit C: extra VRAM for the MTP row's reader cache in MiB (its experts are read on every draft and catch-up step
  // and are not plane types). 0: the activation bytes the v60b layout held for unreachable rows (act_freed, logged); more
  // comes out of the bank (an A/B knob, --mtp-cache-mb)
  int mtp_cache_mb = 0;
  // v60 E: the draft's head through a sketch (kern/tg2_planes.h g_q6_ksk): 2 = the Q6_K head's top two code planes with its
  // scales (~38 % of the head's bytes, 159 MB for this model, taken before the bank is sized), the top kHeadSketchTop rows
  // re-scored by the full product's own kernel. 0: the full head per draft. Only one-token drafts with a device argmax take
  // it; verify, the catch-up steps and any caller that reads logits run the full head unchanged. check: the full head runs
  // too and the sketch's recall (its top rows hold the full head's argmax) is counted per draft (EngineMtpTiming)
  int mtp_head_sketch = 0;
  bool mtp_head_sketch_check = false;
  // v60 F2: the bits the MTP row's plane matrices are packed and read with (0: the slider's plane_bits). The drafter has no
  // verify behind it and feels a lost bit directly (v60n MEASURED: 3 of 4 bits kept 84.1 / 72.1 % vs 86.8 / 74.5 % at 4), and
  // its planes are ~1/41 of the whole, so the default keeps it at 4. Not for a Q4_TG2 MTP block (one decode grid per launch)
  int mtp_plane_bits = 4;
  // v60 H (Fix 4 A/B, item 5): the one-token fused MoE half at __launch_bounds__(256, 2) (moe_blocks 2: registers capped at 128,
  // 40 B of spill at compile time, two blocks per SM; the copy pipeline off, since two blocks' buffers do not fit in 100 KB of
  // shared memory) against the (256, 1) default; rx_pipe 0 turns the v57 copy pipeline off on the default kernel (the A/B's other
  // half: what the pipeline is worth today). Both are measurement knobs; the loser leaves (one engine path).
  int moe_blocks = 1;
  int rx_pipe = 1;
  // v60 commit C2: the pinned-planes read probe at open (copy engine, GB/s, logged). pin_read_min_gbs > 0: below it the
  // planes are freed, pinned and packed again, up to pin_read_tries times (v60d: three of five arms read them 2-3x slower)
  double pin_read_min_gbs = 0.0;  // v60 C3: applies to the SM one-line-per-page read (GB/s of 128 B lines)
  int pin_read_tries = 3;
  // v24 tie rule (author 18:19): every router call orders experts by floor((logit - max) / router_tie, lower index first
  // within a band (router_topk's tie_band; the CPU reference's ForwardOptions::router_tie_band). 0: by probability.
  float router_tie = 0.0f;
  // v26: the layers of a step (1-4 tokens) recorded once as a CUDA graph per (tokens, DeltaNet tail parity, verify
  // checkpoints) and replayed (one launch instead of ~1,200). The engine then runs on its own stream; the position is read
  // on the device (rope, attention, the K / V store), so a replay serves any position. The first step of each kind runs
  // plainly (it builds the plans), the second is captured. Falls back to plain launches if capture fails (summary()).
  bool graph = false;
  // v55b verbose GPU probe (author 03:24 "full verbose mode"): each kernel's registers / local memory / blocks per SM at open,
  // and GPU-clock stamps inside the fused MoE half (start, after phases A, B, C, block 0's end) summed over the one-token decode
  // steps (moe_phase_report); no extra launches, so the fused path is the one measured
  bool verbose_gpu = false;
  // v25 trace (author 19:15): for every step, per layer the picks, the experts read (once per expert) and their bytes from
  // VRAM / the file, and the stage's launch times (GPU stamps), and per step the wall time; read with trace_take(). Off by
  // default (stamps cost launches; a layer with a side output runs the separate kernels, not the fused half).
  bool trace = false;
  // v49 (accuracy tail step 2a): copy each step's router logits to pinned host memory and keep, per position, every layer's
  // router margins (logit 8th - 9th pick: the edge of the picks; logit 1st - 2nd); read with router_margins(). Same note.
  bool router_margins = false;
  // v28: pinned ranges are merged across gaps smaller than this (bytes; v27 used 8 MiB; 0 = exact ranges, one per matrix
  // run)
  std::uint64_t pin_gap = std::uint64_t{8} << 20;
  bool pin_small_pages = true;  // v34: madvise small pages on the ranges registered for the GPU (Linux; see hal/host_alloc_pages.h)
  // v58 (bloat audit, author 15:00 "okay 1-3"): the measured fast path as one set, the starting point of engine-check and
  // generate --engine. v60: the switches that were A/B'd and measured are locked in the engine itself (the picked-expert
  // reader with the R3 expert stage, rows staged for RAM experts with the v57 copy pipeline and shared activations, the shared
  // expert folded in, the fused MoE half, the F32 fast products, the size-pooled cache, the GPU prompt on R3, every 1-4-token
  // step with the weights read once for its rows, the L3 dense half for one token, the lean output head, the MTP block on
  // the fused launch); the CPU cold path, the doorbell, the split, prefetch, the plane tail and the int8 bank left with them.
  static EngineOptions production() {
    EngineOptions o;
    o.router_tie = 0.01f;
    o.max_hot_per_layer = 40;
    o.graph = true;
    o.auto_vram = true;
    return o;
  }
};

struct EngineStats {
  double seconds = 0.0;         // the whole step
  std::uint64_t hot_picks = 0;  // every pick runs on the GPU (v24); counted for the summaries
};

// v20: where an MTP greedy run spends its time (engine_generate_greedy_mtp): drafting (the MTP block, one token per
// draft), the trunk's verify by batch size (index 1 .. 8), keep(), the MTP block's catch-up on the kept tokens, and the
// host's argmax over the vocabulary
struct EngineMtpTiming {
  double draft_s = 0.0, keep_s = 0.0, catchup_s = 0.0, argmax_s = 0.0, prompt_s = 0.0;
  // v60 D: GPU time (globaltimer stamps) of the MTP block's steps, split at the head: the drafts, the catch-up steps, and
  // the catch-up over the prompt (the "prompt tax"). total >= layer + head (the rest: the input gather / norms / eh_proj)
  double draft_gpu_s = 0.0, draft_layer_s = 0.0, draft_head_s = 0.0;
  double catchup_gpu_s = 0.0, catchup_layer_s = 0.0, catchup_head_s = 0.0;
  double prompt_mtp_gpu_s = 0.0, prompt_mtp_layer_s = 0.0, prompt_mtp_head_s = 0.0;
  double verify_s[10] = {};
  int verify_n[10] = {};
  int drafts = 0;
  // v60 E: drafts whose argmax went through the head sketch, and (with EngineOptions::mtp_head_sketch_check) how many of
  // them had the full head's argmax among the sketch's top rows
  std::uint64_t sketch_drafts = 0, sketch_recall_hit = 0, sketch_recall_n = 0;
  // v60 E2: the sketch chain's GPU time split over the drafts (stamps inside the chain): the sketch product, the top-k select,
  // the gather + re-score + argmax + pick
  double sketch_dot_s = 0.0, sketch_select_s = 0.0, sketch_rescore_s = 0.0;
  // v60 H: the drafts' block split (stamps inside the block): 0 the input (embedding gather, cat, norms), 1 eh_proj, 2 the
  // attention, 3 the MoE half (sums over the drafts; v60n-p MEASURED the whole block swinging 0.40-0.69 ms per draft)
  double draft_block_s[4] = {0.0, 0.0, 0.0, 0.0};
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
  // if only those k had run (bit for bit), the attention cache's later rows dropped. v60 G: keep() queues one launch on the
  // stream (every DeltaNet layer at once) and returns without a sync; the next step or host read orders after it.
  Status verify(const std::int32_t* tokens, int n, float* logits, float* hidden_out, EngineStats* st = nullptr);
  Status keep(int k);
  // The MTP block (EngineOptions::mtp): tokens[0..n) at MTP positions mtp_pos() .., token t paired with the hidden row
  // h_prev + t x n_embd (host); as model::mtp_forward. logits (n x n_vocab) and h_out (n x n_embd) may be null.
  Status mtp_step(const std::int32_t* tokens, const float* h_prev, int n, float* logits, float* h_out, EngineStats* st = nullptr);
  void mtp_truncate(std::int64_t pos);
  // v56: the same steps with each row's argmax computed on the device (first index of the largest logit, as a host scan):
  // top gets one int per row the logits would have had (n, or 1 with last_only); no logits are copied to the host
  Status step_top(const std::int32_t* tokens, int n, std::int32_t* top, bool last_only, EngineStats* st = nullptr, float* hidden_out = nullptr);
  Status verify_top(const std::int32_t* tokens, int n, std::int32_t* top, float* hidden_out, EngineStats* st = nullptr);
  Status mtp_step_top(const std::int32_t* tokens, const float* h_prev, int n, std::int32_t* top, float* h_out, EngineStats* st = nullptr);
  [[nodiscard]] std::int64_t mtp_pos() const;
  // v60 commit C: the MTP greedy loop keeps the hidden rows, the draft chain and the MTP block's input on the device
  // (engine_generate_greedy_mtp reaches the engine's device buffers; no API for host callers changed)
  friend Status engine_generate_greedy_mtp(GpuEngine& eng, const Qwen35MoeSpec& s, const std::vector<std::int32_t>& prompt, int n_predict, int n_draft,
                                           const std::vector<std::int32_t>& stops, std::vector<std::int32_t>& out, MtpStats* stats, EngineStats* es,
                                           const std::function<void(std::int32_t)>& on_token, EngineMtpTiming* tm);
  [[nodiscard]] std::uint64_t device_bytes() const;
  [[nodiscard]] int max_batch() const;
  [[nodiscard]] int bank_experts() const;      // experts held on the GPU (all layers)
  [[nodiscard]] std::uint64_t bank_bytes() const;
  [[nodiscard]] std::string summary() const;   // one line: device bytes, dense, bank, cache, threads
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
  // v55b: with EngineOptions::verbose_gpu, the fused MoE half's phases per token (ms, decode one-token steps); empty if off
  std::string moe_phase_report() const;
  std::string mtp_path_report() const;  // v58: MTP block steps on the fused launch vs the old path

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
Result<std::vector<std::vector<int>>> hot_order_from_routes(const std::string& path, std::uint32_t n_layer, std::uint32_t n_expert,
                                                            std::vector<std::vector<std::uint64_t>>* counts = nullptr);

}  // namespace pelias::model
