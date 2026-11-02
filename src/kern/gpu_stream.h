// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kern/gpu_stream.h - GP3-10 GPU: R3, the streaming decode kernel (one token, M = 1), moved from the measured probe
// tools/probes/r3/r3.cu into src. One cooperative launch walks a table of matrices (jobs) and streams their rows back to
// back so the memory pipeline stays full across matrix boundaries. The math is the probe's, unchanged (MEASURED on the
// RTX 4070, 2026-10-29, 40 distinct layers: 4.75 ms streaming, 5.685 ms with the real token's barriers):
//   - grid path 2: the Q4_TG2 grid replicated 32 ways in shared memory (bank = lane) and int8 -> float by the magic
//     number (fix 2);
//   - route 2: each warp's job lookup walks forward from its last job (incremental, at most a few L1 probes);
//   - item 7: per-expert counters: a job may signal a group (each of its tasks adds 1) and a later job may wait for a
//     group (its tasks start once every task of that group is done), so a layer's experts and their down projections
//     share one stage; between stages, a counter barrier per warp (fix 6) instead of a grid-wide sync.
// Formats (r3_formats.h, now kern): dense Q7_ZT (fold route, dp4a), expert Q4_TG2 (grid), expert down Q5_K, output head
// Q6_K. The weights are repacked once at upload (Q7_ZT into base / header / mask / H arrays, Q6_K into 208-byte blocks + a
// d array; Q4_TG2 and Q5_K as stored). Added 2026-10-30 for the stock GGUF files (dp4a on the int8 activations): Q8_0
// (repacked qs[256] | d[8] per 256), Q4_K (as stored), IQ4_XS (repacked to 144 bytes, its levels by __byte_perm). Activations: the engine's int8 per 32 with an fp32 scale and the per-32 sum, as
// act_q8x32 (kern/gpu_ops.h) writes them.
// Host API only: no CUDA types here, so host code includes it in every build (the stubs report "no GPU backend").
#pragma once
#include <cstdint>

#include "kern/gpu_ops.h"

namespace pelias::kern::gpu {

// One matrix on the device in the streaming layout. Filled by stream_upload, freed by stream_release.
struct StreamWeights {
  std::uint32_t type = 0;  // quant::Type id: Q7_ZT (202), Q4_TG2 (204), Q5_K (13), Q6_K (14), Q8_0 (8), Q4_K (12) or IQ4_XS (23)
  std::int64_t n = 0, k = 0;
  void* w = nullptr;     // Q7_ZT base (224 bytes per 256) / Q4_TG2, Q5_K, Q4_K as stored / Q6_K 208-byte blocks / Q8_0 qs[256] | d[8] / IQ4_XS 144-byte blocks
  void* hdr = nullptr;   // Q7_ZT: u64 per 256 (fp16 d | eight 6-bit levels << 16)
  void* mask = nullptr;  // Q7_ZT: u8 per 256 (the gated 32-blocks)
  void* hb = nullptr;    // Q7_ZT: two u32 per 256 (the H words of the gated blocks)
  void* d6 = nullptr;    // Q6_K: fp16 d per 256
  std::uint64_t bytes = 0;  // device bytes of the matrix (what one token reads)
};

// The seven types, and k one of 256, 512, 1024, 2048, 4096 (a row is split over k / 128 lanes of a warp)
bool stream_type_ok(std::uint32_t type, std::int64_t k);
// rows: n rows of k values in GGUF storage (row bytes = quant::row_bytes(type, k)), host memory; repacked and copied
bool stream_upload(std::uint32_t type, const void* rows, std::int64_t n, std::int64_t k, StreamWeights* out);
void stream_release(StreamWeights* w);
// why the last stream_* call or StreamPlan call failed
const char* stream_last_error();

// One matrix-vector product of the token: y[r] = sum_i W[r][i] * q[i] * scale[i / 32]
struct StreamJob {
  const StreamWeights* w = nullptr;
  const std::int8_t* xq = nullptr;  // k int8 values (16-byte aligned)
  const float* xs = nullptr;        // k / 32 scales
  const int* xsum = nullptr;        // k / 32 sums of the int8 values
  float* y = nullptr;               // n outputs
  int signal = -1;                  // group this job's tasks count into, -1 none
  int wait = -1;                    // group that must be complete before this job's tasks start, -1 none
  // v26 (option B): tokens of the job (1-4) and their strides (0: k bytes, k / 32 scales, n outputs). A row's weights are
  // read once per warp and used for every token; each token's arithmetic is the one-token job's, bit for bit.
  int ntok = 1;
  std::int64_t xq_stride = 0, xs_stride = 0, y_stride = 0;
};

// A token's table, built once (the device copy is kept) and run any number of times. stage_end[s] is one past the last
// job of stage s (the last equals njobs); stages run in order with a barrier between them. A job's wait group must be
// signalled only by jobs before it in the table (no deadlock: tasks are taken in table order on co-resident blocks).
class StreamPlan {
 public:
  StreamPlan() = default;
  ~StreamPlan();
  StreamPlan(const StreamPlan&) = delete;
  StreamPlan& operator=(const StreamPlan&) = delete;

  bool build(const StreamJob* jobs, int njobs, const int* stage_end, int nstages, int ngroups);
  // Runs of one plan go on one stream, one after another (they share the plan's counters).
  bool run(Stream s = nullptr);           // the token: stages, barriers, group waits (one cooperative launch)
  bool run_unstaged(Stream s = nullptr);  // every job at once, no barriers or waits (independent jobs only: the read ceiling)
  bool run_reset(Stream s = nullptr);     // v26: as run, the counters cleared on the stream first (graph capture)

  int jobs() const { return njobs_; }
  long long tasks() const { return ntasks_; }
  std::uint64_t bytes() const { return bytes_; }  // weight bytes one run reads
  // v55g (L3): the device job table (for a fused launch that runs this plan's tasks as one of its stages) and the one format
  // every job has (one-token plans only; -1: mixed formats or several tokens)
  const void* jobs_dev() const;
  int fmt1() const { return fmt1_; }

 private:
  void* dev_ = nullptr;           // device copy: job table, stage ends (in tasks), groups, counters
  int njobs_ = 0, nstages_ = 0, ngroups_ = 0;
  long long ntasks_ = 0;
  std::uint64_t bytes_ = 0;
  unsigned long long epoch_ = 0;  // runs so far: the counters are never reset, run e waits for e x the stage's tasks
  int grid_ = 0;                  // blocks of the cooperative launch (occupancy x SMs)
  bool mt_ = false;               // v47: a job has 2-4 tokens: the kernel that loads each row's weights once for all of them
  int only_ = -1;                 // v55d: every job one format (Q8_0 or Q6_K) and one token: run_unstaged uses that format's lean kernel
  int grid_lean_ = 0;             // its blocks (its own occupancy x SMs)
  int fmt1_ = -1;                 // v55g: every job's one format (one token), whatever the lean switch
  std::uint64_t off_jobs_ = 0, off_stage_ = 0, off_grp_ = 0, off_dep_ = 0, off_gtasks_ = 0, off_done_ = 0, off_gdone_ = 0;
};

// v46a-1 (THE PLAN, 46a): the routed and shared experts of one MoE layer on R3, in one cooperative launch. moe_reader3's
// grouping picks each expert's source (bank, cache slot, file mapping) and on a miss the cache slot it is written into;
// a small planning kernel turns the groups into R3 jobs on the device: per expert gate and up (signal the expert's
// counter), silu(g) x u rounded to int8 per 32 (the rule of the reader's silu_round and act_q8x32, waits for gate and up),
// down (waits for that); then the reader's combine. Experts read as stored: Q4_TG2, Q5_K, Q4_K, Q6_K (210-byte blocks)
// and R3's repacked Q8_0 (the shared expert); a miss's bytes are copied into its slot by the lanes that load them.
// Inputs: the int8 activations act_q8x32 wrote for xn (q, scale, sum; H per token). Scratch from rx_scratch_bytes.
bool rx_type_ok(std::uint32_t type, std::int64_t k);  // an expert matrix the stage reads (gpu_decode.h g_* ids)
std::uint64_t rx_scratch_bytes(int max_picks);         // the plan buffer for up to max_picks picks (routed + shared)
bool moe_r3(const ExpertPtr* tab, const ExpertPtr* shx, const std::int32_t* sel, const float* wsel, const float* gsh, int n_used,
            const std::int8_t* xq, const float* xs, const int* xsum, std::int64_t tokens, std::int64_t H, std::int64_t F, const ReaderCache* cache,
            float* gu, std::int8_t* hq, float* hs, int* hsum, float* yd, int* grp, ExpertPtr* gsrc, ExpertPtr* gfill, void* plan, int max_picks,
            float* y, unsigned long long* ts, Stream s = nullptr, int load_mode = 0);
// v52 (token kernel 3a): the MoE half of a layer in one cooperative launch, the same operations and order as the separate
// kernels it replaces, so the same bits: x += a (residual), post rms_norm, the 8-bit rounding (act_q8x32), the router's and
// the shared gate's F32 products, the router's picks (k_router_fast), the shared gate's sigmoid, the grouping with the reader
// cache, the plan, the R3 expert stage, the combine into y, x += y. Three grid barriers instead of twelve launches. Every
// block normalizes the rows itself (reads only) for its share of the products; x is written once, at the end.
struct MoeHalfArgs {
  float* x = nullptr;              // n x H residual stream (in: before the attention's output is added; out: after the layer)
  const float* a = nullptr;        // n x H the attention / DeltaNet block's output (may alias y)
  const float* post_w = nullptr;   // H post-attention norm weight
  float eps = 0.0f;
  int n = 0, H = 0, F = 0, ne = 0, nu = 0;
  float* xn = nullptr;             // n x H normalized rows (written for later readers)
  float* xr = nullptr;             // n x H rounded floats, as act_q8x32
  std::int8_t* xq = nullptr;       // n x H
  float* xs = nullptr;             // n x H / 32
  int* xsum = nullptr;             // n x H / 32
  const float* wr = nullptr;       // ne x H router (F32, rows of H x 4 bytes, 16-byte aligned)
  float* logits = nullptr;         // n x ne
  const float* wsh = nullptr;      // H shared-expert gate (F32); null: no shared expert folded in
  float* gsh = nullptr;            // n
  float wscale = 0.0f, band = 0.0f;
  std::int32_t* sel = nullptr;     // n x nu
  float* wsel = nullptr;           // n x nu
  const ExpertPtr* tab = nullptr;  // ne
  const ExpertPtr* shx = nullptr;  // the shared expert (with wsh)
  ReaderCache cache;               // K 0: none
  int* grp = nullptr;
  ExpertPtr* gsrc = nullptr;
  ExpertPtr* gfill = nullptr;
  float* gu = nullptr;
  std::int8_t* hq = nullptr;
  float* hs = nullptr;
  int* hsum = nullptr;
  float* yd = nullptr;
  void* plan = nullptr;
  int max_picks = 0, load_mode = 0;
  float* y = nullptr;              // n x H the MoE output (as moe_r3's y)
  // v55f (the 1-bit route): null, or the expert inputs' bit planes (n x H / 4 words: 8 per 32 values) and the down inputs'
  // (max_picks x F / 4 words); written by the MoE half itself (rounding, silu x up), read by the planed experts' decode
  std::uint32_t* xb = nullptr;
  std::uint32_t* hb = nullptr;
  unsigned long long* ph = nullptr;  // v55b: null, or 5 GPU-clock stamps by block 0 (start, after phases A, B, C, end)
  // v55e3 diagnosis (temporary): null, or kRxStatN device counters the one-token expert stage adds to (a counting kernel):
  // 0 warp cycles waiting for a group, 1 cycles running tasks, 2 tasks, 3 the warps' loop spans, 4 warps, 5 staged tasks,
  // 6 filling tasks, 7 cycles of staged tasks, 8 + f cycles of format f (DJob fmt 0..10, 11 other), 20 + f tasks of format f
  unsigned long long* st = nullptr;
};
constexpr int kRxStatN = 32;
bool moe_half(const MoeHalfArgs& args, Stream s = nullptr);
// v55g (L3, the author's order 03:13; game plan "Lowering the launches per token"): the dense half of a DeltaNet layer for
// one token in one cooperative launch, the same operations and order as the separate kernels it replaces, so the same bits:
// (0) attn rms_norm + the 8-bit rounding (k_rms_norm_q8x32's arithmetic; block 0); (1) the qkv + gate R3 tasks (the
// direct plan, outputs in place) and the beta / alpha F32 rows with their gates (k_dn_beta_alpha_gates' sums); (2) per
// (value head, 64 state rows): the conv of the head's q / k / v channels + the conv tail, the l2 norms of q and k, the
// recurrence (k_dn_rec_reg's per-row arithmetic), and the gated norm + rounding of a head by the unit that finishes it
// last; (3) the ssm_out R3 tasks. Three grid barriers instead of eleven launches. Activations made inside the launch are
// read with L1-bypassing loads.
struct DnHalfArgs {
  const float* x = nullptr;        // H the residual stream (read)
  const float* norm_w = nullptr;   // H attn norm weight
  float eps = 0.0f;                // rms eps (the attn norm, the l2 norms, the gated norm)
  int H = 0, rot_in = 0, rot_out = 0;
  float* xn = nullptr;             // H normalized row (written: later readers)
  float* xr = nullptr;             // rounded floats (as act_q8x32; H, then nv x ds)
  std::int8_t* q = nullptr;        // the R3 plans' int8 inputs, scales, sums (stage 0 for p_in, stage 2 for p_out)
  float* scale = nullptr;
  int* sum = nullptr;
  const StreamPlan* p_in = nullptr;   // qkv + gate, direct (outputs mixed and zg)
  const StreamPlan* p_out = nullptr;  // ssm_out, direct (output y)
  const float* wb = nullptr;       // nv x H beta (F32, rows rbb bytes, 16-byte aligned)
  const float* wa = nullptr;       // nv x H alpha
  std::uint64_t rbb = 0, rba = 0;
  float* beta = nullptr;           // nv (out: sigmoid)
  float* alpha = nullptr;          // nv (out: softplus(alpha + dt) x a)
  const float* dt = nullptr;
  const float* ssm_a = nullptr;
  int nk = 0, nv = 0, ds = 0, ch = 0, taps = 0;
  const float* mixed = nullptr;    // ch (stage 1's qkv output)
  const float* zg = nullptr;       // nv x ds (stage 1's gate output)
  const float* tail = nullptr;     // (taps - 1) x ch
  float* new_tail = nullptr;       // (taps - 1) x ch
  const float* conv_w = nullptr;   // ch x taps
  float* state = nullptr;          // nv x ds x ds
  float* core = nullptr;           // nv x ds (out: after the gated norm)
  const float* gn_w = nullptr;     // ds gated norm weight
  float qscale = 0.0f;             // 1 / sqrt(ds) as the host computes it
  unsigned* hcnt = nullptr;        // nv counters, zero between launches (the last unit of a head resets its own)
};
// whether dn_half takes these shapes and pointers (ds 128, F32 beta / alpha on the fast path's terms, one-token direct plans)
bool dn_half_ok(const DnHalfArgs& args);
bool dn_half(const DnHalfArgs& args, Stream s = nullptr);
// v55b: the expert stage holds one Q4_TG2 grid in shared memory (8 KB instead of 24 KB, so more blocks fit per SM): the grid
// of the bits every Q4_TG2 job of the engine reads (4: stored or 4 / 5 planes; 3 or 2: the plane slider's base grid). Set
// once at engine setup; before any expert-stage launch.
bool rx_set_grid_bits(int bits);
// v55b: the kernels' resource use for the log (registers, local memory, blocks per SM), one line per kernel
void stream_kernel_report();
// v55e (temporary switch, --no-lean-dense): v55d's lean one-format dense kernels on (default) or off; set before plans are built
void set_lean_dense(bool on);
// v55f: the expert stage's per-warp staging buffer, sized to the largest block any staged matrix has (rx_stage_block_bytes of
// each staged expert matrix's type at the slider's bits); set before plans and launches; default: Q8_0's 272
int rx_stage_block_bytes(std::uint32_t type, int plane_bits);
void rx_set_stage_bytes(int max_block_bytes);
// v46a-2 load_mode: 0 each lane loads its own 16-byte pieces (v46a-1); 1 a warp first copies its task's rows contiguously into
// shared memory for experts read from RAM (every 32-byte sector over PCIe once; the slot filled by the same copy); 2 the same
// for every expert. The arithmetic is the same in all three, so the outputs are equal bit for bit.

}  // namespace pelias::kern::gpu
