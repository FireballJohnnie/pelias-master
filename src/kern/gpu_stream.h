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

 private:
  void* dev_ = nullptr;           // device copy: job table, stage ends (in tasks), groups, counters
  int njobs_ = 0, nstages_ = 0, ngroups_ = 0;
  long long ntasks_ = 0;
  std::uint64_t bytes_ = 0;
  unsigned long long epoch_ = 0;  // runs so far: the counters are never reset, run e waits for e x the stage's tasks
  int grid_ = 0;                  // blocks of the cooperative launch (occupancy x SMs)
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
// v46a-2 load_mode: 0 each lane loads its own 16-byte pieces (v46a-1); 1 a warp first copies its task's rows contiguously into
// shared memory for experts read from RAM (every 32-byte sector over PCIe once; the slot filled by the same copy); 2 the same
// for every expert. The arithmetic is the same in all three, so the outputs are equal bit for bit.

}  // namespace pelias::kern::gpu
