// provenance: original (the layer order and every step are forward_impl's in model/arch_qwen35moe.cpp, as GpuForward runs
// them, with the state carried between calls as forward_decode carries it)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// gpu_engine.cpp - see gpu_engine.h.
#include "model/gpu_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <thread>

#include "hal/host_alloc_pages.h"
#include "kern/cpu_dot.h"
#include "kern/tg2_planes.h"
#include "kern/gpu_ops.h"
#include "kern/gpu_decode.h"
#include "kern/gpu_prompt.h"
#include "kern/gpu_stream.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"

namespace pelias::model {

namespace g = pelias::kern::gpu;
using std::int64_t;

namespace {

using Clock = std::chrono::steady_clock;
inline double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }
inline std::size_t z(int64_t v) { return static_cast<std::size_t>(v); }
inline std::size_t al(std::size_t x) { return (x + 255) / 256 * 256; }

bool is_float_type(std::uint32_t t) {
  return t == static_cast<std::uint32_t>(quant::Type::f32) || t == static_cast<std::uint32_t>(quant::Type::f16) ||
         t == static_cast<std::uint32_t>(quant::Type::bf16);
}
Status gfail(const std::string& what) { return Status::fail(Code::internal, what + ": " + g::last_error()); }


// a bump allocator over one device block
struct Arena {
  std::uint8_t* base = nullptr;
  std::size_t cap = 0, used = 0;
  void* take(std::size_t bytes) {
    const std::size_t at = al(used);
    if (at + bytes > cap) return nullptr;
    used = at + bytes;
    return base + at;
  }
};

struct DevW {
  const WeightView* v = nullptr;
  const void* d = nullptr;   // a plain device copy: float weights (matmul / the F32 fast products) and, for the shared
                             // expert's fold, R3's copy where the stage reads that layout (g_q8_0r) or a stored copy
  int64_t n_out = 0, k = 0;
  std::uint32_t dtype = 0;   // the type d is read as (g_q8_0r for Q8_0 held in R3's layout)
  std::uint64_t drb = 0;     // its row bytes
  bool r3 = false;           // sw holds this weight for R3 (every quantized product; v60: the only quantized path)
  g::StreamWeights sw;
};

struct Layer {
  DevW wq, wk, wv, wo, wqkv, wgate, wbeta, walpha, ssm_out, gate_inp, gate_sh, up_sh, down_sh;
  const float *attn_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *conv1d = nullptr, *ssm_dt = nullptr,
              *ssm_a = nullptr, *ssm_norm = nullptr, *gate_inp_sh = nullptr;
  // v58b: an F32 copy of the router for the fused launch when the file stores it otherwise (the MTP block's; 0: gate_inp
  // is F32 itself)
  const float* router_f32 = nullptr;
  // the bank: typed stacks per projection (each with its expert map), and which experts it holds
  std::vector<std::uint8_t> hot;  // n_expert flags
  int n_hot = 0;
  void* bank_mem = nullptr;
  // state
  float *K = nullptr, *V = nullptr;          // full attention: max_ctx x kv_w each
  float* state = nullptr;                    // DeltaNet: nv x ds x ds
  float* tail[2] = {nullptr, nullptr};       // DeltaNet: (taps - 1) x ch, ping-pong
  int cur = 0;
  float* ckpt = nullptr;        // DeltaNet, verify: the state after each of the first max_verify - 1 tokens
  float* mixed_save = nullptr;  // DeltaNet, verify: the batch's conv inputs (max_verify x ch), to rebuild the tail of a prefix
  std::vector<const std::uint8_t*> bank_at[3];  // v24: per projection (gate, up, down), each hot expert's matrix in the bank
};

// one routed expert's rows of one projection
struct ExpRows {
  quant::Type t = quant::Type::f32;
  const std::uint8_t* base = nullptr;
  std::uint64_t rb = 0;
};
ExpRows exp_rows(const WeightView& v, int e) {
  if (!v.slots.empty()) return {static_cast<quant::Type>(v.slots[z(e)].type), v.slots[z(e)].data, v.slots[z(e)].row_bytes};
  return {static_cast<quant::Type>(v.type), v.data + static_cast<std::uint64_t>(e) * v.ne[1] * v.row_bytes, v.row_bytes};
}
std::uint32_t exp_type(const WeightView& v, int e) { return v.slots.empty() ? v.type : v.slots[z(e)].type; }


}  // namespace

struct GpuEngine::Impl {
  Qwen35MoeSpec s;
  const Qwen35MoeWeights* w = nullptr;
  EngineOptions o;
  int npad = 0;
  int64_t kmax = 0, kv_w = 0;
  std::int64_t pos = 0;
  std::uint64_t dev_bytes = 0, dense_bytes_total = 0, bank_bytes_total = 0, cache_bytes = 0;
  int bank_count = 0;
  std::vector<Layer> L;
  // device memory
  void* act_mem = nullptr;
  void* dense_mem = nullptr;
  void* state_mem = nullptr;
  Arena arena;  // over dense_mem (resident dense weights, norms, the output layer)
  DevW out;
  const float* out_norm = nullptr;
  const float* rope_ff = nullptr;
  float *x = nullptr, *xn = nullptr, *y = nullptr, *xr = nullptr, *scale = nullptr;
  std::int8_t* q = nullptr;
  int* sum = nullptr;
  float *qg = nullptr, *att = nullptr, *ascr = nullptr;
  float *mixed = nullptr, *zg = nullptr, *beta = nullptr, *alpha = nullptr, *conv = nullptr, *core = nullptr;
  std::int32_t* top_d = nullptr;    // v56: device argmax per output row (max_batch)
  std::int32_t* top_req = nullptr;  // v56: non-null during step_top / verify_top / mtp_step_top: the host destination
  unsigned* dn_hcnt = nullptr;  // v55g (L3): per value head, the dense half's units that finished it (zero between launches)
  std::uint64_t dn_half_runs = 0;
  float *logits_r = nullptr, *wsel = nullptr, *gs = nullptr, *us = nullptr, *sh = nullptr, *gsh = nullptr, *galpha = nullptr, *zero2 = nullptr;
  std::int32_t* sel = nullptr;
  float* lg = nullptr;
  // v24 picked-expert reader
  static constexpr int kReaderMax = 4;   // steps of up to this many tokens use it
  static constexpr int kRxLoad = 1;      // v46a-2 rx_load "miss" (rows staged for experts read from RAM; v60: locked, the pipeline stages every expert anyway)
  bool reader = false;
  std::string reader_note = "off";
  std::vector<void*> reg_bases;          // v27: the registered ranges of the model file's mapping (host)
  void* planes_h = nullptr;              // v54: the RAM experts' Q4_TG2 matrices in bit planes (pinned, mapped)
  std::uint64_t planes_b = 0;
  std::string planes_note;
  g::ExpertPtr* rtab = nullptr;          // n_layer x n_expert entries (device)
  float* rgu = nullptr;                  // scratch: kReaderMax x nu x 2F
  float* ryd = nullptr;                  // v25 scratch: kReaderMax x nu x H (each pick's down row)
  int* rgrp = nullptr;                   // v25 scratch: the picks grouped by expert
  std::vector<g::ExpertPtr> rtab_h;      // host copy of the table (the trace's byte counts)
  // v26 reader v3: the shared expert per layer (device), the groups' sources / fills, the cache
  g::ExpertPtr *rshx = nullptr, *rgsrc = nullptr, *rgfill = nullptr;
  bool fold_ok = false;
  std::uint64_t cache_reserved = 0, cslot = 0, mtp_cache_bonus = 0;  // v60 commit C: the MTP row's extra cache bytes
  int rc_eff = 0;      // v54b: the reader cache's slots per layer (--reader-cache, or the VRAM probe's)
  std::string probe_note;
  void* cache_mem = nullptr;
  void* cache_meta = nullptr;
  std::vector<g::ReaderCache> rcache;
  // v46a-1 (THE PLAN 46a): the R3 expert stage instead of the reader v3 kernels (EngineOptions::r3_experts): the int8
  // silu x up per pick, its scales and sums, the plan buffer; on only if every expert matrix is one the stage reads
  std::int8_t* rhq = nullptr;
  std::uint32_t *rxb = nullptr, *rhb = nullptr;  // v55f: the expert inputs' and the down inputs' bit planes (the 1-bit route)
  float* rhs = nullptr;
  int* rhsum = nullptr;
  void* rxplan = nullptr;
  int rx_maxp = 0;
  bool rx_ok = false;
  unsigned long long *ph_h = nullptr, *ph_d = nullptr;  // v55b: n_layer x 5 phase stamps of the fused MoE half (verbose_gpu)
  unsigned long long *dph_h = nullptr, *dph_d = nullptr;  // v56: n_layer x 5 stage stamps of the DeltaNet dense half (L3; verbose_gpu)
  double dph_sum[4] = {0, 0, 0, 0};
  std::uint64_t dph_steps = 0, dph_layers = 0;
  // v56: the same stamps of verify steps (2-4 tokens; MTP), per step size: [n - 2][phase]
  double vph_sum[3][4] = {}, vdph_sum[3][4] = {};
  std::uint64_t vph_steps[3] = {}, vdph_steps[3] = {};
  unsigned long long* rst_d = nullptr;                    // v55e3 diagnosis: the expert stage's counters (g::kRxStatN, verbose_gpu)
  unsigned long long* rst_mtp_d = nullptr;                // v60 commit C: the MTP row's own (its draft / catch-up launches)
  double ph_sum[4] = {0, 0, 0, 0};
  std::uint64_t ph_steps = 0, ph_layers = 0;
  std::string rx_note = "off";
  g::ExpertPtr *trc_src_h = nullptr, *trc_src_d = nullptr, *trc_fill_h = nullptr, *trc_fill_d = nullptr;
  // v25 trace: pinned mapped, per layer 6 reader stamps, the picks (kReaderMax x nu) and the group count
  void* trc_hv = nullptr;
  unsigned long long *trc_ts_h = nullptr, *trc_ts_d = nullptr;
  std::int32_t *trc_sel_h = nullptr, *trc_sel_d = nullptr;
  int *trc_u_h = nullptr, *trc_u_d = nullptr;
  std::string trace_text;
  float *rlg_h = nullptr, *rlg_d = nullptr;  // v49: router logits per layer (kReaderMax x n_expert each), pinned mapped
  std::vector<float> rmarg;                  // v49: per position 2 x n_layer router margins
  std::vector<std::uint8_t> rmarg_have;
  std::uint64_t trace_seq = 0;
  static constexpr int kOutRows = 4;  // v60 commit C: every step is 1-4 tokens (kReaderMax); the head runs at most 4 rows
  // verify / keep (stage 3)
  bool ckpt_mode = false;
  bool in_prompt = false;   // v48b: a step of a split prompt (its products of 2-4 tokens go through R3, as option B's)
  bool no_head = false;      // v48: a prompt step whose logits nobody reads (an inner step of a split prompt): skip the head
  int last_n = 0;            // tokens of the last verify batch (0: none pending)
  std::int64_t verify_pos0 = 0;
  // the MTP block (stage 3)
  bool has_mtp = false;
  Layer M;
  // v58: the MTP block's MoE half on the reader as its layer n_layer (EngineOptions::mtp_fused); the reader's per-layer
  // tables (expert table, shared expert, cache, planes) then have n_layer + 1 rows
  bool mtp_rd = false;
  std::uint64_t mtp_fused_steps = 0;
  std::string mtp_router_note;  // v58b: the MTP router's stored type, and whether an F32 copy was made
  std::uint32_t nlr() const { return s.n_layer + (mtp_rd ? 1u : 0u); }
  const Qwen35MoeLayer& WL(std::uint32_t il) const { return il < s.n_layer ? w->layers[il] : w->mtp.layer; }
  Layer& LL(std::uint32_t il) { return il < s.n_layer ? L[il] : M; }
  DevW eh, head;
  // v60 E: the head sketch (EngineOptions::mtp_head_sketch; kern/tg2_planes.h g_q6_ksk): the head's top two code planes +
  // scales (hsk), a kHeadSketchTop-row Q6_K scratch in R3's layout that the full product's own kernel re-scores (head_top:
  // gathered per draft, fixed addresses so its direct plan is built once), the candidate ids, the re-scored argmax's index,
  // and the recall counters (hit, total) of the check mode
  static constexpr int kHeadSketchTop = 64;
  void* hsk_mem = nullptr;
  std::uint8_t* hsk = nullptr;
  // v60 G (item 4): the keep() stitch's table - one g::DnKeepLayer per DeltaNet layer with checkpoints (device, fixed at open)
  g::DnKeepLayer* dnk_d = nullptr;
  int dnk_n = 0;
  std::size_t hsk_bytes = 0;
  DevW head_top;
  void* htop_w = nullptr;
  std::uint16_t* htop_d6 = nullptr;
  std::int32_t* cand_d = nullptr;
  std::int32_t* idx_d = nullptr;
  float* lg_top = nullptr;
  unsigned* sk_cnt_d = nullptr;
  std::uint64_t sk_drafts = 0, sk_hit = 0, sk_total = 0;
  std::string sketch_note;
  const float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
  std::int64_t mtp_pos = 0;
  float* cat = nullptr;  // kReaderMax x 2 n_embd
  std::vector<float> cat_h;
  // v60 commit C: the MTP path on the device. hid_d: the trunk's post-norm hidden rows of a prompt chunk (row 0 the row
  // before the chunk, then max_batch rows) or of a verify batch; h_d / hn_d: the draft chain's seed and output; mtp_top_d:
  // each draft's argmax (draft i reads row i - 1 as its id: no host in the chain); mtp_emb_dev: the MTP block's embedding
  // rows registered for the GPU gather (the trunk's emb_dev when the file shares the tensor)
  float *hid_d = nullptr, *h_d = nullptr, *hn_d = nullptr;
  float* hidden_dev_req = nullptr;  // v60 commit C: set around step() / verify(): the post-norm hidden rows go here (device), no host copy
  // v60 commit D (author 11-06 12:00, "see why the extra bank didn't do anything"): where a draft's time goes. Four GPU
  // timestamps (globaltimer ns, g::stamp into mapped memory) per MTP step at its ring slot: 0 start (after the position),
  // 1 after the MoE half (= the block: gather, norms, eh_proj, attention, MoE), 2 before the head product, 3 after the
  // argmax; the generate loop sums them after its sync (mtp_slots lists the steps)
  unsigned long long *mtp_ts_h = nullptr, *mtp_ts_d = nullptr;
  // v60 E2: 0 start, 1 after the MoE half, 2 before the head, 3 after the sketch dot, 4 after the top-k select, 5 after the
  // argmax / pick; v60 H: 6 after the input (gather, cat, norms), 7 after eh_proj, 8 after the attention (the MoE half ends at 1)
  static constexpr int kMtpStamps = 9;
  std::vector<unsigned> mtp_slots;
  std::int32_t* mtp_top_d = nullptr;
  const std::uint8_t* mtp_emb_dev = nullptr;
  std::uint64_t mtp_copy_runs_saved = 0;  // v60 commit C: MTP attention on the direct plan (knew / vnew), counted
  double pin_read_gbs = -1.0;  // v60 commit C2: the pinned planes read rate at open (GB/s; -1: not probed)
  int pin_read_tries = 0;
  std::uint64_t act_freed = 0;  // v60 commit C: activation bytes the v60b layout held for rows no step can reach (to the MTP row's cache)

  std::vector<float> xe_h;  // embed(): the host rows before the upload
  // v23 probe
  std::int64_t probe_pos = -1;
  bool probe_have = false;
  float* probe_xd = nullptr;
  std::int32_t* probe_seld = nullptr;
  float* probe_wd = nullptr;
  float *probe_bd = nullptr, *probe_ld = nullptr, *probe_md = nullptr;  // v25 per-operation rows
  std::vector<float> probe_bh, probe_lh, probe_mh;
  std::vector<float> probe_xh, probe_wh;
  std::vector<std::int32_t> probe_selh;

  // ---- weights
  static std::size_t view_bytes(const WeightView& v) {
    if (!v.slots.empty()) {
      std::size_t b = 0;
      for (const ExpertSlot& sl : v.slots) b += al(static_cast<std::size_t>(v.ne[1] * sl.row_bytes));
      return b;
    }
    return static_cast<std::size_t>(v.ne[1] * v.ne[2] * v.row_bytes);
  }
  bool use_r3 = true;  // v60: always true (kept for r3_takes)
  // R3's copy of these types is the stored layout (Q8_0 repacked, read as g_q8_0r): the shared expert's fold reads it too
  static bool r3_shared(std::uint32_t t) {
    return t == static_cast<std::uint32_t>(quant::Type::q8_0) || t == static_cast<std::uint32_t>(quant::Type::q4_k) ||
           t == static_cast<std::uint32_t>(quant::Type::q5_k) || t == static_cast<std::uint32_t>(quant::Type::q4_tg2);
  }
  bool r3_takes(const WeightView& v) const {
    return use_r3 && v.present() && !v.rot32 && v.slots.empty() && v.ne[2] == 1 && g::stream_type_ok(v.type, static_cast<int64_t>(v.ne[0]));
  }
  // v60 (commit B): every quantized weight is held once, in R3's copy (counted by put_dense from stream_upload); the int8
  // side tables and the raw copies for the batched kernels left with them (v55 did this for the output head: -0.48 GB).
  // A plain copy stays for float weights and for a shared-expert matrix whose stored layout R3 does not keep (the fold)
  std::size_t dense_bytes(const WeightView& v, bool fold = false) const {
    if (!v.present()) return 0;
    if (r3_takes(v) && (r3_shared(v.type) || !fold)) return 256;
    return al(view_bytes(v)) + 256;
  }
  static std::size_t vec_bytes(const std::vector<float>& f) { return f.empty() ? 0 : al(f.size() * sizeof(float)) + 256; }
  std::size_t layer_dense_bytes(const Qwen35MoeLayer& l) const {
    std::size_t b = 0;
    for (const WeightView* v : {&l.wq, &l.wk, &l.wv, &l.wo, &l.wqkv, &l.wgate, &l.wbeta, &l.walpha, &l.ssm_out, &l.gate_inp}) b += dense_bytes(*v);
    for (const WeightView* v : {&l.gate_shexp, &l.up_shexp, &l.down_shexp}) b += dense_bytes(*v, true);
    for (const std::vector<float>* f : {&l.attn_norm, &l.post_norm, &l.q_norm, &l.k_norm, &l.conv1d, &l.ssm_dt, &l.ssm_a, &l.ssm_norm, &l.gate_inp_shexp})
      b += vec_bytes(*f);
    return b;
  }
  Status put_dense(const WeightView& v, DevW& dw, bool fold = false) {
    dw = DevW{};
    if (!v.present()) return Status::good();
    dw.v = &v;
    dw.n_out = static_cast<int64_t>(v.ne[1]);
    dw.k = static_cast<int64_t>(v.ne[0]);
    dw.dtype = v.type;
    dw.drb = v.row_bytes;
    if (r3_takes(v)) {
      if (!g::stream_upload(v.type, v.data, dw.n_out, dw.k, &dw.sw)) return Status::fail(Code::internal, std::string("engine: R3 upload ") + v.name + ": " + g::stream_last_error());
      dw.r3 = true;
      dev_bytes += dw.sw.bytes;
      dense_bytes_total += dw.sw.bytes;
      if (r3_shared(v.type)) {  // the shared expert's fold reads R3's copy as stored (Q8_0 in R3's layout)
        dw.d = dw.sw.w;
        if (v.type == static_cast<std::uint32_t>(quant::Type::q8_0)) {
          dw.dtype = g::g_q8_0r;
          dw.drb = static_cast<std::uint64_t>(dw.k / 256) * 272;
        }
      }
      if (dw.d || !fold) return Status::good();  // v60: held once; prod() refuses any product of it that is not on R3
    }
    const std::size_t bytes = view_bytes(v);
    void* dd = arena.take(bytes);
    if (!dd || !g::to_device(dd, v.data, bytes)) return Status::fail(Code::internal, "engine: weight upload " + v.name);
    dw.d = dd;
    return Status::good();
  }
  Status put_vec(const std::vector<float>& f, const float*& out_ptr) {
    out_ptr = nullptr;
    if (f.empty()) return Status::good();
    void* d = arena.take(f.size() * sizeof(float));
    if (!d || !g::to_device(d, f.data(), f.size() * sizeof(float))) return Status::fail(Code::internal, "engine: vector upload");
    out_ptr = static_cast<const float*>(d);
    return Status::good();
  }
  Status put_layer(const Qwen35MoeLayer& l, Layer& D) {
    Status st;
    for (auto [v, d] : {std::pair{&l.wq, &D.wq}, {&l.wk, &D.wk}, {&l.wv, &D.wv}, {&l.wo, &D.wo}, {&l.wqkv, &D.wqkv}, {&l.wgate, &D.wgate},
                        {&l.wbeta, &D.wbeta}, {&l.walpha, &D.walpha}, {&l.ssm_out, &D.ssm_out}, {&l.gate_inp, &D.gate_inp}})
      if (!(st = put_dense(*v, *d))) return st;
    for (auto [v, d] : {std::pair{&l.gate_shexp, &D.gate_sh}, {&l.up_shexp, &D.up_sh}, {&l.down_shexp, &D.down_sh}})
      if (!(st = put_dense(*v, *d, true))) return st;
    for (auto [f, d] : {std::pair{&l.attn_norm, &D.attn_norm}, {&l.post_norm, &D.post_norm}, {&l.q_norm, &D.q_norm}, {&l.k_norm, &D.k_norm},
                        {&l.conv1d, &D.conv1d}, {&l.ssm_dt, &D.ssm_dt}, {&l.ssm_a, &D.ssm_a}, {&l.ssm_norm, &D.ssm_norm},
                        {&l.gate_inp_shexp, &D.gate_inp_sh}})
      if (!(st = put_vec(*f, *d))) return st;
    return Status::good();
  }

  // ---- the bank
  // device bytes of expert e's three projections in the bank (matrices and their share of the side tables)
  // v54c: the plane format a stored matrix takes when the slider is on (Q4_TG2 / Q5_K rows), or 0
  std::uint32_t planed_type(quant::Type t, std::uint64_t rb) const {
    if (o.plane_bits <= 0) return 0;
    if (t == quant::Type::q4_tg2 && rb % 144 == 0) return kern::gpu::g_q4_tg2p;
    if (t == quant::Type::q5_k && rb % 176 == 0) return kern::gpu::g_q5_kp;
    if (t == quant::Type::q4_k && rb % 144 == 0) return kern::gpu::g_q4_kp;  // v60 F: the MTP block's experts (and stock Q4_K files)
    return 0;
  }
  // v60 F2: the bits a row's plane matrices are packed and read with: the slider for the trunk; the MTP row (il == n_layer)
  // its own (EngineOptions::mtp_plane_bits; 0 = the slider). The drafter has no verify behind it, so it feels a lost bit
  // directly (v60m / v60n MEASURED: the MTP block at 3 of 4 bits kept 84.1 / 72.1 % against 86.8 / 74.5 % at 4); its planes
  // are ~1/41 of the whole. Every site that sizes, packs or reads a row's planes asks this one function
  int row_bits(std::uint32_t il) const { return il == s.n_layer && o.mtp_plane_bits > 0 ? o.mtp_plane_bits : o.plane_bits; }
  // v54c: an expert's matrices at the row's bits (the bytes a bank copy or a cache slot of it takes)
  std::size_t planed_bytes(const Qwen35MoeLayer& l, int e, int bits) const {
    std::size_t b = 0;
    for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
      const ExpRows r = exp_rows(*v, e);
      const std::uint32_t pt = planed_type(r.t, r.rb);
      b += z(static_cast<int64_t>(pt ? kern::gpu::plane_prefix(pt, r.rb, bits) : r.rb)) * z(static_cast<int64_t>(v->ne[1]));
    }
    return b;
  }
  // the bytes a bank copy or a cache slot of an expert takes (v54c: the planed bank holds the matrices alone)
  std::size_t expert_bytes(std::uint32_t il, int e) const { return al(planed_bytes(WL(il), e, row_bits(il))); }
  // an expert can sit in the bank when the R3 expert stage reads every projection's type (v60: the bank is read by the
  // stage alone; the int8-step condition of the stacked bank is gone, the type set is the same for this file's types)
  bool bankable(const Qwen35MoeLayer& l, int e) const {
    for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps})
      if (!g::rx_type_ok(exp_type(*v, e), static_cast<int64_t>(v->ne[0])) || v->rot32) return false;
    return true;
  }
  // v54c: the bank in bit planes (the slider on): each hot expert's matrices back to back in VRAM, plane matrices as the
  // slider's prefix of each row (packed), the rest as stored. Only the R3 expert stage reads it (no stacks, no tables)
  Status build_bank_planes(std::uint32_t il, Layer& D) {
    const Qwen35MoeLayer& l = WL(il);
    const int bits = row_bits(il);  // v60 F2
    const int ne = static_cast<int>(s.n_expert);
    std::size_t need_b = 0;
    for (int e = 0; e < ne; ++e)
      if (D.hot[z(e)]) need_b += al(planed_bytes(l, e, bits));
    for (auto& ba : D.bank_at) ba.assign(z(ne), nullptr);
    if (need_b == 0) return Status::good();
    D.bank_mem = g::dalloc(need_b);
    if (!D.bank_mem) return gfail("engine: bank memory (planes)");
    dev_bytes += need_b;
    bank_bytes_total += need_b;
    auto* base = static_cast<std::uint8_t*>(D.bank_mem);
    std::size_t off = 0;
    std::vector<std::uint8_t> row, host;
    for (int e = 0; e < ne; ++e) {
      if (!D.hot[z(e)]) continue;
      host.clear();
      const WeightView* vv[3] = {&l.gate_exps, &l.up_exps, &l.down_exps};
      std::size_t moff[3];
      for (int pj = 0; pj < 3; ++pj) {
        const ExpRows r = exp_rows(*vv[pj], e);
        const std::uint32_t pt = planed_type(r.t, r.rb);
        const std::uint64_t rows = static_cast<std::uint64_t>(vv[pj]->ne[1]);
        moff[pj] = host.size();
        if (!pt) {
          host.insert(host.end(), r.base, r.base + r.rb * rows);
          continue;
        }
        const std::uint64_t pf = kern::gpu::plane_prefix(pt, r.rb, bits);
        row.resize(z(static_cast<int64_t>(r.rb)));
        for (std::uint64_t rr = 0; rr < rows; ++rr) {
          kern::gpu::plane_row(static_cast<std::uint32_t>(r.t), r.base + rr * r.rb, row.data(), r.rb);
          host.insert(host.end(), row.begin(), row.begin() + static_cast<std::ptrdiff_t>(pf));
        }
      }
      if (!g::to_device(base + off, host.data(), host.size())) return gfail("engine: bank upload (planes)");
      for (int pj = 0; pj < 3; ++pj) D.bank_at[z(pj)][z(e)] = base + off + moff[pj];
      off += al(host.size());
    }
    return Status::good();
  }

  // ---- products: as GpuForward::prod (quantized weights round their input to gpu256x8 first)
  Status prod(const DevW& W, const float* xin, int64_t n, float* yout) {
    const WeightView& v = *W.v;
    if (is_float_type(v.type)) {
      if (v.rot32) return Status::fail(Code::unsupported, "engine: a rotated float weight (" + v.name + ")");
      if (!g::matmul(W.dtype, W.d, W.drb, W.n_out, W.k, xin, n, W.k, yout, W.n_out)) return gfail("engine: product " + v.name);
      return Status::good();
    }
    // v60: every quantized product runs on R3 (1-4 rows; the weights read once for the step's rows, each row's arithmetic
    // the one-token path's). The batched kernels (int8 tables, matmul) left with the CPU cold path.
    if (W.r3 && n >= 1 && n <= kReaderMax) return prod_r3({{&W, yout}}, xin, n);
    return Status::fail(Code::internal, "engine: " + v.name + ": a product of " + std::to_string(n) + " rows is not on R3 (v60: steps are 1-4 tokens)");
  }
  // one token through R3: the products of one input, grouped into one plan (built once per group, kept), outputs into a
  // fixed scratch and then copied to where each belongs (the attention cache's row moves with the position)
  struct R3Group {
    std::unique_ptr<g::StreamPlan> plan;
    std::vector<int64_t> off;  // each job's output offset in r3y
    std::vector<float*> yo;    // v55: direct plans: the outputs the plan writes (a call with other outputs takes the scratch plan)
  };
  // v55 (L1): a product's outputs go straight to where they belong when the call's outputs are the plan's (no scratch + copy),
  // and a one-stage group with no waits runs as a plain launch (no counter reset, no cooperative launch); same tasks, same sums
  std::uint64_t r3_direct_runs = 0, r3_copy_runs = 0;
  std::string r3_direct_note() const {
    return use_r3 ? "; R3 products: outputs written in place, one-stage groups as plain launches (v55)" : "";
  }
  std::map<std::pair<const DevW*, int>, R3Group> r3plans;  // v26: per group and token count
  // v26 CUDA graph
  g::Stream stm = nullptr;
  int* pos_d = nullptr;
  // v60 (commit B): the token embedding gathered on the GPU (g::gather_rows) from the file's rows, registered for the GPU
  // (pinned; no VRAM), the ids written to a mapped ring the kernel reads in place: no host dequantization, no upload, no
  // sync before the layers. The ring (kRing slots of kReaderMax ids, and kRing positions) lets inner prompt steps skip the
  // step-end sync: a slot is reused only after kRing steps, and a sync is forced before the ring wraps.
  static constexpr int kRing = 64;
  const std::uint8_t* emb_dev = nullptr;  // tok_embd's rows as the GPU sees them (null: the host path)
  void* ring_hv = nullptr;                // mapped: kRing x kReaderMax ids, then kRing positions
  std::int32_t* ring_tok_h = nullptr;
  std::int32_t* ring_tok_d = nullptr;
  int* ring_pos_h = nullptr;
  unsigned ring_at = 0;       // the next slot
  unsigned unsynced = 0;      // steps enqueued since the last host sync
  float *knew = nullptr, *vnew = nullptr;
  bool graph_ok = false;
  bool trunk_dp = false;  // the trunk's step reads the device position (not the MTP block's own attention)
  std::string graph_note = "off";
  std::map<std::tuple<int, int, int, int>, void*> graphs;
  std::map<std::tuple<int, int, int, int>, int> graph_seen;
  std::map<std::tuple<int, int, int, int>, std::size_t> graph_nodes;  // v60 I (0.3): launches per captured trunk step
  // v60 I (0.4): the MTP draft chain's graphs, keyed by (hidden input, argmax output, hidden output): the ids are gathered and the
  // position copied before the graph (they vary per call); the chain's stamps land in the slot it was captured with
  struct MtpGraph {
    void* exec = nullptr;
    unsigned slot = 0;
    int seen = 0;
    std::size_t nodes = 0;
  };
  // v60 I2: keyed by (hidden input, argmax output, hidden output, tokens) so the catch-up steps (1-4 tokens, no head) and the
  // prompt catch-up chunks (16 hidden-row pointers x 4 tokens) replay too; kMtpGraphMax caps the table (a new shape past the cap
  // runs plainly)
  std::map<std::tuple<const void*, const void*, const void*, int>, MtpGraph> mtp_graphs;
  static constexpr std::size_t kMtpGraphMax = 96;
  bool mtp_graph_ok = false;
  std::string mtp_graph_note = "off";
  std::uint64_t mtp_graph_captures = 0, mtp_graph_replays = 0;
  std::uint64_t graph_replays = 0, graph_captures = 0;
  float* r3y = nullptr;
  std::size_t r3y_floats = 0;
  Status prod_r3(std::initializer_list<std::pair<const DevW*, float*>> jobs, const float* xin, int64_t n = 1) {
    const DevW& W0 = *jobs.begin()->first;
    if (n < 1 || n > kReaderMax) return Status::fail(Code::internal, "engine: R3 takes 1-4 tokens");
    // v55: the direct plan (built with this call's outputs) when the outputs match; otherwise the scratch plan and copies
    R3Group* Dp = nullptr;
    if (Status st = r3_group(jobs, n, 0x200, &Dp); !st) return st;
    bool same = Dp->yo.size() == jobs.size();
    {
      std::size_t i = 0;
      for (const auto& [W, yo] : jobs) {
        (void)W;
        same = same && Dp->yo[i++] == yo;
      }
    }
    if (!g::act_q8x32(xin, W0.k, n, W0.k, W0.v->rot32, xr, W0.k, q, scale, sum)) return gfail("engine: rounding for R3");
    if (same) {
      if (!Dp->plan->run_unstaged()) return Status::fail(Code::internal, std::string("engine: R3 run: ") + g::stream_last_error());
      ++r3_direct_runs;
      return Status::good();
    }
    R3Group* Gp = nullptr;
    if (Status st = r3_group(jobs, n, 0, &Gp); !st) return st;
    R3Group& G = *Gp;
    if (!G.plan->run_unstaged()) return Status::fail(Code::internal, std::string("engine: R3 run: ") + g::stream_last_error());
    std::size_t i = 0;
    for (const auto& [W, yo] : jobs)
      if (!g::dcopy(yo, r3y + G.off[i++], z(n * W->n_out) * sizeof(float))) return gfail("engine: R3 output");
    ++r3_copy_runs;
    return Status::good();
  }
  // v26 / v50: the R3 plan of a group for `rows` input rows (tag separates the halves' plans from plain ones of the same size)
  Status r3_group(std::initializer_list<std::pair<const DevW*, float*>> jobs, int64_t n, int tag, R3Group** gout) {
    const DevW& W0 = *jobs.begin()->first;
    R3Group& G = r3plans[{&W0, static_cast<int>(n) | tag}];
    *gout = &G;
    if (!G.plan) {
      std::vector<g::StreamJob> js;
      int64_t at = 0;
      for (const auto& [W, yo] : jobs) {
        if (!W->r3 || W->k != W0.k || W->v->rot32 != W0.v->rot32) return Status::fail(Code::internal, "engine: an R3 group with unlike inputs");
        g::StreamJob j;
        j.w = &W->sw;
        j.xq = q;
        j.xs = scale;
        j.xsum = sum;
        if (tag & 0x200) {  // v55: a direct plan writes the call's outputs
          j.y = yo;
          G.yo.push_back(yo);
        } else {
          if (static_cast<std::size_t>(at + n * W->n_out) > r3y_floats) return Status::fail(Code::internal, "engine: R3 scratch too small");
          j.y = r3y + at;
        }
        j.ntok = static_cast<int>(n);  // v26: rows of n tokens back to back (stride n_out)
        j.xq_stride = W0.k;
        j.xs_stride = W0.k / 32;
        j.y_stride = W->n_out;
        G.off.push_back(at);
        at += (n * W->n_out + 63) / 64 * 64;
        js.push_back(j);
      }
      const int stage_end[1] = {static_cast<int>(js.size())};
      G.plan = std::make_unique<g::StreamPlan>();
      if (!G.plan->build(js.data(), static_cast<int>(js.size()), stage_end, 1, 0))
        return Status::fail(Code::internal, std::string("engine: R3 plan: ") + g::stream_last_error());
    }
    return Status::good();
  }

  // dev_pos: the step's position is in pos_d (the trunk under a graph; v60 commit C: every MTP step too): new K / V rows go
  // to knew / vnew (fixed addresses, so the R3 direct plan matches every step) and kv_store puts them at *pos_d
  Status attention_block(Layer& D, int64_t n, int64_t p0, bool dev_pos) {
    const int64_t hd = s.head_dim, nh = s.n_head, nkv = s.n_head_kv;
    const int64_t qh = s.attn_gate ? 2 * hd : hd, q_w = qh * nh;
    const bool gp = dev_pos;  // v26: new rows in scratch, stored at the device position; rope / attention read it there
    float* kn = gp ? knew : D.K + p0 * kv_w;  // the new keys and values go straight into the cache
    float* vn = gp ? vnew : D.V + p0 * kv_w;
    Status st;
    if (D.wq.r3 && D.wk.r3 && D.wv.r3 && D.wq.k == D.wk.k && D.wk.k == D.wv.k) {
      if (!(st = prod_r3({{&D.wq, qg}, {&D.wk, kn}, {&D.wv, vn}}, xn, n))) return st;
    } else if (!(st = prod(D.wq, xn, n, qg)) || !(st = prod(D.wk, xn, n, kn)) || !(st = prod(D.wv, xn, n, vn))) {
      return st;
    }
    if (D.q_norm && !g::rms_norm(qg, D.q_norm, qg, hd, nh, qh, q_w, n, s.rms_eps)) return gfail("engine: q norm");
    if (D.k_norm && !g::rms_norm(kn, D.k_norm, kn, hd, nkv, hd, kv_w, n, s.rms_eps)) return gfail("engine: k norm");
    const int sec[4] = {s.rope_sections[0], s.rope_sections[1], s.rope_sections[2], s.rope_sections[3]};
    if (!g::rope(qg, nh, qh, q_w, n, p0, static_cast<int>(s.n_rot), s.rope_freq_base, sec, s.rope_neox, rope_ff, nullptr, gp ? pos_d : nullptr) ||
        !g::rope(kn, nkv, hd, kv_w, n, p0, static_cast<int>(s.n_rot), s.rope_freq_base, sec, s.rope_neox, rope_ff, nullptr, gp ? pos_d : nullptr))
      return gfail("engine: rope");
    if (gp && !g::kv_store(D.K, D.V, kn, vn, kv_w, n, pos_d)) return gfail("engine: K / V store");
    const float scl = s.attn_scale != 0.0f ? s.attn_scale : 1.0f / std::sqrt(static_cast<float>(hd));
    if (!g::attention(qg, q_w, qh, D.K, D.V, kv_w, nh, nkv, hd, n, p0, scl, s.attn_gate, att, ascr, nullptr, gp ? pos_d : nullptr, o.max_ctx))
      return gfail("engine: attention");
    if (Status ps = prod(D.wo, att, n, y); !ps) return ps;
    return Status::good();
  }

  // v55g (L3): a DeltaNet layer's dense half for one token in one cooperative launch (attn norm + rounding, qkv + gate,
  // beta / alpha + gates, conv + tail, l2 norms, recurrence, gated norm + rounding, ssm_out), the same bits as the separate
  // launches; *took false (nothing run) where it does not apply, and the caller runs the attn norm and deltanet_block
  Status dn_half_layer(Layer& D, int64_t n, bool* took) {
    *took = false;
    // v60: one token only. The several-token L3 (v57 --dn-half-mt) produced wrong hidden states (v59e E3 MEASURED: 0 of 126
    // drafts kept, 5 of 128 tokens equal) and is retired; 2-4-token steps run the separate launches below.
    if (n != 1) return Status::good();
    const int64_t ch = s.conv_channels(), ds = s.ssm_d_state, nk = s.ssm_n_group, nv = s.ssm_dt_rank, taps = s.ssm_d_conv, e = s.n_embd;
    if (!dn_hcnt || !D.wqkv.r3 || !D.wgate.r3 || !D.ssm_out.r3 || D.wqkv.k != e || D.wgate.k != e || D.ssm_out.k != nv * ds ||
        D.wqkv.v->rot32 != D.wgate.v->rot32 || D.wbeta.v->rot32 || D.walpha.v->rot32 || D.wbeta.k != e || D.walpha.k != e || D.wbeta.n_out != nv ||
        D.walpha.n_out != nv || !is_float_type(D.wbeta.v->type) || !is_float_type(D.walpha.v->type) ||
        !g::deltanet_ba_fast(D.wbeta.dtype, D.wbeta.d, D.wbeta.drb, D.walpha.dtype, D.walpha.d, D.walpha.drb, nv, e, xn, n, e))
      return Status::good();
    R3Group* G1 = nullptr;
    R3Group* G2 = nullptr;
    if (Status st = r3_group({{&D.wqkv, mixed}, {&D.wgate, zg}}, n, 0x200, &G1); !st) return st;
    if (Status st = r3_group({{&D.ssm_out, y}}, n, 0x200, &G2); !st) return st;
    if (G1->yo.size() != 2 || G1->yo[0] != mixed || G1->yo[1] != zg || G2->yo.size() != 1 || G2->yo[0] != y) return Status::good();
    g::DnHalfArgs A;
    A.x = x;
    A.norm_w = D.attn_norm;
    A.eps = s.rms_eps;
    A.H = static_cast<int>(e);
    A.rot_in = D.wqkv.v->rot32 ? 1 : 0;
    A.rot_out = D.ssm_out.v->rot32 ? 1 : 0;
    A.xn = xn;
    A.xr = xr;
    A.q = q;
    A.scale = scale;
    A.sum = sum;
    A.p_in = G1->plan.get();
    A.p_out = G2->plan.get();
    A.wb = static_cast<const float*>(D.wbeta.d);
    A.wa = static_cast<const float*>(D.walpha.d);
    A.rbb = D.wbeta.drb;
    A.rba = D.walpha.drb;
    A.beta = beta;
    A.alpha = alpha;
    A.dt = D.ssm_dt;
    A.ssm_a = D.ssm_a;
    A.nk = static_cast<int>(nk);
    A.nv = static_cast<int>(nv);
    A.ds = static_cast<int>(ds);
    A.ch = static_cast<int>(ch);
    A.taps = static_cast<int>(taps);
    A.mixed = mixed;
    A.zg = zg;
    A.tail = D.tail[D.cur];
    A.new_tail = D.tail[1 - D.cur];
    A.conv_w = D.conv1d;
    A.state = D.state;
    A.core = core;
    A.gn_w = D.ssm_norm;
    A.qscale = 1.0f / std::sqrt(static_cast<float>(ds));  // as deltanet_recurrent_reg computes it
    A.hcnt = dn_hcnt;
    A.n = static_cast<int>(n);
    A.smact = A.rot_in == 0 ? 1 : 0;  // v56b (v60: locked on; MEASURED better)
    const bool ck = ckpt_mode && n > 1;  // v56: a verify step keeps every prefix's state (k_dn_rec_reg_ck) and the batch's conv inputs
    A.ckpt = ck ? D.ckpt : nullptr;
    A.msave = ck ? D.mixed_save : nullptr;
    if (ck && (!D.ckpt || !D.mixed_save)) return Status::good();
    A.ph = dph_d && !in_prompt ? dph_d + static_cast<int64_t>(&D - L.data()) * 5 : nullptr;  // v56: stage stamps (verbose_gpu; v60 C: not prompt steps)
    if (!g::dn_half_ok(A)) return Status::good();
    if (!g::dn_half(A)) return Status::fail(Code::internal, std::string("engine: dense half (L3): ") + g::stream_last_error());
    D.cur = 1 - D.cur;
    ++dn_half_runs;
    *took = true;
    return Status::good();
  }

  Status deltanet_block(Layer& D, int64_t n) {
    const int64_t ch = s.conv_channels(), ds = s.ssm_d_state, nk = s.ssm_n_group, nv = s.ssm_dt_rank, taps = s.ssm_d_conv;
    Status st;
    if (D.wqkv.r3 && D.wgate.r3 && D.wqkv.k == D.wgate.k) {
      if (!(st = prod_r3({{&D.wqkv, mixed}, {&D.wgate, zg}}, xn, n))) return st;
    } else if (!(st = prod(D.wqkv, xn, n, mixed)) || !(st = prod(D.wgate, xn, n, zg))) {
      return st;
    }
    // v55g (L2): beta, alpha and the gates in one launch where both products take the F32 fast path (the same sums)
    const bool ba = !D.wbeta.v->rot32 && !D.walpha.v->rot32 && D.wbeta.k == D.walpha.k && D.wbeta.n_out == nv && D.walpha.n_out == nv &&
                    is_float_type(D.wbeta.v->type) && is_float_type(D.walpha.v->type) &&
                    g::deltanet_beta_alpha_gates(D.wbeta.dtype, D.wbeta.d, D.wbeta.drb, D.walpha.dtype, D.walpha.d, D.walpha.drb, nv, D.wbeta.k, xn, n,
                                                 D.wbeta.k, beta, alpha, D.ssm_dt, D.ssm_a);
    if (!ba) {
      if (!(st = prod(D.wbeta, xn, n, beta)) || !(st = prod(D.walpha, xn, n, alpha))) return st;
    }
    if (!ba && !g::deltanet_gates(beta, alpha, D.ssm_dt, D.ssm_a, n, nv)) return gfail("engine: deltanet gates");
    const bool ck = ckpt_mode && n > 1;
    if (ck && !g::dcopy(D.mixed_save, mixed, z(n * ch) * sizeof(float))) return gfail("engine: verify inputs");
    if (!g::deltanet_conv(mixed, D.tail[D.cur], D.conv1d, n, ch, taps, conv, D.tail[1 - D.cur])) return gfail("engine: deltanet conv");
    D.cur = 1 - D.cur;
    if (!g::l2_norm(conv, ds, 2 * nk, ds, ch, n, s.rms_eps)) return gfail("engine: deltanet l2 norm");
    const bool pow2 = ds >= 32 && ds <= 1024 && (ds & (ds - 1)) == 0;
    const bool ok = ck ? g::deltanet_recurrent_reg_ckpt(conv, ch, beta, alpha, D.state, n, nk, nv, ds, core, D.ckpt)
                       : pow2 ? g::deltanet_recurrent_reg(conv, ch, beta, alpha, D.state, n, nk, nv, ds, core)
                              : g::deltanet_recurrent(conv, ch, beta, alpha, D.state, n, nk, nv, ds, core);
    if (!ok) return gfail("engine: deltanet recurrence");
    if (!g::gated_norm(core, zg, D.ssm_norm, n, nv, ds, s.rms_eps)) return gfail("engine: gated norm");
    if (Status ps = prod(D.ssm_out, core, n, y); !ps) return ps;
    return Status::good();
  }

  Status run(const std::int32_t* tokens, int n, float* logits, bool last_only, EngineStats* stats, float* hidden_out, float* hidden_dev = nullptr);
  Status mtp_run(const std::int32_t* tokens_h, const std::int32_t* tokens_d, const float* h_prev_d, int n, std::int32_t* top_d_out, float* h_out_d,
                 float* logits_h, std::int32_t* top_h);  // v60 commit C
  // embedding rows on the host (the reference's embed_row without planes), then one copy
  Status embed(const WeightView& et, const std::int32_t* tokens, int n, float* dst, int64_t e) {
    xe_h.resize(z(n) * z(e));
    for (int t = 0; t < n; ++t) {
      const std::int32_t tok = tokens[t];
      if (tok < 0 || static_cast<std::uint64_t>(tok) >= s.n_vocab) return Status::fail(Code::invalid_argument, "a token id is outside the vocabulary");
      const Status qs = quant::dequantize_row(static_cast<quant::Type>(et.type), et.data + static_cast<std::uint64_t>(tok) * et.row_bytes, xe_h.data() + z(t) * z(e), e);
      if (!qs.ok()) return Status::fail(Code::unsupported, "engine: the embedding type cannot be read on the host");
    }
    return g::to_device(dst, xe_h.data(), xe_h.size() * sizeof(float)) ? Status::good() : gfail("engine: embedding upload");
  }

  // v24: the MoE block of a 1-4 token step with the picked-expert reader: router, rounding, every routed expert on the GPU
  // (bank experts from VRAM, the rest from the registered file mapping), then the shared expert; nothing waits on the CPU
  // v52: whether this layer's MoE half can run as one launch (g::moe_half): the R3 expert stage with the shared expert folded
  // in, an F32 router, the shared-memory picks; none of the side outputs the separate kernels feed (router margins, op
  // stamps, traces, probes, the CPU split, the plane tail, prefetch, the halves)
  // v58b: the first condition that keeps a layer's MoE half off the fused launch, or nullptr when it can run there (v58
  // MEASURED: the MTP block's took the old path on every step; this names why)
  const char* fused_why(const Layer& D, int64_t n, bool probe_row) const {
    const int64_t e = s.n_embd;
    auto k_ok = [](int64_t k) { return k == 256 || k == 512 || k == 1024 || k == 2048 || k == 4096; };
    if (!k_ok(e) || !g::fused_width_ok(static_cast<int>(e)) || !k_ok(s.n_ff_exp)) return "a width the fused kernels are not built for";
    if (s.n_expert > 1024 || s.n_expert_used > 8) return "more than 1024 experts or 8 picks";
    if (!rx_ok) return "the R3 expert stage is off";
    if (!fold_ok) return "the shared expert is not folded";
    if (rlg_d || trc_src_d || trc_sel_d || probe_row) return "a side output (router margins, trace, probe)";
    if (n < 1 || n > 4) return "a step of more than 4 tokens";
    const float* wr = router_of(D);
    if (!wr) return "the router is not F32 (and has no F32 copy)";
    if (!D.gate_inp_sh) return "no shared-expert gate";
    if (!D.post_norm) return "no post-attention norm";
    if ((reinterpret_cast<std::uintptr_t>(wr) & 15u) != 0 || (reinterpret_cast<std::uintptr_t>(D.gate_inp_sh) & 15u) != 0) return "router rows not 16-byte aligned";
    if (!g::router_fast_for(s.n_expert, o.router_tie)) return "the fast router does not take this expert count / tie band";
    return nullptr;
  }
  // the router rows the fused launch reads: gate_inp itself when stored as plain F32 rows, else the layer's F32 copy
  const float* router_of(const Layer& D) const {
    if (D.router_f32) return D.router_f32;
    const int64_t e = s.n_embd;
    if (D.gate_inp.v && !D.gate_inp.v->rot32 && D.gate_inp.dtype == static_cast<std::uint32_t>(quant::Type::f32) &&
        D.gate_inp.drb == static_cast<std::uint64_t>(e) * sizeof(float))
      return static_cast<const float*>(D.gate_inp.d);
    return nullptr;
  }
  bool fused_ok(const Layer& D, int64_t n, bool probe_row) const { return fused_why(D, n, probe_row) == nullptr; }

  // v52: x += y (the block's output, in y), then the MoE half, then x += y, in one launch
  Status moe_half_fused(Layer& D, int64_t n, std::uint32_t il) {
    g::MoeHalfArgs A;
    A.x = x;
    A.a = y;
    A.post_w = D.post_norm;
    A.eps = s.rms_eps;
    A.n = static_cast<int>(n);
    A.H = static_cast<int>(s.n_embd);
    A.F = static_cast<int>(s.n_ff_exp);
    A.ne = static_cast<int>(s.n_expert);
    A.nu = static_cast<int>(s.n_expert_used);
    A.xn = xn;
    A.xr = xr;
    A.xq = q;
    A.xs = scale;
    A.xsum = sum;
    A.wr = router_of(D);  // v58b
    A.logits = logits_r;
    A.wsh = D.gate_inp_sh;
    A.gsh = gsh;
    A.wscale = s.expert_weights_scale;
    A.band = o.router_tie;
    A.sel = sel;
    A.wsel = wsel;
    A.tab = rtab + static_cast<int64_t>(il) * s.n_expert;
    A.shx = rshx + il;
    if (!rcache.empty()) A.cache = rcache[il];
    A.grp = rgrp;
    A.gsrc = rgsrc;
    A.gfill = rgfill;
    A.gu = rgu;
    A.hq = rhq;
    A.hs = rhs;
    A.hsum = rhsum;
    A.yd = ryd;
    A.plan = rxplan;
    A.max_picks = rx_maxp;
    A.load_mode = kRxLoad;
    A.y = y;
    A.ph = ph_d && il < s.n_layer && !in_prompt ? ph_d + static_cast<int64_t>(il) * 5 : nullptr;  // v55b (v58: not the MTP row; v60 C: not prompt steps, whose stamps nobody reads)
    // v57 copy pipeline + shared activations (v60: locked on; v59d X3 MEASURED no gain without it; X2 / v59g G5-G6 neutral-to-
    // better). v60 H (Fix 4 A/B): off for the (256, 2) instantiation (two blocks' buffers do not fit) or with rx_pipe 0
    A.pipe = o.rx_pipe != 0 && o.moe_blocks != 2 ? 1 : 0;
    A.sact = A.pipe;
    A.blocks = o.moe_blocks;
    A.st = il < s.n_layer ? rst_d : rst_mtp_d;                    // v55e3 (v60 C: the MTP row has its own counters)
    // v55f / v55f2: the bit planes only feed planed experts' 1-bit route; with no planes (--plane-bits 0) nothing reads
    // them, so they are not built (v55f2 S0 +0.33 ms vs g55e3, MEASURED)
    // v60 I (0.5): the plane formats' dp4a route by default (no activation bit planes built or read); rx_bit_route 1 = the 1-bit route
    A.xb = o.plane_bits > 0 && o.rx_bit_route != 0 ? rxb : nullptr;
    A.hb = o.plane_bits > 0 && o.rx_bit_route != 0 ? rhb : nullptr;
    if (!g::moe_half(A)) return Status::fail(Code::internal, std::string("engine: fused MoE half: ") + g::stream_last_error());
    return Status::good();
  }
  Status moe_block_reader(Layer& D, int64_t n, std::uint32_t il) {
    const int64_t e = s.n_embd, ne = s.n_expert, nu = s.n_expert_used, ffs = s.n_ff_shexp;
    Status st;
    if (!(st = prod(D.gate_inp, xn, n, logits_r))) return st;
    if (rlg_d && !g::dcopy(rlg_d + static_cast<int64_t>(il) * kReaderMax * ne, logits_r, z(n * ne) * sizeof(float))) return gfail("engine: router margins");  // v49
    if (!g::router_topk(logits_r, ne, n, static_cast<int>(nu), s.expert_weights_scale, sel, wsel, nullptr, o.router_tie)) return gfail("engine: router");
    if (!g::act_q8x32(xn, e, n, e, false, xr, e, q, scale, sum)) return gfail("engine: expert input rounding");
    const bool fold = fold_ok;
    if (fold && (!g::matmul(static_cast<std::uint32_t>(quant::Type::f32), D.gate_inp_sh, static_cast<std::uint64_t>(e) * sizeof(float), 1, e, xn, n, e, gsh, 1) ||
                 !g::deltanet_gates(gsh, galpha, zero2, zero2, n, 1)))
      return gfail("engine: shared gate");
    unsigned long long* tsl = trc_ts_d ? trc_ts_d + 6 * static_cast<int64_t>(il) : nullptr;
    // v46a-1: the R3 expert stage (v60: the only expert path; the reader v3 kernels, the split and the plane test are gone)
    if (!g::moe_r3(rtab + static_cast<int64_t>(il) * ne, fold ? rshx + il : nullptr, sel, wsel, gsh, static_cast<int>(nu), q, scale, sum, n, e, s.n_ff_exp,
                   rcache.empty() ? nullptr : &rcache[il], rgu, rhq, rhs, rhsum, ryd, rgrp, rgsrc, rgfill, rxplan, rx_maxp, y, tsl, nullptr, kRxLoad))
      return Status::fail(Code::internal, std::string("engine: R3 expert stage: ") + g::stream_last_error());
    if (trc_src_d) {
      const int64_t P = kReaderMax * (nu + 1);
      if (!g::dcopy(trc_src_d + static_cast<int64_t>(il) * P, rgsrc, z(n * (nu + 1)) * sizeof(g::ExpertPtr)) ||
          !g::dcopy(trc_fill_d + static_cast<int64_t>(il) * P, rgfill, z(n * (nu + 1)) * sizeof(g::ExpertPtr)))
        return gfail("engine: trace copy");
    }
    if (trc_sel_d && (!g::dcopy(trc_sel_d + static_cast<int64_t>(il) * kReaderMax * nu, sel, z(n * nu) * sizeof(std::int32_t)) ||
                      !g::dcopy(trc_u_d + il, rgrp, sizeof(int))))
      return gfail("engine: trace copy");
    if (!fold) {
      if (D.gate_sh.r3 && D.up_sh.r3 && D.gate_sh.k == D.up_sh.k) {
        if (!(st = prod_r3({{&D.gate_sh, gs}, {&D.up_sh, us}}, xn, n))) return st;
      } else if (!(st = prod(D.gate_sh, xn, n, gs)) || !(st = prod(D.up_sh, xn, n, us))) {
        return st;
      }
      if (!g::silu_mul(gs, us, gs, n * ffs)) return gfail("engine: shared silu");
      if (!(st = prod(D.down_sh, gs, n, sh))) return st;
      if (!g::matmul(static_cast<std::uint32_t>(quant::Type::f32), D.gate_inp_sh, static_cast<std::uint64_t>(e) * sizeof(float), 1, e, xn, n, e, gsh, 1) ||
          !g::deltanet_gates(gsh, galpha, zero2, zero2, n, 1))
        return gfail("engine: shared gate");
      if (!g::add_rows(y, e, sh, e, gsh, e, n)) return gfail("engine: shared add");
    }
    return Status::good();
  }
  // v25 trace of one reader step (after the step's sync): S line, then one L line per layer
  void trace_step(int n, double wall_s) {
    const int64_t nu = s.n_expert_used, ne = s.n_expert, mp = kReaderMax * nu;
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    char b[256];
    std::snprintf(b, sizeof b, "S\t%llu\t%d\t%.3f\t%lld\n", static_cast<unsigned long long>(trace_seq++), n, wall_s * 1e3, static_cast<long long>(now_ms));
    trace_text += b;
    for (std::uint32_t il = 0; il < s.n_layer; ++il) {
      const std::int32_t* sp = trc_sel_h + static_cast<int64_t>(il) * mp;
      std::vector<int> seen;
      std::uint64_t bv = 0, bh = 0;
      int n_hot = 0, n_cold = 0;
      std::string picks;
      for (int64_t pi = 0; pi < n * nu; ++pi) {
        const int ex = sp[pi];
        picks += (pi > 0 ? (pi % nu == 0 ? ";" : ",") : "") + std::to_string(ex);
        if (ex < 0 || ex >= ne) continue;
        if (L[il].hot[z(ex)]) ++n_hot;
        else ++n_cold;
        if (std::find(seen.begin(), seen.end(), ex) != seen.end()) continue;
        seen.push_back(ex);
        const g::ExpertPtr& X = rtab_h[z(static_cast<int64_t>(il) * ne + ex)];
        const std::uint64_t bytes = X.rb[0] * s.n_ff_exp + X.rb[1] * s.n_ff_exp + X.rb[2] * s.n_embd;
        (L[il].hot[z(ex)] ? bv : bh) += bytes;
      }
      const unsigned long long* qs = trc_ts_h + 6 * il;
      auto dus = [&](int a, int c) { return static_cast<double>(static_cast<std::int64_t>(qs[c] - qs[a])) * 1e-3; };
      std::snprintf(b, sizeof b, "L\t%llu\t%u\t%d\t%d\t%d\t%d\t%llu\t%llu\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t", static_cast<unsigned long long>(trace_seq - 1), il,
                    trc_u_h[il], static_cast<int>(seen.size()), n_hot, n_cold, static_cast<unsigned long long>(bv), static_cast<unsigned long long>(bh), dus(0, 1), dus(1, 2),
                    dus(2, 3), dus(3, 4), dus(4, 5));
      trace_text += b;
      trace_text += picks;
      if (trc_src_h) {  // v26: groups served from VRAM by the cache, and misses written into a slot
        const int64_t P = kReaderMax * (nu + 1);
        const int U = trc_u_h[il];
        int hits = 0, fills = 0;
        std::uint64_t vram_b = 0, file_b = 0;
        for (int u = 0; u < U && u < P; ++u) {
          const g::ExpertPtr& X = trc_src_h[static_cast<int64_t>(il) * P + u];
          const std::uint64_t bytes = X.rb[0] * s.n_ff_exp + X.rb[1] * s.n_ff_exp + X.rb[2] * s.n_embd;
          (X.vram ? vram_b : file_b) += bytes;
          if (trc_fill_h[static_cast<int64_t>(il) * P + u].w[0]) ++fills;
        }
        for (int u = 0; u < U && u < P; ++u) {
          const g::ExpertPtr& X = trc_src_h[static_cast<int64_t>(il) * P + u];
          bool bank = false;
          for (int64_t ex = 0; ex < ne; ++ex)
            if (rtab_h[z(static_cast<int64_t>(il) * ne + ex)].w[0] == X.w[0] && rtab_h[z(static_cast<int64_t>(il) * ne + ex)].vram) bank = true;
          if (X.vram && !bank && u < static_cast<int>(seen.size())) ++hits;
        }
        std::snprintf(b, sizeof b, "\t%d\t%d\t%llu\t%llu", hits, fills, static_cast<unsigned long long>(vram_b), static_cast<unsigned long long>(file_b));
        trace_text += b;
      }
      trace_text += "\n";
    }
  }
  // v24: register the span of the model file holding the trunk's routed experts, and build the reader's tables
  void setup_reader() {
    const int64_t ne = s.n_expert;
    if (s.n_embd % 32 != 0 || s.n_ff_exp % 32 != 0) {
      reader_note = "off (shapes not a multiple of 32)";
      return;
    }
    std::uintptr_t lo = UINTPTR_MAX, hi = 0;
    for (std::uint32_t il = 0; il < nlr(); ++il)  // v58: nlr() = n_layer (+ the MTP block's row with mtp_fused)
      for (const WeightView* v : {&WL(il).gate_exps, &WL(il).up_exps, &WL(il).down_exps}) {
        if (v->rot32) {
          reader_note = "off (a rotated expert weight)";
          return;
        }
        for (int64_t ex = 0; ex < ne; ++ex) {
          const ExpRows r = exp_rows(*v, static_cast<int>(ex));
          if (!g::type_supported(static_cast<std::uint32_t>(r.t)) && il >= s.n_layer) {  // v58: the MTP block keeps the old path
            mtp_rd = false;
            break;
          }
          if (!g::type_supported(static_cast<std::uint32_t>(r.t))) {
            reader_note = "off (an expert type the reader cannot decode)";
            return;
          }
          const auto a = reinterpret_cast<std::uintptr_t>(r.base);
          lo = std::min(lo, a);
          hi = std::max(hi, a + static_cast<std::uintptr_t>(r.rb * v->ne[1]));
        }
      }
    if (hi <= lo) return;
    // v54: the bit planes: every Q4_TG2 matrix of an expert outside the bank (and not the CPU's) copied row by row into the
    // plane layout, in pinned mapped memory; those matrices' file bytes are then not registered
    std::vector<std::uint64_t> poff;  // per (layer, matrix, expert): offset in planes_h + 1 (0: not in planes)
    void* planes_dev = nullptr;
    if (o.plane_bits > 0) {
      poff.assign(z(static_cast<int64_t>(nlr()) * 3 * ne), 0);
      std::uint64_t tot = 0;
      struct PJob {
        const std::uint8_t* src;
        std::uint64_t off, rb, pf, rows;  // v60: pf = the prefix (the slider's bits) each row is packed to
        std::uint32_t t;
      };
      std::vector<PJob> jobs;
      for (std::uint32_t il = 0; il < nlr(); ++il) {
        const WeightView* vv[3] = {&WL(il).gate_exps, &WL(il).up_exps, &WL(il).down_exps};
        for (int pj = 0; pj < 3; ++pj)
          for (int64_t ex = 0; ex < ne; ++ex) {
            if (LL(il).hot[z(ex)]) continue;
            const ExpRows r = exp_rows(*vv[pj], static_cast<int>(ex));
            const std::uint32_t pt = planed_type(r.t, r.rb);
            if (!pt) continue;  // v54c: Q4_TG2 and Q5_K
            // v60 (Fix 2): rows packed at the prefix pf (the slider's bits), as the bank and the slots hold them: the pinned
            // planes shrink (13.24 GB -> the prefix share at 3 bits) and a task's rows are one contiguous run
            const std::uint64_t rows = static_cast<std::uint64_t>(vv[pj]->ne[1]), pf = kern::gpu::plane_prefix(pt, r.rb, row_bits(il)), b = pf * rows;  // v60 F2: the row's bits
            poff[z((static_cast<int64_t>(il) * 3 + pj) * ne + ex)] = tot + 1;
            jobs.push_back({r.base, tot, r.rb, pf, rows, static_cast<std::uint32_t>(r.t)});
            tot += (b + 255) / 256 * 256;  // each matrix 256-byte aligned
          }
      }
      if (tot > 0) {
        const Clock::time_point tp = Clock::now();
        // v60 commit C2: the pack as a function (the pinned-read probe below may re-pin and pack again)
        auto pack = [&]() -> bool {
          planes_h = g::host_alloc_mapped(static_cast<std::size_t>(tot), &planes_dev);
          if (!planes_h || !planes_dev) {
            planes_h = nullptr;
            planes_dev = nullptr;
            return false;
          }
          auto* ph = static_cast<std::uint8_t*>(planes_h);
          const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
          std::atomic<std::size_t> next{0};
          std::vector<std::thread> th;
          for (unsigned t = 0; t < nt; ++t)
            th.emplace_back([&]() {
              std::vector<std::uint8_t> row;
              for (std::size_t i = next.fetch_add(1); i < jobs.size(); i = next.fetch_add(1)) {
                const PJob& J = jobs[i];
                row.resize(z(static_cast<int64_t>(J.rb)));
                for (std::uint64_t rr = 0; rr < J.rows; ++rr) {  // the plane layout is plane-major: the prefix is the row's first pf bytes
                  kern::gpu::plane_row(J.t, J.src + rr * J.rb, row.data(), J.rb);
                  std::memcpy(ph + J.off + rr * J.pf, row.data(), static_cast<std::size_t>(J.pf));
                }
              }
            });
          for (auto& t : th) t.join();
          return true;
        };
        if (!pack()) {
          reader_note = "off (pinned memory for the bit planes: " + std::to_string(tot >> 20) + " MiB)";
          return;
        }
        planes_b = tot;
        const double pack_s = since(tp);
        // v60 commit C2 (v60d MEASURED: three of five arms read the pinned planes 2-3x slower, GPU 99 % busy at 84 W, same
        // binary, same VM): the pinned-read probe. The copy engine reads `pb_bytes` of the planes into a VRAM scratch (the
        // mapping the expert stage's loads go through); best of two passes in GB/s. Below o.pin_read_min_gbs (0: log only)
        // the planes are freed, pinned and packed again, up to o.pin_read_tries times; every try's number is logged
        // v60 C3 (v60e MEASURED: the copy engine read 26 GB/s in the slow arm too): three readers of the same 512 MB - the copy
        // engine, SM streaming loads, SM one-line-per-page loads (the expert stage's kind); the floor applies to the page-line
        // number (GB/s of 128 B lines), the one that should see a badly mapped range
        double gbs_copy = 0.0, gbs_sm = 0.0, gbs_pg = 0.0, pg_first = 0.0;
        int tries = 1;
        {
          const std::size_t pb_bytes = std::min<std::size_t>(std::size_t{512} << 20, static_cast<std::size_t>(tot)) & ~std::size_t{4095};
          void* scratch = pb_bytes >= 4096 ? g::dalloc(pb_bytes + 256) : nullptr;
          unsigned* live = scratch ? reinterpret_cast<unsigned*>(static_cast<std::uint8_t*>(scratch) + pb_bytes) : nullptr;
          auto timed = [&](const std::function<bool()>& f, double bytes_moved) -> double {
            double best = 0.0;
            for (int pass = 0; pass < 2; ++pass) {
              if (!g::sync()) return -1.0;
              const Clock::time_point t1 = Clock::now();
              if (!f() || !g::sync()) return -1.0;
              const double sec = since(t1);
              if (sec > 0.0) best = std::max(best, bytes_moved / sec / 1e9);
            }
            return best;
          };
          auto probe = [&]() {
            if (!scratch) {
              gbs_copy = gbs_sm = gbs_pg = -1.0;
              return;
            }
            gbs_copy = timed([&]() { return g::dcopy_any(scratch, planes_dev, pb_bytes); }, static_cast<double>(pb_bytes));
            gbs_sm = timed([&]() { return g::probe_read_sm(planes_dev, pb_bytes, 0, live); }, static_cast<double>(pb_bytes));
            gbs_pg = timed([&]() { return g::probe_read_sm(planes_dev, pb_bytes, 1, live); }, static_cast<double>(pb_bytes / 4096) * 128.0);
          };
          probe();
          pg_first = gbs_pg;
          while (o.pin_read_min_gbs > 0.0 && gbs_pg >= 0.0 && gbs_pg < o.pin_read_min_gbs && tries < std::max(1, o.pin_read_tries)) {
            g::host_free(planes_h);
            planes_h = nullptr;
            planes_dev = nullptr;
            ++tries;
            if (!pack()) {
              reader_note = "off (pinned memory for the bit planes, re-pin " + std::to_string(tries) + ")";
              g::dfree(scratch);
              return;
            }
            probe();
          }
          g::dfree(scratch);
          pin_read_gbs = gbs_pg;
          pin_read_tries = tries;
        }
        char pb[520], mb2[64] = "";
        if (o.mtp && row_bits(s.n_layer) != o.plane_bits) std::snprintf(mb2, sizeof mb2, ", the MTP row with %d (v60 F2)", row_bits(s.n_layer));
        std::snprintf(pb, sizeof pb,
                      "; bit planes: %zu Q4_TG2 / Q5_K / Q4_K matrices outside the bank, %.2f GB pinned (rows packed at each row's prefix, v60), built in %.2f s; every pick read with %d bits%s (at most each matrix's own), bank and slots sized for them; pinned read (512 MB, best of 2): copy engine %.1f, SM streaming %.1f, SM one line per page %.1f GB/s (first pin %.1f, %d pin%s, floor %.0f on the page lines)",
                      jobs.size(), static_cast<double>(tot) / 1e9, pack_s, o.plane_bits, mb2, gbs_copy, gbs_sm, gbs_pg, pg_first, tries, tries == 1 ? "" : "s", o.pin_read_min_gbs);
        planes_note = pb;
      }
    }
    // v27: pin only the experts the bank does not hold (the GPU reads the bank's from VRAM): their matrices as page-aligned
    // ranges, merged where the gap between them is under o.pin_gap (v27: 8 MiB; fewer, larger registrations); each range's
    // device address. v54: not the matrices held in bit planes.
    const std::uintptr_t pg = 4096;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> want;
    for (std::uint32_t il = 0; il < nlr(); ++il)
      for (const WeightView* v : {&WL(il).gate_exps, &WL(il).up_exps, &WL(il).down_exps})
        for (int64_t ex = 0; ex < ne; ++ex) {
          if (LL(il).hot[z(ex)]) continue;
          const int pjv = v == &WL(il).gate_exps ? 0 : v == &WL(il).up_exps ? 1 : 2;
          if (!poff.empty() && poff[z((static_cast<int64_t>(il) * 3 + pjv) * ne + ex)]) continue;
          const ExpRows r = exp_rows(*v, static_cast<int>(ex));
          const auto a = reinterpret_cast<std::uintptr_t>(r.base);
          want.emplace_back(a & ~(pg - 1), (a + static_cast<std::uintptr_t>(r.rb * v->ne[1]) + pg - 1) & ~(pg - 1));
        }
    std::sort(want.begin(), want.end());
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> runs;
    for (const auto& iv : want) {
      if (!runs.empty() && iv.first <= runs.back().second + static_cast<std::uintptr_t>(o.pin_gap)) runs.back().second = std::max(runs.back().second, iv.second);
      else runs.push_back(iv);
    }
    std::vector<const std::uint8_t*> run_dev;
    std::uint64_t pinned = 0;
    std::string why;
    const Clock::time_point tr = Clock::now();
    std::size_t small_ok = 0;
    if (o.pin_small_pages)
      for (const auto& rn : runs)
        if (hal::advise_small_pages(reinterpret_cast<void*>(rn.first), static_cast<std::size_t>(rn.second - rn.first))) ++small_ok;
    for (const auto& rn : runs) {
      void* dv = nullptr;
      if (!g::host_register(reinterpret_cast<void*>(rn.first), static_cast<std::size_t>(rn.second - rn.first), &dv, &why)) {
        for (void* rb2 : reg_bases) g::host_unregister(rb2);
        reg_bases.clear();
        reader_note = "off (registering " + std::to_string((rn.second - rn.first) >> 20) + " MiB of the file mapping failed: " + why + ")";
        return;
      }
      reg_bases.push_back(reinterpret_cast<void*>(rn.first));
      run_dev.push_back(static_cast<const std::uint8_t*>(dv));
      pinned += rn.second - rn.first;
    }
    auto dev_of = [&](const std::uint8_t* hp) -> const std::uint8_t* {
      const auto a = reinterpret_cast<std::uintptr_t>(hp);
      auto it = std::upper_bound(runs.begin(), runs.end(), std::make_pair(a, UINTPTR_MAX));
      if (it == runs.begin()) return nullptr;
      --it;
      if (a >= it->second) return nullptr;
      return run_dev[z(it - runs.begin())] + (a - it->first);
    };
    const std::uintptr_t b0 = 0, b1 = pinned;
    const double reg_s = since(tr);
    std::vector<g::ExpertPtr> tab(z(static_cast<int64_t>(nlr()) * ne));
    for (std::uint32_t il = 0; il < nlr(); ++il) {
      const Qwen35MoeLayer& l = WL(il);
      Layer& Li = LL(il);
      const WeightView* vv[3] = {&l.gate_exps, &l.up_exps, &l.down_exps};
      for (int64_t ex = 0; ex < ne; ++ex) {
        g::ExpertPtr& X = tab[z(static_cast<int64_t>(il) * ne + ex)];
        for (int pj = 0; pj < 3; ++pj) {
          const ExpRows r = exp_rows(*vv[pj], static_cast<int>(ex));
          const std::uint8_t* bank = Li.hot[z(ex)] && !Li.bank_at[pj].empty() ? Li.bank_at[pj][z(ex)] : nullptr;
          X.w[pj] = bank ? bank : dev_of(r.base);
          X.type[pj] = static_cast<std::uint32_t>(r.t);
          X.rb[pj] = r.rb;
          X.rs[pj] = r.rb;
          if (bank) X.vram = 1;
          if (!bank && !poff.empty() && poff[z((static_cast<int64_t>(il) * 3 + pj) * ne + ex)]) {  // v54: from the bit planes
            X.w[pj] = static_cast<const std::uint8_t*>(planes_dev) + (poff[z((static_cast<int64_t>(il) * 3 + pj) * ne + ex)] - 1);
            X.type[pj] = planed_type(r.t, r.rb);
            X.rs[pj] = kern::gpu::plane_prefix(X.type[pj], r.rb, row_bits(il));  // v60: packed at the prefix (F2: the row's bits)
          }
          if (bank && planed_type(r.t, r.rb)) {  // v54c: the planed bank holds the slider's prefix of each row
            X.type[pj] = planed_type(r.t, r.rb);
            X.rb[pj] = kern::gpu::plane_prefix(X.type[pj], r.rb, row_bits(il));
            X.rs[pj] = X.rb[pj];
          }
        }
      }
    }
    for (std::size_t ti = 0; ti < tab.size(); ++ti) {
      const g::ExpertPtr& X = tab[ti];
      if (!X.w[0] || !X.w[1] || !X.w[2]) {
        reader_note = "off (an expert outside the pinned ranges)";
        return;
      }
    }
    // v26: the shared expert as a pick (its three matrices as the batched kernels read them, in VRAM)
    std::vector<g::ExpertPtr> shx_h;
    if (s.n_ff_shexp == s.n_ff_exp) {  // v60: always folded when the shapes allow
      bool ok = true;
      for (std::uint32_t il = 0; il < nlr() && ok; ++il) {
        const Layer& D = LL(il);
        g::ExpertPtr X;
        const DevW* dw[3] = {&D.gate_sh, &D.up_sh, &D.down_sh};
        for (int pj = 0; pj < 3; ++pj) {
          const DevW& W = *dw[pj];
          if (!W.d || !W.v || W.v->rot32 || !g::type_supported(W.dtype) || W.k != (pj < 2 ? s.n_embd : s.n_ff_shexp)) ok = false;
          X.w[pj] = static_cast<const std::uint8_t*>(W.d);
          X.type[pj] = W.dtype;
          X.rb[pj] = W.drb;
          X.rs[pj] = W.drb;
        }
        if (!ok && il >= s.n_layer) {  // v58: the MTP block's shared expert cannot be folded: its MoE half stays on the old path
          mtp_rd = false;
          ok = true;
          break;
        }
        X.vram = 1;
        shx_h.push_back(X);
      }
      if (ok) fold_ok = true;
      else shx_h.clear();
    }
    const int64_t mp = kReaderMax * static_cast<int64_t>(s.n_expert_used), ff = s.n_ff_exp;
    const int64_t P = kReaderMax * (static_cast<int64_t>(s.n_expert_used) + 1);  // v26: room for the shared expert's picks
    (void)mp;
    rtab = static_cast<g::ExpertPtr*>(g::dalloc(tab.size() * sizeof(g::ExpertPtr)));
    rgu = static_cast<float*>(g::dalloc(z(P * 2 * ff) * sizeof(float)));
    ryd = static_cast<float*>(g::dalloc(z(P * s.n_embd) * sizeof(float)));
    rgrp = static_cast<int*>(g::dalloc(z(1 + 6 * P) * sizeof(int)));
    rgsrc = static_cast<g::ExpertPtr*>(g::dalloc(z(P) * sizeof(g::ExpertPtr)));
    rgfill = static_cast<g::ExpertPtr*>(g::dalloc(z(P) * sizeof(g::ExpertPtr)));
    if (!shx_h.empty()) {
      rshx = static_cast<g::ExpertPtr*>(g::dalloc(shx_h.size() * sizeof(g::ExpertPtr)));
      if (!rshx || !g::to_device(rshx, shx_h.data(), shx_h.size() * sizeof(g::ExpertPtr))) fold_ok = false;
    }
    if (!rtab || !rgu || !ryd || !rgrp || !rgsrc || !rgfill || !g::to_device(rtab, tab.data(), tab.size() * sizeof(g::ExpertPtr))) {
      reader_note = "off (device memory for the reader's tables)";
      fold_ok = false;
      return;
    }
    dev_bytes += tab.size() * sizeof(g::ExpertPtr) + z(P * 2 * ff + P * s.n_embd) * sizeof(float) + z(1 + 6 * P) * sizeof(int) + z(2 * P) * sizeof(g::ExpertPtr);
    // v26 cache: the bytes reserved before the bank was filled
    std::string cache_note;
    if (cache_reserved > 0) {
      int K = rc_eff;
      const std::size_t nl = nlr();
      const std::uint64_t budget0 = cslot * static_cast<std::uint64_t>(rc_eff);  // per layer, as before (v54b: rc_eff, the probe's or --reader-cache)
      // v60 commit C: the MTP row (the last region) gets mtp_cache_bonus more; the trunk regions are budget0 apart
      auto budget_of = [&](std::size_t il) { return budget0 + (il == s.n_layer ? mtp_cache_bonus : 0); };
      // v51 size pools: per layer the file experts (not in the bank) by real size; each size's share of the budget follows
      // its experts' routes counts x size (counts + 1); at most kMaxPools sizes (the smallest-demand size merges upward)
      std::vector<g::ReaderCache> pc(nl);
      bool pools_ok = true;  // v51b
      std::vector<signed char> pool_h(nl * z(ne), -1);
      int kpool_max = K;
      {  // v51 size pools (v60: locked on; replay52c MEASURED misses x0.896)
        kpool_max = 0;
        for (std::size_t il = 0; il < nl; ++il) {
          const Qwen35MoeLayer& l = WL(static_cast<std::uint32_t>(il));
          Layer& Li = LL(static_cast<std::uint32_t>(il));
          const std::uint64_t budget = budget_of(il);
          std::vector<std::uint64_t> esz(z(ne), 0);
          std::map<std::uint64_t, double> dem;
          for (int ex = 0; ex < static_cast<int>(ne); ++ex) {
            if (Li.hot[z(ex)]) continue;
            std::uint64_t b = 0;
            for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
              const ExpRows r = exp_rows(*v, ex);
              // v54b: a matrix held in bit planes takes the slider's prefix of each row in its slot (as rd3_group_one lays it)
              const std::uint32_t pt = planed_type(r.t, r.rb);
              b += (pt ? kern::gpu::plane_prefix(pt, r.rb, row_bits(static_cast<std::uint32_t>(il))) : r.rb) * static_cast<std::uint64_t>(v->ne[1]);  // v60 F2
            }
            esz[z(ex)] = (b + 255) & ~std::uint64_t{255};
            const double cn = il < o.hot_count.size() && z(ex) < o.hot_count[il].size() ? static_cast<double>(o.hot_count[il][z(ex)]) : 0.0;
            dem[esz[z(ex)]] += (cn + 1.0) * static_cast<double>(esz[z(ex)]);
          }
          while (dem.size() > static_cast<std::size_t>(g::ReaderCache::kMaxPools)) {  // merge the smallest-demand size (not the largest) upward
            auto lw = dem.begin();
            for (auto it = dem.begin(); std::next(it) != dem.end(); ++it)
              if (it->second < lw->second) lw = it;
            const double d = lw->second;
            auto up = std::next(lw);
            dem.erase(lw);
            up->second += d;
          }
          double tot = 0.0;
          for (const auto& kv : dem) tot += kv.second;
          // v51b: slot counts first, then the layout. Each size gets floor(budget x share / size) slots (at least 1 while it
          // fits); if the forced ones push the layer past its budget, the size with the most slots gives one back until it fits
          // (v51 laid the pools out as it went and could run past the budget into the next layer's region: G51's garbage)
          std::vector<std::uint64_t> szs;
          std::vector<double> dms;
          std::vector<int> cnt;
          for (const auto& kv : dem) {
            szs.push_back(kv.first);
            dms.push_back(kv.second);
            cnt.push_back(tot > 0 ? static_cast<int>(static_cast<double>(budget) * kv.second / tot / static_cast<double>(kv.first)) : 0);
          }
          auto total = [&]() {
            std::uint64_t u = 0;
            for (std::size_t pi = 0; pi < szs.size(); ++pi) u += static_cast<std::uint64_t>(cnt[pi]) * szs[pi];
            return u;
          };
          for (std::size_t pi = 0; pi < szs.size(); ++pi)
            if (cnt[pi] < 1 && total() + szs[pi] <= budget) cnt[pi] = 1;
          while (total() > budget) {
            std::size_t mx = 0;
            for (std::size_t pi = 1; pi < szs.size(); ++pi)
              if (cnt[pi] > cnt[mx]) mx = pi;
            if (cnt[mx] == 0) break;
            --cnt[mx];
          }
          // leftover bytes: one more slot at a time to the size of most demand per byte per slot that still fits
          for (bool more = true; more;) {
            more = false;
            int best = -1;
            double bd = -1.0;
            for (std::size_t pi = 0; pi < szs.size(); ++pi)
              if (total() + szs[pi] <= budget && dms[pi] / static_cast<double>(szs[pi]) / (cnt[pi] + 1) > bd) {
                bd = dms[pi] / static_cast<double>(szs[pi]) / (cnt[pi] + 1);
                best = static_cast<int>(pi);
              }
            if (best >= 0) {
              ++cnt[static_cast<std::size_t>(best)];
              more = true;
            }
          }
          g::ReaderCache& c = pc[il];
          std::uint64_t used = 0;
          int k = 0;
          for (std::size_t pi = 0; pi < szs.size(); ++pi) {
            c.pbytes[pi] = szs[pi];
            c.pboff[pi] = used;
            c.pk0[pi] = k;
            used += static_cast<std::uint64_t>(cnt[pi]) * szs[pi];
            k += cnt[pi];
          }
          c.np = static_cast<int>(szs.size());
          c.pk0[c.np] = k;
          if (used > budget) pools_ok = false;  // never expected after the loop above; checked, not assumed
          c.K = c.pk0[c.np];
          kpool_max = std::max(kpool_max, c.K);
          for (int ex = 0; ex < static_cast<int>(ne); ++ex) {
            if (esz[z(ex)] == 0) continue;
            for (int pq = 0; pq < c.np; ++pq)
              if (c.pbytes[pq] >= esz[z(ex)] && c.pk0[pq + 1] > c.pk0[pq]) {
                pool_h[il * z(ne) + z(ex)] = static_cast<signed char>(pq);
                break;
              }
          }
        }
        K = kpool_max;
        if (!pools_ok) {  // v51b: fall back to the fixed slots rather than run past a layer's budget
          K = rc_eff;
          pc.assign(nl, g::ReaderCache{});
        }
      }
      const bool use_pools = pools_ok;
      const std::size_t meta_b = nl * (static_cast<std::size_t>(K) * 8 + 16) + (use_pools ? nl * z(ne) : 0);
      cache_mem = g::dalloc(cache_reserved);
      cache_meta = g::dalloc(meta_b);
      if (cache_mem && cache_meta) {
        std::vector<std::uint8_t> init(meta_b, 0);
        auto* ip = reinterpret_cast<int*>(init.data());
        for (std::size_t i = 0; i < nl * static_cast<std::size_t>(K); ++i) ip[i] = -1;  // tags: empty
        if (use_pools) std::memcpy(init.data() + nl * (static_cast<std::size_t>(K) * 8 + 16), pool_h.data(), pool_h.size());
        if (g::to_device(cache_meta, init.data(), meta_b)) {
          auto* mb = static_cast<std::uint8_t*>(cache_meta);
          int kmin = K, kall = 0;
          for (std::size_t il = 0; il < nl; ++il) {
            g::ReaderCache c = use_pools ? pc[il] : g::ReaderCache{};
            if (!use_pools) c.K = K;
            c.tag = reinterpret_cast<int*>(mb) + il * static_cast<std::size_t>(K);
            c.use = reinterpret_cast<unsigned*>(mb + nl * static_cast<std::size_t>(K) * 4) + il * static_cast<std::size_t>(K);
            c.stamp = reinterpret_cast<unsigned*>(mb + nl * static_cast<std::size_t>(K) * 8) + il;
            c.base = static_cast<std::uint8_t*>(cache_mem) + il * budget0;  // (the MTP row is last: its bonus extends the end)
            c.slot_bytes = cslot;
            if (use_pools) c.pool_of = reinterpret_cast<const signed char*>(mb + nl * (static_cast<std::size_t>(K) * 8 + 16)) + il * z(ne);
            if (o.plane_bits > 0) c.plane_bits = row_bits(static_cast<std::uint32_t>(il));  // v54b (v60 F2: the row's bits; the grouping reads this row's entry)
            kmin = std::min(kmin, c.K);
            kall += c.K;
            rcache.push_back(c);
          }
          dev_bytes += cache_reserved + meta_b;
          char cb[320];
          if (use_pools)
            std::snprintf(cb, sizeof cb, "; cache %.2f GB in size pools (%d slots of %d before; %d-%d slots per layer, %.1f average)",
                          static_cast<double>(cache_reserved) / 1e9, rc_eff, rc_eff, kmin, K, static_cast<double>(kall) / static_cast<double>(nl));
          else
            std::snprintf(cb, sizeof cb, "; cache %d slots per layer, %.2f GB (size pools off: layout over budget)", K, static_cast<double>(cache_reserved) / 1e9);
          cache_note = cb;
          if (use_pools && mtp_rd) {  // v60 C: the MTP row's extra cache
            char mnote[120];
            std::snprintf(mnote, sizeof mnote, "; %.0f MB more for the MTP row = %d slots (v60 C)", static_cast<double>(mtp_cache_bonus) / 1048576.0, rcache.back().K);
            cache_note += mnote;
          }
        }
      }
      if (rcache.empty()) cache_note = "; cache off (device memory)";
    }
    if (o.plane_bits > 0 && rcache.empty()) {  // v54b: no slots, but the grouping still needs the slider (F2: per row)
      rcache.assign(nlr(), g::ReaderCache{});
      for (std::uint32_t il = 0; il < nlr(); ++il) rcache[il].plane_bits = row_bits(il);
    }
    rtab_h = tab;
    // v46a-1: the R3 expert stage. It reads the experts as stored (Q4_TG2 / Q5_K / Q4_K with 16-byte aligned rows, Q6_K
    // 2-byte aligned) and the shared expert in R3's Q8_0 layout; each matrix's start 16-byte aligned (a miss's bytes go to
    // the same offsets in its slot, so the slots keep that alignment too)
    {  // v46a-1 (v60: the only expert path)
      const int64_t H = s.n_embd, Fx = s.n_ff_exp;
      auto mat_ok = [&](const g::ExpertPtr& X) {
        for (int pj = 0; pj < 3; ++pj) {
          if (!g::rx_type_ok(X.type[pj], pj < 2 ? H : Fx) || !X.w[pj]) return false;
          if (reinterpret_cast<std::uintptr_t>(X.w[pj]) % 16 != 0) return false;
          if (X.type[pj] == g::g_q6_k ? X.rs[pj] % 2 != 0 : X.rs[pj] % 16 != 0) return false;
        }
        return true;
      };
      bool ok = true;
      for (std::size_t ti = 0; ti < tab.size(); ++ti) {
        if (mat_ok(tab[ti])) continue;
        if (ti >= z(static_cast<int64_t>(s.n_layer) * s.n_expert)) mtp_rd = false;  // v58: only the MTP row: its MoE half keeps the old path
        else ok = false;
      }
      if (!ok) rx_note = "off (an expert matrix the stage cannot read: type or alignment)";
      for (const g::ExpertPtr& X : shx_h)
        if (ok && !mat_ok(X)) {
          ok = false;
          rx_note = "off (the shared expert's type or alignment)";
        }
      for (const g::ReaderCache& c : rcache)
        if (ok && (c.slot_bytes % 16 != 0 || reinterpret_cast<std::uintptr_t>(c.base) % 16 != 0)) {
          ok = false;
          rx_note = "off (cache slots not 16-byte aligned)";
        }
      if (ok) {
        rx_maxp = static_cast<int>(P);
        rhq = static_cast<std::int8_t*>(g::dalloc(z(P * Fx)));
        rhs = static_cast<float*>(g::dalloc(z(P * Fx / 32) * sizeof(float)));
        rhsum = static_cast<int*>(g::dalloc(z(P * Fx / 32) * sizeof(int)));
        const std::uint64_t pb = g::rx_scratch_bytes(rx_maxp);
        rxplan = g::dalloc(pb);
        // v55f: bit planes for the 1-bit route (8 words per 32 inputs = 1 byte per input): up to 4 tokens of H, every pick's
        // F; without them the stage reads planed experts by dp4a as before
        rxb = static_cast<std::uint32_t*>(g::dalloc(z(4 * s.n_embd)));
        rhb = static_cast<std::uint32_t*>(g::dalloc(z(P * Fx)));
        if (!rxb || !rhb) {
          g::dfree(rxb);
          g::dfree(rhb);
          rxb = rhb = nullptr;
        } else {
          dev_bytes += z(4 * s.n_embd) + z(P * Fx);
        }
        if (rhq && rhs && rhsum && rxplan) {
          dev_bytes += z(P * Fx) + z(P * Fx / 32) * 8 + pb;
          rx_ok = true;
          rx_note = std::string("on (gate / up, silu x up, down of every pick in one R3 launch per layer)") +
                    (o.moe_blocks == 2 ? "; one-token kernel at __launch_bounds__(256, 2), copy pipeline off (v60 H A/B)"
                                       : o.rx_pipe == 0 ? "; copy pipeline off (v60 H A/B)" : "") +
                    (o.plane_bits > 0 ? (o.rx_bit_route != 0 ? "; plane formats on the 1-bit route (v60 I A/B)" : "; plane formats on the dp4a route (v60 I)") : "");
          // v55b: the stage's one grid in shared memory: the slider's bits for every Q4_TG2 job (planes), else the stored grid
          {  // v55f: the staging buffer sized to the largest block of any matrix the stage stages (load mode 1: the experts
             // not in VRAM; 2: every expert and the shared one), at the slider's bits for the plane matrices
            int mb = 0;
            auto take = [&](const g::ExpertPtr& X) {
              for (int pj = 0; pj < 3; ++pj) mb = std::max(mb, g::rx_stage_block_bytes(X.type[pj], std::max(o.plane_bits, row_bits(s.n_layer))));  // v60 F2
            };
            for (const g::ExpertPtr& X : tab) take(X);  // v57: the pipeline stages every expert
            for (const g::ExpertPtr& X : shx_h) take(X);
            g::rx_set_stage_bytes(mb);
          }
          if (!g::rx_set_grid_bits(o.plane_bits > 0 ? std::min(o.plane_bits, 4) : 4)) {
            rx_ok = false;
            rx_note = std::string("off (grid bits: ") + g::stream_last_error() + ")";
          }
          if (rx_ok && o.verbose_gpu) {
            g::stream_kernel_report();
            void* phd = nullptr;
            ph_h = static_cast<unsigned long long*>(g::host_alloc_mapped(static_cast<std::size_t>(s.n_layer) * 5 * sizeof(unsigned long long), &phd));
            ph_d = static_cast<unsigned long long*>(phd);
            if (!ph_h || !ph_d) ph_h = ph_d = nullptr;
            void* dphd = nullptr;  // v56
            dph_h = static_cast<unsigned long long*>(g::host_alloc_mapped(static_cast<std::size_t>(s.n_layer) * 5 * sizeof(unsigned long long), &dphd));
            dph_d = static_cast<unsigned long long*>(dphd);
            if (!dph_h || !dph_d) dph_h = dph_d = nullptr;
            rst_d = static_cast<unsigned long long*>(g::dalloc(g::kRxStatN * sizeof(unsigned long long)));  // v55e3
            if (rst_d && !g::dzero(rst_d, g::kRxStatN * sizeof(unsigned long long))) {
              g::dfree(rst_d);
              rst_d = nullptr;
            }
            if (rst_d) {  // v60 commit C: the MTP row's counters apart from the trunk's
              rst_mtp_d = static_cast<unsigned long long*>(g::dalloc(g::kRxStatN * sizeof(unsigned long long)));
              if (rst_mtp_d && !g::dzero(rst_mtp_d, g::kRxStatN * sizeof(unsigned long long))) {
                g::dfree(rst_mtp_d);
                rst_mtp_d = nullptr;
              }
            }
          }
        } else {
          rx_note = "off (device memory)";
        }
      }
    }
    if (o.trace) {
      const std::size_t nl = s.n_layer, b_ts = nl * 6 * sizeof(unsigned long long), b_sel = nl * z(mp) * sizeof(std::int32_t), b_u = nl * sizeof(int);
      const std::size_t b_src = nl * z(P) * sizeof(g::ExpertPtr);
      void* dv = nullptr;
      trc_hv = g::host_alloc_mapped(b_ts + b_sel + b_u + 64 + 2 * b_src, &dv);
      if (trc_hv) {
        auto* hb = static_cast<std::uint8_t*>(trc_hv);
        auto* db = static_cast<std::uint8_t*>(dv);
        trc_ts_h = reinterpret_cast<unsigned long long*>(hb);
        trc_ts_d = reinterpret_cast<unsigned long long*>(db);
        trc_sel_h = reinterpret_cast<std::int32_t*>(hb + b_ts);
        trc_sel_d = reinterpret_cast<std::int32_t*>(db + b_ts);
        trc_u_h = reinterpret_cast<int*>(hb + b_ts + b_sel);
        trc_u_d = reinterpret_cast<int*>(db + b_ts + b_sel);
        const std::size_t o2 = (b_ts + b_sel + b_u + 63) & ~std::size_t{63};
        trc_src_h = reinterpret_cast<g::ExpertPtr*>(hb + o2);
        trc_src_d = reinterpret_cast<g::ExpertPtr*>(db + o2);
        trc_fill_h = reinterpret_cast<g::ExpertPtr*>(hb + o2 + b_src);
        trc_fill_d = reinterpret_cast<g::ExpertPtr*>(db + o2 + b_src);
      }
    }
    if (o.router_margins) {  // v49
      void* dv = nullptr;
      rlg_h = static_cast<float*>(g::host_alloc_mapped(z(s.n_layer) * kReaderMax * z(ne) * sizeof(float), &dv));
      rlg_d = static_cast<float*>(dv);
      if (rlg_h) {
        rmarg.assign(z(o.max_ctx) * 2 * z(s.n_layer), 0.0f);
        rmarg_have.assign(z(o.max_ctx), 0);
      }
    }
    reader = true;
    char b[160];
    char b2[640];
    std::snprintf(b2, sizeof b2, "on (R3 expert stage, steps of 1-%d tokens; %.2f GB of the file (the experts outside the bank) registered in %zu ranges (gap %llu KiB; small pages advised on %zu) in %.2f s%s%s)",
                  kReaderMax, static_cast<double>(b1 - b0) / 1e9, runs.size(), static_cast<unsigned long long>(o.pin_gap >> 10), small_ok, reg_s,
                  fold_ok ? "; shared expert folded in" : "", cache_note.c_str());
    reader_note = b2;
    (void)b;
  }

  Status reset_state() {
    pos = 0;
    mtp_pos = 0;
    last_n = 0;
    unsynced = 0;  // v60 commit C (the ring guard counts from a synced state; dzero below is stream-ordered)
    for (Layer& D : L) {
      D.cur = 0;
      if (D.state && !g::dzero(D.state, z(s.ssm_dt_rank * s.ssm_d_state * s.ssm_d_state) * sizeof(float))) return gfail("engine: state reset");
      for (float* tl : D.tail)
        if (tl && !g::dzero(tl, z((s.ssm_d_conv - 1) * s.conv_channels()) * sizeof(float))) return gfail("engine: conv tail reset");
    }
    return g::sync() ? Status::good() : gfail("engine: reset sync");
  }
};

GpuEngine::GpuEngine(std::unique_ptr<Impl> p) : p_(std::move(p)) {}


GpuEngine::~GpuEngine() {
  if (!p_) return;
  for (auto& kv : p_->graphs) g::graph_free(kv.second);
  p_->graphs.clear();
  for (auto& kv : p_->mtp_graphs) g::graph_free(kv.second.exec);  // v60 I
  p_->mtp_graphs.clear();
  for (void* rb2 : p_->reg_bases) g::host_unregister(rb2);
  g::dfree(p_->rtab);
  g::dfree(p_->dnk_d);  // v60 G
  g::dfree(p_->rgu);
  g::dfree(p_->ryd);
  g::dfree(p_->rgrp);
  g::dfree(p_->rgsrc);
  g::dfree(p_->rgfill);
  g::dfree(p_->rshx);
  g::dfree(p_->rhq);
  g::dfree(p_->rxb);  // v55f
  g::dfree(p_->rhb);
  g::dfree(p_->rhs);
  g::dfree(p_->rhsum);
  g::dfree(p_->rxplan);
  g::dfree(p_->cache_mem);
  g::dfree(p_->cache_meta);
  g::host_free(p_->trc_hv);
  g::host_free(p_->planes_h);  // v54
  g::host_free(p_->ph_h);  // v55b
  g::host_free(p_->dph_h);  // v56
  g::dfree(p_->rst_d);      // v55e3
  g::dfree(p_->rst_mtp_d);  // v60 C
  g::host_free(p_->rlg_h);  // v49
  g::dfree(p_->probe_bd);
  g::dfree(p_->probe_ld);
  g::dfree(p_->probe_md);
  g::dfree(p_->probe_xd);
  g::dfree(p_->probe_seld);
  g::dfree(p_->probe_wd);
  for (Layer& D : p_->L) g::dfree(D.bank_mem);
  g::dfree(p_->M.bank_mem);
  p_->r3plans.clear();
  auto rel = [](DevW& d) {
    if (d.r3) g::stream_release(&d.sw);
  };
  for (Layer& D : p_->L)
    for (DevW* d : {&D.wq, &D.wk, &D.wv, &D.wo, &D.wqkv, &D.wgate, &D.wbeta, &D.walpha, &D.ssm_out, &D.gate_inp, &D.gate_sh, &D.up_sh, &D.down_sh}) rel(*d);
  for (DevW* d : {&p_->M.wq, &p_->M.wk, &p_->M.wv, &p_->M.wo, &p_->M.gate_inp, &p_->M.gate_sh, &p_->M.up_sh, &p_->M.down_sh, &p_->eh, &p_->out}) rel(*d);
  if (p_->M.router_f32) g::dfree(const_cast<float*>(p_->M.router_f32));  // v58b: the MTP router's F32 copy (device memory)
  if (p_->head.r3 && p_->head.sw.w != p_->out.sw.w) rel(p_->head);
  g::dfree(p_->act_mem);
  g::dfree(p_->hsk_mem);  // v60 E: the head sketch (head_top's plan went with r3plans; its weights live here)
  g::dfree(p_->dense_mem);
  g::dfree(p_->state_mem);
  if (p_->stm) {  // v26
    g::sync();
    g::set_stream(nullptr);
    g::stream_free(p_->stm);
  }
  g::dfree(p_->pos_d);
  g::dfree(p_->knew);
  g::dfree(p_->vnew);
  g::host_free(p_->ring_hv);
}

std::int64_t GpuEngine::pos() const { return p_ ? p_->pos : 0; }
std::uint64_t GpuEngine::device_bytes() const { return p_ ? p_->dev_bytes : 0; }
int GpuEngine::bank_experts() const { return p_ ? p_->bank_count : 0; }
int GpuEngine::max_batch() const { return p_ ? p_->o.max_batch : 0; }
std::uint64_t GpuEngine::bank_bytes() const { return p_ ? p_->bank_bytes_total : 0; }
Status GpuEngine::reset() { return p_->reset_state(); }
void GpuEngine::probe_set(std::int64_t pos) {
  Impl& I = *p_;
  I.probe_have = false;
  I.probe_pos = pos;
  if (pos >= 0 && !I.probe_xd) {
    const std::size_t nl = I.s.n_layer, nuq = I.s.n_expert_used;
    I.probe_xd = static_cast<float*>(g::dalloc(nl * I.s.n_embd * sizeof(float)));
    I.probe_seld = static_cast<std::int32_t*>(g::dalloc(nl * nuq * sizeof(std::int32_t)));
    I.probe_wd = static_cast<float*>(g::dalloc(nl * nuq * sizeof(float)));
    I.probe_bd = static_cast<float*>(g::dalloc(nl * I.s.n_embd * sizeof(float)));
    I.probe_ld = static_cast<float*>(g::dalloc(nl * I.s.n_expert * sizeof(float)));
    I.probe_md = static_cast<float*>(g::dalloc(nl * I.s.n_embd * sizeof(float)));
    if (!I.probe_xd || !I.probe_seld || !I.probe_wd || !I.probe_bd || !I.probe_ld || !I.probe_md) I.probe_pos = -1;
  }
}
bool GpuEngine::probe_get(std::vector<float>* x_layers, std::vector<std::int32_t>* picks, std::vector<float>* pick_w) {
  Impl& I = *p_;
  if (!I.probe_have) return false;
  if (x_layers) *x_layers = I.probe_xh;
  if (picks) *picks = I.probe_selh;
  if (pick_w) *pick_w = I.probe_wh;
  return true;
}
bool GpuEngine::probe_get_ops(std::vector<float>* block_out, std::vector<float>* router_logits, std::vector<float>* moe_out) {
  Impl& I = *p_;
  if (I.probe_bh.empty()) return false;
  if (block_out) *block_out = I.probe_bh;
  if (router_logits) *router_logits = I.probe_lh;
  if (moe_out) *moe_out = I.probe_mh;
  return true;
}
std::string GpuEngine::moe_phase_report() const {  // v55b
  const Impl& I = *p_;
  if (!I.ph_h || I.ph_steps == 0) return "";
  const double st = static_cast<double>(I.ph_steps);
  char b[400];
  std::snprintf(b, sizeof b,
                "fused MoE half, ms per token over %llu decode steps (%.1f layers per step; GPU clock, block 0): A norm + router + shared gate %.3f, "
                "B picks + grouping + plan (block 0) %.3f, C expert stage %.3f, D combine + residual %.3f; total %.3f",
                static_cast<unsigned long long>(I.ph_steps), static_cast<double>(I.ph_layers) / st, I.ph_sum[0] / st, I.ph_sum[1] / st, I.ph_sum[2] / st,
                I.ph_sum[3] / st, (I.ph_sum[0] + I.ph_sum[1] + I.ph_sum[2] + I.ph_sum[3]) / st);
  std::string r = b;
  if (I.dph_h && I.dph_steps > 0) {  // v56
    const double ds2 = static_cast<double>(I.dph_steps);
    char d[400];
    std::snprintf(d, sizeof d,
                  "\nDeltaNet dense half (L3), ms per token over %llu decode steps (%.1f layers per step; GPU clock, block 0): 0 norm + rounding %.3f, "
                  "1 qkv + gate + beta / alpha %.3f, 2 conv + l2 + recurrence + gated norm %.3f, 3 ssm_out %.3f; total %.3f",
                  static_cast<unsigned long long>(I.dph_steps), static_cast<double>(I.dph_layers) / ds2, I.dph_sum[0] / ds2, I.dph_sum[1] / ds2,
                  I.dph_sum[2] / ds2, I.dph_sum[3] / ds2, (I.dph_sum[0] + I.dph_sum[1] + I.dph_sum[2] + I.dph_sum[3]) / ds2);
    r += d;
  }
  for (int k = 0; k < 3; ++k) {  // v56: verify steps (MTP) by size
    char d[400];
    if (I.vph_steps[k] > 0) {
      const double st2 = static_cast<double>(I.vph_steps[k]);
      std::snprintf(d, sizeof d, "\nverify steps of %d tokens (%llu): fused MoE half ms per step: A %.3f, B %.3f, C expert stage %.3f, D %.3f; total %.3f", k + 2,
                    static_cast<unsigned long long>(I.vph_steps[k]), I.vph_sum[k][0] / st2, I.vph_sum[k][1] / st2, I.vph_sum[k][2] / st2, I.vph_sum[k][3] / st2,
                    (I.vph_sum[k][0] + I.vph_sum[k][1] + I.vph_sum[k][2] + I.vph_sum[k][3]) / st2);
      r += d;
    }
    if (I.vdph_steps[k] > 0) {
      const double st2 = static_cast<double>(I.vdph_steps[k]);
      std::snprintf(d, sizeof d, "\nverify steps of %d tokens (%llu): DeltaNet dense half ms per step: 0 %.3f, 1 %.3f, 2 %.3f, 3 %.3f; total %.3f", k + 2,
                    static_cast<unsigned long long>(I.vdph_steps[k]), I.vdph_sum[k][0] / st2, I.vdph_sum[k][1] / st2, I.vdph_sum[k][2] / st2,
                    I.vdph_sum[k][3] / st2, (I.vdph_sum[k][0] + I.vdph_sum[k][1] + I.vdph_sum[k][2] + I.vdph_sum[k][3]) / st2);
      r += d;
    }
  }
  // v55e3 diagnosis: the one-token expert stage's counters (SM cycles; every one-token step since open). v60 commit C: the
  // trunk's launches and the MTP row's (its draft and catch-up steps) reported apart
  auto counters = [&](const unsigned long long* src, const char* who) {
    if (!src) return;
    unsigned long long c[g::kRxStatN] = {};
    if (g::sync() && g::to_host(c, src, sizeof c)) {
      if (c[4] == 0) return;
      r += who;
      const double wt = static_cast<double>(c[0]), wk = static_cast<double>(c[1]), sp = static_cast<double>(c[3]);
      const double tk = static_cast<double>(c[2]), wp = static_cast<double>(c[4]);
      char e[600];
      std::snprintf(e, sizeof e,
                    "\nexpert stage counters (one-token launches, SM cycles summed over warps): warps %llu, tasks %llu (%.1f per warp), loop span %.0f cycles "
                    "per warp: running tasks %.1f %%, waiting for a group %.1f %%, rest (job lookup, signals) %.1f %%; %.0f cycles per task; staged tasks (host "
                    "RAM over PCIe) %llu at %.0f cycles each = %.1f %% of task cycles (VRAM tasks %.0f cycles each); filling tasks %llu",
                    c[4], c[2], wp > 0 ? tk / wp : 0.0, wp > 0 ? sp / wp : 0.0, sp > 0 ? 100.0 * wk / sp : 0.0, sp > 0 ? 100.0 * wt / sp : 0.0,
                    sp > 0 ? 100.0 * (sp - wk - wt) / sp : 0.0, tk > 0 ? wk / tk : 0.0, c[5], c[5] ? static_cast<double>(c[7]) / static_cast<double>(c[5]) : 0.0,
                    wk > 0 ? 100.0 * static_cast<double>(c[7]) / wk : 0.0, tk > static_cast<double>(c[5]) ? (wk - static_cast<double>(c[7])) / (tk - static_cast<double>(c[5])) : 0.0,
                    c[6]);
      r += e;
      static const char* fn[12] = {"Q7_ZT", "Q4_TG2", "Q5_K", "Q6_K dense", "Q8_0", "Q4_K", "IQ4_XS", "Q6_K", "silu x up", "Q4_TG2 planes", "Q5_K planes", "other"};
      for (int f = 0; f < 12; ++f) {
        if (c[20 + f] == 0) continue;
        std::snprintf(e, sizeof e, "\n  %-13s tasks %10llu, %8.0f cycles per task, %5.1f %% of task cycles", fn[f], c[20 + f],
                      static_cast<double>(c[8 + f]) / static_cast<double>(c[20 + f]), wk > 0 ? 100.0 * static_cast<double>(c[8 + f]) / wk : 0.0);
        r += e;
      }
    }
  };
  counters(I.rst_d, "");
  counters(I.rst_mtp_d, "\nMTP row:");
  return r;
}
std::string GpuEngine::mtp_path_report() const {
  const Impl& I = *p_;
  if (!I.has_mtp) return {};
  std::string r = "MTP block MoE half (v58): on the fused launch (reader row n_layer), " + std::to_string(I.mtp_fused_steps) + " steps";
  if (!I.mtp_router_note.empty()) r += "; " + I.mtp_router_note;
  if (I.mtp_emb_dev) r += "; v60 C: input gathered and concatenated on the GPU, hidden rows and the draft chain on the device, K / V through the direct plan";
  r += "; R3 dense runs since open: " + std::to_string(I.r3_direct_runs) + " on the direct plan, " + std::to_string(I.r3_copy_runs) + " through scratch copies";
  if (I.hsk) {  // v60 E
    r += "; head sketch: " + std::to_string(I.sk_drafts) + " drafts through the two-plane sketch + top-" + std::to_string(Impl::kHeadSketchTop) + " re-score";
    if (I.o.mtp_head_sketch_check) r += ", recall " + std::to_string(I.sk_hit) + " / " + std::to_string(I.sk_total) + " (the full head's argmax among the sketch's top rows)";
  }
  r += "; MTP step graphs (v60 I / I2): " + I.mtp_graph_note;  // 0.4 + I2
  if (I.mtp_graph_captures) {
    std::size_t nd = 0, nc = 0, draft_nodes = 0, catch_nodes = 0;
    for (const auto& kv : I.mtp_graphs)
      if (kv.second.exec) {
        if (std::get<1>(kv.first)) {
          ++nd;
          draft_nodes = kv.second.nodes;
        } else {
          ++nc;
          catch_nodes = kv.second.nodes;
        }
      }
    r += ", " + std::to_string(I.mtp_graph_captures) + " captured (" + std::to_string(nd) + " draft, " + std::to_string(nc) + " catch-up shapes), " +
         std::to_string(I.mtp_graph_replays) + " replays, " + std::to_string(draft_nodes) + " nodes per draft, " + std::to_string(catch_nodes) + " per catch-up step";
  }
  return r;
}

std::string GpuEngine::graph_report() const {
  std::string r = p_->graph_note + "; captures " + std::to_string(p_->graph_captures) + ", replays " + std::to_string(p_->graph_replays);
  if (!p_->graph_nodes.empty()) {  // v60 I (0.3): launches per captured step, by (tokens, parity, verify checkpoints)
    r += "; nodes per step:";
    for (const auto& kv : p_->graph_nodes)
      r += " " + std::to_string(std::get<0>(kv.first)) + "-token" + (std::get<2>(kv.first) ? " verify" : "") + " " + std::to_string(kv.second);
  }
  return r + "; DeltaNet dense halves in one launch (L3, v55g) " + std::to_string(p_->dn_half_runs);
}
bool GpuEngine::router_margins(std::vector<float>* m, std::vector<std::uint8_t>* have) const {
  if (!p_->rlg_h) return false;
  if (m) *m = p_->rmarg;
  if (have) *have = p_->rmarg_have;
  return true;
}
std::string GpuEngine::trace_take() {
  std::string out;
  out.swap(p_->trace_text);
  return out;
}

std::string GpuEngine::summary() const {
  const Impl& I = *p_;
  char b[2048];
  std::snprintf(b, sizeof b, "engine: %.2f GB on the device (dense + output %.2f GB, bank %.2f GB = %d experts of %u, cache + state %.2f GB); ctx %d, batch %d; router tie band %g; expert reader %s; graph %s",
                static_cast<double>(I.dev_bytes) / 1e9, static_cast<double>(I.dense_bytes_total) / 1e9, static_cast<double>(I.bank_bytes_total) / 1e9, I.bank_count,
                I.s.n_layer * I.s.n_expert, static_cast<double>(I.cache_bytes) / 1e9, I.o.max_ctx, I.o.max_batch, static_cast<double>(I.o.router_tie),
                I.reader_note.c_str(), I.graph_note.c_str());
  std::string r = b;
  r += "; prompt on the GPU in steps of 1-4 tokens";  // v48 (v60: the only prompt path)
  r += "; R3 expert stage " + I.rx_note + ", rows staged for experts read from RAM";  // v46a-1 / v46a-2 (v60: locked)
  r += I.planes_note;  // v54
  r += I.sketch_note;  // v60 E
  r += I.probe_note;   // v54b
  r += "; dense weights held once (R3's copies; v55 / v60: no raw copies or int8 tables)";
  {  // v60 commit C: the activation layout (steps of 1-4 rows) and what the v60b layout held beyond it
    char ab[200];
    std::snprintf(ab, sizeof ab, "; activations for %d-row steps (v60b layout held %.0f MB more, freed before the VRAM probe)", Impl::kReaderMax,
                  static_cast<double>(I.act_freed) / 1048576.0);
    r += ab;
  }
  r += I.r3_direct_note();  // v55
  if (I.rx_ok && I.fold_ok) r += "; MoE half fused (one launch per layer where no side output needs the separate kernels)";  // v52
  return r;
}

Result<std::unique_ptr<GpuEngine>> GpuEngine::open(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, const EngineOptions& o) {
  g::set_fast_f32(true);  // v52 (v60: locked on; same bits)
  using R = Result<std::unique_ptr<GpuEngine>>;
  if (!g::available()) return R::fail(Status::fail(Code::no_device, std::string("engine: ") + g::last_error()));
  if (s.family != Family::qwen35moe || s.dense) return R::fail(Status::fail(Code::unsupported, "engine: qwen35moe only in this first cut"));
  if (o.max_ctx < 1 || o.max_ctx > (1 << 20) || o.max_batch < 1 || o.max_batch > 4096)
    return R::fail(Status::fail(Code::invalid_argument, "engine: max_ctx 1 .. 2^20, max_batch 1 .. 4096"));
  if (w.layers.size() != s.n_layer) return R::fail(Status::fail(Code::invalid_argument, "engine: the weights were bound for another spec"));
  if (!w.output.present() || w.output_norm.empty()) return R::fail(Status::fail(Code::invalid_argument, "engine: no output layer"));
  if (!o.hot_order.empty() && o.hot_order.size() != s.n_layer) return R::fail(Status::fail(Code::invalid_argument, "engine: hot order for another layer count"));
  for (const Qwen35MoeLayer& l : w.layers) {
    if (!l.bq.empty() || !l.attn_sub_norm.empty()) return R::fail(Status::fail(Code::unsupported, "engine: biases / sub-norms (qwen35moe only)"));
    for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps})
      if (v->rot32) return R::fail(Status::fail(Code::unsupported, "engine: rotated routed experts (" + v->name + ")"));
  }
  auto p = std::make_unique<Impl>();
  Impl& I = *p;
  I.s = s;
  I.w = &w;
  I.o = o;
  // v60 commit C: every step is 1-4 tokens (run() and mtp_step reject more; step() splits a prompt), so the activations hold
  // kReaderMax rows. v60b held (max_batch + 63) / 64 * 64 + 64 rows (192 at max_batch 64) of which 4 were reachable; the
  // difference (act_freed, logged) goes to the MTP row's cache (see the VRAM probe)
  I.npad = Impl::kReaderMax;
  const std::size_t np_old = z((o.max_batch + 63) / 64 * 64 + 64);
  const int64_t e = s.n_embd, hd = s.head_dim, nh = s.n_head, nkv = s.n_head_kv, qh = s.attn_gate ? 2 * hd : hd;
  const int64_t ch = s.conv_channels(), di = s.ssm_d_inner, ds = s.ssm_d_state, nv = s.ssm_dt_rank, ne = s.n_expert, nu = s.n_expert_used;
  const int64_t ffs = s.n_ff_shexp, ff = s.n_ff_exp, taps = s.ssm_d_conv;
  I.kv_w = hd * nkv;
  I.has_mtp = o.mtp;
  I.use_r3 = true;  // v60: R3 is the only dense path
  // v55 (author 02:36-02:41, build audit A): with planes on, every product of the output head runs on R3 (one-token steps,
  // the GPU prompt's steps of 1-4 tokens on R3, no MTP block, no verify batches off R3), so the head's raw copy and its int8
  // tables (~0.48 GB for this model's Q6_K head) are not loaded; the VRAM probe gives that room to the experts
  if (o.mtp && !w.mtp.present) return R::fail(Status::fail(Code::unsupported, "engine: this file has no MTP block"));
  if (o.max_verify > 1 && !(ds >= 32 && ds <= 1024 && (ds & (ds - 1)) == 0))
    return R::fail(Status::fail(Code::unsupported, "engine: verify batches need the DeltaNet d_state a power of two (the checkpointing recurrence)"));
  if (o.max_verify > o.max_batch) return R::fail(Status::fail(Code::invalid_argument, "engine: max_verify above max_batch"));
  if (o.plane_bits < 2 || o.plane_bits > 5) return R::fail(Status::fail(Code::invalid_argument, "engine: plane_bits 2, 3, 4 or 5 (5: every code as stored; v60: always on)"));
  if (o.mtp_plane_bits != 0 && (o.mtp_plane_bits < 2 || o.mtp_plane_bits > 5))
    return R::fail(Status::fail(Code::invalid_argument, "engine: mtp_plane_bits 0 (the slider) or 2-5 (v60 F2)"));
  if ((o.moe_blocks != 1 && o.moe_blocks != 2) || (o.rx_pipe != 0 && o.rx_pipe != 1))
    return R::fail(Status::fail(Code::invalid_argument, "engine: moe_blocks 1 or 2, rx_pipe 0 or 1 (v60 H)"));
  if (o.rx_bit_route != 0 && o.rx_bit_route != 1) return R::fail(Status::fail(Code::invalid_argument, "engine: rx_bit_route 0 (dp4a) or 1 (1-bit) (v60 I)"));
  if (o.mtp && I.row_bits(s.n_layer) != o.plane_bits) {  // v60 F2: the Q4_TG2 decode grid (c_rx_bits) is one per launch, so a planed
    // Q4_TG2 matrix in the MTP row cannot read with other bits than the trunk's; this file's MTP experts are Q4_K (no grid)
    for (const WeightView* v : {&w.mtp.layer.gate_exps, &w.mtp.layer.up_exps, &w.mtp.layer.down_exps})
      if (v->present() && v->type == static_cast<std::uint32_t>(quant::Type::q4_tg2))
        return R::fail(Status::fail(Code::unsupported, "engine: mtp_plane_bits differs from plane_bits but the MTP block's experts are Q4_TG2 (one decode grid per launch); use the slider's bits"));
  }
  I.kmax = std::max<int64_t>({e, hd * nh, di, ffs, ff, o.mtp ? 2 * e : e});
  // activations
  const std::size_t np = z(I.npad);
  struct Buf {
    void** ptr;
    std::size_t bytes;
  };
  // the buffers for `rows` step rows and `hrows` head rows (v60 commit C: called once for the layout and once, with the v60b
  // sizes, to count what that layout held for rows no step can reach)
  auto bufs_for = [&](std::size_t rows, std::size_t hrows) {
    std::vector<Buf> b = {
        {reinterpret_cast<void**>(&I.x), rows * z(e) * 4},          {reinterpret_cast<void**>(&I.xn), rows * z(e) * 4},
        {reinterpret_cast<void**>(&I.y), rows * z(e) * 4},          {reinterpret_cast<void**>(&I.xr), rows * z(I.kmax) * 4},
        {reinterpret_cast<void**>(&I.q), rows * z(I.kmax)},         {reinterpret_cast<void**>(&I.scale), rows * z(I.kmax / 32) * 4},
        {reinterpret_cast<void**>(&I.sum), rows * z(I.kmax / 32) * 4},
        {reinterpret_cast<void**>(&I.qg), rows * z(qh * nh) * 4},   {reinterpret_cast<void**>(&I.att), rows * z(hd * nh) * 4},
        {reinterpret_cast<void**>(&I.ascr), rows * z(nh) * z(o.max_ctx) * 4},
        {reinterpret_cast<void**>(&I.mixed), rows * z(ch) * 4},     {reinterpret_cast<void**>(&I.zg), rows * z(di) * 4},
        {reinterpret_cast<void**>(&I.beta), rows * z(nv) * 4},      {reinterpret_cast<void**>(&I.alpha), rows * z(nv) * 4},
        {reinterpret_cast<void**>(&I.conv), rows * z(ch) * 4},      {reinterpret_cast<void**>(&I.core), rows * z(di) * 4},
        {reinterpret_cast<void**>(&I.dn_hcnt), z(nv > 0 ? nv : 1) * 4},
        {reinterpret_cast<void**>(&I.top_d), rows * 4 + 64},
        {reinterpret_cast<void**>(&I.logits_r), rows * z(ne) * 4},  {reinterpret_cast<void**>(&I.sel), rows * z(nu) * 4},
        {reinterpret_cast<void**>(&I.wsel), rows * z(nu) * 4},      {reinterpret_cast<void**>(&I.gs), rows * z(ffs) * 4},
        {reinterpret_cast<void**>(&I.us), rows * z(ffs) * 4},       {reinterpret_cast<void**>(&I.sh), rows * z(e) * 4},
        {reinterpret_cast<void**>(&I.gsh), rows * 4},               {reinterpret_cast<void**>(&I.galpha), rows * 4},
        {reinterpret_cast<void**>(&I.zero2), 256},
        {reinterpret_cast<void**>(&I.cat), o.mtp ? rows * z(2 * e) * 4 : 256},
        // v60 commit C: the MTP path's device rows (hid_d: 1 + max_batch rows; h_d / hn_d one row each; the draft chain's ids)
        {reinterpret_cast<void**>(&I.hid_d), o.mtp ? (z(o.max_batch) + 1) * z(e) * 4 : 256},
        {reinterpret_cast<void**>(&I.h_d), o.mtp ? z(e) * 4 : 256},
        {reinterpret_cast<void**>(&I.hn_d), o.mtp ? z(e) * 4 : 256},
        {reinterpret_cast<void**>(&I.mtp_top_d), o.mtp ? z(o.max_batch) * 4 + 64 : 256},
        {reinterpret_cast<void**>(&I.r3y), 4 * (z(s.n_vocab) + z(qh * nh) + z(4 * I.kv_w) + z(2 * ch) + z(2 * di) + z(4 * ffs) + 1024) * 4},
        {reinterpret_cast<void**>(&I.lg), hrows * z(s.n_vocab) * 4}};
    return b;
  };
  std::vector<Buf> bufs = bufs_for(np, z(Impl::kOutRows));
  std::size_t total = 0;
  for (const Buf& b : bufs) total += al(b.bytes);
  {  // v60 commit C: the v60b layout for the same buffers (np_old rows, 64 head rows) plus dig and yc (never read)
    std::size_t old_total = al(np_old * z(I.kmax / 32) * 4) + al(np_old * z(e) * 4);
    for (const Buf& b : bufs_for(np_old, 64)) old_total += al(b.bytes);
    // the four commit-C buffers did not exist in v60b: take them back out of the old figure
    old_total -= al(o.mtp ? (z(o.max_batch) + 1) * z(e) * 4 : 256) + 2 * al(o.mtp ? z(e) * 4 : 256) + al(o.mtp ? z(o.max_batch) * 4 + 64 : 256);
    I.act_freed = old_total > total ? old_total - total : 0;
  }
  I.r3y_floats = 4 * (z(s.n_vocab) + z(qh * nh) + z(4 * I.kv_w) + z(2 * ch) + z(2 * di) + z(4 * ffs) + 1024);  // v26: up to 4 tokens
  I.act_mem = g::dalloc(total);
  if (!I.act_mem || !g::dzero(I.act_mem, total)) return R::fail(Status::fail(Code::over_ceiling, "engine: activation memory"));
  auto* cur = static_cast<std::uint8_t*>(I.act_mem);
  for (const Buf& b : bufs) {
    *b.ptr = cur;
    cur += al(b.bytes);
  }
  I.dev_bytes += total;
  // the cache and the state
  I.L.resize(s.n_layer);
  {
    std::size_t sb = 0;
    const int64_t nck = std::max(0, o.max_verify - 1);
    for (const Qwen35MoeLayer& l : w.layers)
      sb += l.recurrent ? al(z(nv * ds * ds) * 4) + 2 * al(z((taps - 1) * ch) * 4) + (nck > 0 ? al(z(nck * nv * ds * ds) * 4) + al(z(o.max_verify) * z(ch) * 4) : 0)
                        : 2 * al(z(o.max_ctx) * z(I.kv_w) * 4);
    if (o.mtp) sb += 2 * al(z(o.max_ctx) * z(I.kv_w) * 4);
    I.state_mem = g::dalloc(sb);
    if (!I.state_mem) return R::fail(Status::fail(Code::over_ceiling, "engine: cache and state (" + std::to_string(sb >> 20) + " MiB)"));
    I.dev_bytes += sb;
    I.cache_bytes = sb;
    auto* c = static_cast<std::uint8_t*>(I.state_mem);
    for (std::uint32_t il = 0; il < s.n_layer; ++il) {
      Layer& D = I.L[il];
      if (w.layers[il].recurrent) {
        D.state = reinterpret_cast<float*>(c);
        c += al(z(nv * ds * ds) * 4);
        for (float*& tl : D.tail) {
          tl = reinterpret_cast<float*>(c);
          c += al(z((taps - 1) * ch) * 4);
        }
        if (nck > 0) {
          D.ckpt = reinterpret_cast<float*>(c);
          c += al(z(nck * nv * ds * ds) * 4);
          D.mixed_save = reinterpret_cast<float*>(c);
          c += al(z(o.max_verify) * z(ch) * 4);
        }
      } else {
        D.K = reinterpret_cast<float*>(c);
        c += al(z(o.max_ctx) * z(I.kv_w) * 4);
        D.V = reinterpret_cast<float*>(c);
        c += al(z(o.max_ctx) * z(I.kv_w) * 4);
      }
    }
    if (o.mtp) {
      I.M.K = reinterpret_cast<float*>(c);
      c += al(z(o.max_ctx) * z(I.kv_w) * 4);
      I.M.V = reinterpret_cast<float*>(c);
      c += al(z(o.max_ctx) * z(I.kv_w) * 4);
    }
  }
  // v60 G (item 4): the keep() stitch's table - every DeltaNet layer's state, checkpoints, saved inputs and tails, one upload
  {
    std::vector<g::DnKeepLayer> kt;
    for (std::uint32_t il = 0; il < s.n_layer; ++il) {
      const Layer& D = I.L[il];
      if (!D.state || !D.ckpt || !D.mixed_save) continue;
      g::DnKeepLayer ke;
      ke.state = D.state;
      ke.ckpt = D.ckpt;
      ke.msave = D.mixed_save;
      ke.tail[0] = D.tail[0];
      ke.tail[1] = D.tail[1];
      kt.push_back(ke);
    }
    if (!kt.empty()) {
      I.dnk_d = static_cast<g::DnKeepLayer*>(g::dalloc(kt.size() * sizeof(g::DnKeepLayer)));
      if (!I.dnk_d || !g::to_device(I.dnk_d, kt.data(), kt.size() * sizeof(g::DnKeepLayer))) return R::fail(gfail("engine: keep table"));
      I.dnk_n = static_cast<int>(kt.size());
    }
  }
  // (v60 commit C: the batched MoE scratch (g::moe_scratch_alloc) is gone: nothing read it since the R3 stage became the only
  // expert path; its bytes were taken before the VRAM probe and never used)
  // the dense weights of every layer and the output layer, resident
  {
    std::size_t db = I.dense_bytes(w.output) + Impl::vec_bytes(w.output_norm) + Impl::vec_bytes(w.rope_freqs) + 1024;
    for (const Qwen35MoeLayer& l : w.layers) db += I.layer_dense_bytes(l);
    const bool head_is_out = w.mtp.head.data == w.output.data;
    if (o.mtp)
      db += I.layer_dense_bytes(w.mtp.layer) + I.dense_bytes(w.mtp.eh_proj) + Impl::vec_bytes(w.mtp.enorm) + Impl::vec_bytes(w.mtp.hnorm) +
            Impl::vec_bytes(w.mtp.head_norm) + (head_is_out ? 0 : I.dense_bytes(w.mtp.head));
    I.dense_mem = g::dalloc(db);
    if (!I.dense_mem) return R::fail(Status::fail(Code::over_ceiling, "engine: dense weights (" + std::to_string(db >> 20) + " MiB)"));
    I.dev_bytes += db;
    I.dense_bytes_total = db;
    I.arena = Arena{static_cast<std::uint8_t*>(I.dense_mem), db, 0};
    Status st = I.put_dense(w.output, I.out);
    if (st) st = I.put_vec(w.output_norm, I.out_norm);
    if (st) st = I.put_vec(w.rope_freqs, I.rope_ff);
    for (std::uint32_t il = 0; st && il < s.n_layer; ++il) st = I.put_layer(w.layers[il], I.L[il]);
    if (st && o.mtp) {
      st = I.put_layer(w.mtp.layer, I.M);
      if (st) st = I.put_dense(w.mtp.eh_proj, I.eh);
      if (st) st = I.put_vec(w.mtp.enorm, I.enorm);
      if (st) st = I.put_vec(w.mtp.hnorm, I.hnorm);
      if (st) st = I.put_vec(w.mtp.head_norm, I.head_norm);
      if (st) {
        if (head_is_out) I.head = I.out;
        else st = I.put_dense(w.mtp.head, I.head);
      }
      I.M.hot.assign(z(ne), 0);  // the MTP block has no bank: its experts come from the planes / the file mapping (v58)
    }
    if (!st) return R::fail(st);
  }
  // v60 E: the head sketch for the drafts, taken before the bank is sized (the bank gives up its bytes; ~100 experts here)
  if (o.mtp && o.mtp_head_sketch > 0) {
    if (o.mtp_head_sketch != 2) return R::fail(Status::fail(Code::unsupported, "engine: mtp_head_sketch takes 0 or 2 (the two-plane sketch is the one built; v60 E)"));
    if (!I.head.r3 || !I.head.v || I.head.v->type != static_cast<std::uint32_t>(quant::Type::q6_k) || !I.head.sw.d6)
      return R::fail(Status::fail(Code::unsupported, std::string("engine: the head sketch reads a Q6_K head held by R3 (this file's head is ") +
                                                        (I.head.v ? quant::type_info(I.head.v->type).name : "absent") + ")"));
    const int64_t nvoc = s.n_vocab, hk = I.head.k, nbr = hk / 256;
    const std::size_t sk_b = al(z(nvoc) * static_cast<std::size_t>(kern::gpu::q6k_sketch_row_bytes(hk)));
    const std::size_t top_w = al(z(Impl::kHeadSketchTop) * z(nbr) * 208), top_d = al(z(Impl::kHeadSketchTop) * z(nbr) * 2);
    const std::size_t ids_b = al(z(Impl::kHeadSketchTop) * 4);
    I.hsk_bytes = sk_b + top_w + top_d + ids_b * 2 + al(64) + al(256);
    I.hsk_mem = g::dalloc(I.hsk_bytes);
    if (!I.hsk_mem || !g::dzero(I.hsk_mem, I.hsk_bytes))
      return R::fail(Status::fail(Code::over_ceiling, "engine: head sketch memory (" + std::to_string(I.hsk_bytes >> 20) + " MiB)"));
    I.dev_bytes += I.hsk_bytes;
    auto* c = static_cast<std::uint8_t*>(I.hsk_mem);
    I.hsk = c;
    c += sk_b;
    I.htop_w = c;
    c += top_w;
    I.htop_d6 = reinterpret_cast<std::uint16_t*>(c);
    c += top_d;
    I.cand_d = reinterpret_cast<std::int32_t*>(c);
    c += ids_b;
    I.lg_top = reinterpret_cast<float*>(c);
    c += ids_b;
    I.idx_d = reinterpret_cast<std::int32_t*>(c);
    c += al(64);
    I.sk_cnt_d = reinterpret_cast<unsigned*>(c);
    if (!g::head_sketch_build(I.head.sw.w, nvoc, hk, I.hsk)) return R::fail(gfail("engine: head sketch build"));
    // the re-score weight: kHeadSketchTop rows in R3's Q6_K layout at fixed addresses, filled per draft, run by the same kernel
    // and plan kind as the head itself (prod_r3 keys the plan on this DevW)
    I.head_top = DevW{};
    I.head_top.v = I.head.v;
    I.head_top.n_out = Impl::kHeadSketchTop;
    I.head_top.k = hk;
    I.head_top.dtype = I.head.dtype;
    I.head_top.drb = I.head.drb;
    I.head_top.r3 = true;
    I.head_top.sw.type = I.head.sw.type;
    I.head_top.sw.n = Impl::kHeadSketchTop;
    I.head_top.sw.k = hk;
    I.head_top.sw.w = I.htop_w;
    I.head_top.sw.d6 = I.htop_d6;
    I.head_top.sw.bytes = z(Impl::kHeadSketchTop) * z(nbr) * 210;
    char sb[240];
    std::snprintf(sb, sizeof sb, "; head sketch (v60 E): the Q6_K head's top two code planes + scales, %.1f MB (%lld rows x %llu B, built on the GPU), top %d re-scored by the head's own kernel%s",
                  static_cast<double>(sk_b) / 1e6, static_cast<long long>(nvoc), static_cast<unsigned long long>(kern::gpu::q6k_sketch_row_bytes(hk)), Impl::kHeadSketchTop,
                  o.mtp_head_sketch_check ? ", recall checked against the full head" : "");
    I.sketch_note = sb;
  }
  // the bank: what is left of the budget, filled rank by rank across the layers
  for (Layer& D : I.L) D.hot.assign(z(ne), 0);
  std::size_t free_b = 0, total_b = 0;
  if (!g::sync() || !g::mem_info(&free_b, &total_b)) return R::fail(Status::fail(Code::internal, std::string("engine: device memory: ") + g::last_error()));
  std::uint64_t left = free_b > o.vram_reserve ? free_b - o.vram_reserve : 0;
  if (o.vram_budget > 0) left = std::min<std::uint64_t>(left, o.vram_budget > I.dev_bytes ? o.vram_budget - I.dev_bytes : 0);
  // v58: the MTP block's MoE half on the reader (its row n_layer: planes / file mapping and its own cache slots, no bank);
  // v60: the only MTP path (the old path ran the block's experts on the CPU)
  I.mtp_rd = o.mtp && w.mtp.present;
  if (I.mtp_rd) {  // v58b: the fused launch reads the router as F32 rows; the MTP block's may be stored otherwise
    const WeightView& rv = w.mtp.layer.gate_inp;
    I.mtp_router_note = std::string("MTP router stored as ") + quant::type_info(rv.type).name + (rv.rot32 ? " (rotated)" : "");
    if (!I.router_of(I.M) && rv.present()) {
      std::vector<float> rf(z(ne) * z(e));
      bool ok = true;
      for (int64_t r = 0; r < ne && ok; ++r) {
        float* rptr = rf.data() + z(r) * z(e);
        ok = quant::dequantize_row(static_cast<quant::Type>(rv.type), rv.data + static_cast<std::uint64_t>(r) * rv.row_bytes, rptr, e).ok();
        if (ok && rv.rot32) {
          quant::hadamard32(rptr, e);
        }
      }
      void* rd = ok ? g::dalloc(rf.size() * sizeof(float)) : nullptr;
      if (rd && g::to_device(rd, rf.data(), rf.size() * sizeof(float))) {
        I.M.router_f32 = static_cast<const float*>(rd);
        I.dev_bytes += rf.size() * sizeof(float);
        left = left > rf.size() * sizeof(float) ? left - rf.size() * sizeof(float) : 0;
        I.mtp_router_note += ", an F32 copy made for the fused launch (decoded from the stored rows; draft picks only: verify still decides every token)";
      } else {
        I.mtp_router_note += ", no F32 copy (dequantize or device memory)";
      }
    }
  }
  I.rc_eff = o.reader_cache;
  int hot_cap = o.max_hot_per_layer;
  if (o.reader_cache > 0 || o.auto_vram) {  // v26: the reader cache's bytes before the bank takes its share
    // v60 (Fix 1): the slot is the largest expert at the slider's bits (a slot holds the planed prefix of each row, as the
    // size pools lay it out), not the file's raw rows (v59 audit: "cslot ignores the bit-plane slider")
    std::uint64_t mx = 0;
    for (std::uint32_t il = 0; il < I.nlr(); ++il)  // v58: with the MTP row too
      for (int ex = 0; ex < static_cast<int>(ne); ++ex) mx = std::max<std::uint64_t>(mx, I.expert_bytes(il, ex));
    const std::uint64_t slot_b = (mx + 255) & ~std::uint64_t{255};
    if (o.auto_vram) {  // v54b VRAM probe: the free bytes after the dense weights split cache_share / the rest
      const std::uint64_t slack = (3 * 4 * (2048 + 256 * 4) + 64 * 1024) * static_cast<std::uint64_t>(s.n_layer);
      const std::uint64_t ex_b = left > slack ? left - slack : 0;
      // the cache share, capped so bank + cache never ask for more than every expert's bytes (a model whose experts all fit
      // in the bank gets no cache)
      std::uint64_t all_b = 0;
      for (std::uint32_t il = 0; il < I.nlr(); ++il)  // v60 C: the MTP row's experts counted too
        for (int ex = 0; ex < static_cast<int>(ne); ++ex) all_b += I.expert_bytes(il, ex);
      double cache_b = static_cast<double>(ex_b) * o.cache_share;
      if (static_cast<double>(all_b) <= static_cast<double>(ex_b)) cache_b = 0.0;
      else cache_b = std::min(cache_b, static_cast<double>(all_b) - static_cast<double>(ex_b) * (1.0 - o.cache_share));
      I.rc_eff = static_cast<int>(cache_b / static_cast<double>(slot_b * I.nlr()));
      hot_cap = static_cast<int>(ne);
      char pn[240];
      std::snprintf(pn, sizeof pn, "; VRAM probe: %.2f GB free after the dense weights (reserve %.2f GB kept), %.2f GB for experts: cache %.1f %% = %d slots of %.2f MB per layer, the bank the rest",
                    static_cast<double>(free_b) / 1e9, static_cast<double>(o.vram_reserve) / 1e9, static_cast<double>(ex_b) / 1e9, 100.0 * o.cache_share, I.rc_eff,
                    static_cast<double>(slot_b) / 1e6);
      I.probe_note = pn;
    }
    // v60 commit C: the MTP row's cache gets more than a trunk layer's budget: the activation bytes v60b held for unreachable
    // rows (act_freed, DERIVED above) or, when set, --mtp-cache-mb (an A/B knob: more takes from the bank). Its experts are
    // read on every draft and every catch-up step and are not plane types (raw rows over PCIe when they miss)
    const std::uint64_t mtp_bonus = I.mtp_rd ? (o.mtp_cache_mb > 0 ? static_cast<std::uint64_t>(o.mtp_cache_mb) << 20 : I.act_freed) & ~std::uint64_t{255} : 0;
    const std::uint64_t need = slot_b * static_cast<std::uint64_t>(I.rc_eff) * I.nlr() + mtp_bonus;
    if (need < left) {
      left -= need;
      I.cache_reserved = need;
      I.cslot = slot_b;
      I.mtp_cache_bonus = mtp_bonus;
    }
  }
  if (!o.hot_order.empty()) {
    // stack overhead per layer (alignment, maps, table rounding): ESTIMATE, kept out of the budget
    const std::uint64_t per_layer_slack = 3 * 4 * (2048 + 256 * 4) + 64 * 1024;
    left = left > per_layer_slack * s.n_layer ? left - per_layer_slack * s.n_layer : 0;
    std::size_t max_rank = 0;
    for (const auto& v : o.hot_order) max_rank = std::max(max_rank, v.size());
    const std::size_t cap = std::min<std::size_t>(max_rank, static_cast<std::size_t>(std::max(0, hot_cap)));  // v54b: the probe lifts the cap
    for (std::size_t r = 0; r < cap && left > 0; ++r)
      for (std::uint32_t il = 0; il < s.n_layer; ++il) {
        if (r >= o.hot_order[il].size()) continue;
        const int ex = o.hot_order[il][r];
        if (ex < 0 || ex >= ne || I.L[il].hot[z(ex)] || !I.bankable(w.layers[il], ex)) continue;
        const std::uint64_t b = I.expert_bytes(il, ex);
        if (b > left) continue;
        left -= b;
        I.L[il].hot[z(ex)] = 1;
        ++I.L[il].n_hot;
        ++I.bank_count;
      }
    for (std::uint32_t il = 0; il < s.n_layer; ++il) {
      const Status st = I.build_bank_planes(il, I.L[il]);  // v54c (v60: the planed bank is the only bank)
      if (!st) return R::fail(st);
    }
  }
  {  // v60: the mapped ring for the step's ids and position, and the token embedding registered for the GPU gather
    void* rdev = nullptr;
    const std::size_t ring_ids = static_cast<std::size_t>(Impl::kRing) * (Impl::kReaderMax * sizeof(std::int32_t) + sizeof(int));
    const std::size_t ring_ts = (ring_ids + 255) / 256 * 256;  // v60 D: the MTP stamps start here, kMtpStamps x u64 per slot (E2: 6; H: 9)
    const std::size_t ring_b = ring_ts + static_cast<std::size_t>(Impl::kRing) * Impl::kMtpStamps * sizeof(unsigned long long) + 256;
    I.ring_hv = g::host_alloc_mapped(ring_b, &rdev);
    if (!I.ring_hv || !rdev) return R::fail(Status::fail(Code::internal, std::string("engine: mapped memory for the step ring: ") + g::last_error()));
    std::memset(I.ring_hv, 0, ring_b);
    I.ring_tok_h = static_cast<std::int32_t*>(I.ring_hv);
    I.ring_tok_d = static_cast<std::int32_t*>(rdev);
    I.ring_pos_h = reinterpret_cast<int*>(static_cast<std::uint8_t*>(I.ring_hv) + Impl::kRing * Impl::kReaderMax * sizeof(std::int32_t));
    if (o.mtp) {
      I.mtp_ts_h = reinterpret_cast<unsigned long long*>(static_cast<std::uint8_t*>(I.ring_hv) + ring_ts);
      I.mtp_ts_d = reinterpret_cast<unsigned long long*>(static_cast<std::uint8_t*>(rdev) + ring_ts);
    }
    const WeightView& te = w.tok_embd;
    const std::uintptr_t pg = 4096;
    if (te.present() && !te.rot32 && te.slots.empty() && g::type_supported(te.type)) {
      const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(te.data), b = a + static_cast<std::uintptr_t>(te.row_bytes * te.ne[1]);
      void* dv = nullptr;
      std::string why;
      const std::uintptr_t ra = a & ~(pg - 1), rl = ((b + pg - 1) & ~(pg - 1)) - ra;
      // v60c: the same small-pages advise the reader's file ranges get before registering (v27): registering the mapping
      // on huge pages spun in the kernel (v60b GP5: 290 s of system time in open, no progress)
      if (o.pin_small_pages) hal::advise_small_pages(reinterpret_cast<void*>(ra), static_cast<std::size_t>(rl));
      if (g::host_register(reinterpret_cast<void*>(ra), static_cast<std::size_t>(rl), &dv, &why)) {
        I.reg_bases.push_back(reinterpret_cast<void*>(ra));
        I.emb_dev = static_cast<const std::uint8_t*>(dv) + (a - (a & ~(pg - 1)));
        I.probe_note += "; token embedding gathered on the GPU from the registered file rows (v60)";
      } else {
        I.probe_note += "; token embedding on the host path (registering its rows failed: " + why + ")";
      }
    } else {
      I.probe_note += "; token embedding on the host path (type without a GPU decoder, rotated or sliced)";
    }
    // v60 commit C: the MTP block's embedding for the GPU gather (the trunk's registration when the file shares the tensor)
    if (I.mtp_rd) {
      const WeightView& me = w.mtp.embed_tokens;
      if (me.present() && me.data == te.data && I.emb_dev) {
        I.mtp_emb_dev = I.emb_dev;
      } else if (me.present() && !me.rot32 && me.slots.empty() && g::type_supported(me.type)) {
        const std::uintptr_t ma = reinterpret_cast<std::uintptr_t>(me.data), mb = ma + static_cast<std::uintptr_t>(me.row_bytes * me.ne[1]);
        const std::uintptr_t ra = ma & ~(pg - 1), rl = ((mb + pg - 1) & ~(pg - 1)) - ra;
        void* dv = nullptr;
        std::string why;
        if (o.pin_small_pages) hal::advise_small_pages(reinterpret_cast<void*>(ra), static_cast<std::size_t>(rl));  // the v33 stall (v60c)
        if (g::host_register(reinterpret_cast<void*>(ra), static_cast<std::size_t>(rl), &dv, &why)) {
          I.reg_bases.push_back(reinterpret_cast<void*>(ra));
          I.mtp_emb_dev = static_cast<const std::uint8_t*>(dv) + (ma - ra);
        } else {
          I.probe_note += "; MTP embedding not registered (" + why + "): the MTP block cannot run (v60 C needs the GPU gather)";
        }
      } else {
        I.probe_note += "; MTP embedding without a GPU decoder, rotated or sliced: the MTP block cannot run (v60 C)";
      }
      if (I.mtp_emb_dev) I.probe_note += "; MTP input built on the GPU (embedding gather + hidden rows, v60 C)";
    }
  }
  I.setup_reader();  // v24
  if (!I.reader || !I.rx_ok)  // v60: the reader with the R3 expert stage is the engine's only expert path
    return R::fail(Status::fail(Code::internal, "engine: the R3 expert stage could not be set up (" + I.rx_note + "; reader " + I.reader_note + ")"));
  if (o.mtp && !I.mtp_rd) return R::fail(Status::fail(Code::unsupported, "engine: the MTP block's experts cannot run on the R3 expert stage (type or alignment)"));
  const Status rs = I.reset_state();
  if (!rs) return R::fail(rs);
  {  // v26: the device position and the K / V landing rows (v60 commit C: always, the MTP block's attention uses them too)
    const std::size_t kvb = z(Impl::kReaderMax) * z(I.kv_w) * sizeof(float);  // a step is 1-4 rows
    I.pos_d = static_cast<int*>(g::dalloc(256));
    I.knew = static_cast<float*>(g::dalloc(kvb));
    I.vnew = static_cast<float*>(g::dalloc(kvb));
    if (!I.pos_d || !I.knew || !I.vnew) return R::fail(Status::fail(Code::over_ceiling, "engine: device position / K V rows"));
    I.dev_bytes += 256 + 2 * kvb;
  }
  if (o.graph) {  // v26: the engine's own stream (a graph cannot record the default stream)
    I.graph_note = "off (setup failed)";
    if (g::sync() && (I.stm = g::stream_new()) != nullptr) {
      g::set_stream(I.stm);
      I.graph_ok = true;
      I.graph_note = "on (reader steps of 1-4 tokens)";
      if (o.mtp && o.mtp_graph) {  // v60 I (0.4): the draft chain's graphs ride on the same capture machinery
        I.mtp_graph_ok = true;
        I.mtp_graph_note = "on (drafts and catch-up steps: the chain from the input copies on, per hidden input / output / tokens; the id gather and the position before it)";
      } else if (o.mtp) {
        I.mtp_graph_note = "off (--no-mtp-graph)";
      }
    }
  }
  return R::good(std::unique_ptr<GpuEngine>(new GpuEngine(std::move(p))));
}

Status GpuEngine::step(const std::int32_t* tokens, int n, float* logits, bool last_only, EngineStats* stats, float* hidden_out) {
  Impl& I = *p_;
  I.ckpt_mode = false;
  I.last_n = 0;
  if (n <= Impl::kReaderMax) return I.run(tokens, n, logits, last_only, stats, hidden_out, I.hidden_dev_req);
  // v48: the prompt as steps of up to kReaderMax tokens on the GPU-only path; each step's rows land where the one batch
  // would have put them (logits row c0 + r, hidden row c0 + r); with last_only only the final step runs the head
  if (n > I.o.max_batch || !tokens || !logits) return Status::fail(Code::invalid_argument, "engine: 1 .. max_batch tokens and an output");
  const std::int64_t nvoc = I.s.n_vocab, e = I.s.n_embd;
  EngineStats sum;
  for (int c0 = 0; c0 < n; c0 += Impl::kReaderMax) {
    const int m = std::min(Impl::kReaderMax, n - c0);
    const bool last = c0 + m == n;
    EngineStats es;
    I.no_head = last_only && !last;
    I.in_prompt = true;
    const Status st = I.run(tokens + c0, m, last_only ? logits : logits + static_cast<std::int64_t>(c0) * nvoc, last_only, &es,
                            hidden_out ? hidden_out + static_cast<std::int64_t>(c0) * e : nullptr,
                            I.hidden_dev_req ? I.hidden_dev_req + static_cast<std::int64_t>(c0) * e : nullptr);
    I.no_head = false;
    I.in_prompt = false;
    if (!st) return st;
    sum.seconds += es.seconds;
    sum.hot_picks += es.hot_picks;
  }
  if (stats) *stats = sum;
  return {};
}

Status GpuEngine::verify(const std::int32_t* tokens, int n, float* logits, float* hidden_out, EngineStats* stats) {
  Impl& I = *p_;
  if (n > std::max(1, I.o.max_verify)) return Status::fail(Code::invalid_argument, "engine: verify batch above max_verify");
  I.ckpt_mode = true;
  I.verify_pos0 = I.pos;
  const Status st = I.run(tokens, n, logits, false, stats, hidden_out, I.hidden_dev_req);
  I.ckpt_mode = false;
  I.last_n = st ? n : 0;
  return st;
}

Status GpuEngine::keep(int k) {
  Impl& I = *p_;
  if (I.last_n == 0) return Status::fail(Code::invalid_argument, "engine: keep() without a verify batch");
  if (k < 1 || k > I.last_n) return Status::fail(Code::invalid_argument, "engine: keep 1 .. the verify batch's tokens");
  const int n = I.last_n;
  I.last_n = 0;
  if (k == n) return Status::good();
  const Qwen35MoeSpec& s = I.s;
  const int64_t ch = s.conv_channels(), taps = s.ssm_d_conv, ss = s.ssm_dt_rank * s.ssm_d_state * s.ssm_d_state;
  // v60 G (item 4, the keep() stitch): one launch for every DeltaNet layer - the state from the checkpoint after token k - 1,
  // the conv tail rebuilt from the tail before the batch and the saved inputs - queued on the stream behind the verify; the
  // next step or host read orders after it (no conv output computed, no sync here). The per-layer form it replaces (a state
  // copy + a conv over k rows whose output was discarded + a tail kernel per layer, then a sync: 90 launches) MEASURED 1.2-1.4
  // ms per rejecting step (v60o: keep 0.011-0.041 s over 9-30 such steps). Every DeltaNet layer runs every step, so their
  // ping-pong indices agree; if ever not (or no table), the per-layer form runs, synced as before.
  int cur = -1;
  bool same = I.dnk_d != nullptr && I.dnk_n > 0;
  for (std::uint32_t il = 0; il < s.n_layer && same; ++il) {
    const Layer& D = I.L[il];
    if (!D.state) continue;
    if (cur < 0) cur = D.cur;
    else if (D.cur != cur) same = false;
  }
  if (same && cur >= 0) {
    if (!g::deltanet_keep(I.dnk_d, I.dnk_n, k, cur, ss, ch, taps)) return gfail("engine: keep (stitch)");
    I.pos = I.verify_pos0 + k;
    return Status::good();
  }
  for (std::uint32_t il = 0; il < s.n_layer; ++il) {
    Layer& D = I.L[il];
    if (!D.state) continue;
    // the state after token k - 1 of the batch; the conv tail rebuilt from the tail before the batch and the kept inputs
    if (!g::dcopy(D.state, D.ckpt + static_cast<int64_t>(k - 1) * ss, z(ss) * sizeof(float))) return gfail("engine: keep state");
    if (!g::deltanet_conv(D.mixed_save, D.tail[1 - D.cur], D.conv1d, k, ch, taps, I.conv, D.tail[D.cur])) return gfail("engine: keep tail");
  }
  I.pos = I.verify_pos0 + k;
  I.unsynced = 0;
  return g::sync() ? Status::good() : gfail("engine: keep sync");
}

Status GpuEngine::Impl::run(const std::int32_t* tokens, int n, float* logits, bool last_only, EngineStats* stats, float* hidden_out, float* hidden_dev) {
  Impl& I = *this;
  const Qwen35MoeWeights& wt = *I.w;
  if (n <= 0 || n > I.o.max_batch || !tokens || !logits) return Status::fail(Code::invalid_argument, "engine: 1 .. max_batch tokens and an output");
  if (I.pos + n > I.o.max_ctx) return Status::fail(Code::invalid_argument, "engine: the cache is full (max_ctx)");
  const Clock::time_point t0 = Clock::now();
  EngineStats es;
  const int64_t e = s.n_embd;
  if (n > Impl::kReaderMax) return Status::fail(Code::internal, "engine: run() takes 1-4 tokens (step() splits a prompt)");  // v60
  for (int t = 0; t < n; ++t)
    if (tokens[t] < 0 || static_cast<std::uint64_t>(tokens[t]) >= s.n_vocab) return Status::fail(Code::invalid_argument, "a token id is outside the vocabulary");
  const unsigned slot = I.ring_at++ % static_cast<unsigned>(Impl::kRing);
  if (I.emb_dev) {  // v60: the embedding rows gathered on the GPU; the ids read from the mapped ring
    std::int32_t* th = I.ring_tok_h + static_cast<std::ptrdiff_t>(slot) * Impl::kReaderMax;
    std::copy_n(tokens, n, th);
    if (!g::gather_rows(wt.tok_embd.type, I.emb_dev, wt.tok_embd.row_bytes, e, I.ring_tok_d + static_cast<std::ptrdiff_t>(slot) * Impl::kReaderMax, n, I.x))
      return gfail("engine: embedding gather");
  } else if (Status es0 = I.embed(wt.tok_embd, tokens, n, I.x, e); !es0) {
    return es0;
  }
  Status st;
  if (I.graph_ok) {  // v26: the step's position for the device-side readers (on the engine's stream, before the layers)
    I.ring_pos_h[slot] = static_cast<int>(I.pos);
    if (!g::h2d_async(I.pos_d, I.ring_pos_h + slot, sizeof(int))) return gfail("engine: position");
  }
  I.trunk_dp = I.graph_ok;
  struct DpReset {
    bool& f;
    ~DpReset() { f = false; }
  } dp_reset{I.trunk_dp};
  auto layers = [&]() -> Status {
  for (std::uint32_t il = 0; il < s.n_layer; ++il) {
    const Qwen35MoeLayer& l = wt.layers[il];
    Layer& D = I.L[il];
    bool dnh = false;  // v55g (L3): the DeltaNet dense half in one launch (one token)
    if (l.recurrent && !(st = I.dn_half_layer(D, n, &dnh))) return st;
    if (!dnh) {
    if (!g::rms_norm(I.x, D.attn_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: attn norm");
    if (!(st = l.recurrent ? I.deltanet_block(D, n) : I.attention_block(D, n, I.pos, I.trunk_dp))) return st;
    }
    const bool probe_row = I.probe_pos >= I.pos && I.probe_pos < I.pos + n && I.probe_bd;
    if (probe_row && !g::dcopy(I.probe_bd + static_cast<int64_t>(il) * e, I.y + (I.probe_pos - I.pos) * e, z(e) * sizeof(float))) return gfail("engine: probe copy");
    const bool fused = I.fused_ok(D, n, probe_row);  // v52: residual + MoE half + residual in one launch
    if (fused) {
      if (!(st = I.moe_half_fused(D, n, il))) return st;
    } else {
    if (!g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("engine: residual");
    if (!g::rms_norm(I.x, D.post_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: post norm");
    if (!(st = I.moe_block_reader(D, n, il))) return st;
    }
    if (probe_row) {
      const int64_t pr = I.probe_pos - I.pos, ne2 = s.n_expert;
      if (!g::dcopy(I.probe_md + static_cast<int64_t>(il) * e, I.y + pr * e, z(e) * sizeof(float)) ||
          !g::dcopy(I.probe_ld + static_cast<int64_t>(il) * ne2, I.logits_r + pr * ne2, z(ne2) * sizeof(float)))
        return gfail("engine: probe copy");
    }
    if (!fused && !g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("engine: residual");
    if (I.probe_pos >= I.pos && I.probe_pos < I.pos + n && I.probe_xd) {  // v23: this row's residual and picks after the layer
      const int64_t r = I.probe_pos - I.pos, nuq = s.n_expert_used;
      if (!g::dcopy(I.probe_xd + static_cast<int64_t>(il) * e, I.x + r * e, z(e) * sizeof(float)) ||
          !g::dcopy(I.probe_seld + static_cast<int64_t>(il) * nuq, I.sel + r * nuq, z(nuq) * sizeof(std::int32_t)) ||
          !g::dcopy(I.probe_wd + static_cast<int64_t>(il) * nuq, I.wsel + r * nuq, z(nuq) * sizeof(float)))
        return gfail("engine: probe copy");
    }
  }
  return Status::good();
  };
  // v26: replay a recorded graph of this kind of step, record it the second time this kind is seen, else plain launches
  const bool probe_here = I.probe_pos >= I.pos && I.probe_pos < I.pos + n && I.probe_xd;
  if (I.graph_ok && !probe_here) {
    int parity = 0;
    for (std::uint32_t il = 0; il < s.n_layer; ++il)
      if (wt.layers[il].recurrent) {
        parity = I.L[il].cur;
        break;
      }
    const std::tuple<int, int, int, int> key{n, parity, I.ckpt_mode && n > 1 ? 1 : 0, 0};  // (tokens, DeltaNet tail parity, verify checkpoints, spare)
    auto flip = [&]() {
      for (std::uint32_t il = 0; il < s.n_layer; ++il)
        if (wt.layers[il].recurrent) I.L[il].cur = 1 - I.L[il].cur;
    };
    auto it = I.graphs.find(key);
    if (it != I.graphs.end()) {
      if (!g::graph_launch(it->second, nullptr)) return gfail("engine: graph replay");
      flip();  // the host side of the DeltaNet tails, as the plain step would have done
      ++I.graph_replays;
    } else if (I.graph_seen[key]++ >= 1) {
      void* ex = nullptr;
      std::vector<int> cur_before(s.n_layer);  // v60 commit C: the tail parities before the attempt (a failure part-way flipped only some)
      for (std::uint32_t il = 0; il < s.n_layer; ++il) cur_before[il] = I.L[il].cur;
      const bool began = g::capture_begin(nullptr);
      const Status cs = began ? layers() : Status::fail(Code::internal, "capture did not begin");
      std::size_t nodes = 0;
      const bool ended = began && g::capture_end(nullptr, &ex, &nodes);  // v60 I (0.3): the step's launch count
      if (cs.ok() && ended && ex) {
        I.graphs[key] = ex;
        I.graph_nodes[key] = nodes;
        ++I.graph_captures;
        if (!g::graph_launch(ex, nullptr)) return gfail("engine: graph launch");
      } else {  // the attempt ran the host side only: restore the tails' parities, stop using graphs, run this step plainly
        for (std::uint32_t il = 0; il < s.n_layer; ++il) I.L[il].cur = cur_before[il];
        g::graph_free(ex);
        I.graph_ok = false;
        I.graph_note = std::string("off after a failed capture (") + (cs.ok() ? g::last_error() : cs.text()) + ")";
        if (!(st = layers())) return st;
      }
    } else if (!(st = layers())) {
      return st;
    }
  } else if (!(st = layers())) {
    return st;
  }
  if (I.probe_pos >= I.pos && I.probe_pos < I.pos + n && I.probe_xd) {
    const std::size_t nl = s.n_layer, nuq = s.n_expert_used;
    I.probe_xh.resize(nl * z(e));
    I.probe_selh.resize(nl * nuq);
    I.probe_wh.resize(nl * nuq);
    if (!g::to_host(I.probe_xh.data(), I.probe_xd, I.probe_xh.size() * sizeof(float)) ||
        !g::to_host(I.probe_selh.data(), I.probe_seld, I.probe_selh.size() * sizeof(std::int32_t)) ||
        !g::to_host(I.probe_wh.data(), I.probe_wd, I.probe_wh.size() * sizeof(float)))
      return gfail("engine: probe read");
    I.probe_bh.resize(nl * z(e));
    I.probe_lh.resize(nl * z(s.n_expert));
    I.probe_mh.resize(nl * z(e));
    if (!g::to_host(I.probe_bh.data(), I.probe_bd, I.probe_bh.size() * sizeof(float)) ||
        !g::to_host(I.probe_lh.data(), I.probe_ld, I.probe_lh.size() * sizeof(float)) ||
        !g::to_host(I.probe_mh.data(), I.probe_md, I.probe_mh.size() * sizeof(float)))
      return gfail("engine: probe read");
    I.probe_have = true;
    I.probe_pos = -1;
  }
  const int64_t nvoc = s.n_vocab;
  const int r_begin = I.no_head ? n : last_only ? n - 1 : 0;  // v48: no_head runs no row through the head
  for (int r0 = hidden_out || hidden_dev ? 0 : r_begin; r0 < n; r0 += Impl::kOutRows) {
    const int m = std::min(Impl::kOutRows, n - r0);
    if (!g::rms_norm(I.x + static_cast<int64_t>(r0) * e, I.out_norm, I.xn, e, 1, e, e, m, s.rms_eps)) return gfail("engine: output norm");
    if (hidden_out && !g::to_host(hidden_out + static_cast<int64_t>(r0) * e, I.xn, z(m) * z(e) * sizeof(float))) return gfail("engine: hidden");
    // v60 commit C: the hidden rows kept on the device (the MTP path's seed); stream-ordered, no host visit, the step stays inner
    if (hidden_dev && !g::dcopy(hidden_dev + static_cast<int64_t>(r0) * e, I.xn, z(m) * z(e) * sizeof(float))) return gfail("engine: hidden (device)");
    if (r0 + m <= r_begin) continue;
    // rows before r_begin (asked for the hidden only) are run through the head with the rest of their chunk; only rows
    // from r_begin on are copied out
    if (!(st = I.prod(I.out, I.xn, m, I.lg))) return st;
    const int c0 = std::max(r0, r_begin);
    if (I.top_req) {  // v56: the argmax on the device
      if (!g::argmax_rows(I.lg + static_cast<int64_t>(c0 - r0) * nvoc, nvoc, r0 + m - c0, nvoc, I.top_d + (c0 - r_begin))) return gfail("engine: argmax");
    } else if (!g::to_host(logits + static_cast<int64_t>(c0 - r_begin) * nvoc, I.lg + static_cast<int64_t>(c0 - r0) * nvoc, z(r0 + m - c0) * z(nvoc) * sizeof(float))) {
      return gfail("engine: logits");
    }
  }
  if (I.top_req && n > r_begin && !g::to_host(I.top_req, I.top_d, z(n - r_begin) * sizeof(std::int32_t))) return gfail("engine: argmax out");
  // v60: an inner prompt step (no logits, hidden, trace, margins or probe read on the host) leaves its work queued; the next
  // step's launches follow it on the stream (the ring above keeps its ids and position intact until it runs)
  const bool inner = I.no_head && !hidden_out && !I.top_req && !I.trc_ts_h && !I.rlg_h && !probe_here;
  if (inner && I.unsynced + 2 < static_cast<unsigned>(Impl::kRing)) {
    ++I.unsynced;
  } else {
    if (!g::sync()) return gfail("engine: sync");
    I.unsynced = 0;
  }
  {  // v25: every pick runs on the GPU
    es.hot_picks += static_cast<std::uint64_t>(n) * s.n_expert_used;
    if (I.trc_ts_h) I.trace_step(n, since(t0));
    // v55b / v56: the fused halves' stamps of this step (layers that ran one fill stamp 0): decode steps into ph / dph,
    // verify steps of 2-4 tokens into vph / vdph by size; prompt steps are not counted (stamps cleared)
    auto take = [&](unsigned long long* base, double* acc, std::uint64_t* nlay) {
      bool any = false;
      for (std::uint32_t il = 0; il < s.n_layer; ++il) {
        unsigned long long* pq = base + static_cast<int64_t>(il) * 5;
        if (pq[0] == 0 || pq[4] < pq[0]) {
          pq[0] = 0;
          continue;
        }
        if (acc)
          for (int k = 0; k < 4; ++k) acc[k] += static_cast<double>(pq[k + 1] - pq[k]) * 1e-6;
        if (nlay) ++*nlay;
        pq[0] = 0;
        any = true;
      }
      return any;
    };
    const bool dec = !I.in_prompt && n >= 1 && n <= 4;
    if (I.ph_h && dec) {
      const bool any = take(I.ph_h, dec ? (n == 1 ? I.ph_sum : I.vph_sum[n - 2]) : nullptr, dec && n == 1 ? &I.ph_layers : nullptr);
      if (any && dec) ++(n == 1 ? I.ph_steps : I.vph_steps[n - 2]);
    }
    if (I.dph_h && dec) {
      const bool any = take(I.dph_h, dec ? (n == 1 ? I.dph_sum : I.vdph_sum[n - 2]) : nullptr, dec && n == 1 ? &I.dph_layers : nullptr);
      if (any && dec) ++(n == 1 ? I.dph_steps : I.vdph_steps[n - 2]);
    }
    if (I.rlg_h) {  // v49: this step's router margins, per token and layer (8th - 9th logit, 1st - 2nd)
      const int64_t ne2 = s.n_expert, nu2 = s.n_expert_used, nl = s.n_layer;
      std::vector<float> row(z(ne2));
      for (int t = 0; t < n; ++t) {
        const int64_t p = I.pos + t;  // pos advances after this (I.pos += n below)
        if (p < 0 || p >= I.o.max_ctx) continue;
        for (int64_t il = 0; il < nl; ++il) {
          const float* lr = I.rlg_h + (il * Impl::kReaderMax + t) * ne2;
          row.assign(lr, lr + ne2);
          std::partial_sort(row.begin(), row.begin() + nu2 + 1, row.end(), std::greater<float>());
          I.rmarg[z(p) * 2 * z(nl) + 2 * z(il)] = row[z(nu2 - 1)] - row[z(nu2)];
          I.rmarg[z(p) * 2 * z(nl) + 2 * z(il) + 1] = row[0] - row[1];
        }
        I.rmarg_have[z(p)] = 1;
      }
    }
  }
  I.pos += n;
  es.seconds = since(t0);
  if (stats) *stats = es;
  return Status::good();
}

// v60 commit C: the MTP block's step with everything on the device. tokens_h (host ids, through the mapped ring) or
// tokens_d (device ids: the draft chain reads the previous draft's argmax); h_prev_d: n hidden rows (device, stride n_embd);
// top_d_out: each row's argmax written on the device (null: no head); h_out_d: the block's post-norm hidden rows (device,
// null: not wanted). The host is not visited: the input is gathered and concatenated on the GPU (gather_rows + strided
// copies), the attention's K / V go through knew / vnew at the device position (the direct R3 plan, no scratch copies), and
// the step stays queued (unsynced) unless the caller syncs. logits_h / top_h: the v58 host outputs for mtp_step() callers.
Status GpuEngine::Impl::mtp_run(const std::int32_t* tokens_h, const std::int32_t* tokens_d, const float* h_prev_d, int n, std::int32_t* top_d_out,
                                float* h_out_d, float* logits_h, std::int32_t* top_h) {
  Impl& I = *this;
  const Qwen35MoeWeights& wt = *I.w;
  if (!I.has_mtp) return Status::fail(Code::unsupported, "engine: opened without the MTP block (EngineOptions::mtp)");
  if (n <= 0 || n > Impl::kReaderMax || (!tokens_h && !tokens_d) || !h_prev_d)
    return Status::fail(Code::invalid_argument, "engine: MTP steps are 1-4 tokens with their hidden rows (v60)");
  if (I.mtp_pos + n > I.o.max_ctx) return Status::fail(Code::invalid_argument, "engine: the MTP cache is full (max_ctx)");
  if (!I.mtp_emb_dev) return Status::fail(Code::unsupported, "engine: the MTP embedding is not registered for the GPU gather (see the open note)");
  const int64_t e = s.n_embd;
  // ids and the position through the mapped ring, as run() does (the slot stays untouched until the GPU reads it)
  const unsigned slot = I.ring_at++ % static_cast<unsigned>(Impl::kRing);
  const std::int32_t* ids = tokens_d;
  if (tokens_h) {
    for (int t = 0; t < n; ++t)
      if (tokens_h[t] < 0 || static_cast<std::uint64_t>(tokens_h[t]) >= s.n_vocab) return Status::fail(Code::invalid_argument, "a token id is outside the vocabulary");
    std::int32_t* th = I.ring_tok_h + static_cast<std::ptrdiff_t>(slot) * Impl::kReaderMax;
    std::copy_n(tokens_h, n, th);
    ids = I.ring_tok_d + static_cast<std::ptrdiff_t>(slot) * Impl::kReaderMax;
  }
  I.ring_pos_h[slot] = static_cast<int>(I.mtp_pos);
  if (!g::h2d_async(I.pos_d, I.ring_pos_h + slot, sizeof(int))) return gfail("engine: MTP position");
  // v60 I (R3 rebuild 0.4): a one-token draft that wants only its device argmax runs its chain (from the input copies to the
  // pick) as a CUDA graph after the first plain run of that (hidden input, argmax output, hidden output) shape - the trunk's
  // policy (plain, then capture on the second sight, then replay). Outside the graph, as in run(): the id gather (it reads
  // the ring slot), the position copy (pos_d is fixed), the host-side stamp clearing and the start stamp. The chain's own
  // stamps land in the slot it was captured with, so a replay records that slot for the fold. v60q MEASURED the eager chain at
  // 0.63-0.90 ms per draft for ~0.2 ms of bytes; the trunk's replays cost 0.7 us per kernel.
  // v60 I2: the catch-up steps (no head, 1-4 tokens) and the prompt catch-up chunks join: v60r MEASURED the eager catch-up
  // steps that follow a graph replay at +11 ms over the run (0.043 -> 0.054 s) while the drafts saved 15 ms
  Impl::MtpGraph* G = nullptr;
  if (I.mtp_graph_ok && !logits_h && !top_h && !I.o.mtp_head_sketch_check && (!top_d_out || n == 1)) {
    const std::tuple<const void*, const void*, const void*, int> key{h_prev_d, top_d_out, h_out_d, n};
    auto it = I.mtp_graphs.find(key);
    if (it != I.mtp_graphs.end()) G = &it->second;
    else if (I.mtp_graphs.size() < Impl::kMtpGraphMax) G = &I.mtp_graphs[key];
  }
  const bool replay = G && G->exec;
  const bool capture = G && !G->exec && G->seen++ >= 1;
  const unsigned sslot = replay ? G->slot : slot;
  unsigned long long* ts = I.mtp_ts_d ? I.mtp_ts_d + static_cast<std::ptrdiff_t>(sslot) * Impl::kMtpStamps : nullptr;  // v60 D (E2: 6 per slot; H: 9)
  if (ts) {
    I.mtp_slots.push_back(sslot);
    // v60 E2: the chain stamps (3, 4) are written on the sketch path only; the slot is reused after kRing steps, so clear them
    // on the host first (mapped memory; the GPU's writes for this step come later in stream order)
    unsigned long long* th = I.mtp_ts_h + static_cast<std::ptrdiff_t>(sslot) * Impl::kMtpStamps;
    th[3] = 0;
    th[4] = 0;
    if (!g::stamp(ts + 0)) return gfail("engine: MTP stamp");
  }
  // [enorm(embedding) | hnorm(h_prev)] -> eh_proj: the embedding rows gathered into xr, both halves copied at the 2 n_embd stride
  const WeightView& et = wt.mtp.embed_tokens;
  if (!g::gather_rows(et.type, I.mtp_emb_dev, et.row_bytes, e, ids, n, I.xr)) return gfail("engine: MTP embedding gather");
  auto chain = [&]() -> Status {
  if (!g::dcopy2d(I.cat, z(2 * e) * sizeof(float), I.xr, z(e) * sizeof(float), z(e) * sizeof(float), z(n)) ||
      !g::dcopy2d(I.cat + e, z(2 * e) * sizeof(float), h_prev_d, z(e) * sizeof(float), z(e) * sizeof(float), z(n)))
    return gfail("engine: MTP input");
  if (!g::rms_norm(I.cat, I.enorm, I.cat, e, 1, e, 2 * e, n, s.rms_eps) || !g::rms_norm(I.cat + e, I.hnorm, I.cat + e, e, 1, e, 2 * e, n, s.rms_eps))
    return gfail("engine: MTP norms");
  if (ts && !g::stamp(ts + 6)) return gfail("engine: MTP stamp");  // v60 H: after the input
  Status st;
  if (!(st = I.prod(I.eh, I.cat, n, I.x))) return st;
  if (ts && !g::stamp(ts + 7)) return gfail("engine: MTP stamp");  // v60 H: after eh_proj
  Layer& D = I.M;
  if (!g::rms_norm(I.x, D.attn_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: MTP attn norm");
  if (!(st = I.attention_block(D, n, I.mtp_pos, true))) return st;  // v60 C: K / V through knew / vnew at *pos_d (the direct plan)
  if (ts && !g::stamp(ts + 8)) return gfail("engine: MTP stamp");  // v60 H: after the attention (the MoE half ends at stamp 1)
  if (!I.fused_ok(D, n, false)) {  // v58b: names the first condition that keeps the block off the fused launch
    const char* why = I.fused_why(D, n, false);
    return Status::fail(Code::internal, std::string("engine: the MTP block's MoE half cannot run on the fused launch: ") + (why ? why : "?"));
  }
  if (!(st = I.moe_half_fused(D, n, s.n_layer))) return st;  // v58: residual + MoE half + residual in one launch, the reader's row n_layer
  ++I.mtp_fused_steps;
  if (ts && !g::stamp(ts + 1)) return gfail("engine: MTP stamp");
  const int64_t nvoc = s.n_vocab;
  if (!g::rms_norm(I.x, I.head_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: MTP head norm");
  if (h_out_d && !g::dcopy(h_out_d, I.xn, z(n) * z(e) * sizeof(float))) return gfail("engine: MTP hidden");
  if (ts && !g::stamp(ts + 2)) return gfail("engine: MTP stamp");
  // v60 E: a one-token draft that wants only its device argmax takes the head sketch (the full head's bytes x ~0.38): the
  // sketch product by the 1-bit route, the top kHeadSketchTop rows gathered and re-scored by the head's own kernel, their
  // argmax mapped back. Anything that reads logits, or more than one row, runs the full head below as before
  const bool sketch = I.hsk && n == 1 && top_d_out && !logits_h;
  if (sketch) {
    const int64_t hk = I.head.k;
    if (I.o.mtp_head_sketch_check) {  // the full head too, its argmax into top_d (the truth the recall counter compares with)
      if (!(st = I.prod(I.head, I.xn, 1, I.lg + nvoc))) return st;
      if (!g::argmax_rows(I.lg + nvoc, nvoc, 1, nvoc, I.top_d)) return gfail("engine: MTP argmax (check)");
    }
    if (!g::act_q8x32(I.xn, hk, 1, hk, I.head.v->rot32, I.xr, hk, I.q, I.scale, I.sum)) return gfail("engine: rounding for the head sketch");
    if (!g::head_sketch_dot(I.hsk, static_cast<const std::uint16_t*>(I.head.sw.d6), nvoc, hk, I.q, I.scale, I.lg)) return gfail("engine: head sketch");
    if (ts && !g::stamp(ts + 3)) return gfail("engine: MTP stamp");  // v60 E2: after the sketch dot
    if (!g::topk_select(I.lg, nvoc, Impl::kHeadSketchTop, I.cand_d)) return gfail("engine: head sketch top rows");
    if (ts && !g::stamp(ts + 4)) return gfail("engine: MTP stamp");  // v60 E2: after the top-k select
    if (!g::gather_q6k_rows(I.head.sw.w, static_cast<const std::uint16_t*>(I.head.sw.d6), hk, I.cand_d, Impl::kHeadSketchTop, I.htop_w, I.htop_d6))
      return gfail("engine: head sketch gather");
    if (!(st = I.prod_r3({{&I.head_top, I.lg_top}}, I.xn, 1))) return st;  // the exact Q6_K product of the top rows
    if (!g::argmax_rows(I.lg_top, Impl::kHeadSketchTop, 1, Impl::kHeadSketchTop, I.idx_d)) return gfail("engine: head sketch argmax");
    if (!g::pick_id(I.cand_d, I.idx_d, top_d_out)) return gfail("engine: head sketch pick");
    if (I.o.mtp_head_sketch_check && !g::recall_count(I.cand_d, Impl::kHeadSketchTop, I.top_d, I.sk_cnt_d, I.sk_cnt_d + 1)) return gfail("engine: head sketch recall");
    ++I.sk_drafts;
    if (top_h && !g::to_host(top_h, top_d_out, sizeof(std::int32_t))) return gfail("engine: MTP argmax out");
  } else if (top_d_out || logits_h || top_h) {
    if (!(st = I.prod(I.head, I.xn, n, I.lg))) return st;
    if (top_d_out && !g::argmax_rows(I.lg, nvoc, n, nvoc, top_d_out)) return gfail("engine: MTP argmax");
    if (top_h) {
      if (!top_d_out && !g::argmax_rows(I.lg, nvoc, n, nvoc, I.top_d)) return gfail("engine: MTP argmax");
      if (!g::to_host(top_h, top_d_out ? top_d_out : I.top_d, z(n) * sizeof(std::int32_t))) return gfail("engine: MTP argmax out");
    }
    if (logits_h && !g::to_host(logits_h, I.lg, z(n) * z(nvoc) * sizeof(float))) return gfail("engine: MTP logits");
  }
  if (ts && !g::stamp(ts + 5)) return gfail("engine: MTP stamp");
  return Status::good();
  };
  if (replay) {
    if (!g::graph_launch(G->exec, nullptr)) return gfail("engine: MTP draft graph replay");
    ++I.mtp_graph_replays;
    ++I.mtp_fused_steps;  // the host-side counts the chain would have kept
    if (I.hsk && top_d_out) ++I.sk_drafts;  // (I2: a catch-up replay has no head)
  } else if (capture) {
    void* ex = nullptr;
    const bool began = g::capture_begin(nullptr);
    const Status cs = began ? chain() : Status::fail(Code::internal, "capture did not begin");
    std::size_t nodes = 0;
    const bool ended = began && g::capture_end(nullptr, &ex, &nodes);
    if (cs.ok() && ended && ex) {
      G->exec = ex;
      G->slot = slot;
      G->nodes = nodes;
      ++I.mtp_graph_captures;
      if (!g::graph_launch(ex, nullptr)) return gfail("engine: MTP draft graph launch");
    } else {  // the attempt ran the host side only: no more draft graphs, run this chain plainly
      g::graph_free(ex);
      I.mtp_graph_ok = false;
      I.mtp_graph_note = std::string("off after a failed capture (") + (cs.ok() ? g::last_error() : cs.text()) + ")";
      if (Status r = chain(); !r) return r;
    }
  } else if (Status r = chain(); !r) {
    return r;
  }
  I.mtp_pos += n;
  // the ring guard, as run(): a step that read nothing on the host stays queued; the slots must not lap the GPU
  if (I.unsynced + 2 < static_cast<unsigned>(Impl::kRing)) {
    ++I.unsynced;
  } else {
    if (!g::sync()) return gfail("engine: MTP sync");
    I.unsynced = 0;
  }
  return Status::good();
}

Status GpuEngine::mtp_step(const std::int32_t* tokens, const float* h_prev, int n, float* logits, float* h_out, EngineStats* stats) {
  Impl& I = *p_;
  const Qwen35MoeSpec& s = I.s;
  if (!I.has_mtp) return Status::fail(Code::unsupported, "engine: opened without the MTP block (EngineOptions::mtp)");
  if (n <= 0 || n > Impl::kReaderMax || !tokens || !h_prev) return Status::fail(Code::invalid_argument, "engine: MTP 1 .. 4 tokens and their hidden rows (v60)");
  const Clock::time_point t0 = Clock::now();
  EngineStats es;
  const int64_t e = s.n_embd;
  // the host rows up (the v58 interface: engine-check's dump path); the device loop in engine_generate_greedy_mtp uses mtp_run
  if (!g::to_device(I.hid_d, h_prev, z(n) * z(e) * sizeof(float))) return gfail("engine: MTP hidden upload");
  Status st = I.mtp_run(tokens, nullptr, I.hid_d, n, I.top_req ? I.top_d : nullptr, h_out ? I.hid_d : nullptr, I.top_req ? nullptr : logits,
                        I.top_req && logits ? I.top_req : nullptr);
  if (!st) return st;
  if (h_out && !g::to_host(h_out, I.hid_d, z(n) * z(e) * sizeof(float))) return gfail("engine: MTP hidden");
  if (!g::sync()) return gfail("engine: MTP sync");
  I.unsynced = 0;
  I.mtp_slots.clear();  // v60 D: the host interface does not fold the stamps
  es.seconds = since(t0);
  if (stats) *stats = es;
  return Status::good();
}

// v56: the argmax variants (Impl::top_req routes the head's rows to the device argmax; the logits pointer is never written)
namespace {
struct TopReq {
  std::int32_t*& r;
  TopReq(std::int32_t*& slot, std::int32_t* v) : r(slot) { r = v; }
  ~TopReq() { r = nullptr; }
};
}  // namespace
Status GpuEngine::step_top(const std::int32_t* tokens, int n, std::int32_t* top, bool last_only, EngineStats* st, float* hidden_out) {
  if (!top) return Status::fail(Code::invalid_argument, "engine: step_top needs an output");
  TopReq g{p_->top_req, top};
  return step(tokens, n, reinterpret_cast<float*>(top), last_only, st, hidden_out);
}
Status GpuEngine::verify_top(const std::int32_t* tokens, int n, std::int32_t* top, float* hidden_out, EngineStats* st) {
  if (!top) return Status::fail(Code::invalid_argument, "engine: verify_top needs an output");
  TopReq g{p_->top_req, top};
  return verify(tokens, n, reinterpret_cast<float*>(top), hidden_out, st);
}
Status GpuEngine::mtp_step_top(const std::int32_t* tokens, const float* h_prev, int n, std::int32_t* top, float* h_out, EngineStats* st) {
  if (!top) return Status::fail(Code::invalid_argument, "engine: mtp_step_top needs an output");
  TopReq g{p_->top_req, top};
  return mtp_step(tokens, h_prev, n, reinterpret_cast<float*>(top), h_out, st);
}


void GpuEngine::mtp_truncate(std::int64_t p) {
  if (p >= 0 && p < p_->mtp_pos) p_->mtp_pos = p;
}
std::int64_t GpuEngine::mtp_pos() const { return p_ ? p_->mtp_pos : 0; }

namespace {
std::int32_t argmax_of(const float* v, std::int64_t n) {
  std::int64_t b = 0;
  for (std::int64_t i = 1; i < n; ++i)
    if (v[i] > v[b]) b = i;
  return static_cast<std::int32_t>(b);
}
void add_stats(EngineStats& a, const EngineStats& b) {
  a.seconds += b.seconds;
  a.hot_picks += b.hot_picks;
}
}  // namespace

Status engine_generate_greedy_mtp(GpuEngine& eng, const Qwen35MoeSpec& s, const std::vector<std::int32_t>& prompt, int n_predict, int n_draft,
                                  const std::vector<std::int32_t>& stops, std::vector<std::int32_t>& out, MtpStats* stats, EngineStats* es,
                                  const std::function<void(std::int32_t)>& on_token, EngineMtpTiming* tm) {
  EngineMtpTiming tm_local;
  EngineMtpTiming& T = tm ? *tm : tm_local;
  auto secs = [](Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); };
  const Clock::time_point tp0 = Clock::now();
  out.clear();
  const auto is_stop = [&stops](std::int32_t t) { return std::find(stops.begin(), stops.end(), t) != stops.end(); };
  if (prompt.empty()) return Status::fail(Code::invalid_argument, "an empty prompt");
  if (n_predict <= 0) return Status::good();
  if (n_draft < 0) return Status::fail(Code::invalid_argument, "the draft length is 0 or more");
  const int64_t e = s.n_embd, nv = s.n_vocab, P = static_cast<int64_t>(prompt.size());
  MtpStats local;
  MtpStats& stt = stats ? *stats : local;
  EngineStats tot;
  if (Status r = eng.reset(); !r) return r;
  // v60 commit C: the whole loop on the device. The trunk's hidden rows go to hid_d (never to the host), the MTP block's input
  // is gathered and concatenated on the GPU, the draft chain feeds each draft's device argmax to the next draft's gather, and
  // the host reads one int per draft (the ids it needs for the verify batch) plus the verify's argmaxes
  GpuEngine::Impl& I = *eng.p_;
  if (n_draft > 0 && !I.mtp_emb_dev) return Status::fail(Code::unsupported, "engine: the MTP block's device input is off (see the open note)");
  const float* const hid0 = I.hid_d;  // row 0: the hidden before a chunk (zeros before the first); rows 1 .. k: the chunk's
  // v60 D: after a sync, the GPU time of the MTP steps issued since the last fold, split at the head (stamps 0-1 layer, 2-3 head)
  auto fold_stamps = [&](double* total_s, double* layer_s, double* head_s, double* block_s) {
    for (unsigned slot : I.mtp_slots) {
      const unsigned long long* t = I.mtp_ts_h + static_cast<std::ptrdiff_t>(slot) * GpuEngine::Impl::kMtpStamps;
      if (t[0] && t[5] >= t[0]) *total_s += static_cast<double>(t[5] - t[0]) * 1e-9;
      if (t[0] && t[1] >= t[0]) *layer_s += static_cast<double>(t[1] - t[0]) * 1e-9;
      // v60 H: the block's split (stamps 6, 7, 8 inside it; drafts only)
      if (block_s && t[0] && t[6] >= t[0] && t[7] >= t[6] && t[8] >= t[7] && t[1] >= t[8]) {
        block_s[0] += static_cast<double>(t[6] - t[0]) * 1e-9;
        block_s[1] += static_cast<double>(t[7] - t[6]) * 1e-9;
        block_s[2] += static_cast<double>(t[8] - t[7]) * 1e-9;
        block_s[3] += static_cast<double>(t[1] - t[8]) * 1e-9;
      }
      if (t[2] && t[5] >= t[2]) *head_s += static_cast<double>(t[5] - t[2]) * 1e-9;
      // v60 E2: the sketch chain's split (stamps 3 and 4 are written only on the sketch path; zero otherwise)
      if (t[2] && t[3] >= t[2] && t[4] >= t[3] && t[5] >= t[4]) {
        T.sketch_dot_s += static_cast<double>(t[3] - t[2]) * 1e-9;
        T.sketch_select_s += static_cast<double>(t[4] - t[3]) * 1e-9;
        T.sketch_rescore_s += static_cast<double>(t[5] - t[4]) * 1e-9;
      }
    }
    I.mtp_slots.clear();
  };
  if (!g::dzero(I.hid_d, z(e) * sizeof(float))) return Status::fail(Code::internal, "engine: MTP seed");
  // the prompt in chunks: the trunk's last logits (host) and every position's hidden (device); the MTP catch-up of each
  // chunk queued right behind it in 4-token steps (v58c: no picks go to a slow path), nothing waits on the host in between
  const int kChunk = eng.max_batch();
  std::vector<float> lg(z(nv));
  for (int64_t t0 = 0; t0 < P; t0 += kChunk) {
    const int k = static_cast<int>(std::min<int64_t>(kChunk, P - t0));
    EngineStats x;
    I.hidden_dev_req = I.hid_d + e;
    const Status r = eng.step(prompt.data() + t0, k, lg.data(), true, &x);
    I.hidden_dev_req = nullptr;
    if (!r) return r;
    add_stats(tot, x);
    if (n_draft > 0)
      for (int u = 0; u < k; u += GpuEngine::Impl::kReaderMax) {  // token t0 + u with the hidden of t0 + u - 1 = row u
        const int m = std::min(GpuEngine::Impl::kReaderMax, k - u);
        if (Status rc = I.mtp_run(prompt.data() + t0 + u, nullptr, hid0 + static_cast<int64_t>(u) * e, m, nullptr, nullptr, nullptr, nullptr); !rc) return rc;
      }
    if (!g::dcopy(I.hid_d, I.hid_d + static_cast<int64_t>(k) * e, z(e) * sizeof(float))) return Status::fail(Code::internal, "engine: hidden carry");
  }
  if (!g::sync()) return Status::fail(Code::internal, std::string("engine: prompt sync: ") + g::last_error());
  I.unsynced = 0;
  if (I.mtp_ts_h) fold_stamps(&T.prompt_mtp_gpu_s, &T.prompt_mtp_layer_s, &T.prompt_mtp_head_s, nullptr);  // v60 D: the MTP catch-up over the prompt
  // pending: the hidden of the last prompt position (row 0 after the carry) -> h_d, the draft chain's seed
  if (!g::dcopy(I.h_d, I.hid_d, z(e) * sizeof(float))) return Status::fail(Code::internal, "engine: MTP seed");
  T.prompt_s += secs(tp0);
  std::int32_t last = argmax_of(lg.data(), nv);
  out.push_back(last);
  if (on_token) on_token(last);
  std::vector<std::int32_t> drafts;
  while (static_cast<int>(out.size()) < n_predict && !is_stop(last)) {
    ++stt.steps;
    const int room = n_predict - static_cast<int>(out.size()) - 1;
    const int k = std::max(0, std::min(n_draft, room));
    drafts.assign(z(k), 0);
    const std::int64_t m0 = eng.mtp_pos();
    if (k > 0) {
      const Clock::time_point tq = Clock::now();
      for (int i = 0; i < k; ++i) {
        // draft 0: the trunk's token (host id through the ring) with the pending hidden; draft i: the previous draft's device
        // argmax as its id, the previous draft's hidden (hn_d; the read precedes the write on the stream)
        if (Status r = I.mtp_run(i == 0 ? &last : nullptr, i == 0 ? nullptr : I.mtp_top_d + (i - 1), i == 0 ? I.h_d : I.hn_d, 1, I.mtp_top_d + i, I.hn_d,
                                 nullptr, nullptr);
            !r)
          return r;
      }
      if (!g::to_host(drafts.data(), I.mtp_top_d, z(k) * sizeof(std::int32_t))) return Status::fail(Code::internal, "engine: draft ids");
      I.unsynced = 0;
      T.draft_s += secs(tq);
      if (I.mtp_ts_h) fold_stamps(&T.draft_gpu_s, &T.draft_layer_s, &T.draft_head_s, T.draft_block_s);
      T.drafts += k;
      // v58c: draft[0] at position m0 used exact (last, pending), so m0 + 1 is ALREADY valid in the MTP KV cache!
      eng.mtp_truncate(m0 + 1);
      stt.drafted += k;
    }
    const int nb = 1 + k;
    std::vector<std::int32_t> batch{last};
    batch.insert(batch.end(), drafts.begin(), drafts.end());
    std::vector<std::int32_t> tops(z(nb));
    EngineStats x;
    const Clock::time_point tv = Clock::now();
    I.hidden_dev_req = I.hid_d;  // the batch's hidden rows 0 .. nb - 1 on the device
    const Status rv = k > 0 ? eng.verify_top(batch.data(), nb, tops.data(), nullptr, &x) : eng.step_top(batch.data(), 1, tops.data(), true, &x, nullptr);
    I.hidden_dev_req = nullptr;
    if (!rv) return rv;
    T.verify_s[nb > 8 ? 9 : nb] += secs(tv);
    ++T.verify_n[nb > 8 ? 9 : nb];
    add_stats(tot, x);
    const Clock::time_point ta2 = Clock::now();
    int a = 0;
    while (a < k && tops[z(a)] == drafts[z(a)] && !is_stop(drafts[z(a)])) ++a;
    if (a < k && is_stop(drafts[z(a)]) && tops[z(a)] == drafts[z(a)]) ++a;
    const std::int32_t next = tops[z(a)];
    T.argmax_s += secs(ta2);
    const Clock::time_point tk = Clock::now();
    if (k > 0)
      if (Status r = eng.keep(a + 1); !r) return r;
    T.keep_s += secs(tk);
    stt.accepted += a;
    if (n_draft > 0) {  // the MTP block catches up over the accepted tokens (their hidden rows are in hid_d), one queued step
      const Clock::time_point tc = Clock::now();
      Status rc;
      if (k == 0) rc = I.mtp_run(batch.data(), nullptr, I.h_d, 1, nullptr, nullptr, nullptr, nullptr);
      else if (a > 0) rc = I.mtp_run(batch.data() + 1, nullptr, I.hid_d, a, nullptr, nullptr, nullptr, nullptr);
      if (!rc) return rc;
      if (!g::sync()) return Status::fail(Code::internal, std::string("engine: catch-up sync: ") + g::last_error());
      I.unsynced = 0;
      T.catchup_s += secs(tc);
      if (I.mtp_ts_h) fold_stamps(&T.catchup_gpu_s, &T.catchup_layer_s, &T.catchup_head_s, nullptr);
    }
    // pending: the hidden of the accepted token (row a) -> h_d
    if (!g::dcopy(I.h_d, I.hid_d + static_cast<int64_t>(a) * e, z(e) * sizeof(float))) return Status::fail(Code::internal, "engine: pending hidden");
    for (int i = 0; i < a && static_cast<int>(out.size()) < n_predict; ++i) {
      out.push_back(drafts[z(i)]);
      if (on_token) on_token(drafts[z(i)]);
    }
    if (!out.empty() && is_stop(out.back())) break;
    if (static_cast<int>(out.size()) < n_predict) {
      out.push_back(next);
      if (on_token) on_token(next);
    }
    last = next;
  }
  if (I.hsk) {  // v60 E: the sketch's counters (the last step synced before its host read; the counters are plain device ints)
    unsigned cnt[2] = {0, 0};
    if (!g::sync() || !g::to_host(cnt, I.sk_cnt_d, sizeof cnt)) return gfail("engine: head sketch counters");
    I.sk_hit = cnt[0];
    I.sk_total = cnt[1];
    T.sketch_drafts = I.sk_drafts;
    T.sketch_recall_hit = I.sk_hit;
    T.sketch_recall_n = I.sk_total;
  }
  if (es) *es = tot;
  return Status::good();
}

Result<std::vector<std::vector<int>>> hot_order_from_routes(const std::string& path, std::uint32_t n_layer, std::uint32_t n_expert,
                                                            std::vector<std::vector<std::uint64_t>>* counts) {
  using R = Result<std::vector<std::vector<int>>>;
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return R::fail(Status::fail(Code::not_found, "routing record not found: " + path));
  char magic[8];
  std::uint32_t hdr[5];
  std::uint64_t hash = 0;
  bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "PELROUT1", 8) == 0 && std::fread(hdr, 4, 5, f) == 5 && std::fread(&hash, 8, 1, f) == 1;
  if (!ok || hdr[3] != n_layer || hdr[4] > 8) {
    std::fclose(f);
    return R::fail(Status::fail(Code::corrupt, "not a routing record for this model (PELROUT1, " + std::to_string(n_layer) + " layers): " + path));
  }
  const std::uint64_t records = static_cast<std::uint64_t>(hdr[1]) * hdr[2] * hdr[3];
  std::vector<std::vector<std::uint64_t>> cnt(n_layer, std::vector<std::uint64_t>(n_expert, 0));
  std::uint8_t rec[72];
  for (std::uint64_t i = 0; i < records && ok; ++i) {
    ok = std::fread(rec, 1, sizeof rec, f) == sizeof rec;
    if (!ok) break;
    const std::uint32_t il = static_cast<std::uint32_t>(i % n_layer);
    for (int j = 0; j < 8; ++j) {
      std::int16_t ex;
      std::memcpy(&ex, rec + 2 * j, 2);
      if (ex >= 0 && static_cast<std::uint32_t>(ex) < n_expert) ++cnt[il][static_cast<std::size_t>(ex)];
    }
  }
  std::fclose(f);
  if (!ok) return R::fail(Status::fail(Code::corrupt, "routing record cut short: " + path));
  std::vector<std::vector<int>> order(n_layer);
  for (std::uint32_t il = 0; il < n_layer; ++il) {
    std::vector<int>& v = order[il];
    v.resize(n_expert);
    for (std::uint32_t e = 0; e < n_expert; ++e) v[e] = static_cast<int>(e);
    std::stable_sort(v.begin(), v.end(), [&](int a, int b) { return cnt[il][static_cast<std::size_t>(a)] > cnt[il][static_cast<std::size_t>(b)]; });
  }
  if (counts) *counts = std::move(cnt);  // v51
  return R::good(std::move(order));
}

}  // namespace pelias::model
