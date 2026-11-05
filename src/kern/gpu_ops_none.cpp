// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kern/gpu_ops_none.cpp - the GPU kernels' interface in a build without the CUDA backend (PELIAS_CUDA off): every call
// reports that there is no GPU backend, so callers fall back to the CPU path.
#include "kern/gpu_ops.h"
#include "kern/gpu_prompt.h"
#include "kern/gpu_stream.h"

namespace pelias::kern::gpu {
namespace {
constexpr const char* kNone = "built without the CUDA backend (configure with -DPELIAS_CUDA=ON)";
}

bool compiled() { return false; }
bool available() { return false; }
const char* last_error() { return kNone; }
void* dalloc(std::size_t) { return nullptr; }
void dfree(void*) {}
bool to_device(void*, const void*, std::size_t, Stream) { return false; }
bool to_host(void*, const void*, std::size_t, Stream) { return false; }
bool dzero(void*, std::size_t, Stream) { return false; }
bool sync(Stream) { return false; }
bool set_host_wait(int, std::string* why) {
  if (why) *why = "no CUDA in this build";
  return false;
}
bool mem_info(std::size_t*, std::size_t*) { return false; }
bool dcopy(void*, const void*, std::size_t, Stream) { return false; }
bool dcopy2d(void*, std::size_t, const void*, std::size_t, std::size_t, std::size_t, Stream) { return false; }  // v60 C
bool dcopy_any(void*, const void*, std::size_t, Stream) { return false; }  // v60 C2
bool probe_read_sm(const void*, std::size_t, int, unsigned*, Stream) { return false; }  // v60 C3
void* host_alloc_mapped(std::size_t, void**) { return nullptr; }
void host_free(void*) {}
bool bell_ring(unsigned*, unsigned, Stream, unsigned long long*) { return false; }
bool bell_wait(const unsigned*, unsigned, Stream, unsigned long long*) { return false; }
bool bell_ring_dev(unsigned*, const unsigned*, unsigned, Stream) { return false; }
bool bell_wait_dev(const unsigned*, const unsigned*, unsigned, Stream) { return false; }
bool stamp(unsigned long long*, Stream) { return false; }
void router_topk_serial_for_tests(bool) {}
bool host_register(void*, std::size_t, void**, std::string* why) {
  if (why) *why = "built without CUDA";
  return false;
}
void host_unregister(void*) {}
bool read_rate(const void*, std::size_t, int, double*) { return false; }
bool read_chunks_rate(const void*, const std::uint64_t*, int, std::size_t, double*) { return false; }
bool kv_store(float*, float*, const float*, const float*, std::int64_t, std::int64_t, const int*, Stream) { return false; }
void set_stream(Stream) {}
Stream stream_new() { return nullptr; }
void stream_free(Stream) {}
bool h2d_async(void*, const void*, std::size_t, Stream) { return false; }
bool capture_begin(Stream) { return false; }
bool capture_end(Stream, void** exec) {
  *exec = nullptr;
  return false;
}
bool graph_launch(void*, Stream) { return false; }
void graph_free(void*) {}
bool reader_prefetch(const ExpertPtr*, const std::int32_t*, int, const ReaderCache*, std::int64_t, std::int64_t, int*, Stream) { return false; }
void* event_new() { return nullptr; }
void event_free(void*) {}
bool event_record(void*, Stream) { return false; }
bool stream_wait(Stream, void*) { return false; }
bool moe_reader3(const ExpertPtr*, const ExpertPtr*, const std::int32_t*, const float*, const float*, int, const float*, std::int64_t, std::int64_t, std::int64_t,
                 std::uint64_t, const ReaderCache*, float*, float*, float*, int*, ExpertPtr*, ExpertPtr*, float*, unsigned long long*, Stream, int,
                 const std::uint8_t*, int) {
  return false;
}
bool to_host_async(void*, const void*, std::size_t, Stream) { return false; }
bool matmul(std::uint32_t, const void*, std::uint64_t, std::int64_t, std::int64_t, const float*, std::int64_t, std::int64_t, float*, std::int64_t,
            Stream) {
  return false;
}
bool moe_matmul(std::uint32_t, const void*, std::uint64_t, std::uint64_t, std::int64_t, std::int64_t, int, const int*, const int*, const int*,
                const float*, std::int64_t, float*, std::int64_t, int, Stream) {
  return false;
}
bool gather_rows(std::uint32_t, const void*, std::uint64_t, std::int64_t, const std::int32_t*, std::int64_t, float*, Stream) { return false; }
void set_fast_f32(bool) {}
bool router_fast_for(std::int64_t, float) { return false; }
bool act_q8x32(const float*, std::int64_t, std::int64_t, std::int64_t, bool, float*, std::int64_t, std::int8_t*, float*, int*, Stream) {
  return false;
}
bool rms_norm_q8x32(const float*, const float*, std::int64_t, std::int64_t, std::int64_t, float, bool, float*, std::int64_t, float*, std::int64_t,
                    std::int8_t*, float*, int*, Stream) {
  return false;
}
bool rms_norm(const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, float, Stream) {
  return false;
}
bool l2_norm(float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, float, Stream) { return false; }
bool rope(float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, int, float, const int*, bool, const float*, Stream, const int*) {
  return false;
}
bool silu_mul(const float*, const float*, float*, std::int64_t, Stream) { return false; }
bool add_rows(float*, std::int64_t, const float*, std::int64_t, const float*, std::int64_t, std::int64_t, Stream) { return false; }
bool router_topk(const float*, std::int64_t, std::int64_t, int, float, std::int32_t*, float*, Stream, float) { return false; }
bool attention(const float*, std::int64_t, std::int64_t, const float*, const float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
               std::int64_t, std::int64_t, float, bool, float*, float*, Stream, const int*, std::int64_t) {
  return false;
}
bool deltanet_conv(const float*, const float*, const float*, std::int64_t, std::int64_t, std::int64_t, float*, float*, Stream) { return false; }
bool deltanet_gates(float*, float*, const float*, const float*, std::int64_t, std::int64_t, Stream) { return false; }
bool argmax_rows(const float*, std::int64_t, std::int64_t, std::int64_t, std::int32_t*, Stream) { return false; }
bool deltanet_ba_fast(std::uint32_t, const void*, std::uint64_t, std::uint32_t, const void*, std::uint64_t, std::int64_t, std::int64_t, const float*, std::int64_t,
                      std::int64_t) {
  return false;
}
bool deltanet_beta_alpha_gates(std::uint32_t, const void*, std::uint64_t, std::uint32_t, const void*, std::uint64_t, std::int64_t, std::int64_t, const float*,
                               std::int64_t, std::int64_t, float*, float*, const float*, const float*, Stream) {
  return false;
}
bool deltanet_recurrent(const float*, std::int64_t, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                        float*, Stream) {
  return false;
}
bool deltanet_chunked(const float*, std::int64_t, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                      float*, Stream) {
  return false;
}
bool deltanet_recurrent_reg(const float*, std::int64_t, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t,
                            std::int64_t, float*, Stream) {
  return false;
}
bool deltanet_recurrent_reg_ckpt(const float*, std::int64_t, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t,
                                 std::int64_t, float*, float*, Stream) {
  return false;
}
bool deltanet_chunked_rows(const float*, std::int64_t, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t,
                           std::int64_t, int, float*, Stream) {
  return false;
}
bool gated_norm(float*, const float*, const float*, std::int64_t, std::int64_t, std::int64_t, float, Stream) { return false; }

// R3 streaming (kern/gpu_stream.h)
bool stream_type_ok(std::uint32_t, std::int64_t) { return false; }
bool stream_upload(std::uint32_t, const void*, std::int64_t, std::int64_t, StreamWeights*) { return false; }
void stream_release(StreamWeights* w) {
  if (w) *w = StreamWeights{};
}
const char* stream_last_error() { return "built without the CUDA backend (configure with -DPELIAS_CUDA=ON)"; }
StreamPlan::~StreamPlan() {  // nothing was allocated; the fields are named so every compiler sees them used
  dev_ = nullptr;
  njobs_ = nstages_ = ngroups_ = grid_ = 0;
  mt_ = false;
  epoch_ = 0;
  off_jobs_ = off_stage_ = off_grp_ = off_dep_ = off_gtasks_ = off_done_ = off_gdone_ = 0;
}
bool StreamPlan::build(const StreamJob*, int, const int*, int, int) { return false; }
bool StreamPlan::run(Stream) { return false; }
bool StreamPlan::run_unstaged(Stream) { return false; }
bool StreamPlan::run_reset(Stream) { return false; }
bool rx_type_ok(std::uint32_t, std::int64_t) { return false; }
std::uint64_t rx_scratch_bytes(int) { return 0; }
bool moe_half(const MoeHalfArgs&, Stream) { return false; }
bool dn_half_ok(const DnHalfArgs&) { return false; }
bool fused_width_ok(int) { return false; }
bool dn_half(const DnHalfArgs&, Stream) { return false; }
const void* StreamPlan::jobs_dev() const { return nullptr; }
bool rx_set_grid_bits(int) { return false; }
void stream_kernel_report() {}
int rx_stage_block_bytes(std::uint32_t, int) { return 272; }
void rx_set_stage_bytes(int) {}
bool moe_r3(const ExpertPtr*, const ExpertPtr*, const std::int32_t*, const float*, const float*, int, const std::int8_t*, const float*, const int*,
            std::int64_t, std::int64_t, std::int64_t, const ReaderCache*, float*, std::int8_t*, float*, int*, float*, int*, ExpertPtr*, ExpertPtr*, void*,
            int, float*, unsigned long long*, Stream, int) {
  return false;
}

// the int8 prompt path (kern/gpu_prompt.h)
bool prompt_type_ok(std::uint32_t, std::int64_t) { return false; }
bool prompt_prepare(std::uint32_t, const void*, std::int64_t, std::int64_t, PromptWeights*, Stream) { return false; }
std::size_t prompt_table_bytes(std::uint32_t, std::int64_t, std::int64_t) { return 0; }
bool prompt_prepare_at(std::uint32_t, const void*, std::int64_t, std::int64_t, void*, PromptWeights*, Stream) { return false; }
void prompt_release(PromptWeights* w) {
  if (w) *w = PromptWeights{};
}
const char* prompt_last_error() { return "built without the CUDA backend (configure with -DPELIAS_CUDA=ON)"; }
bool act_digits(const float*, const int*, std::int64_t, std::int64_t, float*, Stream) { return false; }
bool prompt_gemm(const PromptWeights&, std::int64_t, const std::int8_t*, const float*, const float*, std::int64_t, float*, Stream) { return false; }
bool moe_scratch_alloc(std::int64_t, int, int, std::int64_t, std::int64_t, MoeScratch*, bool) { return false; }
void moe_scratch_free(MoeScratch* m) {
  if (m) *m = MoeScratch{};
}
bool moe_prompt(const PromptWeights&, const PromptWeights&, const PromptWeights&, const std::int8_t*, const float*, const float*, std::int64_t,
                const std::int32_t*, const float*, MoeScratch&, MoeCombine, int, float*, Stream) {
  return false;
}
bool moe_prompt_multi(MoeStacks, MoeStacks, MoeStacks, const std::int8_t*, const float*, const float*, std::int64_t, const std::int32_t*, const float*,
                      MoeScratch&, MoeCombine, int, float*, Stream) {
  return false;
}

}  // namespace pelias::kern::gpu
