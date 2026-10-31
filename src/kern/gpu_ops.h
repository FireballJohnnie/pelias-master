// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kern/gpu_ops.h - GP3-10 GPU: the GPU kernels of one token's forward pass, as plain functions on device pointers (masterwork
// 14.2 kern/: gemm_gguf, route, elementwise, attn, deltanet). No CUDA type appears here, so host-only builds compile every
// caller; there the functions report "no GPU backend" (gpu_ops_none.cpp). Each operation does what the CPU reference
// (model/arch_qwen35moe.cpp) does, in the same float expressions where the order is the same; where a GPU must sum in
// another order (dot products, norms, softmax) the rung-1 test (tests/gpu/test_gpu_rung1.cpp) measures it against an fp64
// reference under masterwork 16.2's tolerances. Every function returns false with last_error() set on a bad argument or a
// CUDA error; nothing allocates except dalloc (rule 4: no allocation inside a hot loop).
// Layouts (row-major, the reference's): a matrix of n_out rows of k values stores row r at w + r * row_bytes in its GGUF
// type; activations are float rows of ld floats; the KV cache is position-major (position u at u * kv_w); the DeltaNet
// state of value head h is M[j][i] at state + h * ds * ds + j * ds + i (row j = value index), as the CPU's DecodeState.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace pelias::kern::gpu {

using Stream = void*;  // a cudaStream_t; nullptr = the default stream

bool compiled();             // built with the CUDA backend
bool available();            // compiled, and a CUDA device answers
const char* last_error();    // the reason of the last false return ("" if none)

// ---- memory (the only allocating calls)
void* dalloc(std::size_t bytes);
void dfree(void* p);
bool to_device(void* dst, const void* src, std::size_t bytes, Stream s = nullptr);
bool to_host(void* dst, const void* src, std::size_t bytes, Stream s = nullptr);
bool dzero(void* p, std::size_t bytes, Stream s = nullptr);
bool sync(Stream s = nullptr);
// free and total device memory now (cudaMemGetInfo); false when no device answers
bool mem_info(std::size_t* free_bytes, std::size_t* total_bytes);

// ---- products straight from GGUF storage (gemm_gguf)
// y[t * ldy + r] = sum_i W[r][i] * x[t * ldx + i] for r < n_out, t < n_tok; W in one of the types of gpu_decode.h
bool matmul(std::uint32_t type, const void* w, std::uint64_t row_bytes, std::int64_t n_out, std::int64_t k, const float* x,
            std::int64_t n_tok, std::int64_t ldx, float* y, std::int64_t ldy, Stream s = nullptr);
// grouped experts in one launch (P-K2b; O14: unequal token counts). Expert e's matrix starts at w + e * expert_bytes. Its
// slots are slot_off[e] .. slot_off[e + 1] - 1 (device ints, n_expert + 1 of them); slot i reads x row src[i] and writes y
// row dst[i]. Experts with no slot cost nothing; all slots on one expert is fine.
bool moe_matmul(std::uint32_t type, const void* w, std::uint64_t expert_bytes, std::uint64_t row_bytes, std::int64_t n_out,
                std::int64_t k, int n_expert, const int* slot_off, const int* src, const int* dst, const float* x, std::int64_t ldx,
                float* y, std::int64_t ldy, int max_slots_per_expert, Stream s = nullptr);
// rows of a matrix as floats (the token embedding: out row t = W[ids[t]]); ids on the device
bool gather_rows(std::uint32_t type, const void* w, std::uint64_t row_bytes, std::int64_t k, const std::int32_t* ids,
                 std::int64_t n, float* out, Stream s = nullptr);

// ---- activations to the engine's 8-bit format (gpu256x8: per 32 values amax / 127, round half to even, clamp to +-127),
// value for value model::act_round_q8_256x8. rot32: Walsh-Hadamard per 32 first (a weight stored rotated), norm =
// 1 / sqrt(32) as quant::hadamard32 computes it on the host. Writes the rounded floats to xr (may equal x) and, when q is
// not null, the int8 values, the scale and the integer sum per 32 (the R3 kernel's inputs).
bool act_q8x32(const float* x, std::int64_t n, std::int64_t rows, std::int64_t ldx, bool rot32, float* xr, std::int64_t ldr,
               std::int8_t* q, float* scale, int* sum, Stream s = nullptr);

// rms_norm (whole rows: one segment of n, weight w or none) and act_q8x32 in one kernel: the norm's output stays in
// registers and goes straight into the 8-bit rounding, so the normalized row is never written to memory and read back
// (2026-10-30, author 20:30 "fuse the norms"). The same expressions in the same order as rms_norm followed by act_q8x32:
// every output bit for bit equal to the two-kernel path. y (optional, may be null) receives the normalized floats before
// rounding; xr / q / scale / sum as act_q8x32 (xr may be null when only the int8 form is needed; q, scale, sum may be null).
bool rms_norm_q8x32(const float* x, const float* w, std::int64_t n, std::int64_t ldx, std::int64_t rows, float eps, bool rot32, float* y,
                    std::int64_t ldy, float* xr, std::int64_t ldr, std::int8_t* q, float* scale, int* sum, Stream s = nullptr);

// ---- elementwise
// rms norm of segments: segment g of row t is the n floats at x + t * ld + g * seg_stride (n_seg per row); y likewise
// (y may equal x); w (n floats) is applied to every segment, or none when w is null
bool rms_norm(const float* x, const float* w, float* y, std::int64_t n, std::int64_t n_seg, std::int64_t seg_stride, std::int64_t ld,
              std::int64_t rows, float eps, Stream s = nullptr);
// x / sqrt(sum x^2 + eps) per segment, in place, as model l2_norm (rms with eps / n, then 1 / sqrt(n))
bool l2_norm(float* x, std::int64_t n, std::int64_t n_seg, std::int64_t seg_stride, std::int64_t ld, std::int64_t rows, float eps,
             Stream s = nullptr);
// IMRoPE / NEOX / adjacent-pair rope over the first n_rot values of each head: head g of row t at v + t * ld + g * head_stride,
// row t at position pos0 + t. sections: the 4 rope sections (all 0: plain rope); freq_factors: n_rot / 2 device floats or null.
bool rope(float* v, std::int64_t n_heads, std::int64_t head_stride, std::int64_t ld, std::int64_t rows, std::int64_t pos0, int n_rot,
          float freq_base, const int sections[4], bool neox, const float* freq_factors, Stream s = nullptr);
// out[i] = silu(g[i]) * u[i]
bool silu_mul(const float* g, const float* u, float* out, std::int64_t n, Stream s = nullptr);
// y[t * ldy + i] += a[t] * x[t * ldx + i] (a null: weight 1); the weighted sums of the experts and the residual adds
bool add_rows(float* y, std::int64_t ldy, const float* x, std::int64_t ldx, const float* a, std::int64_t n, std::int64_t rows,
              Stream s = nullptr);

// ---- route: per row, softmax over n_expert logits (sum in double), the top n_used by probability (ties: lower index),
// renormalized to sum 1 (sum clamped below at 2^-14), times weights_scale when it is neither 0 nor 1; sel / wsel n_used per row
// tests only (v22): route router_topk through the earlier serial kernel, to check the shared-memory kernel bit for bit
void router_topk_serial_for_tests(bool on);
// tie_band (v24, the author's tie rule): > 0 orders the experts by floor((logit - max logit) / tie_band), lower index first
// within a band (see k_router_fast); 0 = by probability, as before.
bool router_topk(const float* logits, std::int64_t n_expert, std::int64_t rows, int n_used, float weights_scale, std::int32_t* sel,
                 float* wsel, Stream s = nullptr, float tie_band = 0.0f);

// ---- attention (full attention layers): causal, grouped heads, optional per-head output gate sigmoid(q-row second half).
// q: n_tok rows of q_stride floats, head h at h * q_head (q_head = head_dim, or 2 * head_dim with the gate after the query);
// K, V: the cache, positions 0 .. p0 + n_tok - 1, kv_w floats each (head g at g * head_dim); query t sits at p0 + t and
// attends to 0 .. p0 + t. out: n_tok rows of n_head * head_dim. scratch: n_tok * n_head * (p0 + n_tok) floats.
bool attention(const float* q, std::int64_t q_stride, std::int64_t q_head, const float* K, const float* V, std::int64_t kv_w,
               std::int64_t n_head, std::int64_t n_head_kv, std::int64_t head_dim, std::int64_t n_tok, std::int64_t p0, float scale,
               bool gate, float* out, float* scratch, Stream s = nullptr);

// ---- Gated DeltaNet (30 of the 40 layers)
// causal depthwise conv over [taps - 1 rows of tail, n rows of mixed] then SiLU; w: ch x taps (tap kk of channel c at c * taps
// + kk); tail null = zeros (a fresh sequence); new_tail (taps - 1 rows, may be null) receives the last taps - 1 inputs
bool deltanet_conv(const float* mixed, const float* tail, const float* w, std::int64_t n, std::int64_t ch, std::int64_t taps,
                   float* conv, float* new_tail, Stream s = nullptr);
// in place: beta = sigmoid(beta); alpha = softplus(alpha + dt[h]) * a[h]  (n rows of nv)
bool deltanet_gates(float* beta, float* alpha, const float* dt, const float* a, std::int64_t n, std::int64_t nv, Stream s = nullptr);
// the recurrence over n tokens, one value head per block (value head h reads key head h % nk); state carried in and out;
// q at conv row + hk * ds, k at + nk * ds + hk * ds, v at + 2 * nk * ds + h * ds (already L2-normed q, k); core: n rows of
// nv * ds, value head h at h * ds, each value times 1 / sqrt(ds)
bool deltanet_recurrent(const float* conv, std::int64_t ch, const float* beta, const float* alpha, float* state, std::int64_t n,
                        std::int64_t nk, std::int64_t nv, std::int64_t ds, float* core, Stream s = nullptr);
// The same recurrence, 16 tokens at a time (the chunked form for long prompts; same arguments, same state layout, so the two
// can be mixed call by call). Per value head and chunk, with l_t the cumulative log decay inside the chunk:
//   u_t = beta_t (v_t - e^l_t S0 k_t - sum_{s<t} e^(l_t - l_s) (k_t . k_s) u_s)      (forward substitution)
//   o_t = e^l_t S0 q_t + sum_{s<=t} e^(l_t - l_s) (q_t . k_s) u_s,   S_end = e^l_last S0 + sum_s e^(l_last - l_s) u_s k_s^T
// Not bit for bit equal to deltanet_recurrent (other summation order); checked against fp64 and against it (rung 1).
bool deltanet_chunked(const float* conv, std::int64_t ch, const float* beta, const float* alpha, float* state, std::int64_t n,
                      std::int64_t nk, std::int64_t nv, std::int64_t ds, float* core, Stream s = nullptr);
// Variants under measurement (game plan C2: the first chunked kernel ran at 0.94x of the per-token one; both used one block
// per value head, 32 blocks on a 46-SM card). Same arguments, results and state layout as above, so any mix works.
//   deltanet_recurrent_reg: per token, the state in registers: 4 threads per state row (32 values each), 32 rows per block,
//     so a head spans d_state / 32 blocks (128 blocks at 32 heads x 128); no shared memory for the state. d_state a power of
//     two from 32 to 1024.
//   deltanet_chunked_rows: chunked, each head split over d_state / 32 blocks of 32 state rows, the decays computed once per
//     chunk; chunk 16 or 32 tokens. d_state a multiple of 32, at most 256.
bool deltanet_recurrent_reg(const float* conv, std::int64_t ch, const float* beta, const float* alpha, float* state, std::int64_t n,
                            std::int64_t nk, std::int64_t nv, std::int64_t ds, float* core, Stream s = nullptr);
bool deltanet_chunked_rows(const float* conv, std::int64_t ch, const float* beta, const float* alpha, float* state, std::int64_t n,
                           std::int64_t nk, std::int64_t nv, std::int64_t ds, int chunk, float* core, Stream s = nullptr);
// GP3-14 (MTP rollback): deltanet_recurrent_reg, bit for bit, plus the state after each token t < n - 1 written to
// ckpt + t * nv * ds * ds (ckpt may be null when n == 1). Keeping a prefix of k tokens of the batch = copying ckpt[k - 1]
// back into the state (k = n: the state itself).
bool deltanet_recurrent_reg_ckpt(const float* conv, std::int64_t ch, const float* beta, const float* alpha, float* state, std::int64_t n,
                                 std::int64_t nk, std::int64_t nv, std::int64_t ds, float* core, float* ckpt, Stream s = nullptr);
// ---- the doorbell (GP3-14, game plan U2 / masterwork: CPU and GPU hand off through flags in pinned host memory mapped
// into the device; no cudaMemcpy wait, no host callback). host_alloc_mapped: pinned, mapped, zeroed host memory; *dev
// receives its device address. bell_ring: a one-thread kernel, in stream order, makes every earlier write visible system
// wide (__threadfence_system) and then stores value into the flag (a device address of mapped memory). bell_wait: a
// one-thread kernel spins (__nanosleep 64 -> 200 ns backoff) until the flag holds value or later (flags only grow,
// compared modulo 2^32), then fences, so the stream's next work reads what the CPU wrote before ringing.
// to_host_async: device -> pinned host memory, stream ordered, no wait.
// ts (optional, a device address, e.g. of mapped memory; v19 timeline): bell_ring writes the GPU's globaltimer (ns) at the
// ring to ts[0]; bell_wait writes it on entry to ts[0] and on release to ts[1]. Null: no timestamps (the default).
void* host_alloc_mapped(std::size_t bytes, void** dev);
void host_free(void* p);
bool bell_ring(unsigned* flag, unsigned value, Stream s = nullptr, unsigned long long* ts = nullptr);
bool bell_wait(const unsigned* flag, unsigned value, Stream s = nullptr, unsigned long long* ts = nullptr);
bool to_host_async(void* dst, const void* src, std::size_t bytes, Stream s = nullptr);
// v21 (per-operation timeline): a one-thread kernel writes the GPU's globaltimer (ns) to *p (a device address, e.g. of
// mapped memory) when the stream reaches it
bool stamp(unsigned long long* p, Stream s = nullptr);
// v23 pin probe: register existing host memory for GPU access (read-only first, then plain); *why gets the CUDA errors
// on failure. read_rate: one kernel reading `bytes` from a device address (blocks x 256 threads), wall seconds after sync.
bool host_register(void* p, std::size_t bytes, void** dev, std::string* why);
void host_unregister(void* p);
bool read_rate(const void* dev, std::size_t bytes, int blocks, double* seconds);
// v24: n blocks each read one chunk of `chunk` bytes at dev_base + offsets[b] (a routed expert's size, scattered)
bool read_chunks_rate(const void* dev_base, const std::uint64_t* offsets, int n, std::size_t chunk, double* seconds);
// v24 expert reader (1-4 tokens): per routed expert its three matrices (gate, up, down), each a device address (the VRAM
// bank, or the model file's mapping registered with host_register), its GGUF type and row bytes
struct ExpertPtr {
  const std::uint8_t* w[3] = {nullptr, nullptr, nullptr};
  std::uint32_t type[3] = {0, 0, 0};
  std::uint64_t rb[3] = {0, 0, 0};
};
// tab: n_expert entries (device); sel / wsel: tokens x n_used picks and weights (device, as router_topk); xr: tokens x H
// rounded inputs; gu: scratch tokens x n_used x 2F; h: scratch tokens x n_used x F; y: tokens x H, overwritten with the
// weighted sum of the picked experts (pick order)
bool moe_reader_v24(const ExpertPtr* tab, const std::int32_t* sel, const float* wsel, int n_used, const float* xr, std::int64_t tokens, std::int64_t H,
                    std::int64_t F, float* gu, float* h, float* y, Stream s = nullptr);
// v25 grouped reader (the engine's): the picks grouped by expert, each expert's rows read once (16-byte staged) for every
// token that picked it; bit for bit equal to moe_reader_v24. tokens 1-4. row_max: the longest row (bytes) in tab. yd:
// scratch tokens x n_used x H; grp: scratch 1 + 6 x tokens x n_used ints (device). ts (optional, device): 6 GPU stamps
// (start, after grouping, after gate/up, after silu, after down, after combine).
bool moe_reader(const ExpertPtr* tab, const std::int32_t* sel, const float* wsel, int n_used, const float* xr, std::int64_t tokens, std::int64_t H,
                std::int64_t F, std::uint64_t row_max, float* gu, float* h, float* yd, int* grp, float* y, unsigned long long* ts = nullptr,
                Stream s = nullptr);
// device to device copy (stream ordered)
bool dcopy(void* dst, const void* src, std::size_t bytes, Stream s = nullptr);
// per value head: o = rms_norm(o) * w * silu(z), in place on core (n rows of nv * ds), z alike
bool gated_norm(float* core, const float* z, const float* w, std::int64_t n, std::int64_t nv, std::int64_t ds, float eps,
                Stream s = nullptr);

}  // namespace pelias::kern::gpu
