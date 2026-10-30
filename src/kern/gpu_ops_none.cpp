// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kern/gpu_ops_none.cpp - the GPU kernels' interface in a build without the CUDA backend (PELIAS_CUDA off): every call
// reports that there is no GPU backend, so callers fall back to the CPU path.
#include "kern/gpu_ops.h"

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
bool matmul(std::uint32_t, const void*, std::uint64_t, std::int64_t, std::int64_t, const float*, std::int64_t, std::int64_t, float*, std::int64_t,
            Stream) {
  return false;
}
bool moe_matmul(std::uint32_t, const void*, std::uint64_t, std::uint64_t, std::int64_t, std::int64_t, int, const int*, const int*, const int*,
                const float*, std::int64_t, float*, std::int64_t, int, Stream) {
  return false;
}
bool gather_rows(std::uint32_t, const void*, std::uint64_t, std::int64_t, const std::int32_t*, std::int64_t, float*, Stream) { return false; }
bool act_q8x32(const float*, std::int64_t, std::int64_t, std::int64_t, bool, float*, std::int64_t, std::int8_t*, float*, int*, Stream) {
  return false;
}
bool rms_norm(const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, float, Stream) {
  return false;
}
bool l2_norm(float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, float, Stream) { return false; }
bool rope(float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, int, float, const int*, bool, const float*, Stream) {
  return false;
}
bool silu_mul(const float*, const float*, float*, std::int64_t, Stream) { return false; }
bool add_rows(float*, std::int64_t, const float*, std::int64_t, const float*, std::int64_t, std::int64_t, Stream) { return false; }
bool router_topk(const float*, std::int64_t, std::int64_t, int, float, std::int32_t*, float*, Stream) { return false; }
bool attention(const float*, std::int64_t, std::int64_t, const float*, const float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
               std::int64_t, std::int64_t, float, bool, float*, float*, Stream) {
  return false;
}
bool deltanet_conv(const float*, const float*, const float*, std::int64_t, std::int64_t, std::int64_t, float*, float*, Stream) { return false; }
bool deltanet_gates(float*, float*, const float*, const float*, std::int64_t, std::int64_t, Stream) { return false; }
bool deltanet_recurrent(const float*, std::int64_t, const float*, const float*, float*, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                        float*, Stream) {
  return false;
}
bool gated_norm(float*, const float*, const float*, std::int64_t, std::int64_t, std::int64_t, float, Stream) { return false; }

}  // namespace pelias::kern::gpu
