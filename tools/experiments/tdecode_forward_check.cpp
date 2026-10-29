// Run D follow-up (2026-10-28 23:0x): whole forward pass on a synthetic Q7_ZT model (rotated dense), float path vs
// T-decode 1-bit route vs T-decode with H folded into the integers. Build against build/host libs; arg: model.gguf.
#include <cmath>
#include <cstdio>
#include <vector>
#include "io/gguf_reader.h"
#include "model/arch_qwen35moe.h"
#include "model/spec.h"
#include "os/file_map.h"
using namespace pelias;
int main(int, char** argv) {
  os::FileMap m; if (!m.open(argv[1])) return 1;
  io::GgufReader g; if (!g.parse(m)) return 2;
  auto spec = model::read_qwen35moe_spec(g); if (!spec.ok()) return 3;
  auto w = model::bind_qwen35moe(g, m.data(), spec.value); if (!w.ok()) { std::printf("bind fail\n"); return 4; }
  const int n = 64; std::vector<std::int32_t> tok(n); for (int i = 0; i < n; ++i) tok[i] = (i * 37 + 11) % 500;
  const int64_t V = spec.value.n_vocab; std::vector<float> a(n * V), b(n * V), c(n * V);
  model::ForwardOptions fo; fo.act_int8 = {"all"}; fo.act_gpu256 = true; fo.act_int8_gpu = true; fo.act_gpu256_sub = 1; fo.threads = 4;
  if (!model::forward_reference(spec.value, w.value, tok.data(), n, 0, n, a.data(), fo)) return 5;
  fo.t_decode = true;
  if (!model::forward_reference(spec.value, w.value, tok.data(), n, 0, n, b.data(), fo)) return 6;
  fo.t_decode_fold = true;
  if (!model::forward_reference(spec.value, w.value, tok.data(), n, 0, n, c.data(), fo)) return 7;
  double bc = 0; for (size_t i = 0; i < b.size(); ++i) bc = std::max(bc, (double)std::fabs(b[i]-c[i]));
  std::printf("1-bit route vs integer fold (same order): max |logit diff| %.3e\n", bc);
  std::printf("float matmul (dequantized) vs 1-bit route:\n");
  double worst = 0, sc = 0; int at = -1;
  for (int t = 0; t < n; ++t) for (int64_t v = 0; v < V; ++v) { double d = std::fabs(a[t*V+v]-b[t*V+v]); if (d > worst) { worst = d; at = t; } sc = std::max(sc, (double)std::fabs(a[t*V+v])); }
  std::printf("max |logit diff| %.3e (token %d) of max |logit| %.3f\n", worst, at, sc);
  for (int t : {0, 1, 2, 8, 32, 63}) { double d = 0; for (int64_t v = 0; v < V; ++v) d = std::max(d, (double)std::fabs(a[t*V+v]-b[t*V+v])); std::printf("  token %2d: %.3e\n", t, d); }
}
