// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// main.cc - the `pelias` command. Step 2 carries: version, inspect (every tensor of a GGUF with its type, shape and
// bytes), key (SHA-256 model key), plan (the device memory plan against the ceiling, refused above it), device (what
// the GPU reports). Step 3 adds quantize (a GGUF to a stock mix, byte-identical to the reference), convert (a model
// folder of safetensors shards to a GGUF, BF16 or a stock mix, Stage A) and verify (two GGUFs compared tensor by tensor
// and byte for byte, gates P-E6 and P-X6). Step 4 adds tokenize (text to token ids with the model's own tokenizer,
// gate P-D1). Milestone 1 adds kld (the CPU reference forward pass scored against a KL-divergence base file, the same
// statistics as the reference's perplexity tool). The later verbs are listed as not yet built so the user gets a clear
// answer instead of a silent failure.
//
// Exit codes: 0 ok; 1 usage; 2 the input could not be read (missing, corrupt, unsupported); 3 a gate refused (plan
// over the ceiling); 4 no device; 5 verify found differences; 6 quantize or convert failed while writing (disk, I/O).
#include <cerrno>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#if defined(__linux__)
#include <sys/resource.h>
#endif
#include <vector>

#include "hal/device.h"
#include "hal/host_alloc_pages.h"
#include "hal/vram_governor.h"
#include "ingest/stage_a.h"
#include "io/gguf_compare.h"
#include "io/gguf_reader.h"
#include "io/number_text.h"
#include "io/planes_build.h"
#include "io/requantize.h"
#include "io/safetensors.h"
#include "io/sha256.h"
#include "io/tier_build.h"
#include "os/clock.h"
#include "os/file_map.h"
#include "os/membw.h"
#include "pelias/log.h"
#include "pelias/timer.h"
#include "pelias/version.h"
#include "plan/memory_plan.h"
#include "plan/probe_constants.h"
#include "quant/block_types.h"
#include "quant/tdecode.h"
#include "quant/type_rules.h"
#include "io/text_file.h"
#include "kern/cpu_dot.h"
#include "kern/gpu_ops.h"
#include "quant/planes.h"
#include "quant/quant_cpu.h"
#include "model/arch_qwen35moe.h"
#include "model/gpu_forward.h"
#include "model/gpu_engine.h"
#include "model/sampler.h"
#include "model/spec.h"
#include "tok/tokenizer.h"
#include "verify/kl.h"

namespace {

using namespace pelias;

struct Args {
  std::vector<std::string> pos;
  std::vector<std::pair<std::string, std::string>> opts;
  [[nodiscard]] bool has(const std::string& k) const {
    for (const auto& o : opts)
      if (o.first == k) return true;
    return false;
  }
  [[nodiscard]] std::string get(const std::string& k, const std::string& d = "") const {
    for (const auto& o : opts)
      if (o.first == k) return o.second;
    return d;
  }
};

// Sizes typed in GiB become bytes; below 2^34 GiB the byte count fits 64 bits (2^34 x 2^30 = 2^64), so that is the
// limit (DERIVED, not a policy).
constexpr double kGiBTextLimit = 17179869184.0;

// Reads --k as GiB greater than zero. Absent: true, `bytes` unchanged. Bad text: prints why, returns false (exit 1).
bool gib_option(const Args& a, const char* k, std::uint64_t& bytes) {
  if (!a.has(k)) return true;
  const Result<double> r = io::parse_positive_decimal(a.get(k), kGiBTextLimit);
  if (!r.ok()) {
    std::fprintf(stderr, "pelias: --%s %s\n", k, r.status.message.c_str());
    return false;
  }
  bytes = static_cast<std::uint64_t>(r.value * static_cast<double>(plan::kGiB));
  return true;
}

// Reads --index as a whole number (default 0). Bad text: prints why, returns false (exit 1).
bool index_option(const Args& a, int& index) {
  index = 0;
  if (!a.has("index")) return true;
  const Result<std::uint64_t> r = io::parse_whole(a.get("index"), static_cast<std::uint64_t>(std::numeric_limits<int>::max()));
  if (!r.ok()) {
    std::fprintf(stderr, "pelias: --index %s\n", r.status.message.c_str());
    return false;
  }
  index = static_cast<int>(r.value);
  return true;
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string t = argv[i];
    if (t.rfind("--", 0) == 0) {
      const std::size_t eq = t.find('=');
      if (eq != std::string::npos) {
        a.opts.emplace_back(t.substr(2, eq - 2), t.substr(eq + 1));
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        a.opts.emplace_back(t.substr(2), argv[++i]);
      } else {
        a.opts.emplace_back(t.substr(2), "1");
      }
    } else {
      a.pos.push_back(t);
    }
  }
  return a;
}

void usage() {
  std::puts(
      "pelias <command> [options]\n"
      "  version                         build version and backends\n"
      "  device [--index N]              what the GPU reports (CUDA and NVML), and the VRAM ceiling\n"
      "  gpu-check --model FILE [--tokens 64] [--layers N] [--repeat 3] [--threads N] [--fast-cpu] [--no-cpu] [--floor] [--float-ref]\n"
      "  engine-check --model FILE [--routes ROUTES.bin] [--prefill 64] [--decode 32] [--ctx 4096] [--batch 64] [--budget-gb G] [--float-ref]\n"

      "                                  GP3-12: the trunk on the GPU (prefill, fresh sequence) against the CPU reference with the\n"
      "                                  same per-32 int8 rounding: KL, top-1, logits error; GPU time per call and tokens/s\n"
      "  mtp-logits --model FILE --ids 1,2,3 --out OUT.bin [--threads N] [--fast-cpu]\n"
      "                                  GP3-08 gate: the MTP block's logits and hidden for each position (format of tools/oracle/mtp_oracle)\n"
      "  logits --model FILE --ids 1,2,3 --out OUT.bin [--threads N] [--fast-cpu] [--decode]   every position's logits (oracle checks)\n"
      "  bench-dot [--type Q4_TG2] [--mb 1024] [--k 2048] [--threads N] [--reps 5] [--planes K] [--int8] [--tokens N [--groups]] [--reference] [--portable]\n"
      "                                  GP3-10 CPU kernels: GB/s of one type's rows (+ dense planes), decode (one activation row)\n"
      "  membw [--mib N] [--threads 1,2,4,6,8,12,16] [--reps N]\n"
      "                                  probe: RAM read speed by thread count, 4 KB vs 2 MB pages, prefetch on / off,\n"
      "                                  and how a 1 ms foreground job (the user's own work) wakes and runs meanwhile\n"
      "  inspect --model FILE [--json] [--header-only]\n"
      "                                  every tensor of a GGUF or safetensors file: name, type, shape, bytes\n"
      "                                  (--header-only: read a file whose data is cut off, e.g. a header sample)\n"
      "  key --model FILE                SHA-256 of the file (the model key)\n"
      "  plan --model FILE [--bank-gib N] [--ceiling-gib C] [--working-set-gib W] [--index N] [--header-only]\n"
      "                                  the device memory plan; refused above the ceiling (exit 3)\n"
      "  quantize --model IN.gguf --out OUT.gguf [--type Q4_K_M|Q6_K|Q8_0|BF16|F16|F32] [--threads N] [--rotate32 GROUPS]\n"
      "           [--recipe RECIPE.tsv] [--imatrix IMATRIX.gguf]\n"
      "                                  --recipe: exact type per tensor (tools/recipe.py), every tensor named; --imatrix:\n"
      "                                  importance weights as llama-quantize --imatrix (rotated weights need a matrix\n"
      "                                  collected rotated)\n"
      "                                  --rotate32: store the product weights whose names contain one of GROUPS (or 'all')\n"
      "                                  rotated per 32 values (Walsh-Hadamard); the reader rotates their inputs to match\n"
      "                                  a new GGUF in a stock mix (default Q4_K_M), identical to the reference's (exit 6 on a write failure)\n"
      "  quantize --tier ALLOC.tsv --model BASE.gguf --original ORIG.gguf --out OUT.gguf [--imatrix IMATRIX.gguf] [--threads N]\n"
      "                                  usage-tiered experts (docs/PELIAS_RESIDUAL_PLANES.md 6.7; tools/tier_alloc.py): BASE's\n"
      "                                  tensors copied unchanged, except each routed-expert weight ALLOC names, quantized again\n"
      "                                  expert by expert from ORIG (F32 / F16 / BF16) to the type ALLOC gives that expert. Only\n"
      "                                  Pelias reads the result (pelias.tier.*)\n"
      "  convert --model FOLDER --out OUT.gguf [--type BF16|Q4_K_M|Q6_K|Q8_0] [--threads N]\n"
      "                                  a Hugging Face folder (safetensors) to a GGUF: BF16 as the reference converter writes it,\n"
      "                                  or a stock mix as the reference quantizer writes it from that BF16 file (exit 6 on a write failure)\n"
      "  verify --model A.gguf --reference B.gguf\n"
      "                                  every tensor compared by name, then the files byte for byte (exit 5 if anything differs)\n"
      "  tokenize --model FILE (--text S | --file F) [--no-nfc] [--no-special] [--pieces] [--header-only]\n"
      "                                  token ids as a list [a, b, ...] (--pieces: one 'id -> piece' per line);\n"
      "                                  --no-nfc skips NFC (as llama.cpp does), --no-special leaves control tokens as text\n"
      "  kld --model FILE --base BASE.bin [--chunks N] [--threads N] [--expert-stats OUT.csv]\n"
      "                                  KL divergence of the model against a base file written by llama-perplexity\n"
      "                                  --kl-divergence-base, computed by Pelias's fp32 CPU reference forward pass;\n"
      "                                  --expert-stats: per layer and expert, how often it was picked and its routing weight\n"
      "      [--dump-kld OUT.bin] [--act-int8 GROUPS] [--act-int8-path cpu|gpu|gpu256|gpu256x8|gpu256e] [--act-rot h32] [--route-dump OUT.bin] [--dump-codes OUT.bin] [--dump-qk OUT.bin] [--dump-flips OUT.bin]\n"
      "      [--kv-store q8|int4|int4r] [--state-store q8|int4|int4r|int8row] [--state-at P]\n"
      "                                  --dump-codes: per scored token and layer, the state code of the layer's normed input\n"
      "                                  (C6 miss map: sign bits after a fixed sign diagonal and Hadamard per 32, plus one\n"
      "                                  log-energy byte per 32-block; quant::state_code)\n"
      "                                  --dump-qk: every position's query and key heads (normed, roped; fp16) in every\n"
      "                                  full-attention layer, for the C1 KV-page check; use a few --chunks (about 47 MB each)\n"
      "                                  --kv-store / --state-store: store the attention keys and values, or a DeltaNet checkpoint\n"
      "                                  taken at position P (default: the first scored position) and restored, in a lossy form\n"
      "                                  (8-bit or 4-bit per 32; int4r rotated per 32; int8row one scale per 128) and read back\n"
      "                                  --route-dump: per scored token and layer, the experts picked, their weights, the\n"
      "                                  margin to the first unpicked one, and the next layer's pre-gating prediction\n"
      "                                  --dump-kld: every scored token's KL, NLLs, delta-p and top-token match (for kld-compare);\n"
      "                                  --act-int8: round the inputs of quantized weights whose names contain one of the\n"
      "                                  comma-separated GROUPS (or 'all') to 8 bits first: cpu (default) = llama.cpp's CPU\n"
      "                                  format for each weight type (Q8_K / Q8_0 / Q8_1), gpu = q8_1\n"
      "                                  --act-rot h32: rotate each 32-value block (Walsh-Hadamard) before rounding, back after\n"      "      [--h-gate live:F]   (Q8_Z weights: the 8th bit read on the fraction F of 32-blocks with the largest expected loss per token)\n"
      "      [--t-decode]        (Q7_ZT weights with --act-int8-path gpu256x8: the 1-bit route, 8th bit by AND + popcount)\n"
      "      [--mtp-draft N]     (generate, greedy: the MTP block drafts N tokens per step, the trunk keeps those it agrees with)\n"
      "      [--fast-cpu]        (GP3-10 kernels: every row read straight from its storage, AVX2 when the CPU has it; same\n"
      "                           products as the reference, another order of the sum)\n"
"      [--act-bits 4] [--neuron-skip F] [--defer-experts D] [--swap-near-tie M --hot-set FILE] [--act-cold-path gpu256 --hot-set FILE]   (game plan Block M)\n"
      "                                  --act-bits 4: 4-bit activations per 32 (gpu path); --neuron-skip F: zero the smallest\n"
      "                                  F of each routed expert's hidden values (oracle); --defer-experts D: the D lowest\n"
      "                                  experts added one layer later; --swap-near-tie M: use the first unpicked expert when\n"
      "                                  the last pick is not in the hot set, it is, and their probabilities differ by < M\n"
      "      [--planes FILE] [--plane-k K] [--plane-picks N] [--head-rescore N]   (residual planes; see planes and generate)\n"
      "                                  --plane-picks N: the routed experts' planes only for each token's N highest-weighted picks\n"
      "  kld-compare --a A.bin --b B.bin [--name-a S] [--name-b S]\n"
      "                                  paired comparison of two --dump-kld files scored on the same base: mean difference,\n"
      "                                  chunk-clustered standard error, 95 % interval, top-token disagreements, verdict\n"
      "  imatrix --model FILE --file TEXT --out OUT.gguf [--ctx 512] [--chunks N] [--threads N] [--no-nfc]\n"
      "                                  importance matrix from a calibration text, by the fp32 reference pass, in the\n"
      "                                  reference's GGUF imatrix format (llama-quantize --imatrix reads it); weights the\n"
      "                                  file stores rotated (quantize --rotate32) are collected on the rotated inputs\n"
      "  generate --model FILE (--prompt TEXT | --prompt-file F | --prompt-list F --out-dir DIR) [--n-predict 128] [--ctx 32768] [--threads N]\n"
      "      [--temp 1.0] [--top-k 20] [--top-p 0.95] [--min-p 0] [--repeat-penalty 1] [--repeat-last-n 64]\n"
      "      [--presence-penalty 0] [--frequency-penalty 0] [--seed 20261019 | -1] [--ignore-eos] [--show-special]\n"
      "      [--ids-out FILE] [--no-nfc] [--no-special] [--act-int8 GROUPS] [--act-int8-path ...] [--act-rot h32] [--no-mmap] [--chat]\n"
      "      [--engine [--routes ROUTES.bin] [--budget-gb G]]\n"
      "                                  --engine: the GPU engine (GP3-14: resident dense weights, the hot-expert bank chosen by\n"
      "                                  the routing record's pick counts, cold experts on the CPU)\n"
      "                                  --chat wraps the prompt as one user turn in the model family's chat format\n"
      "                                  text from the fp32 reference pass on the CPU (slow; a correctness reference, not the\n"
      "                                  engine): the prompt once, then one token per step on the carried KV cache and\n"
      "                                  DeltaNet state. --temp 0 is greedy; the draw is Philox4x32-10 keyed by --seed.\n"
      "                                  Defaults are the parameter dictionary's. --ids-out writes the prompt and output ids\n"
      "                                  and each step's top-1 minus top-2 raw logit (tools/greedy_compare.py reads it)\n"      "                                  --no-mmap reads the whole file into RAM first (for mounts that do not cache it)\n"
      "      [--planes FILE] [--plane-k K] [--planes-mode all|prefill|start:N] [--head-rescore N]\n"
      "                                  --prompt-list: every line a prompt, all run on the model loaded once; line NN's text and\n"
      "                                  ids go to DIR/pelias_NN.txt and DIR/pelias_NN.ids (nothing is overwritten)\n"
      "                                  residual sign planes (`pelias planes`): read on every token (all), on the prompt only\n"
      "                                  (prefill), or on the prompt and the first N output tokens (start:N); --plane-k reads\n"
      "                                  the first K stored planes; --head-rescore N recomputes the N largest logits exactly from\n"
      "                                  the original output rows (a plane file built with --exact-head)\n"
      "  planes --model FILE --base ORIGINAL.gguf --out PLANES.gguf [--k 2] [--refit 8] [--imatrix IMATRIX.gguf] [--exact-head]\n"
      "         [--threads N] [--keep-mtp] [--k-rules PART=K,...] [--select PART --budget-mb N [--block-weights FILE]]\n"
      "                                  residual sign planes of any model file against its original weights (F32/F16/BF16, same\n"
      "                                  tensor names): per 256 values, K fp16 scales and K sign bits per value, fitted plane by\n"
      "                                  plane then together (--refit rounds); weighted per column by an importance matrix when\n"
      "                                  given. --exact-head also stores the original output rows (BF16) for --head-rescore.\n"
      "                                  The output must not exist; the model file's types are recorded so it is only read\n"
      "                                  with that file. A usage-tiered model file (quantize --tier) is fitted expert by expert.\n"
      "                                  --k-rules: the first PART found in a tensor's name sets its plane count (0 = none; others\n"
      "                                  --k). --select PART --budget-mb N: the tensors whose names contain PART keep planes only on\n"
      "                                  the 256-blocks that remove the most relative error, N MB in all (--block-weights: name<TAB>\n"
      "                                  factor per line, weighting a tensor's blocks)\n"
      "  teacher --model ORIGINAL.gguf --ids A.ids [--ids B.ids ...] --out-dir DIR [--threads N]\n"
      "                                  the original model's view of decoded text (generate --ids-out files): for every output\n"
      "                                  position its own top-1 and top-2 and the held token's logit and log-probability, to\n"
      "                                  DIR/teacher_<name>.tsv; the model is opened once for every file\n"
      "  probe | train | export | serve | bench | web   not built yet (later steps)\n"
      "options: --log debug|info|warn|error (default info)");
}

int fail_input(const Status& s) {
  std::fprintf(stderr, "pelias: %s\n", s.text().c_str());
  return s.code == Code::not_found || s.code == Code::io_error ? 2 : (s.code == Code::corrupt || s.code == Code::unsupported ? 2 : 1);
}

bool looks_like_safetensors(const os::FileMap& m) {
  return m.size() >= 9 && std::memcmp(m.data(), "GGUF", 4) != 0 && m.data()[8] == '{';
}

std::string shape_text(const std::uint64_t* ne, std::uint32_t n) {
  std::string s = "[";
  for (std::uint32_t i = 0; i < n; ++i) s += (i ? "," : "") + std::to_string(ne[i]);
  return s + "]";
}

int cmd_version() {
  std::printf("pelias %s (%s, %s) backends: %s\n", kVersion, kGitCommit, kBuildType, kHasCuda ? "cuda" : "none (host-only build)");
  return 0;
}

int cmd_device(const Args& a) {
  int index = 0;
  if (!index_option(a, index)) return 1;
  const hal::DeviceInfo d = hal::query_device(index);
  std::printf("platform: %s\n", os::platform_name().c_str());
  if (!d.available) {
    std::printf("device: none (%s)\n", d.note.c_str());
  } else {
    std::printf("device %d: %s, compute capability %d.%d, %d SMs\n", d.index, d.name.c_str(), d.cc_major, d.cc_minor, d.sm_count);
    std::printf("VRAM total %s; free after the context (CUDA) %s\n", plan::format_bytes(d.vram_total).c_str(), plan::format_bytes(d.vram_free_cuda).c_str());
  }
  if (d.nvml_available)
    std::printf("NVML: free %s, used %s, driver %s\n", plan::format_bytes(d.vram_free_nvml).c_str(), plan::format_bytes(d.vram_used_nvml).c_str(), d.driver_version.c_str());
  else
    std::printf("NVML: unavailable (%s)\n", d.nvml_note.c_str());
  const hal::Ceiling c = hal::ceiling_from_device(d);
  if (c.bytes) std::printf("VRAM ceiling: %s (basis %s %s minus margin %s; standing rule 3)\n", plan::format_bytes(c.bytes).c_str(), c.basis, plan::format_bytes(c.basis_free).c_str(), plan::format_bytes(c.margin).c_str());
  return d.available ? 0 : 4;
}

// pelias bench-dot: the GP3-10 CPU gate's throughput half. Rows of one type (and optional dense planes) filling --mb
// megabytes, one activation row (decode), every thread on its own share of rows; prints GB/s of weight (+ plane) bytes read.
// With --reference also the reference path (dequantize_row, then a plain sum) on the same rows, for the speedup.
// --tokens N: N activation rows per weight row (a batch: prefill, the MTP verify step), through dot_row_n / dot_row_q8_n
// (each row decoded once); with --groups the rows go through dot_row / dot_row_q8 four tokens at a time instead.
int cmd_bench_dot(const Args& a) {
  std::string tn = a.has("type") ? a.get("type") : "Q4_TG2";
  for (char& ch : tn) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  int id = -1;
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::string n = quant::type_info(i).name ? quant::type_info(i).name : "";
    for (char& ch : n) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if (!n.empty() && n == tn) id = static_cast<int>(i);
  }
  if (id < 0 || !kern::dot_supported(static_cast<quant::Type>(id))) {
    std::fprintf(stderr, "pelias bench-dot: unknown or unsupported --type %s\n", tn.c_str());
    return 1;
  }
  const quant::Type t = static_cast<quant::Type>(id);
  auto num = [&](const char* key, std::uint64_t def, std::uint64_t lo, std::uint64_t hi, std::uint64_t* out) {
    *out = def;
    if (!a.has(key)) return true;
    const Result<std::uint64_t> r = io::parse_whole(a.get(key), hi);
    if (!r.ok() || r.value < lo) {
      std::fprintf(stderr, "pelias bench-dot: --%s needs a whole number from %llu to %llu\n", key, static_cast<unsigned long long>(lo),
                   static_cast<unsigned long long>(hi));
      return false;
    }
    *out = r.value;
    return true;
  };
  std::uint64_t mb, k, threads, reps, kp;
  const std::uint64_t hw = std::max(1u, std::thread::hardware_concurrency());
  if (!num("mb", 1024, 16, 65536, &mb) || !num("k", 2048, 256, 65536, &k) || !num("threads", hw, 1, 1024, &threads) ||
      !num("reps", 5, 1, 1000, &reps) || !num("planes", 0, 0, quant::kPlaneMaxK, &kp))
    return 1;
  std::uint64_t ntok;
  if (!num("tokens", 1, 1, 512, &ntok))
    return 1;
  if (k % 256 != 0) {
    std::fprintf(stderr, "pelias bench-dot: --k must be a multiple of 256\n");
    return 1;
  }
  if (a.has("portable")) kern::force_isa(kern::Isa::portable);
  const std::uint64_t rb = quant::row_bytes(static_cast<std::uint32_t>(id), k);
  const std::uint64_t pb = kp ? static_cast<std::uint64_t>(quant::plane_row_bytes(static_cast<std::int64_t>(k), static_cast<int>(kp))) : 0;
  const std::uint64_t rows = std::max<std::uint64_t>(1, (mb * 1000000ull) / (rb + pb));
  std::printf("bench-dot: %s, k %llu, %llu rows (%llu bytes a row + %llu of planes, %d plane(s)), %.1f MB, %llu threads, path %s%s\n", tn.c_str(),
              static_cast<unsigned long long>(k), static_cast<unsigned long long>(rows), static_cast<unsigned long long>(rb),
              static_cast<unsigned long long>(pb), static_cast<int>(kp), static_cast<double>(rows * (rb + pb)) / 1e6,
              static_cast<unsigned long long>(threads), kern::active_isa() == kern::Isa::avx2 ? "AVX2" : "portable", a.has("int8") ? ", int8 activations" : "");
  if (ntok > 1) std::printf("  %llu tokens per row, %s\n", static_cast<unsigned long long>(ntok), a.has("groups") ? "groups of four" : "blocks of 16 rows, each decoded once");
  // 64 distinct rows, repeated to fill the buffer (the bytes read are what is timed, not their variety)
  const std::uint64_t distinct = std::min<std::uint64_t>(64, rows);
  std::vector<std::uint8_t> w(rows * rb), pl(rows * pb);
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 0.02f);
  std::vector<float> src(k), res(k);
  for (std::uint64_t r = 0; r < distinct; ++r) {
    if (quant::can_quantize(t)) {
      for (float& v : src) v = nd(rng);
      (void)quant::quantize_row(t, src.data(), w.data() + r * rb, static_cast<std::int64_t>(k));
    } else {  // a type Pelias reads but does not write: random bytes with no Inf / NaN scale (bit 6 of each byte cleared)
      for (std::uint64_t i = 0; i < rb; ++i) w[r * rb + i] = static_cast<std::uint8_t>(rng() & 0xBF);
    }
    if (kp) {
      for (float& v : res) v = 0.1f * nd(rng);
      (void)quant::plane_fit_row(res.data(), nullptr, static_cast<std::int64_t>(k), static_cast<int>(kp), 2, pl.data() + r * pb);
    }
  }
  for (std::uint64_t r = distinct; r < rows; ++r) {
    std::memcpy(w.data() + r * rb, w.data() + (r % distinct) * rb, rb);
    if (kp) std::memcpy(pl.data() + r * pb, pl.data() + (r % distinct) * pb, pb);
  }
  std::vector<float> x(k * ntok), y(rows * ntok);
  for (float& v : x) v = nd(rng) * 50.0f;
  std::vector<const float*> xs(ntok);
  std::vector<kern::ActQ8x32> acts(ntok);  // --int8: the engine's int8 activation format, integer products per 32-block
  std::vector<const kern::ActQ8x32*> as(ntok);
  for (std::uint64_t i = 0; i < ntok; ++i) {
    xs[i] = x.data() + i * k;
    kern::act_q8x32_quantize(xs[i], static_cast<std::int64_t>(k), acts[i]);
    as[i] = &acts[i];
  }
  const bool int8 = a.has("int8"), groups = a.has("groups");
  auto pass = [&](bool reference) {
    std::vector<std::thread> pool;
    const Timer tm;
    for (std::uint64_t ti = 0; ti < threads; ++ti)
      pool.emplace_back([&, ti]() {
        const std::uint64_t r0 = rows * ti / threads, r1 = rows * (ti + 1) / threads;
        std::vector<float> row(k);
        if (!reference && ntok > 1 && !groups) {  // blocks of 16 rows, as the forward pass runs a batch (fast_rows)
          for (std::uint64_t b0 = r0; b0 < r1; b0 += 16) {
            const int nr = static_cast<int>(std::min<std::uint64_t>(16, r1 - b0));
            const std::uint8_t* rws[16];
            kern::RowPlanes rps[16];
            const kern::RowPlanes* pps[16];
            for (int r = 0; r < nr; ++r) {
              rws[r] = w.data() + (b0 + static_cast<std::uint64_t>(r)) * rb;
              rps[r] = {kp ? pl.data() + (b0 + static_cast<std::uint64_t>(r)) * pb : nullptr, nullptr, static_cast<int>(kp), static_cast<int>(kp)};
              pps[r] = &rps[r];
            }
            float* yo = &y[b0 * ntok];  // the block's nr x ntok outputs, token-major
            if (int8) kern::dot_rows_q8_n(t, rws, kp ? pps : nullptr, nr, static_cast<std::int64_t>(k), acts.data(), static_cast<std::int64_t>(ntok), yo, nr);
            else kern::dot_rows_n(t, rws, kp ? pps : nullptr, nr, static_cast<std::int64_t>(k), xs.data(), static_cast<std::int64_t>(ntok), yo, nr);
          }
          return;
        }
        for (std::uint64_t r = r0; r < r1; ++r) {
          if (reference) {
            (void)quant::dequantize_row(t, w.data() + r * rb, row.data(), static_cast<std::int64_t>(k));
            if (kp) quant::plane_add_row(pl.data() + r * pb, static_cast<std::int64_t>(k), static_cast<int>(kp), static_cast<int>(kp), row.data());
            for (std::uint64_t tt = 0; tt < ntok; ++tt) {
              float acc = 0.0f;
              for (std::uint64_t i = 0; i < k; ++i) acc += row[i] * xs[tt][i];
              y[r * ntok + tt] = acc;
            }
          } else {
            kern::RowPlanes rp{kp ? pl.data() + r * pb : nullptr, nullptr, static_cast<int>(kp), static_cast<int>(kp)};
            const std::int64_t kk = static_cast<std::int64_t>(k);
            float* yo = &y[r * ntok];
            for (std::uint64_t t0 = 0; t0 < ntok; t0 += 4) {
              const int nt = static_cast<int>(std::min<std::uint64_t>(4, ntok - t0));
              if (int8) kern::dot_row_q8(t, w.data() + r * rb, kk, kp ? &rp : nullptr, as.data() + t0, nt, yo + t0);
              else kern::dot_row(t, w.data() + r * rb, kk, kp ? &rp : nullptr, xs.data() + t0, nt, yo + t0);
            }
          }
        }
      });
    for (std::thread& th : pool) th.join();
    return tm.seconds();
  };
  const double bytes = static_cast<double>(rows * (rb + pb));
  auto report = [&](const char* name, bool reference) {
    (void)pass(reference);  // warm-up
    std::vector<double> ts;
    const bool stamps = a.has("stamps");  // v45b (snoop probe): each rep's end time (unix ms) and rate, to slice by another run's window
    for (std::uint64_t i = 0; i < reps; ++i) {
      ts.push_back(pass(reference));
      if (stamps)
        std::printf("  rep %llu end_ms %lld %.3f GB/s\n", static_cast<unsigned long long>(i),
                    static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()),
                    bytes / ts.back() / 1e9);
    }
    std::sort(ts.begin(), ts.end());
    const double best = ts.front(), med = ts[ts.size() / 2];
    std::printf("  %-9s best %.2f GB/s (%.1f ms), median %.2f GB/s; %.2f G weights/s; %.2f G multiply-adds/s\n", name, bytes / best / 1e9,
                best * 1e3, bytes / med / 1e9, static_cast<double>(rows * k) / best / 1e9, static_cast<double>(rows * k * ntok) / best / 1e9);
    return best;
  };
  const double tf = report("kernels", false);
  if (a.has("reference")) {
    const double tr = report("reference", true);
    std::printf("  speedup %.2fx\n", tr / tf);
  }
  double chk = 0.0;
  for (const float v : y) chk += static_cast<double>(v);
  std::printf("  checksum %.6g\n", chk);
  return 0;
}

// v45b (author 17:19-17:21): does the host back WSL's memory with 2 MB pages? A random pointer chase over 65,536 cache
// lines (4 MB of data, so the data sits in L3 in every case) in three layouts: A dense (4 MB span, 2 MB pages advised),
// B spread (one line per 4 KB page over 256 MB, 2 MB pages advised), C spread with 4 KB pages (MADV_NOHUGEPAGE). B
// needs 128 TLB entries if the 2 MB pages hold through the host, 65,536 if the host backs them with 4 KB pages. So B ~ A:
// the host keeps 2 MB; B ~ C: the host backs with 4 KB pages (each spread access pays a nested page walk).
int cmd_page_probe(const Args& a) {
  (void)a;
  constexpr std::size_t kLines = 65536, kSpan = 256ull << 20, kAlign = 2ull << 20;
  auto anon_huge = []() {
    std::ifstream f("/proc/self/smaps_rollup");
    std::string l;
    while (std::getline(f, l))
      if (l.rfind("AnonHugePages:", 0) == 0) return l;
    return std::string("AnonHugePages: (not found)");
  };
  auto chase = [&](std::uint8_t* buf, std::size_t stride, const char* name) {
    std::vector<std::size_t> off(kLines);
    for (std::size_t i = 0; i < kLines; ++i) off[i] = i * stride + (stride > 64 ? (i % (stride / 64)) * 64 : 0);  // spread lines over the cache sets
    std::vector<std::size_t> ord(kLines);
    for (std::size_t i = 0; i < kLines; ++i) ord[i] = i;
    std::mt19937_64 rng(45);
    for (std::size_t i = kLines - 1; i > 0; --i) std::swap(ord[i], ord[static_cast<std::size_t>(rng() % i)]);  // Sattolo: one cycle
    for (std::size_t i = 0; i < kLines; ++i) {
      const std::uint64_t nx = off[ord[(i + 1) % kLines]];
      std::memcpy(buf + off[ord[i]], &nx, sizeof nx);
    }
    std::uint64_t q = off[ord[0]];
    for (std::size_t i = 0; i < 4 * kLines; ++i) std::memcpy(&q, buf + q, sizeof q);  // warm
    constexpr std::size_t kSteps = 20000000;
    const Timer tm;
    for (std::size_t i = 0; i < kSteps; ++i) std::memcpy(&q, buf + q, sizeof q);
    const double sec = tm.seconds();
    std::printf("  %-38s %.2f ns per access (end %llu)\n", name, sec * 1e9 / static_cast<double>(kSteps), static_cast<unsigned long long>(q % 7));
  };
  std::printf("page-probe: 65,536 lines in a random cycle, 20 M dependent loads per layout; %s at start\n", anon_huge().c_str());
  auto* h = static_cast<std::uint8_t*>(std::aligned_alloc(kAlign, kSpan));
  auto* s4 = static_cast<std::uint8_t*>(std::aligned_alloc(kAlign, kSpan));
  if (!h || !s4) {
    std::fprintf(stderr, "pelias page-probe: allocation failed\n");
    return 1;
  }
  const bool ah = hal::advise_huge_pages(h, kSpan), as = hal::advise_small_pages(s4, kSpan);
  std::memset(h, 1, kSpan);
  std::memset(s4, 1, kSpan);
  std::printf("  advice: 2 MB %s, 4 KB %s; after touching 512 MiB: %s\n", ah ? "taken" : "refused", as ? "taken" : "refused", anon_huge().c_str());
  chase(h, 64, "A0 dense (warm-up, clocks)");
  chase(h, 64, "A dense, 4 MB span, 2 MB pages");
  chase(h, 4096, "B spread, 256 MB span, 2 MB pages");
  chase(s4, 4096, "C spread, 256 MB span, 4 KB pages");
  chase(h, 64, "A again (drift check)");
  std::free(h);
  std::free(s4);
  return 0;
}

int cmd_membw(const Args& a) {
  os::MemBwOptions opt;
  if (a.has("mib")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("mib"), 1u << 20);
    if (!r.ok() || r.value < 64) {
      std::fprintf(stderr, "pelias: --mib needs a whole number of MiB, at least 64\n");
      return 1;
    }
    opt.bytes = static_cast<std::size_t>(r.value) << 20;
  }
  if (a.has("reps")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("reps"), 100);
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --reps needs 1 to 100\n");
      return 1;
    }
    opt.reps = static_cast<int>(r.value);
  }
  if (a.has("threads")) {
    opt.threads.clear();
    const std::string t = a.get("threads");
    std::size_t p = 0;
    while (p <= t.size()) {
      const std::size_t q = t.find(',', p) == std::string::npos ? t.size() : t.find(',', p);
      const Result<std::uint64_t> r = io::parse_whole(t.substr(p, q - p), 1024);
      if (!r.ok() || r.value == 0) {
        std::fprintf(stderr, "pelias: --threads needs a comma-separated list of thread counts\n");
        return 1;
      }
      opt.threads.push_back(static_cast<int>(r.value));
      p = q + 1;
    }
  }
  os::MemBwResult res;
  std::string err;
  if (!os::membw_run(opt, res, err)) {
    std::fprintf(stderr, "pelias: %s\n", err.c_str());
    return 6;
  }
  std::printf("platform: %s; %d logical CPUs, %d physical cores; buffer %zu MiB; median of %d passes per cell\n", os::platform_name().c_str(),
              res.logical_cpus, res.physical_cores, opt.bytes >> 20, opt.reps);
  std::printf("transparent huge pages: %s; AnonHugePages while the 2 MB buffer was in use: %lld kB%s\n",
              res.thp_mode.empty() ? "(unknown)" : res.thp_mode.c_str(), res.huge_kib_seen, res.note.empty() ? "" : (" (" + res.note + ")").c_str());
  std::printf("idle machine, foreground job: wake-up late by %.0f us (p50) / %.0f us (p99); job %.0f us\n", res.idle_wake_p50_us, res.idle_wake_p99_us,
              res.idle_work_us);
  std::printf("\n%-8s %-6s %-9s %9s   %s\n", "threads", "pages", "prefetch", "GB/s", "foreground job meanwhile (wake late p50 / p99, job time vs idle)");
  double best = 0;
  for (const os::MemBwCell& c : res.cells) best = std::max(best, c.gbps);
  for (const os::MemBwCell& c : res.cells) {
    std::printf("%-8d %-6s %-9s %9.2f", c.threads, c.huge ? "2 MB" : "4 KB", c.prefetch ? "on" : "off", c.gbps);
    if (c.work_ratio > 0) std::printf("   %.0f / %.0f us, x%.2f", c.wake_p50_us, c.wake_p99_us, c.work_ratio);
    std::printf("\n");
  }
  for (const os::MemBwCell& c : res.cells)
    if (c.gbps >= 0.95 * best) {
      std::printf("\nfewest threads within 95 %% of the best (%.2f GB/s): %d (%s pages, prefetch %s)\n", best, c.threads, c.huge ? "2 MB" : "4 KB",
                  c.prefetch ? "on" : "off");
      break;
    }
  return 0;
}

int cmd_inspect(const Args& a) {
  const std::string path = a.get("model");
  if (path.empty()) {
    usage();
    return 1;
  }
  os::FileMap m;
  Status s = m.open(path);
  if (!s) return fail_input(s);
  const bool json = a.has("json");
  Timer t;
  if (looks_like_safetensors(m)) {
    io::SafetensorsReader r;
    s = r.parse(m);
    if (!s) return fail_input(s);
    if (json) {
      io::Json root = io::Json::object();
      root.set("format", io::Json::string("safetensors"));
      root.set("file_bytes", io::Json::unsigned_integer(m.size()));
      root.set("header_bytes", io::Json::unsigned_integer(r.header_bytes()));
      io::Json ts = io::Json::array();
      for (const auto& x : r.tensors()) {
        io::Json j = io::Json::object();
        j.set("name", io::Json::string(x.name)).set("dtype", io::Json::string(x.dtype));
        io::Json sh = io::Json::array();
        for (const auto d : x.shape) sh.push(io::Json::unsigned_integer(d));
        j.set("shape", sh).set("bytes", io::Json::unsigned_integer(x.bytes())).set("file_offset", io::Json::unsigned_integer(x.file_offset));
        ts.push(j);
      }
      root.set("tensors", ts);
      std::puts(root.dump(1).c_str());
    } else {
      std::printf("safetensors: %zu tensors, header %llu bytes, file %s\n", r.tensors().size(), static_cast<unsigned long long>(r.header_bytes()), plan::format_bytes(m.size()).c_str());
      for (const auto& x : r.tensors()) {
        std::string sh = "[";
        for (std::size_t i = 0; i < x.shape.size(); ++i) sh += (i ? "," : "") + std::to_string(x.shape[i]);
        std::printf("%-56s %-6s %-24s %14llu bytes\n", x.name.c_str(), x.dtype.c_str(), (sh + "]").c_str(), static_cast<unsigned long long>(x.bytes()));
      }
    }
    return 0;
  }
  io::GgufReader r;
  s = r.parse(m, !a.has("header-only"));
  if (!s) return fail_input(s);
  const double parse_ms = t.ms();
  const std::uint64_t first_mtp = plan::first_mtp_block(r);
  std::map<std::uint32_t, std::uint64_t> by_type_count, by_type_bytes;
  for (const auto& x : r.tensors()) {
    ++by_type_count[x.type];
    by_type_bytes[x.type] += x.bytes;
  }
  if (json) {
    io::Json root = io::Json::object();
    root.set("format", io::Json::string("gguf")).set("version", io::Json::unsigned_integer(r.version())).set("alignment", io::Json::unsigned_integer(r.alignment()));
    root.set("file_bytes", io::Json::unsigned_integer(m.size())).set("data_start", io::Json::unsigned_integer(r.data_start())).set("data_bytes", io::Json::unsigned_integer(r.data_bytes()));
    root.set("n_kv", io::Json::unsigned_integer(r.kv().size())).set("n_tensors", io::Json::unsigned_integer(r.tensors().size()));
    root.set("architecture", io::Json::string(r.get_string("general.architecture")));
    io::Json types = io::Json::array();
    for (const auto& [id, cnt] : by_type_count) {
      io::Json j = io::Json::object();
      j.set("type", io::Json::string(quant::type_info(id).name)).set("id", io::Json::unsigned_integer(id)).set("tensors", io::Json::unsigned_integer(cnt)).set("bytes", io::Json::unsigned_integer(by_type_bytes[id]));
      types.push(j);
    }
    root.set("types", types);
    io::Json ts = io::Json::array();
    for (const auto& x : r.tensors()) {
      io::Json j = io::Json::object();
      j.set("name", io::Json::string(x.name)).set("type", io::Json::string(quant::type_info(x.type).name)).set("type_id", io::Json::unsigned_integer(x.type));
      io::Json sh = io::Json::array();
      for (std::uint32_t i = 0; i < x.n_dims; ++i) sh.push(io::Json::unsigned_integer(x.ne[i]));
      j.set("shape", sh).set("bytes", io::Json::unsigned_integer(x.bytes)).set("file_offset", io::Json::unsigned_integer(x.file_offset)).set("class", io::Json::string(plan::class_name(plan::classify(x.name, first_mtp))));
      ts.push(j);
    }
    root.set("tensors", ts);
    std::puts(root.dump(1).c_str());
    return 0;
  }
  std::printf("GGUF version %u, alignment %u; %zu key-value pairs, %zu tensors; data at byte %llu, %s of data in a %s file; architecture %s\n", r.version(), r.alignment(),
              r.kv().size(), r.tensors().size(), static_cast<unsigned long long>(r.data_start()), plan::format_bytes(r.data_bytes()).c_str(), plan::format_bytes(m.size()).c_str(),
              r.get_string("general.architecture").c_str());
  for (const auto& [id, cnt] : by_type_count)
    std::printf("  %-8s %4llu tensors %14s (%.4g bits per weight)\n", quant::type_info(id).name, static_cast<unsigned long long>(cnt), plan::format_bytes(by_type_bytes[id]).c_str(), quant::bits_per_weight(id));
  for (const auto& x : r.tensors())
    std::printf("%-44s %-8s %-24s %14llu bytes  %s\n", x.name.c_str(), quant::type_info(x.type).name, shape_text(x.ne, x.n_dims).c_str(), static_cast<unsigned long long>(x.bytes),
                plan::class_name(plan::classify(x.name, first_mtp)));
  std::printf("%zu tensors listed; header parsed in %.1f ms\n", r.tensors().size(), parse_ms);
  if (r.data_start() + r.data_bytes() > m.size())
    std::printf("note: header only: the full file would be %llu bytes\n", static_cast<unsigned long long>(r.data_start() + r.data_bytes()));
  else if (r.data_start() + r.data_bytes() != m.size())
    std::printf("note: the file has %llu bytes after the last tensor\n", static_cast<unsigned long long>(m.size() - r.data_start() - r.data_bytes()));
  return 0;
}

int cmd_key(const Args& a) {
  const std::string path = a.get("model");
  if (path.empty()) {
    usage();
    return 1;
  }
  // Plain sequential reads, not the memory map: a failed read then comes back as an error to report, where a mapped
  // page that cannot be read stops the program (SIGBUS, seen on WSL2's /mnt/c under memory pressure in P-M4).
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    const int e = errno;
    return fail_input(Status::fail(e == ENOENT ? Code::not_found : Code::io_error, std::string("cannot open the file: ") + std::strerror(e)));
  }
  Timer t;
  io::Sha256 h;
  std::vector<std::uint8_t> buf(std::size_t{16} << 20);  // 16 MiB reads (P-M4: the faster of the two read sizes)
  std::uint64_t total = 0;
  while (true) {
    const std::size_t got = std::fread(buf.data(), 1, buf.size(), f);
    if (got) {
      h.update(buf.data(), got);
      total += got;
    }
    if (got < buf.size()) break;
  }
  const bool read_error = std::ferror(f) != 0;
  std::fclose(f);
  if (read_error) return fail_input(Status::fail(Code::io_error, "read failed after " + std::to_string(total) + " bytes", total));
  const std::string hex = io::Sha256::hex(h.finish());
  const double sec = t.seconds();
  std::printf("%s  %llu bytes  (%.2f s, %.2f GB/s)\n", hex.c_str(), static_cast<unsigned long long>(total), sec, sec > 0 ? static_cast<double>(total) / sec / 1e9 : 0.0);
  return 0;
}

int cmd_plan(const Args& a) {
  const std::string path = a.get("model");
  if (path.empty()) {
    usage();
    return 1;
  }
  os::FileMap m;
  Status s = m.open(path);
  if (!s) return fail_input(s);
  io::GgufReader r;
  s = r.parse(m, !a.has("header-only"));  // a plan needs only the tensor table; --header-only plans from a header sample
  if (!s) return fail_input(s);
  plan::PlanRequest req;
  std::uint64_t stated_ceiling = 0;
  int index = 0;
  if (!gib_option(a, "ceiling-gib", stated_ceiling) || !gib_option(a, "bank-gib", req.bank_bytes) ||
      !gib_option(a, "working-set-gib", req.working_set_bytes) || !index_option(a, index))
    return 1;
  if (stated_ceiling) {
    const hal::Ceiling c = hal::ceiling_stated(stated_ceiling);
    req.ceiling_bytes = c.bytes;
    req.ceiling_basis = "stated";
  } else {
    const hal::DeviceInfo d = hal::query_device(index);
    const hal::Ceiling c = hal::ceiling_from_device(d);
    if (c.bytes) {
      req.ceiling_bytes = c.bytes;
      req.ceiling_basis = std::string("measured now: ") + c.basis + " free " + plan::format_bytes(c.basis_free) + " minus margin " + plan::format_bytes(c.margin);
    } else {
      std::fprintf(stderr, "pelias: no device (%s); state a ceiling with --ceiling-gib to plan without one\n", d.note.c_str());
      return 4;
    }
  }
  const Result<plan::Plan> res = plan::make_plan(r, req);
  if (res.status.code == Code::invalid_argument) return fail_input(res.status);
  std::fputs(plan::plan_text(res.value).c_str(), stdout);
  return res.ok() ? 0 : 3;
}


int cmd_quantize_tier(const Args& a) {
  const std::string in_path = a.get("model"), orig_path = a.get("original"), out_path = a.get("out");
  if (in_path.empty() || orig_path.empty() || out_path.empty()) {
    usage();
    return 1;
  }
  io::TierBuildOptions opt;
  if (a.has("threads")) {
    const Result<std::uint64_t> t = io::parse_whole(a.get("threads"), 1024);
    if (!t.ok() || t.value == 0) {
      std::fprintf(stderr, "pelias: --threads %s\n", t.ok() ? "must be at least 1" : t.status.message.c_str());
      return 1;
    }
    opt.threads = static_cast<int>(t.value);
  }
  opt.imatrix = a.get("imatrix");
  const Result<std::map<std::string, std::vector<std::uint32_t>>> alloc = io::read_tier_alloc(a.get("tier"));
  if (!alloc.ok()) return fail_input(alloc.status);
  os::FileMap bm, om;
  io::GgufReader br, orr;
  Status s = bm.open(in_path);
  if (s) s = br.parse(bm);
  if (!s) return fail_input(s);
  s = om.open(orig_path);
  if (s) s = orr.parse(om);
  if (!s) return fail_input(s);
  std::printf("tier: %zu weights re-quantized expert by expert from the original; every other tensor copied\n", alloc.value.size());
  std::fflush(stdout);
  Timer timer;
  std::size_t last_pct = 0;
  const Result<io::TierBuildReport> res = io::build_tiered(bm, br, om, orr, alloc.value, out_path, opt, [&](std::size_t i, std::size_t n, const std::string&) {
    const std::size_t pct = i * 100 / n;
    if (pct >= last_pct + 5 || i == n) {
      last_pct = pct;
      std::printf("  %3zu %% (%zu of %zu tensors, %.0f s)\n", pct, i, n, timer.seconds());
      std::fflush(stdout);
    }
  });
  if (!res.ok()) {
    std::fprintf(stderr, "pelias: %s\n", res.status.text().c_str());
    return res.status.code == Code::io_error ? 6 : 2;
  }
  const io::TierBuildReport& rep = res.value;
  std::printf("done in %.1f s: %zu tiered weights (%s in the base file, %s now), %zu tensors copied, %s of tensor data; importance used for %zu weights\n",
              timer.seconds(), rep.tiered, plan::format_bytes(rep.expert_bytes_base).c_str(), plan::format_bytes(rep.expert_bytes_out).c_str(), rep.copied,
              plan::format_bytes(rep.file_bytes).c_str(), rep.with_imatrix);
  for (const auto& [t, n] : rep.experts_by_type) std::printf("  %-7s %llu expert matrices\n", quant::type_info(t).name, static_cast<unsigned long long>(n));
  return 0;
}

int cmd_quantize(const Args& a) {
  if (a.has("tier")) return cmd_quantize_tier(a);
  const std::string in_path = a.get("model"), out_path = a.get("out");
  if (in_path.empty() || out_path.empty()) {
    usage();
    return 1;
  }
  const Result<quant::FileType> ft = quant::file_type_from_name(a.get("type", "Q4_K_M"));
  if (!ft.ok()) return fail_input(ft.status);
  int threads = 0;
  if (a.has("threads")) {
    const Result<std::uint64_t> t = io::parse_whole(a.get("threads"), 1024);
    if (!t.ok() || t.value == 0) {
      std::fprintf(stderr, "pelias: --threads %s\n", t.ok() ? "must be at least 1" : t.status.message.c_str());
      return 1;
    }
    threads = static_cast<int>(t.value);
  }
  os::FileMap m;
  Status s = m.open(in_path);
  if (!s) return fail_input(s);
  io::GgufReader r;
  s = r.parse(m);
  if (!s) return fail_input(s);
  io::RequantOptions opt;
  opt.ftype = ft.value;
  opt.threads = threads;
  if (a.has("rotate32")) {
    const std::string groups = a.get("rotate32");
    if (groups == "1") {
      std::fprintf(stderr, "pelias: --rotate32 needs 'all' or comma-separated parts of tensor names\n");
      return 1;
    }
    for (std::size_t p = 0; p <= groups.size();) {
      const std::size_t q = std::min(groups.find(',', p), groups.size());
      if (q > p) opt.rotate32.push_back(groups.substr(p, q - p));
      p = q + 1;
    }
    if (opt.rotate32.empty()) {
      std::fprintf(stderr, "pelias: --rotate32 needs 'all' or comma-separated parts of tensor names\n");
      return 1;
    }
  }
  if (a.has("recipe")) {
    const Result<std::map<std::string, std::uint32_t>> rc = io::read_recipe(a.get("recipe"));
    if (!rc.ok()) return fail_input(rc.status);
    opt.recipe = rc.value;
    std::printf("recipe: %zu tensors named\n", opt.recipe.size());
  }
  io::ImatrixWeights imw;
  if (a.has("imatrix")) {
    const std::string ip = a.get("imatrix");
    Result<io::ImatrixWeights> im = io::read_imatrix_weights(ip);
    if (!im.ok()) return fail_input(im.status);
    imw = std::move(im.value);
    opt.imatrix = &imw;
    const std::size_t cut = ip.find_last_of("/\\");
    opt.imatrix_name = cut == std::string::npos ? ip : ip.substr(cut + 1);  // rule 7: the file name, not its path
    std::printf("importance matrix: %zu entries over %u chunks%s\n", imw.weights.size(), imw.chunk_count,
                imw.has_rot32 ? (", " + std::to_string(imw.rot32.size()) + " collected rotated").c_str() : "");
  }
  std::printf("quantizing %zu tensors to %s\n", r.tensors().size(), quant::file_type_name(ft.value));
  std::fflush(stdout);
  std::size_t last_pct = 0;
  const Result<io::RequantReport> res = io::requantize(m, r, out_path, opt, [&](std::size_t i, std::size_t n, const std::string&, std::uint32_t) {
    const std::size_t pct = (i + 1) * 100 / n;
    if (pct >= last_pct + 10 || i + 1 == n) {
      last_pct = pct;
      std::printf("  %3zu %% (%zu of %zu tensors)\n", pct, i + 1, n);
      std::fflush(stdout);
    }
  });
  if (!res.ok()) {
    std::fprintf(stderr, "pelias: %s\n", res.status.text().c_str());
    return res.status.code == Code::io_error ? 6 : 2;
  }
  const io::RequantReport& rep = res.value;
  if (opt.imatrix) {
    std::printf("importance weights used for %llu tensors; not in the matrix: %zu", static_cast<unsigned long long>(rep.with_imatrix), rep.imatrix_missing.size());
    for (std::size_t i = 0; i < rep.imatrix_missing.size() && i < 5; ++i) std::printf("%s%s", i ? ", " : " (", rep.imatrix_missing[i].c_str());
    std::printf("%s\n", rep.imatrix_missing.empty() ? "" : rep.imatrix_missing.size() > 5 ? ", ...)" : ")");
  }
  if (rep.rotated) std::printf("rotated (Walsh-Hadamard per 32, recorded in pelias.rot32.tensors): %llu tensors\n", static_cast<unsigned long long>(rep.rotated));
  std::printf("done: %llu tensors (%llu quantized, %llu copied unchanged), %s in, %s out, %.1f s\n", static_cast<unsigned long long>(rep.tensors),
              static_cast<unsigned long long>(rep.quantized), static_cast<unsigned long long>(rep.copied), plan::format_bytes(rep.bytes_in).c_str(),
              plan::format_bytes(rep.bytes_out).c_str(), rep.seconds);
  for (std::uint32_t t = 0; t < 64; ++t)
    if (rep.by_type[t]) std::printf("  %-6s %llu tensors\n", quant::type_info(t).name, static_cast<unsigned long long>(rep.by_type[t]));
  return 0;
}

int cmd_convert(const Args& a) {
  const std::string in_path = a.get("model"), out_path = a.get("out");
  if (in_path.empty() || out_path.empty()) {
    usage();
    return 1;
  }
  const Result<quant::FileType> ft = quant::file_type_from_name(a.get("type", "BF16"));
  if (!ft.ok()) return fail_input(ft.status);
  int threads = 0;
  if (a.has("threads")) {
    const Result<std::uint64_t> t = io::parse_whole(a.get("threads"), 1024);
    if (!t.ok() || t.value == 0) {
      std::fprintf(stderr, "pelias: --threads %s\n", t.ok() ? "must be at least 1" : t.status.message.c_str());
      return 1;
    }
    threads = static_cast<int>(t.value);
  }
  ingest::ConvertOptions opt;
  opt.ftype = ft.value;
  opt.threads = threads;
  std::printf("converting the folder '%s' to %s\n", ingest::folder_name(in_path).c_str(), quant::file_type_name(ft.value));
  std::fflush(stdout);
  std::size_t last_pct = 0;
  const Result<io::RequantReport> res = ingest::convert_folder(in_path, out_path, opt, [&](std::size_t i, std::size_t n, const std::string&, std::uint32_t) {
    const std::size_t pct = (i + 1) * 100 / n;
    if (pct >= last_pct + 10 || i + 1 == n) {
      last_pct = pct;
      std::printf("  %3zu %% (%zu of %zu tensors)\n", pct, i + 1, n);
      std::fflush(stdout);
    }
  });
  if (!res.ok()) {
    std::fprintf(stderr, "pelias: %s\n", res.status.text().c_str());
    return res.status.code == Code::io_error ? 6 : 2;
  }
  const io::RequantReport& rep = res.value;
  if (ft.value == quant::FileType::bf16)
    std::printf("done: %llu tensors (%llu copied as BF16, %llu converted to F32), %s in, %s out, %.1f s\n", static_cast<unsigned long long>(rep.tensors),
                static_cast<unsigned long long>(rep.copied), static_cast<unsigned long long>(rep.quantized), plan::format_bytes(rep.bytes_in).c_str(),
                plan::format_bytes(rep.bytes_out).c_str(), rep.seconds);
  else
    std::printf("done: %llu tensors (%llu quantized, %llu copied unchanged), %s in, %s out, %.1f s\n", static_cast<unsigned long long>(rep.tensors),
                static_cast<unsigned long long>(rep.quantized), static_cast<unsigned long long>(rep.copied), plan::format_bytes(rep.bytes_in).c_str(),
                plan::format_bytes(rep.bytes_out).c_str(), rep.seconds);
  for (std::uint32_t t = 0; t < 64; ++t)
    if (rep.by_type[t]) std::printf("  %-6s %llu tensors\n", quant::type_info(t).name, static_cast<unsigned long long>(rep.by_type[t]));
  return 0;
}

int cmd_tokenize(const Args& a) {
  const std::string path = a.get("model");
  if (path.empty() || (a.has("text") == a.has("file"))) {
    usage();
    return 1;
  }
  os::FileMap m;
  Status s = m.open(path);
  if (!s) return fail_input(s);
  io::GgufReader g;
  s = g.parse(m, !a.has("header-only"));
  if (!s) return fail_input(s);
  Timer t;
  Result<tok::Tokenizer> tk = tok::Tokenizer::from_gguf(g);
  if (!tk.ok()) return fail_input(tk.status);
  const double load_ms = t.ms();
  std::string text;
  if (a.has("file")) {
    Result<std::string> r = io::read_text_file(a.get("file"));
    if (!r.ok()) return fail_input(r.status);
    text = std::move(r.value);
  } else {
    text = a.get("text");
  }
  tok::EncodeOptions opt;
  opt.nfc = !a.has("no-nfc");
  opt.parse_special = !a.has("no-special");
  Timer te;
  const std::vector<std::int32_t> ids = tk.value.encode(text, opt);
  const double enc_ms = te.ms();
  if (a.has("pieces")) {
    for (std::int32_t id : ids) std::printf("%6d -> '%s'\n", id, tk.value.piece(id).c_str());
  } else {
    std::string out = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (i) out += ", ";
      out += std::to_string(ids[i]);
    }
    out += "]";
    std::puts(out.c_str());
  }
  std::fprintf(stderr, "pelias: %zu tokens from %zu bytes; vocabulary %zu (load %.1f ms, encode %.1f ms); NFC %s, specials %s\n",
               ids.size(), text.size(), tk.value.vocab_size(), load_ms, enc_ms, opt.nfc ? "on" : "off",
               opt.parse_special ? "parsed" : "as text");
  return 0;
}

int cmd_verify(const Args& a) {
  const std::string pa = a.get("model"), pb = a.get("reference");
  if (pa.empty() || pb.empty()) {
    usage();
    return 1;
  }
  os::FileMap ma, mb;
  Status s = ma.open(pa);
  if (!s) return fail_input(s);
  s = mb.open(pb);
  if (!s) return fail_input(s);
  io::GgufReader ra, rb;
  s = ra.parse(ma);
  if (!s) return fail_input(s);
  s = rb.parse(mb);
  if (!s) return fail_input(s);
  Timer t;
  const io::GgufCompareReport rep = io::compare_gguf(ma, ra, mb, rb, 64);
  std::printf("tensors: %llu / %llu; identical %llu; type differs %llu; shape differs %llu; data differs %llu; only in one file %llu / %llu\n",
              static_cast<unsigned long long>(rep.tensors_a), static_cast<unsigned long long>(rep.tensors_b), static_cast<unsigned long long>(rep.identical),
              static_cast<unsigned long long>(rep.type_differs), static_cast<unsigned long long>(rep.shape_differs),
              static_cast<unsigned long long>(rep.data_differs), static_cast<unsigned long long>(rep.only_in_a), static_cast<unsigned long long>(rep.only_in_b));
  std::printf("metadata identical: %s; tensor order identical: %s; files byte-identical: %s", rep.kv_identical ? "yes" : "no",
              rep.order_identical ? "yes" : "no", rep.files_identical ? "yes" : "no");
  if (!rep.files_identical && ma.size() == mb.size()) std::printf(" (first difference at byte %llu)", static_cast<unsigned long long>(rep.first_file_difference));
  std::printf("\n");
  for (const std::string& n : rep.notes) std::printf("  %s\n", n.c_str());
  std::printf("compared in %.1f s\n", t.seconds());
  const bool all_tensors = rep.identical == rep.tensors_a && rep.tensors_a == rep.tensors_b;
  return all_tensors && rep.files_identical ? 0 : 5;
}

// --act-int8 GROUPS, --act-int8-path cpu|gpu|gpu256|gpu256x8|gpu256e and --act-rot h32 into `fo` (kld and generate).
// Bad text: prints why, returns false (exit 1).
bool act_options(const Args& a, model::ForwardOptions& fo) {
  fo.fast_cpu = a.has("fast-cpu");  // GP3-10 kernels (kern/cpu_dot.h)
  if (a.has("act-int8")) {
    const std::string groups = a.get("act-int8");
    if (groups == "1") {  // the argument parser's value for a flag given with no value
      std::fprintf(stderr, "pelias: --act-int8 needs a value: 'all' or comma-separated parts of tensor names\n");
      return false;
    }
    for (std::size_t p = 0; p <= groups.size();) {
      const std::size_t q = std::min(groups.find(',', p), groups.size());
      if (q > p) fo.act_int8.push_back(groups.substr(p, q - p));
      p = q + 1;
    }
    if (fo.act_int8.empty()) {
      std::fprintf(stderr, "pelias: --act-int8 needs 'all' or comma-separated parts of tensor names\n");
      return false;
    }
  }
  if (a.has("act-int8-block")) {
    std::fprintf(stderr, "pelias: --act-int8-block was replaced by --act-int8-path cpu|gpu (the format follows each weight's type)\n");
    return false;
  }
  if (a.has("act-int8-path")) {
    const std::string b = a.get("act-int8-path");
    if (fo.act_int8.empty()) {
      std::fprintf(stderr, "pelias: --act-int8-path needs --act-int8\n");
      return false;
    }
    if (b != "cpu" && b != "gpu" && b != "gpu256" && b != "gpu256x8" && b != "gpu256e") {
      std::fprintf(stderr, "pelias: --act-int8-path is cpu (llama.cpp's CPU formats per weight type), gpu (q8_1), gpu256 (the Pelias int8 kernel: one scale per 256), gpu256x8 (eight scales per 256) or gpu256e (one scale per 256, a power-of-two exponent per 32)\n");
      return false;
    }
    fo.act_gpu256 = b.rfind("gpu256", 0) == 0;
    fo.act_int8_gpu = b == "gpu" || fo.act_gpu256;
    fo.act_gpu256_sub = b == "gpu256x8" ? 1 : b == "gpu256e" ? 2 : 0;
  }
  if (a.has("act-rot")) {
    const std::string r = a.get("act-rot");
    if (fo.act_int8.empty()) {
      std::fprintf(stderr, "pelias: --act-rot needs --act-int8\n");
      return false;
    }
    if (r != "h32") {
      std::fprintf(stderr, "pelias: --act-rot is h32 (the 32-value Walsh-Hadamard rotation)\n");
      return false;
    }
    fo.act_rot32 = true;
  }
  return true;
}

// --planes FILE [--plane-k K] [--head-rescore N]: residual sign planes from `pelias planes`, read into RAM and attached to
// the weights (model::attach_planes). Returns 0, or the exit code after printing why.
int plane_options(const Args& a, os::FileMap& pm, io::GgufReader& pg, model::Qwen35MoeWeights& w, model::ForwardOptions& fo, const char* who) {
  if (!a.has("planes")) {
    if (a.has("plane-k") || a.has("head-rescore") || a.has("planes-mode") || a.has("plane-picks")) {
      std::fprintf(stderr, "pelias: --plane-k, --plane-picks, --head-rescore and --planes-mode need --planes FILE\n");
      return 1;
    }
    return 0;
  }
  int k = 0;
  if (a.has("plane-k")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("plane-k"), 4);
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --plane-k needs a whole number from 1 to 4\n");
      return 1;
    }
    k = static_cast<int>(r.value);
  }
  if (a.has("plane-picks")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("plane-picks"), 64);
    if (!r.ok()) {
      std::fprintf(stderr, "pelias: --plane-picks needs a whole number (0 = every pick)\n");
      return 1;
    }
    fo.plane_picks = static_cast<int>(r.value);
  }
  if (a.has("head-rescore")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("head-rescore"), 1u << 20);
    if (!r.ok()) {
      std::fprintf(stderr, "pelias: --head-rescore needs a whole number (0 = off)\n");
      return 1;
    }
    fo.head_rescore = static_cast<int>(r.value);
  }
  Timer tl;
  Status s = pm.load(a.get("planes"));
  if (!s) return fail_input(s);
  if (pm.unread_from() < pm.size()) {
    std::fprintf(stderr, "pelias: the mount refused the end of the plane file; copy it to another drive\n");
    return 2;
  }
  s = pg.parse(pm);
  if (!s) return fail_input(s);
  const Result<int> n = model::attach_planes(pg, pm.data(), w, k);
  if (!n.ok()) return fail_input(n.status);
  int stored = 0;
  for (const model::Qwen35MoeLayer& l : w.layers)
    if (l.down_exps.planes) stored = l.down_exps.plane_k_stored;
  std::size_t sparse = 0;
  for (const model::Qwen35MoeLayer& l : w.layers)
    for (const model::WeightView* p : {&l.gate_exps, &l.up_exps, &l.down_exps}) sparse += p->plane_mask ? 1 : 0;
  std::fprintf(stderr, "pelias %s: residual planes on %d weights (%d of %d stored per row on the routed experts; %zu expert weights block-selective) "
               "from %s, %.2f GB in %.1f s; read for %s; exact head re-score %s\n", who,
               n.value, k > 0 ? std::min(k, stored) : stored, stored, sparse, a.get("planes").c_str(), static_cast<double>(pm.size()) / 1e9, tl.seconds(),
               fo.plane_picks > 0 ? ("each token's top " + std::to_string(fo.plane_picks) + " expert picks").c_str() : "every pick",
               fo.head_rescore > 0 && w.output_exact.present() ? std::to_string(fo.head_rescore).c_str() : "off");
  if (fo.head_rescore > 0 && !w.output_exact.present()) {
    std::fprintf(stderr, "pelias: --head-rescore needs a plane file built with --exact-head\n");
    return 1;
  }
  return 0;
}

int cmd_kld(const Args& a) {
  const std::string path = a.get("model"), base_path = a.get("base");
  if (path.empty() || base_path.empty()) {
    usage();
    return 1;
  }
  int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  if (a.has("threads")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("threads"), 4096);
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --threads needs a whole number from 1 to 4096\n");
      return 1;
    }
    threads = static_cast<int>(r.value);
  }
  os::FileMap m;
  Status s = m.open(path);
  if (!s) return fail_input(s);
  io::GgufReader g;
  s = g.parse(m);
  if (!s) return fail_input(s);
  const Result<model::Qwen35MoeSpec> spec = model::read_qwen35moe_spec(g);
  if (!spec.ok()) return fail_input(spec.status);
  Result<model::Qwen35MoeWeights> w = model::bind_qwen35moe(g, m.data(), spec.value);
  if (!w.ok()) return fail_input(w.status);
  model::ForwardOptions plane_fo;
  os::FileMap pm;
  io::GgufReader pg;
  if (const int rc = plane_options(a, pm, pg, w.value, plane_fo, "kld"); rc != 0) return rc;
  {
    std::size_t nrot = w.value.output.rot32 ? 1 : 0;
    for (const model::Qwen35MoeLayer& l : w.value.layers)
      for (const model::WeightView* p : {&l.wq, &l.wk, &l.wv, &l.wo, &l.wqkv, &l.wgate, &l.wbeta, &l.walpha, &l.ssm_out, &l.up_exps, &l.gate_exps,
                                         &l.down_exps, &l.up_shexp, &l.gate_shexp, &l.down_shexp, &l.ffn_gate, &l.ffn_up, &l.ffn_down})
        nrot += p->rot32 ? 1 : 0;
    if (nrot) std::fprintf(stderr, "pelias: %zu weights are stored rotated (pelias.rot32.tensors); their inputs are rotated to match\n", nrot);
  }
  verify::KlBaseFile base;
  s = base.open(base_path);
  if (!s) return fail_input(s);
  if (static_cast<std::uint32_t>(base.n_vocab()) != spec.value.n_vocab) {
    std::fprintf(stderr, "pelias: the base file's vocabulary (%d) is not the model's (%u)\n", base.n_vocab(), spec.value.n_vocab);
    return 2;
  }
  int n_chunk = base.n_chunk();
  if (a.has("chunks")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("chunks"), static_cast<std::uint64_t>(base.n_chunk()));
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --chunks needs a whole number from 1 to %d (the base file's chunks)\n", base.n_chunk());
      return 1;
    }
    n_chunk = static_cast<int>(r.value);
  }
  // a model that adds BOS gets it in place of each chunk's first token, as the reference's tool does
  bool found = false;
  const bool add_bos = g.find("tokenizer.ggml.add_bos_token") && g.find("tokenizer.ggml.add_bos_token")->b;
  const std::uint64_t bos = g.get_uint("tokenizer.ggml.bos_token_id", &found);
  if (add_bos && !found) {
    std::fprintf(stderr, "pelias: the model adds a BOS token but names none\n");
    return 2;
  }
  const int n_ctx = static_cast<int>(base.n_ctx()), first = base.first(), scored = base.scored();
  const std::int64_t nv = spec.value.n_vocab, rec = base.record_u16();
  std::fprintf(stderr, "pelias kld: %d chunks of %d tokens, positions %d..%d scored, vocabulary %lld, %d threads, fp32 reference pass\n",
               n_chunk, n_ctx, first, n_ctx - 2, static_cast<long long>(nv), threads);
  std::vector<float> logits(static_cast<std::size_t>(scored) * static_cast<std::size_t>(nv));
  std::vector<std::uint16_t> records;
  std::vector<verify::KlToken> per(static_cast<std::size_t>(scored));
  verify::KlStats stats;
  model::ForwardOptions fo;
  fo.threads = threads;
  fo.head_rescore = plane_fo.head_rescore;
  fo.plane_picks = plane_fo.plane_picks;
  if (!act_options(a, fo)) return 1;
  // Block M: speed ideas, each measured for its accuracy cost
  std::uint64_t m_stats[6] = {0, 0, 0, 0, 0, 0};
  fo.m_stats = m_stats;
  if (a.has("act-bits")) {
    const std::string v = a.get("act-bits");
    if ((v != "4" && v != "8") || !fo.act_int8_gpu || (v == "4" && fo.act_gpu256)) {
      std::fprintf(stderr, "pelias: --act-bits is 4 or 8 and needs --act-int8 ... --act-int8-path gpu\n");
      return 1;
    }
    fo.act_bits = v == "4" ? 4 : 8;
  }
  if (a.has("neuron-skip")) {
    const Result<double> f = io::parse_positive_decimal(a.get("neuron-skip"), 0.99);
    if (!f.ok()) {
      std::fprintf(stderr, "pelias: --neuron-skip takes a fraction above 0 and at most 0.99\n");
      return 1;
    }
    fo.neuron_skip = f.value;
  }
  if (a.has("h-gate")) {
    const std::string v = a.get("h-gate");
    const Result<double> f = v.rfind("live:", 0) == 0 ? io::parse_positive_decimal(v.substr(5), 1.0) : Result<double>{};
    if (v.rfind("live:", 0) != 0 || !f.ok()) {
      std::fprintf(stderr, "pelias: --h-gate takes live:F, F the fraction of 32-blocks per row and token that read the 8th bit (0 < F <= 1)\n");
      return 1;
    }
    fo.h_gate = f.value;
    std::fprintf(stderr, "pelias kld: live H gate on Q8_Z weights: the 8th bit read on %.0f %% of each row's 32-blocks per token, the rest filled by 0.5\n",
                 100.0 * fo.h_gate);
  }
  if (a.has("t-decode")) {
    fo.t_decode = true;
    std::fprintf(stderr, "pelias kld: T-decode: Q7_ZT weights with gpu256x8 activations take the 1-bit route (8th bit by AND + popcount)\n");
  }
  if (a.has("defer-experts")) {
    const Result<std::uint64_t> d = io::parse_whole(a.get("defer-experts"), spec.value.n_expert_used > 0 ? spec.value.n_expert_used - 1 : 0);
    if (!d.ok() || d.value == 0) {
      std::fprintf(stderr, "pelias: --defer-experts takes 1 to n_expert_used - 1\n");
      return 1;
    }
    fo.defer_experts = static_cast<int>(d.value);
  }
  if (a.has("act-cold-path")) {
    if (a.get("act-cold-path") != "gpu256" || !fo.act_int8_gpu || !a.has("hot-set")) {
      std::fprintf(stderr, "pelias: --act-cold-path gpu256 needs --act-int8 ... --act-int8-path gpu|gpu256* and --hot-set FILE\n");
      return 1;
    }
    fo.cold_gpu256 = true;
  }
  if (a.has("hot-set") && !a.has("swap-near-tie") && !fo.cold_gpu256) {
    std::fprintf(stderr, "pelias: --hot-set goes with --swap-near-tie or --act-cold-path\n");
    return 1;
  }
  if (a.has("swap-near-tie") && !a.has("hot-set")) {
    std::fprintf(stderr, "pelias: --swap-near-tie and --hot-set go together\n");
    return 1;
  }
  if (a.has("swap-near-tie")) {
    const Result<double> mg = io::parse_positive_decimal(a.get("swap-near-tie"), 1.0);
    if (!mg.ok()) {
      std::fprintf(stderr, "pelias: --swap-near-tie takes a probability margin above 0\n");
      return 1;
    }
    fo.swap_margin = static_cast<float>(mg.value);
  }
  if (a.has("hot-set")) {
    const Result<std::string> hs = io::read_text_file(a.get("hot-set"), std::uint64_t{16} << 20);
    if (!hs.ok()) return fail_input(hs.status);
    const std::uint32_t nl = spec.value.n_layer, nex = spec.value.n_expert;
    fo.hot.assign(static_cast<std::size_t>(nl) * nex, 0);
    std::istringstream in(hs.value);
    std::string line;
    std::uint32_t layers_seen = 0;
    while (std::getline(in, line)) {
      if (line.rfind("layer ", 0) != 0) continue;
      std::istringstream ls(line.substr(6));
      long long li = -1;
      std::string colon;
      ls >> li >> colon;
      if (li < 0 || li >= static_cast<long long>(nl) || colon != ":") {
        std::fprintf(stderr, "pelias: --hot-set: bad line '%s'\n", line.substr(0, 40).c_str());
        return 1;
      }
      long long ex;
      while (ls >> ex) {
        if (ex < 0 || ex >= static_cast<long long>(nex)) {
          std::fprintf(stderr, "pelias: --hot-set: expert %lld out of range\n", ex);
          return 1;
        }
        fo.hot[static_cast<std::size_t>(li) * nex + static_cast<std::size_t>(ex)] = 1;
      }
      ++layers_seen;
    }
    if (layers_seen != nl) {
      std::fprintf(stderr, "pelias: --hot-set names %u layers, the model has %u\n", layers_seen, nl);
      return 1;
    }
  }
  if (fo.act_bits == 4 || fo.neuron_skip > 0 || fo.defer_experts > 0 || fo.swap_margin > 0)
    std::fprintf(stderr, "pelias kld: Block M options: act-bits %d, neuron-skip %.2f, defer-experts %d, swap-near-tie %.4g\n", fo.act_bits, fo.neuron_skip,
                 fo.defer_experts, static_cast<double>(fo.swap_margin));
  const std::string dump_path = a.get("dump-kld");
  if (!dump_path.empty() && std::ifstream(dump_path, std::ios::binary)) {
    std::fprintf(stderr, "pelias: %s already exists (nothing is overwritten)\n", dump_path.c_str());
    return 1;
  }
  verify::KlDump dump;
  if (!fo.act_int8.empty()) {
    std::string list;
    for (const std::string& x : fo.act_int8) list += (list.empty() ? "" : ",") + x;
    std::fprintf(stderr, "pelias kld: 8-bit activation rounding (%s path%s) on: %s\n", fo.act_gpu256 ? (fo.act_gpu256_sub == 1 ? "gpu256x8" : fo.act_gpu256_sub == 2 ? "gpu256e" : "gpu256") : (fo.act_int8_gpu ? "gpu" : "cpu"),
                 fo.act_rot32 ? ", Walsh-Hadamard rotation per 32 values" : "", list.c_str());
  }
  const std::string route_path = a.get("route-dump");
  if (!route_path.empty() && std::ifstream(route_path, std::ios::binary)) {
    std::fprintf(stderr, "pelias: %s already exists (nothing is overwritten)\n", route_path.c_str());
    return 1;
  }
  const std::string codes_path = a.get("dump-codes");
  if (!codes_path.empty() && std::ifstream(codes_path, std::ios::binary)) {
    std::fprintf(stderr, "pelias: %s already exists (nothing is overwritten)\n", codes_path.c_str());
    return 1;
  }
  const std::size_t code_bytes = static_cast<std::size_t>(quant::state_code_bytes(static_cast<std::int64_t>(spec.value.n_embd)));
  const std::size_t code_row = code_bytes * spec.value.n_layer;  // one position, every layer
  std::vector<std::uint8_t> codes_chunk(codes_path.empty() ? 0 : static_cast<std::size_t>(n_ctx) * code_row);
  std::vector<std::uint8_t> codes_all;
  if (!codes_path.empty()) fo.state_codes = codes_chunk.data();
  {
    auto parse_store = [](const std::string& v, model::ForwardOptions::StateStore& out) {
      using K = model::ForwardOptions::StateStore;
      if (v == "q8") out = K::q8; else if (v == "int4") out = K::int4; else if (v == "int4r") out = K::int4r;
      else if (v == "int8row") out = K::int8row; else return false;
      return true;
    };
    if (a.has("kv-store") && (!parse_store(a.get("kv-store"), fo.kv_store) || fo.kv_store == model::ForwardOptions::StateStore::int8row)) {
      std::fprintf(stderr, "pelias: --kv-store needs q8, int4 or int4r\n");
      return 1;
    }
    if (a.has("state-store") && !parse_store(a.get("state-store"), fo.state_store)) {
      std::fprintf(stderr, "pelias: --state-store needs q8, int4, int4r or int8row\n");
      return 1;
    }
    if (a.has("state-at")) {
      const Result<std::uint64_t> r = io::parse_whole(a.get("state-at"), static_cast<std::uint64_t>(n_ctx - 1));
      if (!r.ok()) {
        std::fprintf(stderr, "pelias: --state-at needs a position from 0 to %d\n", n_ctx - 1);
        return 1;
      }
      fo.state_at = static_cast<std::int64_t>(r.value);
    }
  }
  const std::string qk_path = a.get("dump-qk");
  std::vector<std::uint16_t> qk_chunk;
  std::FILE* qk_file = nullptr;
  bool qk_ok = true;
  if (!qk_path.empty()) {
    if (std::ifstream(qk_path, std::ios::binary)) {
      std::fprintf(stderr, "pelias: %s already exists (nothing is overwritten)\n", qk_path.c_str());
      return 1;
    }
    std::vector<std::uint32_t> attn_layers;
    for (std::uint32_t il = 0; il < spec.value.n_layer; ++il)
      if (!w.value.layers[il].recurrent) attn_layers.push_back(il);
    const std::size_t qk_row = static_cast<std::size_t>(spec.value.n_head + spec.value.n_head_kv) * spec.value.head_dim;
    qk_chunk.assign(attn_layers.size() * static_cast<std::size_t>(n_ctx) * qk_row, 0);
    fo.qk_dump = qk_chunk.data();
    // PELQK001: magic, u32 n_ctx, chunks, n_attn_layers, n_head, n_head_kv, head_dim, u64 token hash (as --dump-kld),
    // u32 layer index per full-attention layer; then per chunk, per full-attention layer, per position (all n_ctx):
    // n_head query heads then n_head_kv key heads, head_dim fp16 each
    qk_file = std::fopen(qk_path.c_str(), "wb");
    qk_ok = qk_file != nullptr;
    const std::uint32_t hdr[6] = {static_cast<std::uint32_t>(n_ctx), static_cast<std::uint32_t>(n_chunk), static_cast<std::uint32_t>(attn_layers.size()),
                                  spec.value.n_head, spec.value.n_head_kv, spec.value.head_dim};
    const std::uint64_t hash = verify::kl_token_hash(base.tokens().data(), static_cast<std::size_t>(n_chunk) * static_cast<std::size_t>(n_ctx));
    qk_ok = qk_ok && std::fwrite("PELQK001", 1, 8, qk_file) == 8 && std::fwrite(hdr, 4, 6, qk_file) == 6 && std::fwrite(&hash, 8, 1, qk_file) == 1 &&
            std::fwrite(attn_layers.data(), 4, attn_layers.size(), qk_file) == attn_layers.size();
    if (!qk_ok) {
      std::fprintf(stderr, "pelias: %s could not be written\n", qk_path.c_str());
      return 6;
    }
  }
  // --dump-flips FILE: per scored token, the coin-flip counts of the 8-bit activation rounding (ForwardOptions::flip_counts)
  const std::string flip_path = a.get("dump-flips");
  std::vector<std::uint32_t> flip_chunk, flip_all;
  if (!flip_path.empty()) {
    if (std::ifstream(flip_path, std::ios::binary)) {
      std::fprintf(stderr, "pelias: %s already exists (nothing is overwritten)\n", flip_path.c_str());
      return 6;
    }
    if (fo.act_int8.empty() || !fo.act_gpu256 || fo.act_gpu256_sub != 1 || fo.act_rot32) {
      std::fprintf(stderr, "pelias: --dump-flips counts the gpu256x8 rounding: it needs --act-int8 ... --act-int8-path gpu256x8\n");
      return 1;
    }
    flip_chunk.assign(static_cast<std::size_t>(n_ctx) * 3, 0);
    fo.flip_counts = flip_chunk.data();
  }
  std::vector<model::RouteRecord> route_chunk(route_path.empty() ? 0 : static_cast<std::size_t>(n_ctx) * spec.value.n_layer);
  std::vector<model::RouteRecord> route_all;
  if (!route_path.empty()) fo.route = route_chunk.data();
  const std::string stats_path = a.get("expert-stats");
  const std::size_t n_route = static_cast<std::size_t>(spec.value.n_layer) * spec.value.n_expert;
  std::vector<std::uint64_t> picks(stats_path.empty() ? 0 : n_route, 0);
  std::vector<double> weights(stats_path.empty() ? 0 : n_route, 0.0);
  if (!stats_path.empty()) {
    fo.expert_picks = picks.data();
    fo.expert_weight = weights.data();
  }
  Timer total;
  std::printf("%s", verify::kl_chunk_header().c_str());
  for (int c = 0; c < n_chunk; ++c) {
    Timer tc;
    std::vector<std::int32_t> toks(base.tokens().begin() + static_cast<std::ptrdiff_t>(c) * n_ctx,
                                   base.tokens().begin() + static_cast<std::ptrdiff_t>(c + 1) * n_ctx);
    if (add_bos) toks[0] = static_cast<std::int32_t>(bos);
    std::fill(flip_chunk.begin(), flip_chunk.end(), 0u);
    s = model::forward_reference(spec.value, w.value, toks.data(), n_ctx, first, n_ctx - 1, logits.data(), fo);
    if (!s) return fail_input(s);
    if (!flip_path.empty())  // the scored positions only, in the order of the KL records
      flip_all.insert(flip_all.end(), flip_chunk.begin() + static_cast<std::ptrdiff_t>(first) * 3,
                      flip_chunk.begin() + static_cast<std::ptrdiff_t>(first + scored) * 3);
    s = base.read_chunk(records);
    if (!s) return fail_input(s);
    std::vector<std::int32_t> orig(base.tokens().begin() + static_cast<std::ptrdiff_t>(c) * n_ctx,
                                   base.tokens().begin() + static_cast<std::ptrdiff_t>(c + 1) * n_ctx);
    std::vector<std::thread> pool;
    for (int k = 0; k < threads; ++k)
      pool.emplace_back([&, k] {
        for (int i = k; i < scored; i += threads)
          per[static_cast<std::size_t>(i)] = verify::kl_token(static_cast<std::int32_t>(nv), logits.data() + static_cast<std::ptrdiff_t>(i) * nv,
                                                              records.data() + static_cast<std::ptrdiff_t>(i) * rec, orig[static_cast<std::size_t>(first + i + 1)]);
      });
    for (auto& th : pool) th.join();
    for (const verify::KlToken& t : per) stats.add(t);
    if (!dump_path.empty()) dump.tokens.insert(dump.tokens.end(), per.begin(), per.end());
    if (!route_path.empty())  // the scored positions only, in the order of the KL records
      route_all.insert(route_all.end(), route_chunk.begin() + static_cast<std::ptrdiff_t>(first) * spec.value.n_layer,
                       route_chunk.begin() + static_cast<std::ptrdiff_t>(first + scored) * spec.value.n_layer);
    if (qk_file) qk_ok = qk_ok && std::fwrite(qk_chunk.data(), 2, qk_chunk.size(), qk_file) == qk_chunk.size();
    if (!codes_path.empty())  // the scored positions only, in the order of the KL records
      codes_all.insert(codes_all.end(), codes_chunk.begin() + static_cast<std::ptrdiff_t>(first) * static_cast<std::ptrdiff_t>(code_row),
                       codes_chunk.begin() + static_cast<std::ptrdiff_t>(first + scored) * static_cast<std::ptrdiff_t>(code_row));
    std::printf("%s", verify::kl_chunk_line(c + 1, stats).c_str());
    std::fflush(stdout);
    if (c == 0)
      std::fprintf(stderr, "pelias kld: %.2f seconds per chunk - ETA %.1f minutes\n", tc.seconds(), tc.seconds() * (n_chunk - 1) / 60.0);
  }
  std::printf("\n%s", verify::kl_report(stats).c_str());
  if (!dump_path.empty()) {
    dump.n_ctx = static_cast<std::uint32_t>(n_ctx);
    dump.chunks = static_cast<std::uint32_t>(n_chunk);
    dump.scored = static_cast<std::uint32_t>(scored);
    dump.token_hash = verify::kl_token_hash(base.tokens().data(), static_cast<std::size_t>(n_chunk) * static_cast<std::size_t>(n_ctx));
    s = verify::kl_dump_write(dump_path, dump);
    if (!s) {
      std::fprintf(stderr, "pelias: %s\n", s.text().c_str());
      return 6;
    }
    std::fprintf(stderr, "pelias kld: %zu per-token records written\n", dump.tokens.size());
  }
  if (!route_path.empty()) {
    // PELROUT1: magic, u32 n_ctx, chunks, scored, n_layer, n_used, u64 token hash (as --dump-kld), then per scored
    // token and layer: i16 sel[8], f32 w[8], f32 p_last, f32 p_next, i16 pred[8] (72 bytes), little-endian
    std::FILE* f = std::fopen(route_path.c_str(), "wb");
    bool okw = f != nullptr;
    auto put = [&](const void* p, std::size_t nb) { okw = okw && std::fwrite(p, 1, nb, f) == nb; };
    if (okw) {
      put("PELROUT1", 8);
      const std::uint32_t hdr[5] = {static_cast<std::uint32_t>(n_ctx), static_cast<std::uint32_t>(n_chunk), static_cast<std::uint32_t>(scored),
                                    spec.value.n_layer, spec.value.n_expert_used};
      put(hdr, sizeof hdr);
      const std::uint64_t hash = verify::kl_token_hash(base.tokens().data(), static_cast<std::size_t>(n_chunk) * static_cast<std::size_t>(n_ctx));
      put(&hash, 8);
      for (const model::RouteRecord& r : route_all) {
        put(r.sel, 16);
        put(r.w, 32);
        put(&r.p_last, 4);
        put(&r.p_next, 4);
        put(r.pred, 16);
      }
    }
    if (f) okw = std::fclose(f) == 0 && okw;
    if (!okw) {
      std::fprintf(stderr, "pelias: the routing record could not be written\n");
      return 6;
    }
    std::fprintf(stderr, "pelias kld: routing record written (%zu token-layers)\n", route_all.size());
  }
  if (!flip_path.empty()) {
    // PELFLIP1: magic, u32 n_ctx, chunks, scored, u64 token hash (as --dump-kld), then per scored token u32 x 3: the values
    // within 1e-4, 1e-3, 1e-2 of a step from halfway in the gpu256x8 rounding (trunk and expert inputs, every layer)
    std::FILE* f = std::fopen(flip_path.c_str(), "wb");
    bool okw = f != nullptr;
    const std::uint32_t hdr[3] = {static_cast<std::uint32_t>(n_ctx), static_cast<std::uint32_t>(n_chunk), static_cast<std::uint32_t>(scored)};
    const std::uint64_t hash = verify::kl_token_hash(base.tokens().data(), static_cast<std::size_t>(n_chunk) * static_cast<std::size_t>(n_ctx));
    okw = okw && std::fwrite("PELFLIP1", 1, 8, f) == 8 && std::fwrite(hdr, 4, 3, f) == 3 && std::fwrite(&hash, 8, 1, f) == 1 &&
          std::fwrite(flip_all.data(), 4, flip_all.size(), f) == flip_all.size();
    okw = f != nullptr && std::fclose(f) == 0 && okw;
    if (!okw) {
      std::fprintf(stderr, "pelias: the coin-flip record could not be written\n");
      return 6;
    }
    std::fprintf(stderr, "pelias kld: coin-flip record written (%zu tokens)\n", flip_all.size() / 3);
  }
  if (qk_file) {
    qk_ok = std::fclose(qk_file) == 0 && qk_ok;
    if (!qk_ok) {
      std::fprintf(stderr, "pelias: the q, k record could not be written\n");
      return 6;
    }
    std::fprintf(stderr, "pelias kld: q, k record written (%d chunks)\n", n_chunk);
  }
  if (!codes_path.empty()) {
    // PELCODE1: magic, u32 n_ctx, chunks, scored, n_layer, n_embd, bytes per layer, u64 token hash (as --dump-kld), then
    // per scored token and layer the state code (quant::state_code: n_embd / 8 sign bytes, n_embd / 32 energy bytes)
    std::FILE* f = std::fopen(codes_path.c_str(), "wb");
    bool okw = f != nullptr;
    auto put = [&](const void* p, std::size_t nb) { okw = okw && std::fwrite(p, 1, nb, f) == nb; };
    if (okw) {
      put("PELCODE1", 8);
      const std::uint32_t hdr[6] = {static_cast<std::uint32_t>(n_ctx), static_cast<std::uint32_t>(n_chunk), static_cast<std::uint32_t>(scored),
                                    spec.value.n_layer, static_cast<std::uint32_t>(spec.value.n_embd), static_cast<std::uint32_t>(code_bytes)};
      put(hdr, sizeof hdr);
      const std::uint64_t hash = verify::kl_token_hash(base.tokens().data(), static_cast<std::size_t>(n_chunk) * static_cast<std::size_t>(n_ctx));
      put(&hash, 8);
      put(codes_all.data(), codes_all.size());
    }
    if (f) okw = std::fclose(f) == 0 && okw;
    if (!okw) {
      std::fprintf(stderr, "pelias: the state codes could not be written\n");
      return 6;
    }
    std::fprintf(stderr, "pelias kld: state codes written (%zu bytes)\n", codes_all.size());
  }
  if (!stats_path.empty()) {
    std::string csv = "layer,expert,picks,weight_sum\n";
    for (std::uint32_t il = 0; il < spec.value.n_layer; ++il)
      for (std::uint32_t ex = 0; ex < spec.value.n_expert; ++ex) {
        const std::size_t k = static_cast<std::size_t>(il) * spec.value.n_expert + ex;
        char row[96];
        std::snprintf(row, sizeof row, "%u,%u,%llu,%.6f\n", il, ex, static_cast<unsigned long long>(picks[k]), weights[k]);
        csv += row;
      }
    std::FILE* f = std::fopen(stats_path.c_str(), "wb");
    if (!f || std::fwrite(csv.data(), 1, csv.size(), f) != csv.size()) {
      if (f) std::fclose(f);
      std::fprintf(stderr, "pelias: the expert statistics could not be written\n");
      return 6;
    }
    std::fclose(f);
    std::fprintf(stderr, "pelias kld: expert statistics written (%u layers x %u experts)\n", spec.value.n_layer, spec.value.n_expert);
  }
  std::fprintf(stderr, "pelias kld: %d chunks in %.1f s\n", n_chunk, total.seconds());
  if (fo.swap_margin > 0)
    std::fprintf(stderr, "pelias kld: near-tie swap: %llu swaps of %llu token-layers whose last pick was not resident\n",
                 static_cast<unsigned long long>(m_stats[0]), static_cast<unsigned long long>(m_stats[1]));
  if (fo.neuron_skip > 0)
    std::fprintf(stderr, "pelias kld: neuron skip: %llu of %llu expert hidden values zeroed\n", static_cast<unsigned long long>(m_stats[2]),
                 static_cast<unsigned long long>(m_stats[3]));
  if (fo.cold_gpu256)
    std::fprintf(stderr, "pelias kld: cold experts (CPU format, one scale per 256): %llu of %llu expert rows (%.2f %%)\n",
                 static_cast<unsigned long long>(m_stats[4]), static_cast<unsigned long long>(m_stats[5]),
                 m_stats[5] ? 100.0 * static_cast<double>(m_stats[4]) / static_cast<double>(m_stats[5]) : 0.0);
  return 0;
}

int cmd_kld_compare(const Args& a) {
  const std::string pa = a.get("a"), pb = a.get("b");
  if (pa.empty() || pb.empty()) {
    usage();
    return 1;
  }
  const Result<verify::KlDump> da = verify::kl_dump_read(pa);
  if (!da.ok()) return fail_input(da.status);
  const Result<verify::KlDump> db = verify::kl_dump_read(pb);
  if (!db.ok()) return fail_input(db.status);
  const Result<std::string> r = verify::kl_compare(da.value, db.value, a.has("name-a") ? a.get("name-a") : "A", a.has("name-b") ? a.get("name-b") : "B");
  if (!r.ok()) return fail_input(r.status);
  std::printf("%s", r.value.c_str());
  return 0;
}

int cmd_imatrix(const Args& a) {
  const std::string path = a.get("model"), text_path = a.get("file"), out = a.get("out");
  if (path.empty() || text_path.empty() || out.empty()) {
    usage();
    return 1;
  }
  int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  int n_ctx = 512;
  for (const char* k : {"threads", "ctx"}) {
    if (!a.has(k)) continue;
    const Result<std::uint64_t> r = io::parse_whole(a.get(k), 1u << 20);
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --%s needs a whole number above 0\n", k);
      return 1;
    }
    (std::string(k) == "threads" ? threads : n_ctx) = static_cast<int>(r.value);
  }
  os::FileMap m;
  Status s = m.open(path);
  if (!s) return fail_input(s);
  io::GgufReader g;
  s = g.parse(m);
  if (!s) return fail_input(s);
  const Result<model::Qwen35MoeSpec> spec = model::read_qwen35moe_spec(g);
  if (!spec.ok()) return fail_input(spec.status);
  const Result<model::Qwen35MoeWeights> w = model::bind_qwen35moe(g, m.data(), spec.value);
  if (!w.ok()) return fail_input(w.status);
  Result<tok::Tokenizer> tk = tok::Tokenizer::from_gguf(g);
  if (!tk.ok()) return fail_input(tk.status);
  Result<std::string> text = io::read_text_file(text_path);
  if (!text.ok()) return fail_input(text.status);
  tok::EncodeOptions eo;
  eo.nfc = !a.has("no-nfc");
  eo.parse_special = false;  // the reference tool's default: special-token text stays text
  const std::vector<std::int32_t> ids = tk.value.encode(text.value, eo);
  if (static_cast<std::int64_t>(ids.size()) < 2LL * n_ctx) {
    std::fprintf(stderr, "pelias: the text gives %zu tokens; a context of %d needs at least %d\n", ids.size(), n_ctx, 2 * n_ctx);
    return 2;
  }
  int n_chunk = static_cast<int>(ids.size() / static_cast<std::size_t>(n_ctx));
  if (a.has("chunks")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("chunks"), static_cast<std::uint64_t>(n_chunk));
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --chunks needs a whole number from 1 to %d (what the text holds)\n", n_chunk);
      return 1;
    }
    n_chunk = static_cast<int>(r.value);
  }
  bool found = false;
  const bool add_bos = g.find("tokenizer.ggml.add_bos_token") && g.find("tokenizer.ggml.add_bos_token")->b;
  const std::uint64_t bos = g.get_uint("tokenizer.ggml.bos_token_id", &found);
  model::ImatrixCollector col;
  model::ForwardOptions fo;
  fo.threads = threads;
  fo.imatrix = &col;
  std::fprintf(stderr, "pelias imatrix: %d chunks of %d tokens (%zu in the text), %d threads, fp32 reference pass\n", n_chunk, n_ctx, ids.size(), threads);
  Timer total;
  for (int c = 0; c < n_chunk; ++c) {
    Timer tc;
    std::vector<std::int32_t> toks(ids.begin() + static_cast<std::ptrdiff_t>(c) * n_ctx, ids.begin() + static_cast<std::ptrdiff_t>(c + 1) * n_ctx);
    if (add_bos && found) toks[0] = static_cast<std::int32_t>(bos);
    s = model::forward_reference(spec.value, w.value, toks.data(), n_ctx, 0, n_ctx, nullptr, fo);
    if (!s) return fail_input(s);
    col.finish_chunk();
    if (c == 0) std::fprintf(stderr, "pelias imatrix: %.2f seconds per chunk - ETA %.1f minutes\n", tc.seconds(), tc.seconds() * (n_chunk - 1) / 60.0);
    else std::fprintf(stderr, "pelias imatrix: chunk %d of %d\n", c + 1, n_chunk);
  }
  // the dataset is named by its file name only (rule 7: no paths in outputs)
  const std::size_t cut = text_path.find_last_of("/\\");
  const std::string dataset = cut == std::string::npos ? text_path : text_path.substr(cut + 1);
  s = col.write(out, {dataset}, static_cast<std::uint32_t>(n_ctx));
  if (!s) {
    std::fprintf(stderr, "pelias: %s\n", s.text().c_str());
    return 6;
  }
  std::fprintf(stderr, "pelias imatrix: %zu entries over %u chunks written in %.1f s; %zu collected in the rotated basis (pelias.rot32.tensors)\n",
               col.entries().size(), col.chunks(), total.seconds(), col.rotated().size());
  return 0;
}

// Reads --k as a decimal number in [lo, hi] (negative allowed). Absent: true, `v` unchanged. Bad text: prints why, false.
bool float_option(const Args& a, const char* k, float lo, float hi, float& v) {
  if (!a.has(k)) return true;
  const std::string t = a.get(k);
  char* end = nullptr;
  errno = 0;
  const double d = std::strtod(t.c_str(), &end);
  if (t.empty() || end != t.c_str() + t.size() || errno != 0 || !(d >= static_cast<double>(lo) && d <= static_cast<double>(hi))) {
    std::fprintf(stderr, "pelias: --%s takes a number from %g to %g\n", k, static_cast<double>(lo), static_cast<double>(hi));
    return false;
  }
  v = static_cast<float>(d);
  return true;
}

// GP3-07: text generation on the CPU reference pass. The prompt runs once (prefill), then one token per step against the
// carried KV cache and DeltaNet state; the sampler picks each token (temperature 0 = greedy).
int cmd_planes(const Args& a) {
  const std::string path = a.get("model"), base_path = a.get("base"), out_path = a.get("out");
  if (path.empty() || base_path.empty() || out_path.empty()) {
    usage();
    return 1;
  }
  if (std::ifstream(out_path, std::ios::binary)) {
    std::fprintf(stderr, "pelias: %s already exists (nothing is overwritten)\n", out_path.c_str());
    return 1;
  }
  io::PlaneBuildOptions po;
  if (a.has("k")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("k"), 4);
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --k needs a whole number from 1 to 4\n");
      return 1;
    }
    po.k = static_cast<int>(r.value);
  }
  if (a.has("refit")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("refit"), 64);
    if (!r.ok()) {
      std::fprintf(stderr, "pelias: --refit needs a whole number from 0 to 64\n");
      return 1;
    }
    po.refit_rounds = static_cast<int>(r.value);
  }
  if (a.has("threads")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("threads"), 4096);
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --threads needs a whole number from 1 to 4096\n");
      return 1;
    }
    po.threads = static_cast<int>(r.value);
  }
  po.imatrix = a.get("imatrix");
  po.exact_head = a.has("exact-head");
  po.skip_mtp = !a.has("keep-mtp");
  if (a.has("k-rules")) {  // PART=K,PART=K,...: the first PART found in a tensor's name sets its plane count (0 = none)
    const std::string t = a.get("k-rules");
    for (std::size_t p0 = 0; p0 <= t.size();) {
      const std::size_t p1 = std::min(t.find(',', p0), t.size());
      const std::string item = t.substr(p0, p1 - p0);
      const std::size_t eq = item.rfind('=');
      const Result<std::uint64_t> r = eq == std::string::npos ? Result<std::uint64_t>::fail(Status::fail(Code::invalid_argument, "no =")) : io::parse_whole(item.substr(eq + 1), 4);
      if (!r.ok()) {
        std::fprintf(stderr, "pelias: --k-rules takes PART=K items (K 0 to 4), comma-separated; '%s' is not one\n", item.c_str());
        return 1;
      }
      po.k_rules.emplace_back(item.substr(0, eq), static_cast<int>(r.value));
      p0 = p1 + 1;
    }
  }
  if (a.has("select")) {
    po.select = a.get("select");
    const Result<std::uint64_t> r = io::parse_whole(a.get("budget-mb"), std::uint64_t{1} << 30);
    if (!a.has("budget-mb") || !r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --select needs --budget-mb N (plane data of the selected tensors, megabytes of 10^6 bytes)\n");
      return 1;
    }
    po.budget_bytes = r.value * 1000000ull;
  }
  if (a.has("block-weights")) {  // name<TAB>factor per line (tools/analysis/plane_block_weights.py)
    const Result<std::string> text = io::read_text_file(a.get("block-weights"), std::uint64_t{16} << 20);
    if (!text.ok()) return fail_input(text.status);
    std::size_t p0 = 0;
    while (p0 < text.value.size()) {
      std::size_t p1 = text.value.find('\n', p0);
      if (p1 == std::string::npos) p1 = text.value.size();
      std::string line = text.value.substr(p0, p1 - p0);
      p0 = p1 + 1;
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty()) continue;
      const std::size_t tab = line.find('\t');
      char* end = nullptr;
      const double f = tab == std::string::npos ? 0.0 : std::strtod(line.c_str() + tab + 1, &end);
      if (tab == std::string::npos || end == line.c_str() + tab + 1 || !(f > 0.0)) {
        std::fprintf(stderr, "pelias: --block-weights lines are name<TAB>factor (factor above 0): '%s'\n", line.c_str());
        return 1;
      }
      po.block_weights[line.substr(0, tab)] = f;
    }
  }
  os::FileMap mm, bm;
  Status s = mm.open(path);
  if (!s) return fail_input(s);
  s = bm.open(base_path);
  if (!s) return fail_input(s);
  io::GgufReader mg, bg;
  s = mg.parse(mm);
  if (!s) return fail_input(s);
  s = bg.parse(bm);
  if (!s) return fail_input(s);
  std::fprintf(stderr, "pelias planes: %s against %s: %d plane(s) per row, %d refit round(s), importance %s%s\n", path.c_str(), base_path.c_str(), po.k,
               po.refit_rounds, po.imatrix.empty() ? "none" : po.imatrix.c_str(), po.exact_head ? ", exact output rows" : "");
  Timer t;
  std::size_t last_pct = 0;
  const auto prog = [&](std::size_t done, std::size_t total, const std::string& name) {
    const std::size_t pct = done * 100 / total;
    if (pct / 10 != last_pct / 10 || done == total) {
      std::fprintf(stderr, "pelias planes: %3zu %% (%zu of %zu weights, %s) %.0f s\n", pct, done, total, name.c_str(), t.seconds());
      last_pct = pct;
    }
  };
  const Result<io::PlaneBuildReport> rep = io::build_planes(mm, mg, bm, bg, out_path, po, prog);
  if (!rep.ok()) {
    std::fprintf(stderr, "pelias: %s (a partial %s may be left; it is not removed)\n", rep.status.text().c_str(), out_path.c_str());
    return rep.status.code == Code::invalid_argument ? 1 : 2;
  }
  for (const std::string& x : rep.value.skipped) std::fprintf(stderr, "pelias planes: left out %s\n", x.c_str());
  if (rep.value.tiered) std::fprintf(stderr, "pelias planes: %zu usage-tiered weights fitted expert by expert in each expert's own type\n", rep.value.tiered);
  if (rep.value.blocks_candidate)
    std::fprintf(stderr, "pelias planes: block selection '%s': %llu of %llu blocks kept (%.2f %%); their tensors' residual %.6g -> %.6g (%.4f)\n",
                 po.select.c_str(), static_cast<unsigned long long>(rep.value.blocks_selected), static_cast<unsigned long long>(rep.value.blocks_candidate),
                 100.0 * static_cast<double>(rep.value.blocks_selected) / static_cast<double>(rep.value.blocks_candidate), rep.value.select_err_before,
                 rep.value.select_err_after, rep.value.select_err_before > 0 ? rep.value.select_err_after / rep.value.select_err_before : 0.0);
  if (rep.value.imatrix_basis_skips)
    std::fprintf(stderr, "pelias planes: %zu weights fitted without importance (the matrix was collected rotated and the file is not, or the reverse)\n",
                 rep.value.imatrix_basis_skips);
  std::fprintf(stderr, "pelias planes: %zu weights, %.2f GB of planes in %.0f s; residual (importance-weighted sum) %.6g -> %.6g (%.4f)\n",
               rep.value.tensors, static_cast<double>(rep.value.bytes) / 1e9, t.seconds(), rep.value.err_before, rep.value.err_after,
               rep.value.err_before > 0 ? rep.value.err_after / rep.value.err_before : 0.0);
  return 0;
}

// pelias teacher: the original model's view of decoded text (the split-token grader, author 2026-10-29 20:01-20:46). For
// every --ids file (pelias generate --ids-out format: a "prompt:" and an "output:" line) the model runs the prompt and the
// output once, fresh state, and writes, for each output position, what the model itself would pick there: its top-1 and
// top-2 tokens and logits, and the logit and log-probability of the token the file holds. The model is opened once
// (memory map) for every file.
int cmd_teacher(const Args& a) {
  const std::string path = a.get("model"), dir = a.get("out-dir");
  std::vector<std::string> files;
  for (const auto& o : a.opts)
    if (o.first == "ids") files.push_back(o.second);
  if (path.empty() || dir.empty() || files.empty()) {
    usage();
    return 1;
  }
  int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  if (a.has("threads")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("threads"), 4096);
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --threads needs a whole number from 1 to 4096\n");
      return 1;
    }
    threads = static_cast<int>(r.value);
  }
  std::vector<std::string> outs;
  for (const std::string& f : files) {
    const std::size_t sl = f.find_last_of("/\\");
    std::string base = sl == std::string::npos ? f : f.substr(sl + 1);
    if (base.size() > 4 && base.substr(base.size() - 4) == ".ids") base.resize(base.size() - 4);
    const std::string o = dir + "/teacher_" + base + ".tsv";
    if (std::ifstream(o, std::ios::binary)) {
      std::fprintf(stderr, "pelias: %s already exists (nothing is overwritten)\n", o.c_str());
      return 1;
    }
    outs.push_back(o);
  }
  os::FileMap m;
  Status s = m.open(path);
  if (!s) return fail_input(s);
  io::GgufReader g;
  s = g.parse(m);
  if (!s) return fail_input(s);
  const Result<model::Qwen35MoeSpec> spec = model::read_qwen35moe_spec(g);
  if (!spec.ok()) return fail_input(spec.status);
  const Result<model::Qwen35MoeWeights> w = model::bind_qwen35moe(g, m.data(), spec.value);
  if (!w.ok()) return fail_input(w.status);
  const auto nv = static_cast<std::int64_t>(spec.value.n_vocab);
  model::ForwardOptions fo;
  fo.threads = threads;
  std::fprintf(stderr, "pelias teacher: %s, %zu file(s), %d threads, fp32 reference pass\n", path.c_str(), files.size(), threads);
  for (std::size_t fi = 0; fi < files.size(); ++fi) {
    Result<std::string> txt = io::read_text_file(files[fi]);
    if (!txt.ok()) return fail_input(txt.status);
    std::vector<std::int32_t> prompt, output;
    std::istringstream in(txt.value);
    std::string line;
    while (std::getline(in, line)) {
      std::vector<std::int32_t>* dst = line.rfind("prompt:", 0) == 0 ? &prompt : line.rfind("output:", 0) == 0 ? &output : nullptr;
      if (!dst) continue;
      std::istringstream ls(line.substr(7));
      long long v = 0;
      while (ls >> v) {
        if (v < 0 || v >= nv) {
          std::fprintf(stderr, "pelias: %s holds a token id outside the vocabulary\n", files[fi].c_str());
          return 2;
        }
        dst->push_back(static_cast<std::int32_t>(v));
      }
    }
    if (prompt.empty() || output.empty()) {
      std::fprintf(stderr, "pelias: %s has no prompt or no output ids\n", files[fi].c_str());
      return 2;
    }
    std::vector<std::int32_t> seq = prompt;
    seq.insert(seq.end(), output.begin(), output.end() - 1);  // the last output token is never fed back
    const int n = static_cast<int>(seq.size()), first = static_cast<int>(prompt.size()) - 1, n_out = static_cast<int>(output.size());
    std::vector<float> lg(static_cast<std::size_t>(n_out) * static_cast<std::size_t>(nv));
    Timer t;
    s = model::forward_reference(spec.value, w.value, seq.data(), n, first, first + n_out, lg.data(), fo);
    if (!s) return fail_input(s);
    std::ofstream o(outs[fi], std::ios::binary);
    o << "pos\ttoken\ttop1\ttop1_logit\ttop2\ttop2_logit\ttoken_logit\ttoken_logprob\n";
    int agree = 0;
    double nll = 0.0;
    for (int i = 0; i < n_out; ++i) {
      const float* l = lg.data() + static_cast<std::ptrdiff_t>(i) * nv;
      std::int64_t b1 = 0, b2 = -1;
      for (std::int64_t v = 1; v < nv; ++v)
        if (l[v] > l[b1]) {
          b2 = b1;
          b1 = v;
        } else if (b2 < 0 || l[v] > l[b2]) {
          b2 = v;
        }
      if (b2 < 0) b2 = b1 == 0 ? 1 : 0;
      double mx = l[b1], sum = 0.0;
      for (std::int64_t v = 0; v < nv; ++v) sum += std::exp(static_cast<double>(l[v]) - mx);
      const std::int32_t tk = output[static_cast<std::size_t>(i)];
      const double lp = static_cast<double>(l[tk]) - mx - std::log(sum);
      agree += b1 == tk;
      nll -= lp;
      char buf[256];
      std::snprintf(buf, sizeof buf, "%d\t%d\t%lld\t%.5f\t%lld\t%.5f\t%.5f\t%.6f\n", i, tk, static_cast<long long>(b1), static_cast<double>(l[b1]),
                    static_cast<long long>(b2), static_cast<double>(l[b2]), static_cast<double>(l[tk]), lp);
      o << buf;
    }
    if (!o) {
      std::fprintf(stderr, "pelias: could not write %s\n", outs[fi].c_str());
      return 6;
    }
    std::fprintf(stderr, "pelias teacher: %s: %d tokens, the model's own top-1 on %d of %d, mean NLL %.4f; %.1f s\n", files[fi].c_str(), n, agree, n_out,
                 nll / n_out, t.seconds());
  }
  return 0;
}

// pelias mtp-logits: GP3-08's gate against the reference library (tools/oracle/mtp_oracle.cpp writes the same format):
// the trunk runs the ids, the MTP block runs id p with the trunk's post-norm hidden of p - 1 (zeros before the first), and
// every position's MTP logits and post-norm hidden are written: int32 n, n_vocab, n_embd; n x n_vocab; n x n_embd (fp32).
// pelias logits: every position's logits for a list of token ids (GP3-09 gate against the reference library's
// logits_oracle). One fresh pass, or with --decode the tokens one at a time through the decode state.
// Output: int32 n, int32 n_vocab, then n x n_vocab logits (fp32).
int cmd_logits(const Args& a) {
  const std::string path = a.get("model"), out = a.get("out"), idt = a.get("ids");
  if (path.empty() || out.empty() || idt.empty()) {
    std::fprintf(stderr, "usage: pelias logits --model FILE --ids 1,2,3 --out OUT.bin [--threads N] [--fast-cpu] [--decode]\n");
    return 1;
  }
  if (std::ifstream(out, std::ios::binary)) {
    std::fprintf(stderr, "pelias: %s already exists (nothing is overwritten)\n", out.c_str());
    return 1;
  }
  std::vector<std::int32_t> ids;
  for (std::size_t p = 0; p <= idt.size();) {
    const std::size_t q = std::min(idt.find(',', p), idt.size());
    const Result<std::uint64_t> r = io::parse_whole(idt.substr(p, q - p), 1u << 30);
    if (!r.ok()) {
      std::fprintf(stderr, "pelias: --ids takes comma-separated token ids\n");
      return 1;
    }
    ids.push_back(static_cast<std::int32_t>(r.value));
    p = q + 1;
  }
  model::ForwardOptions fo;
  if (a.has("threads")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("threads"), 4096);
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --threads needs a whole number from 1\n");
      return 1;
    }
    fo.threads = static_cast<int>(r.value);
  }
  if (!act_options(a, fo)) return 1;
  os::FileMap m;
  if (Status s = m.open(path); !s) return fail_input(s);
  io::GgufReader g;
  if (Status s = g.parse(m); !s) return fail_input(s);
  const Result<model::Qwen35MoeSpec> spec = model::read_qwen35moe_spec(g);
  if (!spec.ok()) return fail_input(spec.status);
  Result<model::Qwen35MoeWeights> w = model::bind_qwen35moe(g, m.data(), spec.value);
  if (!w.ok()) return fail_input(w.status);
  const int n = static_cast<int>(ids.size());
  const std::size_t nv = spec.value.n_vocab;
  std::vector<float> lg(static_cast<std::size_t>(n) * nv);
  if (a.has("decode")) {
    model::DecodeState st;
    model::decode_state_reset(spec.value, st);
    for (int i = 0; i < n; ++i)
      if (Status s = model::forward_decode(spec.value, w.value, st, ids.data() + i, 1, lg.data() + static_cast<std::size_t>(i) * nv, true, fo); !s)
        return fail_input(s);
  } else if (Status s = model::forward_reference(spec.value, w.value, ids.data(), n, 0, n, lg.data(), fo); !s) {
    return fail_input(s);
  }
  std::ofstream f(out, std::ios::binary);
  const std::int32_t hdr[2] = {n, static_cast<std::int32_t>(nv)};
  f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
  f.write(reinterpret_cast<const char*>(lg.data()), static_cast<std::streamsize>(lg.size() * sizeof(float)));
  if (!f) {
    std::fprintf(stderr, "pelias: could not write %s\n", out.c_str());
    return 6;
  }
  std::fprintf(stderr, "pelias logits: %s, %d positions%s, written to %s\n", spec.value.arch.c_str(), n, a.has("decode") ? " (one at a time)" : "",
               out.c_str());
  return 0;
}

// GP3-12 rungs 2-3 on a real file: the trunk (all of it, or its first --layers) on the GPU against the CPU reference
// b against a (a the reference side) over n rows of nv logits: rel L2, largest |diff|, KL(a || b) per row, top-1 agreement
struct Cmp {
  double rel, worst, kl_mean, kl_max;
  int top;
};
Cmp compare_logits(const std::vector<float>& ra, const std::vector<float>& rb, int n, std::size_t nv) {
  double num2 = 0.0, den = 0.0, worst = 0.0, kl_sum = 0.0, kl_max = 0.0;
  int top = 0;
  for (int t = 0; t < n; ++t) {
    const float* r = ra.data() + static_cast<std::size_t>(t) * nv;
    const float* gq = rb.data() + static_cast<std::size_t>(t) * nv;
    std::size_t ar = 0, ag = 0;
    double mr = r[0], mg = gq[0];
    for (std::size_t i = 0; i < nv; ++i) {
      const double d = static_cast<double>(gq[i]) - r[i];
      num2 += d * d;
      den += static_cast<double>(r[i]) * r[i];
      worst = std::max(worst, std::fabs(d));
      if (r[i] > r[ar]) ar = i;
      if (gq[i] > gq[ag]) ag = i;
      mr = std::max(mr, static_cast<double>(r[i]));
      mg = std::max(mg, static_cast<double>(gq[i]));
    }
    top += ar == ag ? 1 : 0;
    double sr = 0.0, sg = 0.0;
    for (std::size_t i = 0; i < nv; ++i) {
      sr += std::exp(r[i] - mr);
      sg += std::exp(gq[i] - mg);
    }
    const double lr = mr + std::log(sr), lgg = mg + std::log(sg);
    double k = 0.0;
    for (std::size_t i = 0; i < nv; ++i) {
      const double pa = r[i] - lr, pb = gq[i] - lgg;
      k += std::exp(pa) * (pa - pb);
    }
    kl_sum += k;
    kl_max = std::max(kl_max, k);
  }
  return Cmp{den > 0 ? std::sqrt(num2 / den) : 0.0, worst, kl_sum / n, kl_max, top};
}

int cmd_gpu_check(const Args& a) {
  const std::string path = a.get("model");
  if (path.empty()) {
    std::fprintf(stderr, "usage: pelias gpu-check --model FILE [--tokens 64] [--layers N] [--repeat 3] [--threads N] [--fast-cpu] [--no-cpu] [--floor] [--float-ref]\n");
    return 1;
  }
  auto num = [&](const char* k, std::uint64_t def, std::uint64_t lo, std::uint64_t hi, std::uint64_t& v) {
    v = def;
    if (!a.has(k)) return true;
    const Result<std::uint64_t> r = io::parse_whole(a.get(k), hi);
    if (!r.ok() || r.value < lo) {
      std::fprintf(stderr, "pelias: --%s takes a whole number from %llu to %llu\n", k, static_cast<unsigned long long>(lo), static_cast<unsigned long long>(hi));
      return false;
    }
    v = r.value;
    return true;
  };
  std::uint64_t ntok = 0, nlay = 0, rep = 0, thr = 0;
  if (!num("tokens", 64, 1, 4096, ntok) || !num("layers", 0, 0, 1000, nlay) || !num("repeat", 3, 1, 100, rep) || !num("threads", 0, 0, 4096, thr)) return 1;
  os::FileMap m;
  if (Status s = m.open(path); !s) return fail_input(s);
  io::GgufReader g;
  if (Status s = g.parse(m); !s) return fail_input(s);
  const Result<model::Qwen35MoeSpec> spec = model::read_qwen35moe_spec(g);
  if (!spec.ok()) return fail_input(spec.status);
  Result<model::Qwen35MoeWeights> w = model::bind_qwen35moe(g, m.data(), spec.value);
  if (!w.ok()) return fail_input(w.status);
  model::Qwen35MoeSpec s = spec.value;
  if (nlay > 0 && nlay < s.n_layer) {
    w.value.layers.resize(static_cast<std::size_t>(nlay));
    s.n_layer = static_cast<std::uint32_t>(nlay);
  }
  const int n = static_cast<int>(ntok);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(n));
  for (int t = 0; t < n; ++t) ids[static_cast<std::size_t>(t)] = static_cast<std::int32_t>((static_cast<std::uint64_t>(t) * 7919u + 1011u) % s.n_vocab);
  const std::size_t nv = s.n_vocab;
  std::vector<float> lg_gpu(static_cast<std::size_t>(n) * nv), lg_ref;
  Timer topen;
  auto f = model::GpuForward::open(s, w.value, n);
  if (!f.ok()) return fail_input(f.status);
  std::printf("pelias gpu-check: %s, %u layers, %d tokens; GPU open %.2f s, %.2f GB on the device\n", s.arch.c_str(), s.n_layer, n, topen.seconds(),
              static_cast<double>(f.value->device_bytes()) / 1e9);
  double best = 1e30;
  for (std::uint64_t r = 0; r < rep; ++r) {
    model::GpuForwardStats st;
    if (Status ps = f.value->prefill(ids.data(), n, 0, n, lg_gpu.data(), &st); !ps) return fail_input(ps);
    best = std::min(best, st.seconds);
    std::printf("  GPU prefill call %llu: %.3f s (%.1f tokens/s), weight uploads %.3f s for %.2f GB (%.2f GB/s), compute and the rest %.3f s\n",
                static_cast<unsigned long long>(r + 1), st.seconds, n / st.seconds, st.upload_seconds, static_cast<double>(st.bytes_uploaded) / 1e9,
                st.upload_seconds > 0 ? static_cast<double>(st.bytes_uploaded) / 1e9 / st.upload_seconds : 0.0, st.seconds - st.upload_seconds);
  }
  std::printf("  GPU prefill best: %.3f s = %.1f tokens/s (weights streamed layer by layer: the copies bound it; GP3-14's resident bank removes them)\n", best,
              n / best);
  if (a.has("no-cpu")) return 0;
  model::ForwardOptions o = model::GpuForward::matching_reference_options();
  o.threads = static_cast<int>(thr);
  o.fast_cpu = a.has("fast-cpu");
  lg_ref.resize(lg_gpu.size());
  Timer tcpu;
  if (Status rs = model::forward_reference(s, w.value, ids.data(), n, 0, n, lg_ref.data(), o); !rs) return fail_input(rs);
  const double cpu_s = tcpu.seconds();
  auto compare = [&](const std::vector<float>& ra, const std::vector<float>& rb) { return compare_logits(ra, rb, n, nv); };
  auto show = [&](const char* label, const Cmp& c) {
    std::printf("  %s: logits rel L2 %.3e, largest |diff| %.3e; KL mean %.3e max %.3e; top-1 equal %d / %d\n", label, c.rel, c.worst, c.kl_mean, c.kl_max,
                c.top, n);
  };
  const Cmp main_cmp = compare(lg_ref, lg_gpu);
  std::printf("  CPU reference (same rounding%s): %.2f s = %.2f tokens/s\n", o.fast_cpu ? ", fast CPU kernels" : "", cpu_s, n / cpu_s);
  show("GPU vs reference", main_cmp);
  std::printf("  GPU %.1fx the reference's speed\n", cpu_s / best);
  // --floor: the same rounding through the other CPU kernels (fast <-> plain). The two differ only in fp32 summation order, as the GPU
  // does, so their gap is the floor any second implementation of this rounding meets: int8 rounding flips near boundaries, carried
  // through the layers.
  if (a.has("floor")) {
    model::ForwardOptions o2 = o;
    o2.fast_cpu = !o.fast_cpu;
    std::vector<float> lg_alt(lg_gpu.size());
    Timer t2;
    if (Status rs = model::forward_reference(s, w.value, ids.data(), n, 0, n, lg_alt.data(), o2); !rs) return fail_input(rs);
    std::printf("  floor: CPU reference again through the %s kernels: %.2f s\n", o2.fast_cpu ? "fast" : "plain", t2.seconds());
    show("reference vs the other CPU kernels (floor)", compare(lg_ref, lg_alt));
    show("GPU vs the other CPU kernels", compare(lg_alt, lg_gpu));
  }
  // --float-ref: the model without activation rounding (the default options: fp32 activations into every product). How far each
  // rounded path sits from it says whether the GPU is as close to the model as the CPU reference is.
  if (a.has("float-ref")) {
    model::ForwardOptions o3;
    o3.threads = o.threads;
    o3.fast_cpu = o.fast_cpu;
    std::vector<float> lg_f(lg_gpu.size());
    Timer t3;
    if (Status rs = model::forward_reference(s, w.value, ids.data(), n, 0, n, lg_f.data(), o3); !rs) return fail_input(rs);
    std::printf("  float reference (no activation rounding%s): %.2f s\n", o3.fast_cpu ? ", fast CPU kernels" : "", t3.seconds());
    const Cmp rf = compare(lg_f, lg_ref), gf = compare(lg_f, lg_gpu);
    show("rounded CPU reference vs float", rf);
    show("GPU vs float", gf);
    // the real-model gate (author 2026-10-31 00:5x, DECISIONS_LOG): the GPU as close to the unrounded model as the CPU reference,
    // KL within 1.25x and top-1 within 3 tokens. Two implementations of the same rounding differ by flips the layers carry (v16:
    // CPU fast vs plain KL 5.3e-3 at 40 layers), so a fixed bound on GPU vs reference holds only for one layer and the small rungs.
    const bool ok = gf.kl_mean <= 1.25 * rf.kl_mean && gf.top >= rf.top - 3;
    std::printf("  gate (GPU vs float within 1.25x KL and 3 top-1 of the rounded reference vs float): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 3;
  }
  // without --float-ref: the fixed bound, meant for one layer (KL mean 1e-4, top-1 95 %)
  return main_cmp.kl_mean <= 1e-4 && main_cmp.top * 100 >= n * 95 ? 0 : 3;
}

// GP3-14 stages 1-2: the engine (resident dense weights, the hot-expert bank, CPU cold experts, the cache carried between
// steps) over made-up tokens: prefill in chunks, then decode one token at a time; speed, the bank's share of the picks,
// and the logits of every position against the CPU reference with the same roundings (and, with --float-ref, against the
// unrounded model: the real-model gate of DECISIONS_LOG 2026-10-31 00:5x).
// v19 helpers: a row's two best ids and their logit gap; the exact two-sided McNemar p for b vs c discordant pairs
struct Top2 {
  std::size_t i1 = 0, i2 = 0;
  double margin = 0.0;
};
Top2 top2(const float* r, std::size_t nv) {
  Top2 o;
  o.i1 = 0;
  o.i2 = nv > 1 ? 1 : 0;
  if (nv > 1 && r[1] > r[0]) std::swap(o.i1, o.i2);
  for (std::size_t i = 2; i < nv; ++i) {
    if (r[i] > r[o.i1]) {
      o.i2 = o.i1;
      o.i1 = i;
    } else if (r[i] > r[o.i2]) {
      o.i2 = i;
    }
  }
  o.margin = static_cast<double>(r[o.i1]) - static_cast<double>(r[o.i2]);
  return o;
}
double mcnemar_p(int b, int c) {
  const int nn = b + c, k = std::min(b, c);
  if (nn == 0) return 1.0;
  double sum = 0.0;
  for (int i = 0; i <= k; ++i) sum += std::exp(std::lgamma(nn + 1.0) - std::lgamma(i + 1.0) - std::lgamma(nn - i + 1.0) - nn * std::log(2.0));
  return std::min(1.0, 2.0 * sum);
}
double kl_rows(const float* ra, const float* rb, std::size_t nv) {
  double ma = ra[0], mb = rb[0];
  for (std::size_t i = 0; i < nv; ++i) {
    ma = std::max(ma, static_cast<double>(ra[i]));
    mb = std::max(mb, static_cast<double>(rb[i]));
  }
  double sa = 0.0, sb = 0.0;
  for (std::size_t i = 0; i < nv; ++i) {
    sa += std::exp(ra[i] - ma);
    sb += std::exp(rb[i] - mb);
  }
  const double la = ma + std::log(sa), lb = mb + std::log(sb);
  double k = 0.0;
  for (std::size_t i = 0; i < nv; ++i) {
    const double pa = ra[i] - la;
    k += std::exp(pa) * (pa - (rb[i] - lb));
  }
  return k;
}

// v23 (author 18:06): can the GPU read the cold experts in host memory (the dual bus for MTP drafts)? (1) register a
// region of the model's read-only file mapping for GPU access; (2) register an ordinary private copy of it; for each that
// registers, the GPU's read rate over it; (3) the same read over device memory for scale. Nothing is written to the file.
int cmd_pin_probe(const Args& a) {
  const std::string path = a.get("model");
  if (path.empty()) {
    std::fprintf(stderr, "usage: pelias pin-probe --model FILE [--mb 1024] [--joint] [--small] [--arena]\n");
    return 1;
  }
  std::uint64_t mb = 1024;
  if (a.has("mb")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("mb"), 65536);
    if (!r.ok() || r.value < 64) {
      std::fprintf(stderr, "pelias: --mb takes 64 .. 65536\n");
      return 1;
    }
    mb = r.value;
  }
  os::FileMap m;
  if (Status s = m.open(path); !s) return fail_input(s);
  const std::size_t bytes = static_cast<std::size_t>(mb) << 20;
  if (m.size() < bytes + (std::uint64_t{1} << 30)) {
    std::fprintf(stderr, "pelias: the file is smaller than the region asked for\n");
    return 1;
  }
  // v24 fix: the region must end inside the file (12 GiB from mid-file ran past the end of a 22.7 GB file: segfault)
  const std::uint64_t off = std::min<std::uint64_t>(m.size() / 2, m.size() - bytes) & ~std::uint64_t{4095};
  auto* region = const_cast<std::uint8_t*>(m.data() + off);
  std::printf("pin-probe: %s region of %llu MiB at offset %llu of a %.2f GB file\n", path.c_str(), static_cast<unsigned long long>(mb),
              static_cast<unsigned long long>(off), static_cast<double>(m.size()) / 1e9);
  Timer tw;
  volatile std::uint8_t sink = 0;
  for (std::size_t i = 0; i < bytes; i += 4096) sink = static_cast<std::uint8_t>(sink ^ region[i]);  // pages resident first
  std::printf("  touch (page in): %.2f s\n", tw.seconds());
  auto rate = [&](const void* dev, const char* what) {
    for (int rep = 0; rep < 3; ++rep) {
      double sec = 0.0;
      if (!kern::gpu::read_rate(dev, bytes, 4096, &sec)) {
        std::printf("  %s: read kernel failed: %s\n", what, kern::gpu::last_error());
        return;
      }
      std::printf("  %s: read %llu MiB in %.4f s = %.2f GB/s\n", what, static_cast<unsigned long long>(mb), sec, static_cast<double>(bytes) / sec / 1e9);
    }
  };
  // v24: scattered expert-sized reads in a registered region (one block per chunk, offsets from a fixed LCG, 4 KiB
  // aligned), as the reader's routed experts would be read
  auto scattered = [&](void* dev, const char* tag) {
    for (const std::size_t kb : {std::size_t{256}, std::size_t{1024}, std::size_t{2048}}) {
      const std::size_t chunk = kb << 10;
      if (chunk * 2 > bytes) continue;
      for (const int nchunks : {80, 320}) {
        std::vector<std::uint64_t> offs(static_cast<std::size_t>(nchunks));
        std::uint64_t st = 0x9e3779b97f4a7c15ull ^ (kb * 131 + static_cast<std::uint64_t>(nchunks));
        for (auto& of : offs) {
          st = st * 6364136223846793005ull + 1442695040888963407ull;
          of = ((st >> 17) % ((bytes - chunk) / 4096)) * 4096;
        }
        for (int rep = 0; rep < 2; ++rep) {
          double sec = 0.0;
          if (!kern::gpu::read_chunks_rate(dev, offs.data(), nchunks, chunk, &sec)) {
            std::printf("  %s scattered read failed: %s\n", tag, kern::gpu::last_error());
            return;
          }
          std::printf("  %s scattered: %d chunks of %zu KiB in %.4f s = %.2f GB/s\n", tag, nchunks, kb, sec, static_cast<double>(chunk) * nchunks / sec / 1e9);
        }
      }
    }
  };
  // v46 (author 19:34 "probe it"): --small asks for 4 KB pages on the region first, as the engine does for its registered
  // ranges since v34 (hal::advise_small_pages), so (1) measures the engine's own case
  if (a.has("small")) std::printf("  small-page advice on the file region: %s\n", hal::advise_small_pages(region, bytes) ? "taken" : "refused");
  {  // (1) the file mapping itself
    void* dev = nullptr;
    std::string why;
    Timer tr;
    const bool ok = kern::gpu::host_register(region, bytes, &dev, &why);
    const double reg_s = tr.seconds();
    std::printf("  (1) register the read-only file mapping: %s (%.3f s = %.3f s/GiB)%s%s\n", ok ? "OK" : "REFUSED", reg_s, reg_s * 1024.0 / static_cast<double>(mb),
                ok ? "" : ": ", why.c_str());
    if (ok) {
      rate(dev, "(1) GPU reads the file mapping");
      scattered(dev, "(1s)");
      if (a.has("joint")) {
        // v28 joint read test (author 22:18 "roll it"): the GPU reads expert-sized chunks of the registered region while
        // CPU threads stream a second, unregistered region of the same file mapping (the light experts' path), each
        // alone and then together, for about 2 s per cell
        const std::uint64_t offb = off >= bytes ? off - bytes : off + bytes;
        const std::uint8_t* cpu_reg = m.data() + offb;
        if (offb + bytes > m.size()) {
          std::printf("  (J) no room for the CPU region\n");
        } else {
          Timer tt;
          for (std::size_t i = 0; i < bytes; i += 4096) sink = static_cast<std::uint8_t>(sink ^ cpu_reg[i]);
          std::printf("  (J) CPU region of %llu MiB at offset %llu paged in: %.2f s\n", static_cast<unsigned long long>(mb),
                      static_cast<unsigned long long>(offb), tt.seconds());
          const std::size_t chunk = std::size_t{2} << 20;
          const int nchunks = 320;
          std::vector<std::uint64_t> offs(static_cast<std::size_t>(nchunks));
          std::uint64_t st = 0x2545f4914f6cdd1dull;
          for (auto& of : offs) {
            st = st * 6364136223846793005ull + 1442695040888963407ull;
            of = ((st >> 17) % ((bytes - chunk) / 4096)) * 4096;
          }
          auto gpu_run = [&](double secs, double* gbs) {
            double tot = 0.0, bytes_done = 0.0;
            while (tot < secs) {
              double sec = 0.0;
              if (!kern::gpu::read_chunks_rate(dev, offs.data(), nchunks, chunk, &sec)) return false;
              tot += sec;
              bytes_done += static_cast<double>(chunk) * nchunks;
            }
            *gbs = bytes_done / tot / 1e9;
            return true;
          };
          // one CPU reader thread: random 2 MiB chunks of the CPU region, summed (4 accumulators), until stop
          auto cpu_body = [&](int t, const std::atomic<bool>& stop, double* bytes_read, std::uint64_t* xr) {
            std::uint64_t s0 = 0x9e3779b97f4a7c15ull * (static_cast<std::uint64_t>(static_cast<unsigned>(t)) + 1u);
            std::uint64_t x = 0;
            double n = 0.0;
            while (!stop.load(std::memory_order_relaxed)) {
              s0 = s0 * 6364136223846793005ull + 1442695040888963407ull;
              const std::uint64_t o = ((s0 >> 17) % ((bytes - chunk) / 4096)) * 4096;
              const auto* q = reinterpret_cast<const std::uint64_t*>(cpu_reg + o);
              std::uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
              for (std::size_t i = 0; i < chunk / 8; i += 4) {
                a0 += q[i];
                a1 += q[i + 1];
                a2 += q[i + 2];
                a3 += q[i + 3];
              }
              x += a0 ^ a1 ^ a2 ^ a3;
              n += static_cast<double>(chunk);
            }
            *bytes_read = n;
            *xr = x;
          };
          // CPU threads run while `during` runs (the GPU loop, or a 2 s sleep); returns the CPU rate in GB/s
          auto with_cpu = [&](int nt, const std::function<void()>& during) {
            std::atomic<bool> stop{false};
            std::vector<double> got(static_cast<std::size_t>(nt), 0.0);
            std::vector<std::uint64_t> xs(static_cast<std::size_t>(nt), 0);
            std::vector<std::thread> th;
            Timer tc;
            for (int t = 0; t < nt; ++t) th.emplace_back(cpu_body, t, std::cref(stop), &got[static_cast<std::size_t>(t)], &xs[static_cast<std::size_t>(t)]);
            during();
            stop = true;
            for (auto& h : th) h.join();
            const double secs = tc.seconds();
            double n = 0.0;
            for (double g : got) n += g;
            for (std::uint64_t v : xs) sink = static_cast<std::uint8_t>(sink ^ v);
            return n / secs / 1e9;
          };
          double g_alone = 0.0;
          if (!gpu_run(2.0, &g_alone)) {
            std::printf("  (J) GPU read failed: %s\n", kern::gpu::last_error());
          } else {
            std::printf("  (J) GPU alone: 2 MiB chunks x %d from the registered mapping: %.2f GB/s\n", nchunks, g_alone);
            for (const int nt : {2, 4, 8}) {
              double c_alone = 0.0, c_joint = 0.0, g_joint = 0.0;
              c_alone = with_cpu(nt, [] { std::this_thread::sleep_for(std::chrono::milliseconds(2000)); });
              bool gok = true;
              c_joint = with_cpu(nt, [&] { gok = gpu_run(2.0, &g_joint); });
              if (!gok) std::printf("  (J) GPU read failed: %s\n", kern::gpu::last_error());
              std::printf("  (J) CPU %d threads alone %.2f GB/s | together: CPU %.2f GB/s + GPU %.2f GB/s = %.2f GB/s (CPU %+.1f %%, GPU %+.1f %%)\n", nt,
                          c_alone, c_joint, g_joint, c_joint + g_joint, 100.0 * (c_joint / c_alone - 1.0), 100.0 * (g_joint / g_alone - 1.0));
            }
          }
        }
      }
      kern::gpu::host_unregister(region);
    }
  }
  if (a.has("arena")) {  // (4) v46: the same bytes in a 2 MB-page anonymous arena filled by direct reads (one copy in RAM)
    bool direct = false;
    std::string why;
    Timer tf;
    void* ar = hal::huge_arena_from_file(path.c_str(), off, bytes, &direct, &why);
    if (!ar) {
      std::printf("  (4) arena: FAILED: %s\n", why.c_str());
    } else {
      std::printf("  (4) 2 MB arena filled from the file (%s) in %.2f s; %s\n", direct ? "O_DIRECT: no page-cache copy" : "read through the page cache",
                  tf.seconds(), hal::smaps_rollup_line("AnonHugePages:").c_str());
      void* dev = nullptr;
      std::string why2;
      Timer tr;
      const bool ok = kern::gpu::host_register(ar, bytes, &dev, &why2);
      const double reg_s = tr.seconds();
      std::printf("  (4) register the arena: %s (%.3f s = %.3f s/GiB)%s%s; %s\n", ok ? "OK" : "REFUSED", reg_s, reg_s * 1024.0 / static_cast<double>(mb),
                  ok ? "" : ": ", why2.c_str(), hal::smaps_rollup_line("AnonHugePages:").c_str());
      if (ok) {
        rate(dev, "(4) GPU reads the 2 MB arena");
        scattered(dev, "(4s)");
        kern::gpu::host_unregister(ar);
      }
      hal::huge_arena_free(ar, bytes);
    }
  }
  {  // (2) a private copy
    std::vector<std::uint8_t> copy(region, region + bytes);
    void* dev = nullptr;
    std::string why;
    Timer tr;
    const bool ok = kern::gpu::host_register(copy.data(), bytes, &dev, &why);
    std::printf("  (2) register a private copy: %s (%.3f s)%s%s\n", ok ? "OK" : "REFUSED", tr.seconds(), ok ? "" : ": ", why.c_str());
    if (ok) {
      rate(dev, "(2) GPU reads the pinned copy");
      kern::gpu::host_unregister(copy.data());
    }
  }
  {  // (3) device memory, for scale
    void* d = kern::gpu::dalloc(bytes);
    if (d && kern::gpu::to_device(d, region, bytes)) rate(d, "(3) GPU reads device memory");
    kern::gpu::dfree(d);
  }
  return 0;
}

int cmd_engine_check(const Args& a) {
  if (a.has("host-wait")) {  // v41: before any CUDA call
    const std::string hw = a.get("host-wait");
    const int mode = hw == "spin" ? 1 : hw == "yield" ? 2 : hw == "block" ? 3 : hw == "auto" ? 0 : -1;
    if (mode < 0) {
      std::fprintf(stderr, "pelias: --host-wait takes auto, spin, yield or block\n");
      return 1;
    }
    std::string why;
    const bool hw_ok = kern::gpu::set_host_wait(mode, &why);
    std::printf("pelias engine-check: host wait for the GPU: %s%s%s\n", hw.c_str(), hw_ok ? "" : " REFUSED: ", hw_ok ? "" : why.c_str());
  }
  const std::string path = a.get("model");
  if (path.empty()) {
    std::fprintf(stderr,
                 "usage: pelias engine-check --model FILE [--routes ROUTES.bin] [--prefill 64] [--decode 32] [--ctx 4096] [--batch 64] [--threads N]\n"
                 "                           [--budget-gb G] [--hot-max N] [--mtp-draft K] [--fast-cpu] [--no-cpu] [--float-ref]\n"
                 "                           [--text FILE [--skip-tokens N]] [--timeline] [--dump-top OUT.tsv] [--dump-router OUT.tsv] [--dump-draft OUT.tsv] [--dump-split [--split-sizes 2,3,4] [--split-step]]\n"
                 "                           [--no-doorbell] [--timeline-ops] [--split-ref] [--bank-selftest MIXES] [--probe-pos P]\n"
                 "                           [--router-tie BAND] [--cold-on-gpu] [--trace OUT.tsv] [--reader-v25] [--reader-cache K] [--fold-shared]\n"
                 "                           [--batch-identical] [--graph] [--prefetch] [--plane-tail N] [--cpu-light] [--reader-rpw 1|2] [--r3-experts] [--rx-load lane|miss|all] [--prompt-batched] [--prompt-halves] [--cpu-prompt] [--no-cache-pools] [--no-fast-f32] [--no-fused-moe] [--plane-bits 2|3|4] [--auto-vram [--cache-share F]]\n"
                 "                           [--pin-gap-kib K] [--pin-huge-ok] [--cpu-up-rows N] [--cpu-down2] [--cpu-gu2] [--cpu-tile] [--spin-pause] [--cpu-prefetch] [--cpu-check] [--host-wait auto|spin|yield|block] [--pin-cpus 0,2,4,...] [--pin-host N] [--ref-threads N] [--no-cpu-light] [--split-slow] [--preload-light]\n");
    return 1;
  }
  auto num = [&](const char* k, std::uint64_t d, std::uint64_t lo, std::uint64_t hi, std::uint64_t& out) {
    out = d;
    if (!a.has(k)) return true;
    const Result<std::uint64_t> r = io::parse_whole(a.get(k), hi);
    if (!r.ok() || r.value < lo) {
      std::fprintf(stderr, "pelias: --%s takes a whole number from %llu to %llu\n", k, static_cast<unsigned long long>(lo), static_cast<unsigned long long>(hi));
      return false;
    }
    out = r.value;
    return true;
  };
  std::uint64_t npre = 0, ndec = 0, ctx = 0, batch = 0, thr = 0, hmax = 0, ndraft = 0, skip = 0;
  if (!num("prefill", 64, 1, 1 << 20, npre) || !num("decode", 32, 0, 1 << 20, ndec) || !num("ctx", 4096, 2, 1 << 20, ctx) || !num("batch", 64, 1, 4096, batch) ||
      !num("threads", 0, 0, 4096, thr) || !num("hot-max", 256, 0, 4096, hmax) || !num("mtp-draft", 0, 0, 16, ndraft) ||
      !num("skip-tokens", 0, 0, std::uint64_t{1} << 40, skip))
    return 1;
  const std::string dump_top = a.get("dump-top");
  if (!dump_top.empty()) {
    if (!a.has("float-ref") || a.has("no-cpu")) {
      std::fprintf(stderr, "pelias: --dump-top needs --float-ref (and the CPU reference)\n");
      return 1;
    }
    if (std::ifstream(dump_top).good()) {
      std::fprintf(stderr, "pelias: %s exists (nothing is overwritten)\n", dump_top.c_str());
      return 1;
    }
  }
  const std::string dump_router = a.get("dump-router");  // v49: per position, every layer's router margins
  const std::string dump_draft = a.get("dump-draft");    // v49b: per position, the MTP draft's top-2 for the same prediction
  if (!dump_draft.empty() && std::ifstream(dump_draft).good()) {
    std::fprintf(stderr, "pelias: %s exists (nothing is overwritten)\n", dump_draft.c_str());
    return 1;
  }
  if (!dump_router.empty() && std::ifstream(dump_router).good()) {
    std::fprintf(stderr, "pelias: %s exists (nothing is overwritten)\n", dump_router.c_str());
    return 1;
  }
  if (npre + ndec > ctx) {
    std::fprintf(stderr, "pelias: --prefill + --decode must fit --ctx\n");
    return 1;
  }
  std::uint64_t budget = 0;
  if (!gib_option(a, "budget-gb", budget)) return 1;
  os::FileMap m;
  if (Status s = m.open(path); !s) return fail_input(s);
  io::GgufReader g;
  if (Status s = g.parse(m); !s) return fail_input(s);
  const Result<model::Qwen35MoeSpec> spec = model::read_qwen35moe_spec(g);
  if (!spec.ok()) return fail_input(spec.status);
  Result<model::Qwen35MoeWeights> w = model::bind_qwen35moe(g, m.data(), spec.value);
  if (!w.ok()) return fail_input(w.status);
  const model::Qwen35MoeSpec& s = spec.value;
  model::EngineOptions eo;
  eo.max_ctx = static_cast<int>(ctx);
  eo.max_batch = static_cast<int>(batch);
  eo.cpu_threads = static_cast<int>(thr);
  eo.vram_budget = budget;
  eo.max_hot_per_layer = static_cast<int>(hmax);
  eo.mtp = ndraft > 0 || !a.get("dump-draft").empty();  // v49b: the draft dump needs the MTP block
  eo.max_verify = static_cast<int>(ndraft) + 1;
  eo.timeline = a.has("timeline");
  eo.router_margins = !a.get("dump-router").empty();  // v49
  eo.doorbell = !a.has("no-doorbell");
  eo.timeline_ops = a.has("timeline-ops");
  // v24: the tie rule's band (the engine and the rounded CPU reference; the float reference keeps the plain order) and the
  // picked-expert reader
  float tie_band = 0.0f;
  if (a.has("router-tie")) {
    const std::string tv = a.get("router-tie");
    char* endp = nullptr;
    tie_band = std::strtof(tv.c_str(), &endp);
    if (tv.empty() || !endp || *endp != '\0' || !(tie_band >= 0.0f && tie_band <= 1.0f)) {
      std::fprintf(stderr, "pelias: --router-tie takes a band from 0 to 1\n");
      return 1;
    }
  }
  eo.router_tie = tie_band;
  eo.cold_on_gpu = a.has("cold-on-gpu");
  // v26 switches (each measured on its own in the run)
  eo.reader_v3 = !a.has("reader-v25");
  eo.fold_shared = a.has("fold-shared");
  eo.r3_experts = a.has("r3-experts");  // v46a-1
  eo.gpu_prompt = !a.has("cpu-prompt");  // v49e: the GPU prompt by default (fix forward); --cpu-prompt / --prompt-batched are temporary diagnosis switches
  eo.prompt_r3 = !a.has("prompt-batched");
  eo.prompt_halves = a.has("prompt-halves");  // v50, parked at v51 (temporary switch)                       // v49d: the GPU prompt's steps on the batched kernels
  if (a.has("rx-load")) {  // v46a-2
    const std::string rxl = a.get("rx-load");
    if (rxl == "lane") eo.rx_load = 0;
    else if (rxl == "miss") eo.rx_load = 1;
    else if (rxl == "all") eo.rx_load = 2;
    else {
      std::fprintf(stderr, "pelias: --rx-load takes lane, miss or all\n");
      return 1;
    }
  }
  eo.batch_identical = a.has("batch-identical");
  eo.graph = a.has("graph");
  eo.prefetch = a.has("prefetch");
  if (a.has("plane-tail")) {
    const Result<std::uint64_t> pt = io::parse_whole(a.get("plane-tail"), 8);
    if (!pt.ok()) {
      std::fprintf(stderr, "pelias: --plane-tail takes 0 .. 8 picks\n");
      return 1;
    }
    eo.plane_tail = static_cast<int>(pt.value);
  }
  if (a.has("reader-cache")) {
    const Result<std::uint64_t> rc = io::parse_whole(a.get("reader-cache"), 64);
    if (!rc.ok()) {
      std::fprintf(stderr, "pelias: --reader-cache takes slots per layer, 0 .. 64\n");
      return 1;
    }
    eo.reader_cache = static_cast<int>(rc.value);
  }
  // v28: the author's split (light experts on the CPU, heavy pinned for the GPU), rows per warp, the pin merge gap
  // v28b (author 23:14) made the CPU split the default. v46 step 1 (THE PLAN, author 2026-11-01 17:46-17:54): the GPU does
  // all of the decoding again; the split (light experts on the CPU) only with --cpu-light. --no-cpu-light is still accepted.
  eo.cpu_light = a.has("cpu-light") && !a.has("no-cpu-light");
  eo.split_slow = a.has("split-slow");
  eo.preload_light = a.has("preload-light");
  if (a.has("reader-rpw")) {
    const Result<std::uint64_t> rw = io::parse_whole(a.get("reader-rpw"), 2);
    if (!rw.ok() || rw.value < 1) {
      std::fprintf(stderr, "pelias: --reader-rpw takes 1 or 2\n");
      return 1;
    }
    eo.reader_rows_per_warp = static_cast<int>(rw.value);
  }
  if (a.has("pin-gap-kib")) {
    const Result<std::uint64_t> pgk = io::parse_whole(a.get("pin-gap-kib"), 1u << 20);
    if (!pgk.ok()) {
      std::fprintf(stderr, "pelias: --pin-gap-kib takes 0 (exact ranges) .. 1048576\n");
      return 1;
    }
    eo.pin_gap = pgk.value << 10;
  }
  if (a.has("pin-huge-ok")) eo.pin_small_pages = false;  // v34: leave the registered ranges on 2 MB pages (the v33 stall)
  if (a.has("cpu-up-rows")) {  // v40
    const Result<std::uint64_t> ur = io::parse_whole(a.get("cpu-up-rows"), 512);
    if (!ur.ok() || ur.value < 8) {
      std::fprintf(stderr, "pelias: --cpu-up-rows takes 8 .. 512\n");
      return 1;
    }
    eo.cpu_up_rows = static_cast<int>(ur.value);
  }
  eo.cpu_down2 = a.has("cpu-down2");
  eo.cpu_prefetch = a.has("cpu-prefetch");
  eo.cpu_gu2 = a.has("cpu-gu2");
  eo.cpu_tile = a.has("cpu-tile");
  eo.spin_pause = a.has("spin-pause");
  eo.cpu_check = a.has("cpu-check");
  if (a.has("pin-cpus")) {  // v43: comma list; the first is the doorbell thread, the rest the pool workers in order
    std::string lst = a.get("pin-cpus");
    std::size_t at = 0;
    while (at <= lst.size()) {
      const std::size_t comma = lst.find(',', at);
      const std::string one = lst.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
      const Result<std::uint64_t> c1 = io::parse_whole(one, 1023);
      if (!c1.ok()) {
        std::fprintf(stderr, "pelias: --pin-cpus takes logical CPU numbers separated by commas\n");
        return 1;
      }
      eo.pin_pool.push_back(static_cast<int>(c1.value));
      if (comma == std::string::npos) break;
      at = comma + 1;
    }
  }
  if (a.has("pin-host")) {
    const Result<std::uint64_t> h1 = io::parse_whole(a.get("pin-host"), 1023);
    if (!h1.ok()) {
      std::fprintf(stderr, "pelias: --pin-host takes one logical CPU number\n");
      return 1;
    }
    eo.pin_host = static_cast<int>(h1.value);
  }

  // v25 trace: a new file (nothing is overwritten); the reader's steps, per layer
  const std::string trace_path = a.get("trace");
  if (!trace_path.empty()) {
    if (std::ifstream(trace_path).good()) {
      std::fprintf(stderr, "pelias: %s exists (nothing is overwritten)\n", trace_path.c_str());
      return 1;
    }
    eo.trace = true;
  }
  // v20: the split check's batch sizes (default 1 + drafts)
  std::vector<int> split_sizes;
  if (a.has("split-sizes")) {
    std::stringstream ss(a.get("split-sizes"));
    std::string item;
    while (std::getline(ss, item, ',')) {
      const Result<std::uint64_t> r = io::parse_whole(item, 64);
      if (!r.ok() || r.value < 2) {
        std::fprintf(stderr, "pelias: --split-sizes takes batch sizes from 2 to 64, comma separated\n");
        return 1;
      }
      split_sizes.push_back(static_cast<int>(r.value));
      eo.max_verify = std::max(eo.max_verify, static_cast<int>(r.value));
    }
  } else {
    split_sizes.push_back(static_cast<int>(ndraft) + 1);
  }
  if (eo.max_verify > static_cast<int>(batch)) {
    std::fprintf(stderr, "pelias: the split sizes and drafts need --batch %d or more\n", eo.max_verify);
    return 1;
  }
  if (a.has("routes")) {
    auto ho = model::hot_order_from_routes(a.get("routes"), s.n_layer, s.n_expert, &eo.hot_count);
    if (!ho.ok()) return fail_input(ho.status);
    eo.hot_order = std::move(ho.value);
  }
  eo.cache_pools = !a.has("no-cache-pools");  // v51 (temporary switch)
  eo.fast_f32 = !a.has("no-fast-f32");        // v52 (temporary switch)
  eo.fused_moe = !a.has("no-fused-moe");      // v52 (temporary switch)
  if (a.has("plane-bits")) {  // v54b
    const Result<std::uint64_t> pb = io::parse_whole(a.get("plane-bits"), 4);
    if (!pb.ok() || pb.value < 2) {
      std::fprintf(stderr, "pelias: --plane-bits 2, 3 or 4\n");
      return 1;
    }
    eo.plane_bits = static_cast<int>(pb.value);
  }
  eo.auto_vram = a.has("auto-vram");  // v54b
  if (a.has("cache-share")) {
    char* end = nullptr;
    const std::string cs = a.get("cache-share");
    const double v = std::strtod(cs.c_str(), &end);
    if (!end || *end != 0 || !(v > 0.0 && v < 1.0)) {
      std::fprintf(stderr, "pelias: --cache-share takes a fraction between 0 and 1\n");
      return 1;
    }
    eo.cache_share = v;
  }
  Timer topen;
  auto eng = model::GpuEngine::open(s, w.value, eo);
  if (!eng.ok()) return fail_input(eng.status);
  std::printf("pelias engine-check: %s, %u layers; open %.2f s\n  %s\n", s.arch.c_str(), s.n_layer, topen.seconds(), eng.value->summary().c_str());
  // v25 trace: each phase's lines under a "# phase" line; whatever is left is written when engine-check returns
  auto tflush = [&](const std::string& phase) {
    if (trace_path.empty()) return;
    const std::string txt = eng.value->trace_take();
    if (txt.empty()) return;
    std::ofstream tf(trace_path, std::ios::app);
    tf << "# phase: " << phase << "\n" << txt;
  };
  struct TraceAtExit {
    std::function<void(const std::string&)> f;
    ~TraceAtExit() { f("remaining steps (gate decode / MTP)"); }
  } trace_at_exit{tflush};
  if (!trace_path.empty()) {
    std::ofstream tf(trace_path);
    tf << "# pelias engine-check trace (v25). S: step seq, tokens, wall ms, unix ms. L: step seq, layer, experts read (GPU count), experts read (host count), "
          "picks from the bank, picks from the file, bytes from VRAM, bytes from the file, reader us: grouping, gate/up, silu, down, combine; picks (tokens split by ;)\n"
       << "# " << eng.value->summary() << "\n";
  }
  int selftest_bad = 0;
  if (a.has("bank-selftest")) {  // v22: every bank expert against the CPU path, before any step (MIXES random 8-pick mixes per layer)
    const Result<std::uint64_t> mr = io::parse_whole(a.get("bank-selftest"), 100000);
    if (!mr.ok()) {
      std::fprintf(stderr, "pelias: --bank-selftest takes a number of mixes per layer (0 for none)\n");
      return 1;
    }
    std::string rep;
    Timer tst;
    if (Status st = eng.value->bank_selftest(static_cast<int>(mr.value), &rep, &selftest_bad); !st) return fail_input(st);
    std::printf("  %s (%.1f s)\n", rep.c_str(), tst.seconds());
  }
  const int n = static_cast<int>(npre + ndec);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(n));
  if (a.has("text")) {  // v19: real text, tokens skip .. skip + n of the file (decode steps are fed the text's next tokens)
    Result<tok::Tokenizer> tk = tok::Tokenizer::from_gguf(g);
    if (!tk.ok()) return fail_input(tk.status);
    Result<std::string> txt = io::read_text_file(a.get("text"));
    if (!txt.ok()) return fail_input(txt.status);
    tok::EncodeOptions eopt;
    const std::vector<std::int32_t> all = tk.value.encode(txt.value, eopt);
    if (all.size() < skip + static_cast<std::uint64_t>(n)) {
      std::fprintf(stderr, "pelias: the text has %zu tokens; --skip-tokens %llu + %d do not fit\n", all.size(), static_cast<unsigned long long>(skip), n);
      return 1;
    }
    for (int t = 0; t < n; ++t) ids[static_cast<std::size_t>(t)] = all[static_cast<std::size_t>(skip) + static_cast<std::size_t>(t)];
    std::printf("  text: %zu tokens in the file; positions %llu .. %llu used\n", all.size(), static_cast<unsigned long long>(skip), static_cast<unsigned long long>(skip) + static_cast<unsigned long long>(n) - 1);
  } else {
    for (int t = 0; t < n; ++t) ids[static_cast<std::size_t>(t)] = static_cast<std::int32_t>((static_cast<std::uint64_t>(t) * 7919u + 1011u) % s.n_vocab);
  }
  const std::size_t nv = s.n_vocab;
  std::vector<float> lg(static_cast<std::size_t>(n) * nv);
  model::EngineStats tot_pre, tot_dec;
  auto add = [](model::EngineStats& a2, const model::EngineStats& b2) {
    a2.seconds += b2.seconds;
    a2.cold_seconds += b2.cold_seconds;
    a2.wait_seconds += b2.wait_seconds;
    a2.hot_picks += b2.hot_picks;
    a2.cold_picks += b2.cold_picks;
    a2.cold_bytes += b2.cold_bytes;
  };
  int t = 0;
  const std::size_t ne_ = s.n_embd;
  std::vector<float> hid(dump_draft.empty() ? 0 : static_cast<std::size_t>(n) * ne_);  // v49b: every position's post-norm hidden
  auto hrow = [&](int p) { return hid.empty() ? nullptr : hid.data() + static_cast<std::size_t>(p) * ne_; };
  while (t < static_cast<int>(npre)) {
    const int k = std::min(static_cast<int>(batch), static_cast<int>(npre) - t);
    model::EngineStats es;
    if (Status st = eng.value->step(ids.data() + t, k, lg.data() + static_cast<std::size_t>(t) * nv, false, &es, hrow(t)); !st) return fail_input(st);
    add(tot_pre, es);
    t += k;
  }
  std::vector<double> per;
  eng.value->timeline_reset();
  {  // v30: the light experts' pre-load finishes before the measured decode (so the decode is warm)
    double pls = 0.0;
    std::uint64_t plb = 0;
    if (eng.value->preload_wait(&pls, &plb))
      std::printf("  pre-load of the light experts: %.2f GB touched in place in %.2f s (%.2f GB/s, background from start-up)\n", static_cast<double>(plb) / 1e9, pls,
                  pls > 0 ? static_cast<double>(plb) / 1e9 / pls : 0.0);
  }
#if defined(__linux__)
  struct rusage ru0 {};
  getrusage(RUSAGE_SELF, &ru0);
#endif
  for (; t < n; ++t) {
    model::EngineStats es;
    if (Status st = eng.value->step(ids.data() + t, 1, lg.data() + static_cast<std::size_t>(t) * nv, true, &es, hrow(t)); !st) return fail_input(st);
    add(tot_dec, es);
    per.push_back(es.seconds);
  }
  if (!dump_draft.empty()) {  // v49b: the MTP block at position p (id p with the trunk's hidden of p - 1, zeros at 0) predicts the
    // same next token as the trunk's row p; its top-2 and margin are known before the trunk runs p (the early signal)
    std::FILE* fd = std::fopen(dump_draft.c_str(), "w");
    if (!fd) {
      std::fprintf(stderr, "pelias: cannot write %s\n", dump_draft.c_str());
      return 1;
    }
    std::fprintf(fd, "pos\td1\td2\td_margin\teng1\teng_margin\n");
    const int kc = 4;
    std::vector<float> hp(static_cast<std::size_t>(kc) * ne_), dl(static_cast<std::size_t>(kc) * nv);
    for (int p0 = 0; p0 < n; p0 += kc) {
      const int k = std::min(kc, n - p0);
      for (int j = 0; j < k; ++j) {
        float* d = hp.data() + static_cast<std::size_t>(j) * ne_;
        if (p0 + j == 0) std::fill(d, d + ne_, 0.0f);
        else std::copy(hrow(p0 + j - 1), hrow(p0 + j - 1) + ne_, d);
      }
      if (Status st = eng.value->mtp_step(ids.data() + p0, hp.data(), k, dl.data(), nullptr); !st) return fail_input(st);
      for (int j = 0; j < k; ++j) {
        const Top2 td = top2(dl.data() + static_cast<std::size_t>(j) * nv, nv), te = top2(lg.data() + static_cast<std::size_t>(p0 + j) * nv, nv);
        std::fprintf(fd, "%d\t%zu\t%zu\t%.6f\t%zu\t%.6f\n", p0 + j, td.i1, td.i2, td.margin, te.i1, te.margin);
      }
    }
    std::fclose(fd);
    std::printf("  MTP draft margins: %d positions written to %s\n", n, dump_draft.c_str());
  }
  if (!dump_router.empty()) {  // v49: the main pass's router margins, one line per position (layer il: e8_il = 8th - 9th logit, t1_il = 1st - 2nd)
    std::vector<float> rm;
    std::vector<std::uint8_t> rh;
    if (!eng.value->router_margins(&rm, &rh)) {
      std::fprintf(stderr, "pelias: --dump-router: the router margins are off (needs the expert reader)\n");
      return 1;
    }
    std::FILE* fr = std::fopen(dump_router.c_str(), "w");
    if (!fr) {
      std::fprintf(stderr, "pelias: cannot write %s\n", dump_router.c_str());
      return 1;
    }
    const std::size_t nl = s.n_layer;
    std::fprintf(fr, "pos");
    for (std::size_t il = 0; il < nl; ++il) std::fprintf(fr, "\te8_%zu", il);
    for (std::size_t il = 0; il < nl; ++il) std::fprintf(fr, "\tt1_%zu", il);
    std::fprintf(fr, "\n");
    int rows = 0;
    for (int t2 = 0; t2 < n; ++t2) {
      if (!rh[static_cast<std::size_t>(t2)]) continue;
      const float* r = rm.data() + static_cast<std::size_t>(t2) * 2 * nl;
      std::fprintf(fr, "%d", t2);
      for (std::size_t il = 0; il < nl; ++il) std::fprintf(fr, "\t%.5f", r[2 * il]);
      for (std::size_t il = 0; il < nl; ++il) std::fprintf(fr, "\t%.5f", r[2 * il + 1]);
      std::fprintf(fr, "\n");
      ++rows;
    }
    std::fclose(fr);
    std::printf("  router margins: %d of %d positions written to %s\n", rows, n, dump_router.c_str());
  }
  auto share = [](const model::EngineStats& e2) {
    const double all = static_cast<double>(e2.hot_picks + e2.cold_picks);
    return all > 0 ? 100.0 * static_cast<double>(e2.hot_picks) / all : 0.0;
  };
  std::printf("  prefill %llu tokens: %.3f s = %.1f tokens/s; picks on the GPU %.1f %%; CPU cold work %.3f s, host waits %.3f s\n",
              static_cast<unsigned long long>(npre), tot_pre.seconds, static_cast<double>(npre) / tot_pre.seconds, share(tot_pre), tot_pre.cold_seconds,
              tot_pre.wait_seconds);
  if (ndec > 0) {
    std::vector<double> sorted = per;
    std::sort(sorted.begin(), sorted.end());
    const double med = sorted[sorted.size() / 2];
    std::printf("  decode %llu tokens: %.3f s = %.1f tokens/s (median step %.2f ms = %.1f tokens/s, slowest %.2f ms); picks on the GPU %.1f %%; CPU cold work %.2f ms, host waits %.2f ms per token\n",
                static_cast<unsigned long long>(ndec), tot_dec.seconds, static_cast<double>(ndec) / tot_dec.seconds, med * 1e3, 1.0 / med, sorted.back() * 1e3,
                share(tot_dec), tot_dec.cold_seconds * 1e3 / static_cast<double>(ndec), tot_dec.wait_seconds * 1e3 / static_cast<double>(ndec));
#if defined(__linux__)
    {  // v30: page faults during the decode (major: read from the file / drive; minor: mapped without reading)
      struct rusage ru1 {};
      getrusage(RUSAGE_SELF, &ru1);
      std::printf("  page faults in the decode: %.1f major and %.0f minor per token\n",
                  static_cast<double>(ru1.ru_majflt - ru0.ru_majflt) / static_cast<double>(ndec), static_cast<double>(ru1.ru_minflt - ru0.ru_minflt) / static_cast<double>(ndec));
    }
#endif
    if (tot_dec.cold_bytes > 0 && tot_dec.cold_seconds > 0.0)  // v28: the CPU's rate on its picks (bytes / its cold work time)
      std::printf("  CPU cold experts: %llu picks, %.1f MB per token, %.2f GB/s over its cold work time\n", static_cast<unsigned long long>(tot_dec.cold_picks),
                  static_cast<double>(tot_dec.cold_bytes) / 1e6 / static_cast<double>(ndec), static_cast<double>(tot_dec.cold_bytes) / tot_dec.cold_seconds / 1e9);
  }
  tflush("decode (the speed numbers above)");
  std::printf("  graph: %s\n", eng.value->graph_report().c_str());
  if (a.has("timeline-ops") && ndec > 0) {  // v21: per operation, one-token steps
    const model::EngineTimeline& T = eng.value->timeline();
    if (T.ops_steps == 0) {
      std::printf("  per-operation timeline: no steps recorded\n");
    } else {
      int nl[2] = {0, 0};
      for (std::uint32_t il = 0; il < s.n_layer; ++il) ++nl[w.value.layers[il].recurrent ? 0 : 1];
      std::printf("  per-operation timeline, %d one-token steps (stamps add a launch each; GPU clock), ms per token over %d DeltaNet + %d attention layers:\n",
                  T.ops_steps, nl[0], nl[1]);
      double tot[2] = {0.0, 0.0};
      for (int k = 1; k < model::EngineTimeline::kOps; ++k) {
        const double dn = 1e3 * T.ops[0][k] / T.ops_steps, at = 1e3 * T.ops[1][k] / T.ops_steps;
        tot[0] += dn;
        tot[1] += at;
        std::printf("    %-46s DeltaNet %7.3f (%.3f per layer)   attention %7.3f (%.3f per layer)\n", model::kOpNames[k], dn, nl[0] ? dn / nl[0] : 0.0, at,
                    nl[1] ? at / nl[1] : 0.0);
      }
      std::printf("    %-46s DeltaNet %7.3f   attention %7.3f   all %.3f ms per token\n", "sum", tot[0], tot[1], tot[0] + tot[1]);
    }
  }
  if (a.has("timeline") && ndec > 0) {
    const model::EngineTimeline& T = eng.value->timeline();
    if (T.steps == 0) {
      std::printf("  timeline: no steps recorded (needs the doorbell and mapped memory)\n");
    } else {
      const double k = 1e3 / T.steps;  // ms per token
      auto sum = [](const std::vector<double>& v) {
        double q = 0.0;
        for (double x : v) q += x;
        return q;
      };
      const double cold = sum(T.cpu_cold);
      const double picks = static_cast<double>(T.hot_picks + T.cold_picks);
      if (sum(T.gpu_dense) + sum(T.gpu_hot) + cold <= 0.0) {  // v25: reader steps only (no doorbell): the columns below do not apply
        std::printf("  timeline: %d decode steps, every one with the expert reader (no doorbell, no CPU cold work); see the per-operation lines and the trace\n", T.steps);
      } else {
      std::printf("  timeline, %d decode steps, ms per token (GPU clock): dense %.2f, bank + shared experts %.2f, idle waiting for the CPU %.2f\n", T.steps,
                  sum(T.gpu_dense) * k, sum(T.gpu_hot) * k, sum(T.gpu_idle) * k);
      std::printf("  timeline, ms per token (CPU clock): waiting for the ring %.2f, cold experts %.2f = input rounding %.2f + gate/up %.2f + silu %.2f + down %.2f + combine %.2f\n",
                  sum(T.cpu_wait) * k, cold * k, T.stage[0] * k, T.stage[1] * k, T.stage[2] * k, T.stage[3] * k, T.stage[4] * k);
      std::printf("  timeline: cold %.1f MB per token, %.1f GB/s over the cold time; thread use (item time / (wall x %d threads)) gate/up %.0f %%, down %.0f %%; first worker starts %.1f us after a job opens (mean of %d); picks on the GPU %.1f %%\n",
                  T.cold_bytes / T.steps / 1e6, cold > 0 ? T.cold_bytes / cold / 1e9 : 0.0, T.threads,
                  T.stage[1] > 0 ? 100.0 * T.busy[1] / (T.stage[1] * T.threads) : 0.0, T.stage[3] > 0 ? 100.0 * T.busy[3] / (T.stage[3] * T.threads) : 0.0,
                  T.wake_n ? 1e6 * T.wake / T.wake_n : 0.0, T.wake_n, picks > 0 ? 100.0 * static_cast<double>(T.hot_picks) / picks : 0.0);
      if (T.split_stamps)
        std::printf("  timeline (split path, v39): silu is inside gate/up and the combine inside down; the first item starts %.1f us after a layer's job opens\n",
                    T.wake_n ? 1e6 * T.wake / T.wake_n : 0.0);
      if (T.split_stamps && T.thr_n > 0) {  // v40: per thread, ms per token of item time
        double umin = 1e30, umax = 0.0, dmin = 1e30, dmax = 0.0;
        std::printf("  timeline (v40) per thread, ms per token, gate/up | down:");
        for (int q2 = 0; q2 < T.thr_n; ++q2) {
          const double u = T.thr_up[q2] * k, d = T.thr_dn[q2] * k;
          if (u <= 0.0 && d <= 0.0) continue;
          umin = std::min(umin, u); umax = std::max(umax, u); dmin = std::min(dmin, d); dmax = std::max(dmax, d);
          std::printf(" [%d] %.2f | %.2f", q2, u, d);
        }
        std::printf("\n  timeline (v40): gate/up per thread %.2f .. %.2f ms, down %.2f .. %.2f ms per token\n", umin, umax, dmin, dmax);
      }
      if (T.lag_n > 0)
        std::printf("  timeline (v45): first down item after gate/up ends: mean %.1f us, max %.1f us (%llu thread-layers)\n", 1e6 * T.lag_sum / static_cast<double>(T.lag_n),
                    1e6 * T.lag_max, static_cast<unsigned long long>(T.lag_n));
      if (T.gu2_rows > 0)
        std::printf("  check (v45): paired gate/up vs one-row path: %llu rows compared, %llu differ (bit for bit)\n", static_cast<unsigned long long>(T.gu2_rows),
                    static_cast<unsigned long long>(T.gu2_mismatch));
      if (T.round_n > 0)
        std::printf("  check (v45): AVX2 input rounding vs scalar: %llu vectors compared, %llu differ (bit for bit)\n", static_cast<unsigned long long>(T.round_n),
                    static_cast<unsigned long long>(T.round_mismatch));
      if (T.gx_n > 0)
        std::printf("  check (v45b): GPU rounded input vs CPU rounded input: %llu vectors compared, %llu differ (bit for bit)\n", static_cast<unsigned long long>(T.gx_n),
                    static_cast<unsigned long long>(T.gx_mismatch));
      if (T.down2_rows > 0)
        std::printf("  check (v40): two-row down vs one-row path: %llu rows compared, %llu differ (bit for bit)\n", static_cast<unsigned long long>(T.down2_rows),
                    static_cast<unsigned long long>(T.down2_mismatch));
      for (std::size_t il = 0; il < T.gpu_dense.size(); ++il) {
        std::printf("    layer %2zu: GPU dense %.3f, bank + shared %.3f, idle %.3f | CPU wait %.3f, cold %.3f ms", il, T.gpu_dense[il] * k, T.gpu_hot[il] * k, T.gpu_idle[il] * k,
                    T.cpu_wait[il] * k, T.cpu_cold[il] * k);
        if (T.split_stamps && il < T.cpu_picks.size()) std::printf(" | CPU picks %.2f, expert groups %.2f per token", T.cpu_picks[il] / T.steps, T.cpu_groups[il] / T.steps);
        std::printf("\n");
      }
      }
    }
  }
  // --mtp-draft K: greedy decoding of --decode tokens from the made-up prompt, plain and with MTP drafting (K per step):
  // the same tokens expected (the reference's generate_greedy_mtp contract), the kept share, and the speed of each
  if (ndraft > 0 && ndec > 0) {
    const std::vector<std::int32_t> prompt(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(npre));
    std::vector<std::int32_t> plain, drafted;
    std::vector<float> row(nv);
    if (Status st = eng.value->reset(); !st) return fail_input(st);
    Timer tpl;
    for (std::uint64_t t0 = 0; t0 < npre; t0 += batch) {
      const int k = static_cast<int>(std::min<std::uint64_t>(batch, npre - t0));
      if (Status st = eng.value->step(prompt.data() + t0, k, row.data(), true); !st) return fail_input(st);
    }
    const double pre_s = tpl.seconds();
    const bool split = a.has("dump-split");
    std::vector<float> one_rows(split ? static_cast<std::size_t>(ndec) * nv : 0);
    // v23 --probe-pos P: the row that predicts plain[P] is the step at position npre + P - 1; captured in the plain run
    // (one token at a time), in the first verify replay, and in the CPU reference (--split-ref)
    std::int64_t probe_p = -1;
    if (a.has("probe-pos")) {
      const Result<std::uint64_t> pr = io::parse_whole(a.get("probe-pos"), ndec > 1 ? ndec - 1 : 1);
      if (!pr.ok() || pr.value < 1) {
        std::fprintf(stderr, "pelias: --probe-pos takes 1 .. decode - 1\n");
        return 1;
      }
      probe_p = static_cast<std::int64_t>(npre) + static_cast<std::int64_t>(pr.value) - 1;
      eng.value->probe_set(probe_p);
    }
    std::vector<float> pxa, pxb, pwa, pwb;
    std::vector<std::int32_t> psa, psb;
    std::vector<float> boa, loa, moa, bob, lob, mob;  // v25: per-operation rows at the probe (one-token / batched)
    Timer tdp;
    for (std::uint64_t i = 0; i < ndec; ++i) {
      std::int32_t best = 0;
      for (std::size_t j = 1; j < nv; ++j)
        if (row[j] > row[static_cast<std::size_t>(best)]) best = static_cast<std::int32_t>(j);
      if (split) std::copy(row.begin(), row.end(), one_rows.begin() + static_cast<std::ptrdiff_t>(i * nv));
      plain.push_back(best);
      if (i + 1 == ndec) break;
      if (Status st = eng.value->step(&best, 1, row.data(), true); !st) return fail_input(st);
    }
    const double plain_s = tdp.seconds();
    if (probe_p >= 0 && !eng.value->probe_get(&pxa, &psa, &pwa)) std::printf("  probe: nothing captured in the plain run\n");
    if (probe_p >= 0) eng.value->probe_get_ops(&boa, &loa, &moa);
    tflush("split check: plain, one token at a time");
    model::MtpStats ms;
    model::EngineStats emt;
    model::EngineMtpTiming tmg;
    eng.value->timeline_reset();
    Timer tmt;
    if (Status st = model::engine_generate_greedy_mtp(*eng.value, s, prompt, static_cast<int>(ndec), static_cast<int>(ndraft), {}, drafted, &ms, &emt, {}, &tmg); !st)
      return fail_input(st);
    const double mtp_s = tmt.seconds() - pre_s;  // its prefill (trunk) is the same as the plain run's; the MTP catch-up stays in
    std::size_t same = 0;
    while (same < plain.size() && same < drafted.size() && plain[same] == drafted[same]) ++same;
    std::printf("  greedy %llu tokens: plain %.3f s (%.1f tokens/s); MTP drafting %llu per step %.3f s (%.1f tokens/s, %.2fx), %lld steps, %lld drafted, %lld kept (%.1f %%); first %zu of %zu tokens equal to plain\n",
                static_cast<unsigned long long>(ndec), plain_s, static_cast<double>(ndec) / plain_s, static_cast<unsigned long long>(ndraft), mtp_s,
                static_cast<double>(ndec) / mtp_s, plain_s / mtp_s, static_cast<long long>(ms.steps), static_cast<long long>(ms.drafted),
                static_cast<long long>(ms.accepted), ms.drafted > 0 ? 100.0 * static_cast<double>(ms.accepted) / static_cast<double>(ms.drafted) : 0.0, same,
                plain.size());
    {  // v20: where the MTP run's time went
      double vs = 0.0;
      std::string vb_s;
      for (int b2 = 1; b2 < 10; ++b2) {
        if (tmg.verify_n[b2] == 0) continue;
        vs += tmg.verify_s[b2];
        char buf[96];
        std::snprintf(buf, sizeof buf, "%s%s%d: %d x %.1f ms", vb_s.empty() ? "" : ", ", b2 == 9 ? ">" : "", b2 == 9 ? 8 : b2, tmg.verify_n[b2], 1e3 * tmg.verify_s[b2] / tmg.verify_n[b2]);
        vb_s += buf;
      }
      std::printf("  MTP time: prompt %.3f s; drafts %d x %.1f ms = %.3f s; verify %.3f s (by batch size %s); keep %.3f s; MTP catch-up %.3f s; promote %.3f s; host argmax %.3f s\n",
                  tmg.prompt_s, tmg.drafts, tmg.drafts ? 1e3 * tmg.draft_s / tmg.drafts : 0.0, tmg.draft_s, vs, vb_s.c_str(), tmg.keep_s, tmg.catchup_s, tmg.promote_s,
                  tmg.argmax_s);
      // v28b (author 23:14, the unlisted ~1.5 s): the MTP run's total minus the plain run's prefill still holds the MTP run's own
      // longer prompt phase; here the decode phase alone, and what is left after the listed parts
      const double mtp_dec = mtp_s + pre_s - tmg.prompt_s;
      const double listed = tmg.draft_s + vs + tmg.keep_s + tmg.catchup_s + tmg.promote_s + tmg.argmax_s;
      std::printf("  MTP check: plain prefill %.3f s, MTP prompt %.3f s (%.3f s more, charged to the MTP run above); decode phase alone: plain %.3f s vs MTP %.3f s = %.2fx; listed parts %.3f s, not listed %.3f s\n",
                  pre_s, tmg.prompt_s, tmg.prompt_s - pre_s, plain_s, mtp_dec, mtp_dec > 0 ? plain_s / mtp_dec : 0.0, listed, mtp_dec - listed);
      if (a.has("timeline")) {
        const model::EngineTimeline& T = eng.value->timeline();
        for (int b2 = 1; b2 < 10; ++b2) {
          const double c = T.by_n_steps[b2];
          if (c <= 0) continue;
          std::printf("  timeline, trunk steps of %s%d tokens: %.0f steps, per step wall %.2f ms, GPU busy %.2f, GPU idle %.2f | CPU cold %.2f, CPU wait %.2f ms\n",
                      b2 == 9 ? "more than " : "", b2 == 9 ? 8 : b2, c, 1e3 * T.by_n_wall[b2] / c, 1e3 * T.by_n_gpu_busy[b2] / c, 1e3 * T.by_n_gpu_idle[b2] / c,
                      1e3 * T.by_n_cpu_cold[b2] / c, 1e3 * T.by_n_cpu_wait[b2] / c);
        }
      }
    }
    // v19 / v20 --dump-split: the plain tokens again, through batches of each split size (verify: the MTP's path with the
    // DeltaNet checkpoints; with --split-step also plain step() batches, no checkpoints), each row against the one-token
    // row for the same position: KL by the row's place in its batch, top-1, margins; the one-token margin where plain and
    // drafted first differ
    if (split && ndec > 1) {
      std::vector<float> bat_rows;  // v21: the first verify replay's rows, for --split-ref
      bool bat_have = false;
      for (const int kb : split_sizes)
        for (int mode = 0; mode < (a.has("split-step") ? 2 : 1); ++mode) {
          const bool use_verify = mode == 0;
          if (Status st = eng.value->reset(); !st) return fail_input(st);
          for (std::uint64_t t0 = 0; t0 < npre; t0 += batch) {
            const int k = static_cast<int>(std::min<std::uint64_t>(batch, npre - t0));
            if (Status st = eng.value->step(prompt.data() + t0, k, row.data(), true); !st) return fail_input(st);
          }
          if (probe_p >= 0 && use_verify && !bat_have) eng.value->probe_set(probe_p);
          std::vector<float> vb(static_cast<std::size_t>(kb) * nv);
          std::vector<double> kl_r(static_cast<std::size_t>(kb), 0.0);
          std::vector<int> n_r(static_cast<std::size_t>(kb), 0);
          double kl_s = 0.0, kl_m = 0.0;
          int eq = 0, cnt = 0;
          std::size_t worst_at = 0;
          std::string diffs;
          int split_eq = -1;
          for (int i = 0; i + 1 < static_cast<int>(ndec);) {
            const int mm = std::min(kb, static_cast<int>(ndec) - 1 - i);
            if (use_verify) {
              if (Status st = eng.value->verify(plain.data() + i, mm, vb.data(), nullptr); !st) return fail_input(st);
              if (Status st = eng.value->keep(mm); !st) return fail_input(st);
            } else if (Status st = eng.value->step(plain.data() + i, mm, vb.data(), false); !st) {
              return fail_input(st);
            }
            for (int r = 0; r < mm; ++r) {
              const std::size_t pidx = static_cast<std::size_t>(i + r + 1);  // the row predicts plain[pidx]
              const float* ob = one_rows.data() + pidx * nv;
              const float* bb = vb.data() + static_cast<std::size_t>(r) * nv;
              const double kl = kl_rows(ob, bb, nv);
              kl_s += kl;
              if (kl > kl_m) {
                kl_m = kl;
                worst_at = pidx;
              }
              kl_r[static_cast<std::size_t>(r)] += kl;
              ++n_r[static_cast<std::size_t>(r)];
              if (use_verify && !bat_have) {
                if (bat_rows.empty()) bat_rows.assign(static_cast<std::size_t>(ndec) * nv, 0.0f);
                std::copy(bb, bb + nv, bat_rows.begin() + static_cast<std::ptrdiff_t>(pidx * nv));
              }
              const Top2 to = top2(ob, nv), tb = top2(bb, nv);
              ++cnt;
              if (to.i1 == tb.i1) ++eq;
              else {
                char buf[96];
                std::snprintf(buf, sizeof buf, "%s%zu (one-token margin %.3f)", diffs.empty() ? "" : ", ", pidx, to.margin);
                diffs += buf;
              }
              if (pidx == same) split_eq = to.i1 == tb.i1 ? 1 : 0;
            }
            i += mm;
          }
          std::string byr;
          for (int r = 0; r < kb; ++r) {
            char buf[48];
            std::snprintf(buf, sizeof buf, "%s%.2e", r ? " / " : "", n_r[static_cast<std::size_t>(r)] ? kl_r[static_cast<std::size_t>(r)] / n_r[static_cast<std::size_t>(r)] : 0.0);
            byr += buf;
          }
          std::printf("  split check, %s batches of %d vs one token at a time: KL mean %.3e max %.3e (position %zu); KL by place in the batch %s; top-1 equal %d / %d%s%s\n",
                      use_verify ? "verify" : "step", kb, cnt ? kl_s / cnt : 0.0, kl_m, worst_at, byr.c_str(), eq, cnt, diffs.empty() ? "" : "; differ at ", diffs.c_str());
          if (use_verify && !bat_have && probe_p >= 0 && !eng.value->probe_get(&pxb, &psb, &pwb)) std::printf("  probe: nothing captured in the verify replay\n");
          if (use_verify && !bat_have && probe_p >= 0) eng.value->probe_get_ops(&bob, &lob, &mob);
          tflush(std::string("split check: ") + (use_verify ? "verify" : "step") + " replay, batches of " + std::to_string(kb));
          if (use_verify) bat_have = true;
          if (use_verify && kb == static_cast<int>(ndraft) + 1 && same < plain.size()) {
            const Top2 ts = top2(one_rows.data() + same * nv, nv);
            std::printf("  split check: first MTP difference at token %zu: one-token top-2 %zu / %zu, margin %.4f; the verify batch's top-1 there %s\n", same, ts.i1,
                        ts.i2, ts.margin, split_eq < 0 ? "(token 0: from the prefill, same path)" : split_eq ? "agrees with one-token" : "differs from one-token");
          }
        }
      // v21 --split-ref: the CPU reference (the engine's rounding) over the prompt and the plain tokens; which of the
      // one-token and batched rows is off where they disagree
      if (a.has("split-ref") && bat_have) {
        model::ForwardOptions orf = model::GpuForward::matching_reference_options();
        orf.threads = static_cast<int>(thr);
        orf.fast_cpu = a.has("fast-cpu");
        orf.router_tie_band = tie_band;
        std::vector<std::int32_t> seq(prompt);
        seq.insert(seq.end(), plain.begin(), plain.end() - 1);
        std::vector<float> rr(static_cast<std::size_t>(ndec) * nv);
        std::vector<float> pxr;
        std::vector<model::RouteRecord> rrec;
        if (probe_p >= 0) {
          pxr.assign(static_cast<std::size_t>(s.n_layer) * s.n_embd, 0.0f);
          rrec.resize(seq.size() * s.n_layer);
          orf.probe_pos = probe_p;
          orf.probe_x = pxr.data();
          orf.route = rrec.data();
        }
        Timer trf;
        if (Status rs = model::forward_reference(s, w.value, seq.data(), static_cast<int>(seq.size()), static_cast<int>(npre) - 1,
                                                 static_cast<int>(npre) - 1 + static_cast<int>(ndec), rr.data(), orf);
            !rs)
          return fail_input(rs);
        double ko = 0.0, kb2 = 0.0, kmx = -1.0;
        int to = 0, tb = 0, cntr = 0;
        std::size_t at = 0;
        for (std::size_t i = 1; i < static_cast<std::size_t>(ndec); ++i) {
          const float* rf = rr.data() + i * nv;
          const float* ob = one_rows.data() + i * nv;
          const float* bb = bat_rows.data() + i * nv;
          const double k1 = kl_rows(rf, ob, nv), k2 = kl_rows(rf, bb, nv), kob = kl_rows(ob, bb, nv);
          ko += k1;
          kb2 += k2;
          const Top2 tr = top2(rf, nv);
          to += top2(ob, nv).i1 == tr.i1 ? 1 : 0;
          tb += top2(bb, nv).i1 == tr.i1 ? 1 : 0;
          ++cntr;
          if (kob > kmx) {
            kmx = kob;
            at = i;
          }
        }
        const float* rf = rr.data() + at * nv;
        const Top2 tr = top2(rf, nv), t1 = top2(one_rows.data() + at * nv, nv), t2 = top2(bat_rows.data() + at * nv, nv);
        std::printf("  split ref (CPU reference, %.1f s): over %d positions KL ref->one-token %.3e, ref->batched %.3e; top-1 equal to the reference: one-token %d, batched %d of %d\n",
                    trf.seconds(), cntr, cntr ? ko / cntr : 0.0, cntr ? kb2 / cntr : 0.0, to, tb, cntr);
        std::printf("  split ref at position %zu (largest one-token vs batched gap, KL %.3e): reference top-2 %zu / %zu margin %.4f; one-token %zu / %zu (KL from ref %.3e); batched %zu / %zu (KL from ref %.3e) -> %s is off\n",
                    at, kmx, tr.i1, tr.i2, tr.margin, t1.i1, t1.i2, kl_rows(rf, one_rows.data() + at * nv, nv), t2.i1, t2.i2, kl_rows(rf, bat_rows.data() + at * nv, nv),
                    kl_rows(rf, one_rows.data() + at * nv, nv) > kl_rows(rf, bat_rows.data() + at * nv, nv) ? "the one-token path" : "the batched path");
        if (probe_p >= 0 && !pxa.empty() && !pxb.empty()) {  // v23: layer by layer at the probe position
          const std::size_t e2 = s.n_embd, nuq = s.n_expert_used;
          std::printf("  probe at position %lld (predicts plain[%lld]): residual after each layer vs the CPU reference (relative L2), router picks (* = differs from the reference), the reference's last-pick margin\n",
                      static_cast<long long>(probe_p), static_cast<long long>(probe_p - static_cast<std::int64_t>(npre) + 1));
          for (std::size_t il = 0; il < s.n_layer; ++il) {
            auto rel = [&](const std::vector<float>& v) {
              double nn2 = 0.0, dd2 = 0.0;
              for (std::size_t i = 0; i < e2; ++i) {
                const double d = static_cast<double>(v[il * e2 + i]) - pxr[il * e2 + i];
                nn2 += d * d;
                dd2 += static_cast<double>(pxr[il * e2 + i]) * pxr[il * e2 + i];
              }
              return dd2 > 0 ? std::sqrt(nn2 / dd2) : 0.0;
            };
            const model::RouteRecord& rc = rrec[static_cast<std::size_t>(probe_p) * s.n_layer + il];
            std::vector<int> ref_s, a_s, b_s;
            for (std::size_t j = 0; j < nuq; ++j) {
              ref_s.push_back(rc.sel[j]);
              a_s.push_back(psa[il * nuq + j]);
              b_s.push_back(psb[il * nuq + j]);
            }
            std::sort(ref_s.begin(), ref_s.end());
            std::sort(a_s.begin(), a_s.end());
            std::sort(b_s.begin(), b_s.end());
            double amax = 0.0;
            for (std::size_t i = 0; i < e2; ++i) amax = std::max(amax, std::fabs(static_cast<double>(pxr[il * e2 + i])));
            std::string pa, pb;
            for (std::size_t j = 0; j < nuq; ++j) {
              pa += std::to_string(a_s[j]) + (std::find(ref_s.begin(), ref_s.end(), a_s[j]) == ref_s.end() ? "*" : "") + (j + 1 < nuq ? "," : "");
              pb += std::to_string(b_s[j]) + (std::find(ref_s.begin(), ref_s.end(), b_s[j]) == ref_s.end() ? "*" : "") + (j + 1 < nuq ? "," : "");
            }
            std::printf("    layer %2zu: one-token %.3e  batched %.3e | max |x| %.1f | margin %.2e | picks one-token %s | batched %s\n", il, rel(pxa), rel(pxb), amax,
                        static_cast<double>(rc.p_last - rc.p_next), pa.c_str(), pb.c_str());
          }
          // v25: one-token vs batched per operation at the probe row: "=" bit for bit, else the relative L2 gap. The first
          // operation that is not bit for bit is where the two paths part (every later one sees different inputs).
          if (!boa.empty() && boa.size() == bob.size() && loa.size() == lob.size() && moa.size() == mob.size()) {
            const std::size_t ne2 = s.n_expert;
            auto cmp_op = [](const std::vector<float>& va, const std::vector<float>& vb2, std::size_t off, std::size_t len, bool* eqp) {
              double nn2 = 0.0, dd2 = 0.0;
              *eqp = std::memcmp(va.data() + off, vb2.data() + off, len * sizeof(float)) == 0;
              for (std::size_t i = 0; i < len; ++i) {
                const double d = static_cast<double>(va[off + i]) - vb2[off + i];
                nn2 += d * d;
                dd2 += static_cast<double>(va[off + i]) * va[off + i];
              }
              return dd2 > 0 ? std::sqrt(nn2 / dd2) : 0.0;
            };
            std::printf("  per operation at the probe row, one-token vs batched (= bit for bit, else relative L2):\n");
            std::string first;
            for (std::size_t il = 0; il < s.n_layer; ++il) {
              bool sb = false, sl = false, sm = false;
              const double gb = cmp_op(boa, bob, il * e2, e2, &sb), gl = cmp_op(loa, lob, il * ne2, ne2, &sl), gm = cmp_op(moa, mob, il * e2, e2, &sm);
              char fb[32], fl[32], fm[32];
              std::snprintf(fb, sizeof fb, sb ? "=" : "%.2e", gb);
              std::snprintf(fl, sizeof fl, sl ? "=" : "%.2e", gl);
              std::snprintf(fm, sizeof fm, sm ? "=" : "%.2e", gm);
              const char* blk = w.value.layers[il].recurrent ? "DeltaNet" : "attention";
              std::printf("    layer %2zu: %-9s block %-9s | router logits %-9s | MoE (reader / bank + shared) %s\n", il, blk, fb, fl, fm);
              if (first.empty()) {
                if (!sb) first = "layer " + std::to_string(il) + ", the " + blk + " block";
                else if (!sl) first = "layer " + std::to_string(il) + ", the router logits (input: the post norm of an equal residual)";
                else if (!sm) first = "layer " + std::to_string(il) + ", the MoE block (the reader, the shared expert or their sum)";
              }
            }
            std::printf("  per operation: first difference %s\n", first.empty() ? "none (one-token and batched equal bit for bit at every layer)" : first.c_str());
          }
        }
      }
    }
  }
  if (a.has("no-cpu")) return selftest_bad > 0 ? 4 : 0;
  model::ForwardOptions o = model::GpuForward::matching_reference_options();
  o.threads = static_cast<int>(thr);
  if (a.has("ref-threads")) {  // v28: the CPU reference's threads apart from the engine's cold-path workers (--threads)
    const Result<std::uint64_t> rt = io::parse_whole(a.get("ref-threads"), 4096);
    if (!rt.ok()) {
      std::fprintf(stderr, "pelias: --ref-threads takes a whole number\n");
      return 1;
    }
    o.threads = static_cast<int>(rt.value);
  }
  o.fast_cpu = a.has("fast-cpu");
  o.router_tie_band = tie_band;
  std::vector<float> ref(lg.size());
  Timer tcpu;
  if (Status rs = model::forward_reference(s, w.value, ids.data(), n, 0, n, ref.data(), o); !rs) return fail_input(rs);
  std::printf("  CPU reference (same rounding%s%s): %.2f s\n", o.fast_cpu ? ", fast CPU kernels" : "", tie_band > 0.0f ? ", the same tie band" : "", tcpu.seconds());
  auto show = [&](const char* label, const Cmp& c) {
    std::printf("  %s: logits rel L2 %.3e, largest |diff| %.3e; KL mean %.3e max %.3e; top-1 equal %d / %d\n", label, c.rel, c.worst, c.kl_mean, c.kl_max, c.top, n);
  };
  const Cmp main_cmp = compare_logits(ref, lg, n, nv);
  show("engine vs reference", main_cmp);
  if (a.has("float-ref")) {
    model::ForwardOptions o3;
    o3.threads = o.threads;
    o3.fast_cpu = o.fast_cpu;
    std::vector<float> lf(lg.size());
    if (Status rs = model::forward_reference(s, w.value, ids.data(), n, 0, n, lf.data(), o3); !rs) return fail_input(rs);
    const Cmp rf = compare_logits(lf, ref, n, nv), ef = compare_logits(lf, lg, n, nv);
    show("rounded CPU reference vs float", rf);
    show("engine vs float", ef);
    {  // v19: paired top-1 (McNemar on the positions only one side gets right) and the margins of the flips
      int b = 0, c = 0;
      int fe[3] = {}, fc[3] = {}, fec[3] = {}, ne = 0, nc = 0, nec = 0;
      const double th[3] = {0.1, 0.5, 1.0};
      std::FILE* fo = nullptr;
      if (!dump_top.empty()) {
        fo = std::fopen(dump_top.c_str(), "w");
        if (!fo) {
          std::fprintf(stderr, "pelias: cannot write %s\n", dump_top.c_str());
          return 1;
        }
        std::fprintf(fo, "pos\tinput\teng1\teng2\teng_margin\tcpu1\tcpu2\tcpu_margin\tflt1\tflt2\tflt_margin\teng_kl\tcpu_kl\n");  // v49d: per-row KL vs float
      }
      double klp[2] = {0.0, 0.0}, kld[2] = {0.0, 0.0};  // v49d: KL vs float summed over the prompt rows / the decode rows (engine, CPU)
      for (int t2 = 0; t2 < n; ++t2) {
        const std::size_t off = static_cast<std::size_t>(t2) * nv;
        const Top2 te = top2(lg.data() + off, nv), tc = top2(ref.data() + off, nv), tf = top2(lf.data() + off, nv);
        const bool eok = te.i1 == tf.i1, cok = tc.i1 == tf.i1;
        if (eok && !cok) ++b;
        if (!eok && cok) ++c;
        for (int q = 0; q < 3; ++q) {
          if (!eok && tf.margin < th[q]) ++fe[q];
          if (!cok && tf.margin < th[q]) ++fc[q];
          if (te.i1 != tc.i1 && tc.margin < th[q]) ++fec[q];
        }
        ne += eok ? 0 : 1;
        nc += cok ? 0 : 1;
        nec += te.i1 != tc.i1 ? 1 : 0;
        const double ke = kl_rows(lf.data() + off, lg.data() + off, nv), kc = kl_rows(lf.data() + off, ref.data() + off, nv);
        (t2 < static_cast<int>(npre) ? klp : kld)[0] += ke;
        (t2 < static_cast<int>(npre) ? klp : kld)[1] += kc;
        if (fo)
          std::fprintf(fo, "%d\t%d\t%zu\t%zu\t%.6f\t%zu\t%zu\t%.6f\t%zu\t%zu\t%.6f\t%.6e\t%.6e\n", t2, ids[static_cast<std::size_t>(t2)], te.i1, te.i2, te.margin, tc.i1,
                       tc.i2, tc.margin, tf.i1, tf.i2, tf.margin, ke, kc);
      }
      if (fo) std::fclose(fo);
      {
        const double np2 = static_cast<double>(std::min<std::uint64_t>(npre, static_cast<std::uint64_t>(n))), nd2 = static_cast<double>(n) - np2;
        std::printf("  KL vs float by rows: prompt rows (%.0f) engine %.3e, CPU %.3e; decode rows (%.0f) engine %.3e, CPU %.3e\n", np2, np2 > 0 ? klp[0] / np2 : 0.0,
                    np2 > 0 ? klp[1] / np2 : 0.0, nd2, nd2 > 0 ? kld[0] / nd2 : 0.0, nd2 > 0 ? kld[1] / nd2 : 0.0);
      }
      std::printf("  top-1 paired vs float: engine right / CPU wrong %d, engine wrong / CPU right %d, McNemar exact p %.3f\n", b, c, mcnemar_p(b, c));
      std::printf("  float's top-2 margin (logits) where the engine misses float: < 0.1 %d, < 0.5 %d, < 1.0 %d of %d; where the CPU misses: %d, %d, %d of %d; "
                  "CPU's margin where engine and CPU differ: %d, %d, %d of %d%s%s\n",
                  fe[0], fe[1], fe[2], ne, fc[0], fc[1], fc[2], nc, fec[0], fec[1], fec[2], nec, dump_top.empty() ? "" : "; per position in ", dump_top.c_str());
    }
    const bool ok = ef.kl_mean <= 1.25 * rf.kl_mean && ef.top >= rf.top - 3;
    std::printf("  gate (engine vs float within 1.25x KL and 3 top-1 of the rounded reference vs float): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 3;
  }
  return main_cmp.kl_mean <= 1e-4 && main_cmp.top * 100 >= n * 95 ? 0 : 3;
}

int cmd_mtp_logits(const Args& a) {
  const std::string path = a.get("model"), out = a.get("out"), idt = a.get("ids");
  if (path.empty() || out.empty() || idt.empty()) {
    std::fprintf(stderr, "usage: pelias mtp-logits --model FILE --ids 1,2,3 --out OUT.bin [--threads N] [--fast-cpu]\n");
    return 1;
  }
  if (std::ifstream(out, std::ios::binary)) {
    std::fprintf(stderr, "pelias: %s already exists (nothing is overwritten)\n", out.c_str());
    return 1;
  }
  std::vector<std::int32_t> ids;
  for (std::size_t p = 0; p <= idt.size();) {
    const std::size_t q = std::min(idt.find(',', p), idt.size());
    const Result<std::uint64_t> r = io::parse_whole(idt.substr(p, q - p), 1u << 30);
    if (!r.ok()) {
      std::fprintf(stderr, "pelias: --ids takes comma-separated token ids\n");
      return 1;
    }
    ids.push_back(static_cast<std::int32_t>(r.value));
    p = q + 1;
  }
  model::ForwardOptions fo;
  if (a.has("threads")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("threads"), 4096);
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --threads needs a whole number from 1\n");
      return 1;
    }
    fo.threads = static_cast<int>(r.value);
  }
  if (!act_options(a, fo)) return 1;
  os::FileMap m;
  if (Status s = m.open(path); !s) return fail_input(s);
  io::GgufReader g;
  if (Status s = g.parse(m); !s) return fail_input(s);
  const Result<model::Qwen35MoeSpec> spec = model::read_qwen35moe_spec(g);
  if (!spec.ok()) return fail_input(spec.status);
  Result<model::Qwen35MoeWeights> w = model::bind_qwen35moe(g, m.data(), spec.value);
  if (!w.ok()) return fail_input(w.status);
  const int n = static_cast<int>(ids.size());
  const std::size_t e = spec.value.n_embd, nv = spec.value.n_vocab;
  std::vector<float> hid(static_cast<std::size_t>(n) * e), hp(hid.size(), 0.0f), lg(static_cast<std::size_t>(n) * nv), ho(hid.size());
  model::ForwardOptions oh = fo;
  oh.hidden_out = hid.data();
  if (Status s = model::forward_reference(spec.value, w.value, ids.data(), n, n - 1, n, nullptr, oh); !s) return fail_input(s);
  std::copy(hid.begin(), hid.end() - static_cast<std::ptrdiff_t>(e), hp.begin() + static_cast<std::ptrdiff_t>(e));
  model::MtpState ms;
  model::mtp_state_reset(ms);
  if (Status s = model::mtp_forward(spec.value, w.value, ms, ids.data(), hp.data(), n, lg.data(), ho.data(), fo); !s) return fail_input(s);
  std::ofstream f(out, std::ios::binary);
  const std::int32_t hdr[3] = {n, static_cast<std::int32_t>(nv), static_cast<std::int32_t>(e)};
  f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
  f.write(reinterpret_cast<const char*>(lg.data()), static_cast<std::streamsize>(lg.size() * sizeof(float)));
  f.write(reinterpret_cast<const char*>(ho.data()), static_cast<std::streamsize>(ho.size() * sizeof(float)));
  if (!f) {
    std::fprintf(stderr, "pelias: could not write %s\n", out.c_str());
    return 6;
  }
  std::fprintf(stderr, "pelias mtp-logits: %d positions, MTP logits and hidden written to %s\n", n, out.c_str());
  return 0;
}

int cmd_generate(const Args& a) {
  const std::string path = a.get("model");
  const int prompt_sources = (a.has("prompt") ? 1 : 0) + (a.has("prompt-file") ? 1 : 0) + (a.has("prompt-list") ? 1 : 0);
  if (path.empty() || prompt_sources != 1 || a.has("prompt-list") != a.has("out-dir") || (a.has("prompt-list") && a.has("ids-out"))) {
    usage();
    return 1;
  }
  int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  if (a.has("threads")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("threads"), 4096);
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --threads needs a whole number from 1 to 4096\n");
      return 1;
    }
    threads = static_cast<int>(r.value);
  }
  int n_predict = 128, n_ctx = 32768;
  if (a.has("n-predict")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("n-predict"), 1u << 20);
    if (!r.ok() || r.value == 0) {
      std::fprintf(stderr, "pelias: --n-predict needs a whole number from 1\n");
      return 1;
    }
    n_predict = static_cast<int>(r.value);
  }
  if (a.has("ctx")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("ctx"), 262144);
    if (!r.ok() || r.value < 2) {
      std::fprintf(stderr, "pelias: --ctx needs a whole number from 2 to 262144\n");
      return 1;
    }
    n_ctx = static_cast<int>(r.value);
  }
  model::SamplerConfig sc;
  if (!float_option(a, "temp", 0.0f, 2.0f, sc.temperature) || !float_option(a, "top-p", 0.01f, 1.0f, sc.top_p) ||
      !float_option(a, "min-p", 0.0f, 1.0f, sc.min_p) || !float_option(a, "repeat-penalty", 1.0f, 2.0f, sc.repeat_penalty) ||
      !float_option(a, "presence-penalty", -2.0f, 2.0f, sc.presence_penalty) ||
      !float_option(a, "frequency-penalty", -2.0f, 2.0f, sc.frequency_penalty))
    return 1;
  if (a.has("top-k")) {
    const Result<std::uint64_t> r = io::parse_whole(a.get("top-k"), 1000);
    if (!r.ok()) {
      std::fprintf(stderr, "pelias: --top-k needs a whole number from 0 (off) to 1000\n");
      return 1;
    }
    sc.top_k = static_cast<int>(r.value);
  }
  if (a.has("repeat-last-n")) {
    const std::string t = a.get("repeat-last-n");
    const Result<std::uint64_t> r = io::parse_whole(t, 65536);
    if (t != "-1" && !r.ok()) {
      std::fprintf(stderr, "pelias: --repeat-last-n is -1 (all), 0 (none) or up to 65536\n");
      return 1;
    }
    sc.repeat_last_n = t == "-1" ? -1 : static_cast<int>(r.value);
  }
  if (a.has("seed")) {
    const std::string t = a.get("seed");
    if (t == "-1") {
      std::random_device rd;
      sc.seed = (static_cast<std::uint64_t>(rd()) << 32 | rd()) & 0x7fffffffffffffffull;
    } else {
      const Result<std::uint64_t> r = io::parse_whole(t, 0x7fffffffffffffffull);
      if (!r.ok()) {
        std::fprintf(stderr, "pelias: --seed is -1 (random) or a whole number from 0 to 2^63-1\n");
        return 1;
      }
      sc.seed = r.value;
    }
  }
  if (Status cs = model::check_sampler_config(sc); !cs) {
    std::fprintf(stderr, "pelias: %s\n", cs.message.c_str());
    return 1;
  }
  model::ForwardOptions fo;
  fo.threads = threads;
  if (!act_options(a, fo)) return 1;
  const std::string ids_out = a.get("ids-out");
  if (!ids_out.empty() && std::ifstream(ids_out, std::ios::binary)) {
    std::fprintf(stderr, "pelias: %s already exists (nothing is overwritten)\n", ids_out.c_str());
    return 1;
  }
  os::FileMap m;
  Status s;
  if (a.has("no-mmap")) {
    Timer tl;
    s = m.load(path);
    if (!s) return fail_input(s);
    std::fprintf(stderr, "pelias generate: model read into RAM (--no-mmap): %.2f GB in %.1f s\n", static_cast<double>(m.size()) / 1e9, tl.seconds());
    if (m.unread_from() < m.size())
      std::fprintf(stderr, "pelias generate: the mount refused the file's last %llu bytes (from byte %llu); they are left zero and checked below\n",
                   static_cast<unsigned long long>(m.size() - m.unread_from()), static_cast<unsigned long long>(m.unread_from()));
  } else {
    s = m.open(path);
    if (!s) return fail_input(s);
  }
  io::GgufReader g;
  s = g.parse(m);
  if (!s) return fail_input(s);
  const Result<model::Qwen35MoeSpec> spec = model::read_qwen35moe_spec(g);
  if (!spec.ok()) return fail_input(spec.status);
  if (m.unread_from() < m.size()) {  // only tensors the reference pass does not run (the MTP block) may lie in the unread bytes
    const std::string mtp = "blk." + std::to_string(spec.value.n_layer) + ".";
    for (const io::GgufTensor& t : g.tensors())
      if (t.file_offset + t.bytes > m.unread_from() && t.name.rfind(mtp, 0) != 0) {
        std::fprintf(stderr, "pelias: tensor %s lies in the bytes the mount refused; run without --no-mmap or from another drive\n", t.name.c_str());
        return 2;
      }
  }
  Result<model::Qwen35MoeWeights> w = model::bind_qwen35moe(g, m.data(), spec.value);
  if (!w.ok()) return fail_input(w.status);
  // --planes-mode: all (every token), prefill (the prompt only), start:N (the prompt and the first N output tokens)
  int planes_until = -1;  // output tokens whose step reads the planes; -1 = all
  if (a.has("planes-mode")) {
    const std::string pmode = a.get("planes-mode");
    if (pmode == "all") planes_until = -1;
    else if (pmode == "prefill") planes_until = 1;  // output token 0 comes from the prompt pass
    else if (pmode.rfind("start:", 0) == 0) {
      const Result<std::uint64_t> r = io::parse_whole(pmode.substr(6), 1u << 20);
      if (!r.ok() || r.value == 0) {
        std::fprintf(stderr, "pelias: --planes-mode start:N needs N from 1\n");
        return 1;
      }
      planes_until = static_cast<int>(r.value);
    } else {
      std::fprintf(stderr, "pelias: --planes-mode is all, prefill or start:N\n");
      return 1;
    }
  }
  os::FileMap pm;
  io::GgufReader pg;
  if (const int rc = plane_options(a, pm, pg, w.value, fo, "generate"); rc != 0) return rc;
  if (a.has("planes"))
    std::fprintf(stderr, "pelias generate: planes read on %s\n",
                 planes_until < 0 ? "every token" : planes_until == 1 ? "the prompt only" : ("the prompt and the first " + std::to_string(planes_until) + " output tokens").c_str());
  Result<tok::Tokenizer> tk = tok::Tokenizer::from_gguf(g);
  if (!tk.ok()) return fail_input(tk.status);
  // the prompts: one (--prompt / --prompt-file, ids to --ids-out, text to stdout), or every line of --prompt-list, each run
  // on the model and planes loaded once (author 2026-10-29 20:38: split-token tests keep the model in RAM), the text and
  // ids of line NN to OUT_DIR/pelias_NN.txt and pelias_NN.ids
  struct Job {
    std::string text, ids_path, txt_path;
  };
  std::vector<Job> jobs;
  if (a.has("prompt-list")) {
    Result<std::string> r = io::read_text_file(a.get("prompt-list"));
    if (!r.ok()) return fail_input(r.status);
    const std::string dir = a.get("out-dir");
    std::size_t at = 0;
    for (int nn = 1; at < r.value.size(); ++nn) {
      std::size_t eol = r.value.find('\n', at);
      if (eol == std::string::npos) eol = r.value.size();
      std::string line = r.value.substr(at, eol - at);
      at = eol + 1;
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty()) {
        std::fprintf(stderr, "pelias: --prompt-list line %d is empty\n", nn);
        return 1;
      }
      char tag[16];
      std::snprintf(tag, sizeof tag, "%02d", nn);
      Job j{line, dir + "/pelias_" + tag + ".ids", dir + "/pelias_" + tag + ".txt"};
      for (const std::string& f : {j.ids_path, j.txt_path})
        if (std::ifstream(f, std::ios::binary)) {
          std::fprintf(stderr, "pelias: %s already exists (nothing is overwritten)\n", f.c_str());
          return 1;
        }
      jobs.push_back(j);
    }
  } else {
    std::string text;
    if (a.has("prompt-file")) {
      Result<std::string> r = io::read_text_file(a.get("prompt-file"));
      if (!r.ok()) return fail_input(r.status);
      text = std::move(r.value);
    } else {
      text = a.get("prompt");
    }
    jobs.push_back({text, ids_out, ""});
  }
  // --engine (GP3-14): the GPU engine with the hot-expert bank and CPU cold experts, opened once for every prompt
  std::unique_ptr<model::GpuEngine> engine;
  if (a.has("engine")) {
    if (a.has("planes") || a.has("planes-mode") || a.has("act-int8")) {
      std::fprintf(stderr, "pelias: --engine runs the engine's own rounding (per-32 int8); planes and --act-int8 are reference options\n");
      return 1;
    }
    model::EngineOptions eo2;
    eo2.max_ctx = n_ctx;
    eo2.max_batch = 64;
    eo2.cpu_threads = threads;
    if (!gib_option(a, "budget-gb", eo2.vram_budget)) return 1;
    if (a.has("mtp-draft")) {
      const Result<std::uint64_t> r = io::parse_whole(a.get("mtp-draft"), 16);
      if (!r.ok()) {
        std::fprintf(stderr, "pelias: --mtp-draft takes 0 to 16 drafted tokens per step\n");
        return 1;
      }
      eo2.mtp = r.value > 0;
      eo2.max_verify = static_cast<int>(r.value) + 1;
    }
    if (a.has("routes")) {
      auto ho = model::hot_order_from_routes(a.get("routes"), spec.value.n_layer, spec.value.n_expert);
      if (!ho.ok()) return fail_input(ho.status);
      eo2.hot_order = std::move(ho.value);
    }
    Timer te;
    auto eng = model::GpuEngine::open(spec.value, w.value, eo2);
    if (!eng.ok()) return fail_input(eng.status);
    engine = std::move(eng.value);
    std::fprintf(stderr, "pelias generate: %s (open %.1f s)\n", engine->summary().c_str(), te.seconds());
  }
  for (std::size_t job = 0; job < jobs.size(); ++job) {
  // --chat: one user turn in the family's chat format, the assistant's turn opened (READ-SRC: the files' own
  // tokenizer.chat_template, the plain case without a system message or tools). Qwen (qwen35moe, qwen3, qwen2): ChatML;
  // llama: the Llama 3 header format; bitnet: "User: ... <|eot_id|>Assistant: " (BOS added below when the file asks for it).
  std::string chat_text;
  if (a.has("chat")) {
    if (spec.value.family == model::Family::llama)
      chat_text = "<|start_header_id|>user<|end_header_id|>\n\n" + jobs[job].text + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    else if (spec.value.family == model::Family::bitnet)  // BitNet b1.58 2B4T's template: "User: " text "<|eot_id|>Assistant: "
      chat_text = "User: " + jobs[job].text + "<|eot_id|>Assistant: ";
    else
      chat_text = "<|im_start|>user\n" + jobs[job].text + "<|im_end|>\n<|im_start|>assistant\n";
  }
  const std::string& text = a.has("chat") ? chat_text : jobs[job].text;
  const std::string& ids_path = jobs[job].ids_path;
  std::FILE* tout = stdout;
  if (!jobs[job].txt_path.empty()) {
    tout = std::fopen(jobs[job].txt_path.c_str(), "wb");
    if (!tout) {
      std::fprintf(stderr, "pelias: cannot create %s\n", jobs[job].txt_path.c_str());
      return 6;
    }
  }
  fo.planes = true;  // every prompt's prefill reads the planes (--planes-mode decides the decode steps)
  Timer tjob;
  tok::EncodeOptions eo;
  eo.nfc = !a.has("no-nfc");
  eo.parse_special = !a.has("no-special");
  std::vector<std::int32_t> ids = tk.value.encode(text, eo);
  if (tk.value.add_bos() && tk.value.bos_id() >= 0) ids.insert(ids.begin(), tk.value.bos_id());
  if (ids.empty()) {
    std::fprintf(stderr, "pelias: the prompt is empty after tokenizing\n");
    return 1;
  }
  const int n_prompt = static_cast<int>(ids.size());
  if (n_prompt + n_predict > n_ctx) {
    std::fprintf(stderr, "pelias: the prompt (%d tokens) and --n-predict %d do not fit --ctx %d\n", n_prompt, n_predict, n_ctx);
    return 1;
  }
  std::vector<std::int32_t> stop;
  if (!a.has("ignore-eos")) {
    if (tk.value.eos_id() >= 0) stop.push_back(tk.value.eos_id());
    for (const char* t : {"<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>"})
      if (const std::int32_t id = tk.value.find(t); id >= 0 && std::find(stop.begin(), stop.end(), id) == stop.end()) stop.push_back(id);
  }
  const bool show_special = a.has("show-special");
  std::fprintf(stderr, "pelias generate: %d prompt tokens, up to %d new, %d threads, fp32 reference pass; %s (seed %llu)\n", n_prompt, n_predict, threads,
               sc.temperature <= 0.0f ? "greedy" : "sampling: temperature, top-k, top-p, min-p",
               static_cast<unsigned long long>(sc.seed));
  if (!fo.act_int8.empty()) {
    std::string list;
    for (const std::string& x : fo.act_int8) list += (list.empty() ? "" : ",") + x;
    std::fprintf(stderr, "pelias generate: 8-bit activation rounding (%s path%s) on: %s\n",
                 fo.act_gpu256 ? (fo.act_gpu256_sub == 1 ? "gpu256x8" : fo.act_gpu256_sub == 2 ? "gpu256e" : "gpu256") : (fo.act_int8_gpu ? "gpu" : "cpu"),
                 fo.act_rot32 ? ", Walsh-Hadamard rotation per 32 values" : "", list.c_str());
  }
  model::DecodeState st;
  model::decode_state_reset(spec.value, st);
  if (a.has("mtp-draft")) {  // GP3-08: greedy decoding with MTP drafting (the same tokens as plain greedy decoding)
    const Result<std::uint64_t> r = io::parse_whole(a.get("mtp-draft"), 16);
    if (!r.ok()) {
      std::fprintf(stderr, "pelias: --mtp-draft takes 0 to 16 drafted tokens per step\n");
      return 1;
    }
    if (sc.temperature > 0.0f || planes_until >= 0) {
      std::fprintf(stderr, "pelias: --mtp-draft is built for greedy decoding (--temp 0) with --planes-mode all\n");
      return 1;
    }
    model::MtpStats ms;
    std::vector<std::int32_t> out;
    Timer tg;
    model::EngineStats emt;
    if (engine) {
      if (static_cast<int>(ids.size()) + n_predict + static_cast<int>(r.value) > n_ctx) {
        std::fprintf(stderr, "pelias: the prompt and --n-predict do not fit --ctx (%d)\n", n_ctx);
        return 1;
      }
      s = model::engine_generate_greedy_mtp(*engine, spec.value, ids, n_predict, static_cast<int>(r.value), stop, out, &ms, &emt);
    } else {
      s = model::generate_greedy_mtp(spec.value, w.value, ids, n_predict, static_cast<int>(r.value), stop, out, &ms, fo);
    }
    if (!s) return fail_input(s);
    if (engine) {
      const double all = static_cast<double>(emt.hot_picks + emt.cold_picks);
      std::fprintf(stderr, "pelias generate: engine picks on the GPU %.1f %% (trunk and MTP block); CPU cold work %.2f s, host waits %.2f s\n",
                   all > 0 ? 100.0 * static_cast<double>(emt.hot_picks) / all : 0.0, emt.cold_seconds, emt.wait_seconds);
    }
    const double gen_s = tg.seconds();
    for (const std::int32_t id : out)
      if (std::find(stop.begin(), stop.end(), id) == stop.end() && (show_special || tk.value.type(id) != tok::TokenType::control)) {
        const std::string piece = tk.value.piece(id);
        std::fwrite(piece.data(), 1, piece.size(), tout);
      }
    std::fputs("\n", tout);
    std::fflush(tout);
    if (tout != stdout) std::fclose(tout);
    std::fprintf(stderr, "pelias generate: MTP drafting (%llu per step): %zu tokens out in %.2f s including the prefill (%.3f tok/s); %lld steps, %lld drafted, %lld kept (%.1f %%)\n",
                 static_cast<unsigned long long>(r.value), out.size(), gen_s, static_cast<double>(out.size()) / std::max(gen_s, 1e-9),
                 static_cast<long long>(ms.steps), static_cast<long long>(ms.drafted), static_cast<long long>(ms.accepted),
                 ms.drafted > 0 ? 100.0 * static_cast<double>(ms.accepted) / static_cast<double>(ms.drafted) : 0.0);
    if (!ids_path.empty()) {
      std::ofstream f(ids_path, std::ios::binary);
      std::string line = "prompt:";
      for (std::int32_t id : ids) line += " " + std::to_string(id);
      line += "\noutput:";
      for (std::int32_t id : out) line += " " + std::to_string(id);
      line += "\ngap:\n";  // not recorded on the drafting path
      f << line;
      if (!f) {
        std::fprintf(stderr, "pelias: could not write %s\n", ids_path.c_str());
        return 6;
      }
    }
    continue;
  }
  model::Sampler sampler(sc);
  const auto nv = static_cast<std::int64_t>(spec.value.n_vocab);
  std::vector<float> logits(static_cast<std::size_t>(nv));
  std::vector<std::int32_t> history = ids, out;
  std::vector<float> gap;  // per output token: the largest raw logit minus the second (how near a greedy tie was)
  Timer tp;
  model::EngineStats es_tot;
  auto engine_step = [&](const std::int32_t* tk2, int n2) -> Status {
    model::EngineStats es;
    const Status r2 = engine->step(tk2, n2, logits.data(), true, &es);
    es_tot.cold_seconds += es.cold_seconds;
    es_tot.wait_seconds += es.wait_seconds;
    es_tot.hot_picks += es.hot_picks;
    es_tot.cold_picks += es.cold_picks;
    return r2;
  };
  if (engine) {
    if (n_prompt + n_predict > n_ctx) {
      std::fprintf(stderr, "pelias: the prompt and --n-predict do not fit --ctx (%d)\n", n_ctx);
      return 1;
    }
    s = engine->reset();
    for (int t0 = 0; s && t0 < n_prompt; t0 += 64) s = engine_step(ids.data() + t0, std::min(64, n_prompt - t0));
  } else {
    s = model::forward_decode(spec.value, w.value, st, ids.data(), n_prompt, logits.data(), true, fo);
  }
  if (!s) return fail_input(s);
  const double prefill_s = tp.seconds();
  Timer td;
  const char* why = "the --n-predict limit";
  for (int i = 0; i < n_predict; ++i) {
    float l1 = -INFINITY, l2 = -INFINITY;
    for (const float x : logits)
      if (x > l1) {
        l2 = l1;
        l1 = x;
      } else if (x > l2) {
        l2 = x;
      }
    gap.push_back(l1 - l2);
    const std::int32_t next = sampler.sample(logits.data(), nv, history);
    out.push_back(next);
    history.push_back(next);
    if (std::find(stop.begin(), stop.end(), next) != stop.end()) {
      why = "an end-of-generation token";
      break;
    }
    if (show_special || tk.value.type(next) != tok::TokenType::control) {
      const std::string piece = tk.value.piece(next);
      std::fwrite(piece.data(), 1, piece.size(), tout);
      std::fflush(tout);
    }
    if (i + 1 == n_predict) break;
    fo.planes = planes_until < 0 || i + 1 < planes_until;  // this step gives output token i + 1
    s = engine ? engine_step(&next, 1) : model::forward_decode(spec.value, w.value, st, &next, 1, logits.data(), true, fo);
    if (!s) return fail_input(s);
  }
  const double decode_s = td.seconds();
  if (engine) {
    const double all = static_cast<double>(es_tot.hot_picks + es_tot.cold_picks);
    std::fprintf(stderr, "pelias generate: engine picks on the GPU %.1f %%; CPU cold work %.2f s, host waits %.2f s\n",
                 all > 0 ? 100.0 * static_cast<double>(es_tot.hot_picks) / all : 0.0, es_tot.cold_seconds, es_tot.wait_seconds);
  }
  std::fputs("\n", tout);
  std::fflush(tout);
  if (tout != stdout) std::fclose(tout);
  const int steps = static_cast<int>(out.size()) - 1;  // forward steps after the prefill (the last token is not run)
  std::fprintf(stderr, "pelias generate: prefill %d tokens in %.2f s (%.2f tok/s); %zu tokens out, stopped by %s; %d decode steps in %.2f s (%.3f tok/s)\n",
               n_prompt, prefill_s, n_prompt / std::max(prefill_s, 1e-9), out.size(), why, steps, decode_s,
               steps > 0 ? steps / std::max(decode_s, 1e-9) : 0.0);
  if (!ids_path.empty()) {
    std::ofstream f(ids_path, std::ios::binary);
    std::string line = "prompt:";
    for (std::int32_t id : ids) line += " " + std::to_string(id);
    line += "\noutput:";
    for (std::int32_t id : out) line += " " + std::to_string(id);
    line += "\ngap:";
    char buf[32];
    for (const float x : gap) {
      std::snprintf(buf, sizeof buf, " %.5f", static_cast<double>(x));
      line += buf;
    }
    line += "\n";
    f << line;
    if (!f) {
      std::fprintf(stderr, "pelias: could not write %s\n", ids_path.c_str());
      return 6;
    }
  }
  if (jobs.size() > 1)
    std::fprintf(stderr, "pelias generate: prompt %zu of %zu done in %.1f s (model and planes kept loaded)\n", job + 1, jobs.size(), tjob.seconds());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse_args(argc, argv);
  if (a.pos.empty() || a.has("help")) {
    usage();
    return a.pos.empty() ? 1 : 0;
  }
  const std::string lv = a.get("log", "info");
  log::set_level(lv == "debug" ? log::Level::debug : lv == "warn" ? log::Level::warn : lv == "error" ? log::Level::error : log::Level::info);
  const std::string cmd = a.pos[0];
  if (cmd == "version") return cmd_version();
  if (cmd == "device") return cmd_device(a);
  if (cmd == "membw") return cmd_membw(a);
  if (cmd == "bench-dot") return cmd_bench_dot(a);
  if (cmd == "logits") return cmd_logits(a);
  if (cmd == "mtp-logits") return cmd_mtp_logits(a);
  if (cmd == "gpu-check") return cmd_gpu_check(a);
  if (cmd == "engine-check") return cmd_engine_check(a);
  if (cmd == "pin-probe") return cmd_pin_probe(a);
  if (cmd == "page-probe") return cmd_page_probe(a);
  if (cmd == "inspect") return cmd_inspect(a);
  if (cmd == "key") return cmd_key(a);
  if (cmd == "plan") return cmd_plan(a);
  if (cmd == "quantize") return cmd_quantize(a);
  if (cmd == "convert") return cmd_convert(a);
  if (cmd == "verify") return cmd_verify(a);
  if (cmd == "tokenize") return cmd_tokenize(a);
  if (cmd == "kld") return cmd_kld(a);
  if (cmd == "kld-compare") return cmd_kld_compare(a);
  if (cmd == "imatrix") return cmd_imatrix(a);
  if (cmd == "generate") return cmd_generate(a);
  if (cmd == "planes") return cmd_planes(a);
  if (cmd == "teacher") return cmd_teacher(a);
  for (const char* later : {"probe", "train", "export", "serve", "bench", "web"})
    if (cmd == later) {
      std::fprintf(stderr, "pelias: '%s' is not built yet (a later step of the order of operations)\n", later);
      return 1;
    }
  std::fprintf(stderr, "pelias: unknown command '%s'\n", cmd.c_str());
  usage();
  return 1;
}
