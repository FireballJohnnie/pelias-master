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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "hal/device.h"
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
#include "model/arch_qwen35moe.h"
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
      "      [--dump-kld OUT.bin] [--act-int8 GROUPS] [--act-int8-path cpu|gpu|gpu256|gpu256x8|gpu256e] [--act-rot h32] [--route-dump OUT.bin] [--dump-codes OUT.bin] [--dump-qk OUT.bin]\n"
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
      "      [--ids-out FILE] [--no-nfc] [--no-special] [--act-int8 GROUPS] [--act-int8-path ...] [--act-rot h32] [--no-mmap]\n"
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
                                         &l.down_exps, &l.up_shexp, &l.gate_shexp, &l.down_shexp})
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
    s = model::forward_reference(spec.value, w.value, toks.data(), n_ctx, first, n_ctx - 1, logits.data(), fo);
    if (!s) return fail_input(s);
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
  for (std::size_t job = 0; job < jobs.size(); ++job) {
  const std::string& text = jobs[job].text;
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
    for (const char* t : {"<|im_end|>", "<|endoftext|>"})
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
  model::Sampler sampler(sc);
  const auto nv = static_cast<std::int64_t>(spec.value.n_vocab);
  std::vector<float> logits(static_cast<std::size_t>(nv));
  std::vector<std::int32_t> history = ids, out;
  std::vector<float> gap;  // per output token: the largest raw logit minus the second (how near a greedy tie was)
  Timer tp;
  s = model::forward_decode(spec.value, w.value, st, ids.data(), n_prompt, logits.data(), true, fo);
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
    s = model::forward_decode(spec.value, w.value, st, &next, 1, logits.data(), true, fo);
    if (!s) return fail_input(s);
  }
  const double decode_s = td.seconds();
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
