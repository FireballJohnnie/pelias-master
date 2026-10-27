// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// main.cc - the `pelias` command. Step 2 carries: version, inspect (every tensor of a GGUF with its type, shape and
// bytes), key (SHA-256 model key), plan (the device memory plan against the ceiling, refused above it), device (what
// the GPU reports). Step 3 adds quantize (a GGUF to a stock mix, byte-identical to the reference), convert (a model
// folder of safetensors shards to a GGUF, BF16 or a stock mix, Stage A) and verify (two GGUFs compared tensor by tensor
// and byte for byte, gates P-E6 and P-X6). The later verbs are listed as not yet built so the user gets a clear answer
// instead of a silent failure.
//
// Exit codes: 0 ok; 1 usage; 2 the input could not be read (missing, corrupt, unsupported); 3 a gate refused (plan
// over the ceiling); 4 no device; 5 verify found differences; 6 quantize or convert failed while writing (disk, I/O).
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "hal/device.h"
#include "hal/vram_governor.h"
#include "ingest/stage_a.h"
#include "io/gguf_compare.h"
#include "io/gguf_reader.h"
#include "io/number_text.h"
#include "io/requantize.h"
#include "io/safetensors.h"
#include "io/sha256.h"
#include "os/clock.h"
#include "os/file_map.h"
#include "pelias/log.h"
#include "pelias/timer.h"
#include "pelias/version.h"
#include "plan/memory_plan.h"
#include "plan/probe_constants.h"
#include "quant/block_types.h"
#include "quant/type_rules.h"

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
      "  inspect --model FILE [--json] [--header-only]\n"
      "                                  every tensor of a GGUF or safetensors file: name, type, shape, bytes\n"
      "                                  (--header-only: read a file whose data is cut off, e.g. a header sample)\n"
      "  key --model FILE                SHA-256 of the file (the model key)\n"
      "  plan --model FILE [--bank-gib N] [--ceiling-gib C] [--working-set-gib W] [--index N] [--header-only]\n"
      "                                  the device memory plan; refused above the ceiling (exit 3)\n"
      "  quantize --model IN.gguf --out OUT.gguf [--type Q4_K_M|Q6_K|Q8_0|BF16|F16|F32] [--threads N]\n"
      "                                  a new GGUF in a stock mix (default Q4_K_M), identical to the reference's (exit 6 on a write failure)\n"
      "  convert --model FOLDER --out OUT.gguf [--type BF16|Q4_K_M|Q6_K|Q8_0] [--threads N]\n"
      "                                  a Hugging Face folder (safetensors) to a GGUF: BF16 as the reference converter writes it,\n"
      "                                  or a stock mix as the reference quantizer writes it from that BF16 file (exit 6 on a write failure)\n"
      "  verify --model A.gguf --reference B.gguf\n"
      "                                  every tensor compared by name, then the files byte for byte (exit 5 if anything differs)\n"
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


int cmd_quantize(const Args& a) {
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
  if (cmd == "inspect") return cmd_inspect(a);
  if (cmd == "key") return cmd_key(a);
  if (cmd == "plan") return cmd_plan(a);
  if (cmd == "quantize") return cmd_quantize(a);
  if (cmd == "convert") return cmd_convert(a);
  if (cmd == "verify") return cmd_verify(a);
  for (const char* later : {"probe", "train", "export", "serve", "bench", "web"})
    if (cmd == later) {
      std::fprintf(stderr, "pelias: '%s' is not built yet (a later step of the order of operations)\n", later);
      return 1;
    }
  std::fprintf(stderr, "pelias: unknown command '%s'\n", cmd.c_str());
  usage();
  return 1;
}
