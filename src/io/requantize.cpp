// provenance: original (procedure READ-SRC, see requantize.h)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/requantize.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#include "io/gguf_writer.h"
#include "pelias/timer.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"

namespace pelias::io {
namespace {

constexpr std::uint32_t kQuantizationVersion = 2;  // the reference's GGML_QNT_VERSION

bool is_float_type(std::uint32_t t) {
  return t == static_cast<std::uint32_t>(quant::Type::f32) || t == static_cast<std::uint32_t>(quant::Type::f16) ||
         t == static_cast<std::uint32_t>(quant::Type::bf16);
}

// Converts `n` elements of type `t` at `in` to f32 at `out`.
void to_f32(std::uint32_t t, const std::uint8_t* in, float* out, std::int64_t n) {
  (void)quant::dequantize_row(static_cast<quant::Type>(t), in, out, n);  // checked by the caller beforehand
}

// The quantized data is refused if a scale or a value is not finite (the reference validates every written row too).
bool row_data_finite(std::uint32_t t, const std::uint8_t* p, std::uint64_t bytes) {
  auto half_finite = [](const std::uint8_t* q) {
    std::uint16_t h;
    std::memcpy(&h, q, 2);
    return (h & 0x7C00u) != 0x7C00u;
  };
  switch (static_cast<quant::Type>(t)) {
    case quant::Type::q8_0:
      for (std::uint64_t i = 0; i < bytes; i += 34) if (!half_finite(p + i)) return false;
      return true;
    case quant::Type::q4_k:
      for (std::uint64_t i = 0; i < bytes; i += 144) if (!half_finite(p + i) || !half_finite(p + i + 2)) return false;
      return true;
    case quant::Type::q6_k:
      for (std::uint64_t i = 0; i < bytes; i += 210) if (!half_finite(p + i + 208)) return false;
      return true;
    case quant::Type::f16:
      for (std::uint64_t i = 0; i < bytes; i += 2) if (!half_finite(p + i)) return false;
      return true;
    case quant::Type::bf16:
      for (std::uint64_t i = 0; i < bytes; i += 2) {
        std::uint16_t h;
        std::memcpy(&h, p + i, 2);
        if ((h & 0x7F80u) == 0x7F80u) return false;
      }
      return true;
    case quant::Type::f32:
      for (std::uint64_t i = 0; i < bytes; i += 4) {
        std::uint32_t u;
        std::memcpy(&u, p + i, 4);
        if ((u & 0x7F800000u) == 0x7F800000u) return false;
      }
      return true;
    default:
      return true;
  }
}

// Runs fn(begin, end) over [0, n) split among `threads`.
template <typename F>
void parallel_rows(std::int64_t n, int threads, F fn) {
  const int t = static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>(threads, n)));
  if (t == 1) {
    fn(std::int64_t{0}, n);
    return;
  }
  std::vector<std::thread> pool;
  const std::int64_t per = (n + t - 1) / t;
  for (int k = 0; k < t; ++k) {
    const std::int64_t b = k * per, e = std::min(n, b + per);
    if (b >= e) break;
    pool.emplace_back([=] { fn(b, e); });
  }
  for (auto& th : pool) th.join();
}

}  // namespace

Status write_gguf_file(const std::string& out_path, const std::vector<GgufKvBlob>& kv, const std::vector<GgufOutTensor>& out,
                       const std::function<Status(std::size_t j, GgufFileWriter& w)>& fill) {
  const std::uint64_t total = gguf_file_bytes(kv, out, 32);
  namespace fs = std::filesystem;
  std::error_code ec;
  if (fs::exists(out_path, ec)) return Status::fail(Code::invalid_argument, "the output file already exists (nothing is overwritten)");
  const fs::path dir = fs::absolute(fs::path(out_path), ec).parent_path();
  const fs::space_info sp = fs::space(dir, ec);
  if (!ec && sp.available < total)
    return Status::fail(Code::io_error, "not enough free space: the output needs " + std::to_string(total) + " bytes, the drive has " + std::to_string(sp.available));
  const std::string partial = out_path + ".partial";
  GgufFileWriter w;
  Status s = w.begin(partial, kv, out, 32);
  if (!s) return s;
  for (std::size_t j = 0; j < out.size(); ++j) {
    s = fill(j, w);
    if (!s) {
      w.abandon();
      fs::remove(partial, ec);
      return s;
    }
  }
  s = w.finish();
  if (!s) {
    fs::remove(partial, ec);
    return s;
  }
  fs::rename(partial, out_path, ec);
  if (ec) return Status::fail(Code::io_error, "the output was written as " + fs::path(partial).filename().string() + " but could not be renamed");
  return Status::good();
}

Result<RequantReport> quantize_source(const QuantSource& src, const std::string& out_path, const RequantOptions& opt, const RequantProgress& progress) {
  using R = Result<RequantReport>;
  Timer timer;
  RequantReport rep;

  // ---- metadata: every entry copied as encoded, except the two the reference re-sets at the end and the split keys
  std::vector<GgufKvBlob> kv;
  for (const GgufKvBlob& b : src.kv) {
    const std::string key = gguf_kv_key(b);
    if (key == "general.quantization_version" || key == "general.file_type" || key == "split.no" || key == "split.count" || key == "split.tensors.count")
      continue;
    kv.push_back(b);
  }
  kv.push_back(gguf_kv_u32("general.quantization_version", kQuantizationVersion));
  kv.push_back(gguf_kv_u32("general.file_type", static_cast<std::uint32_t>(opt.ftype)));

  // ---- tensors in the reference's order, and their types by the mix's rule
  const auto& st = src.tensors;
  std::vector<std::string> names;
  names.reserve(st.size());
  for (const auto& t : st) names.push_back(t.name);
  const std::vector<std::size_t> order = reference_order(names);
  std::vector<quant::TensorDesc> descs;
  for (const std::size_t i : order) {
    quant::TensorDesc d;
    d.name = st[i].name;
    std::uint32_t nd = 1;
    for (std::uint32_t k = 0; k < 4; ++k)
      if (st[i].ne[k] > 1) nd = k + 1;
    d.n_dims = nd;  // the reference loads tensors with ggml's own dimension count
    for (int k = 0; k < 4; ++k) d.ne[k] = st[i].ne[k];
    d.type = st[i].type;
    descs.push_back(d);
  }
  const Result<std::vector<std::uint32_t>> types = quant::choose_types(descs, src.facts, opt.ftype);
  if (!types.ok()) return R::fail(types.status);

  std::vector<GgufOutTensor> out;
  for (std::size_t j = 0; j < order.size(); ++j) {
    const SourceTensor& t = st[order[j]];
    const std::uint32_t to = types.value[j];
    if (to != t.type) {
      const quant::TypeInfo ti_in = quant::type_info(t.type), ti_out = quant::type_info(to);
      if (!is_float_type(t.type) && !quant::can_dequantize(static_cast<quant::Type>(t.type)))
        return R::fail(Status::fail(Code::unsupported, std::string("reading ") + ti_in.name + " to re-quantize it is not built yet", 0, t.name));
      if (!quant::can_quantize(static_cast<quant::Type>(to)) || to == static_cast<std::uint32_t>(quant::Type::q8_1) || to == static_cast<std::uint32_t>(quant::Type::q8_k))
        return R::fail(Status::fail(Code::unsupported, std::string("the ") + quant::file_type_name(opt.ftype) + " rule picks " + ti_out.name + " here, and quantizing to " +
                                                           ti_out.name + " is not built yet", 0, t.name));
    }
    GgufOutTensor o;
    o.name = t.name;
    for (int k = 0; k < 4; ++k) o.ne[k] = t.ne[k];
    o.type = to;
    o.bytes = to == t.type ? t.bytes : quant::row_bytes(to, t.ne[0]) * t.ne[1] * t.ne[2] * t.ne[3];
    out.push_back(o);
  }

  const int threads = opt.threads > 0 ? opt.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  constexpr std::uint64_t kSlabElements = std::uint64_t{64} << 20;  // 64 Mi elements (256 MiB of f32) per pass
  std::vector<std::uint8_t> raw;
  std::vector<float> f32;
  std::vector<std::uint8_t> qbuf;
  const Status ws = write_gguf_file(out_path, kv, out, [&](std::size_t j, GgufFileWriter& w) -> Status {
    const SourceTensor& t = st[order[j]];
    const GgufOutTensor& o = out[j];
    const std::uint64_t rows = t.ne[1] * t.ne[2] * t.ne[3];
    const std::uint64_t in_row = quant::row_bytes(t.type, t.ne[0]);
    const std::uint64_t slab_rows = std::max<std::uint64_t>(1, kSlabElements / t.ne[0]);
    for (std::uint64_t r0 = 0; r0 < rows; r0 += slab_rows) {
      const std::uint64_t nr = std::min(slab_rows, rows - r0);
      raw.resize(static_cast<std::size_t>(nr * in_row));
      Status s = src.read(order[j], r0, nr, raw.data());
      if (!s) return s;
      if (o.type == t.type) {
        s = w.write(raw.data(), raw.size());
        if (!s) return s;
        continue;
      }
      const auto ne0 = static_cast<std::int64_t>(t.ne[0]);
      const std::uint64_t out_row = quant::row_bytes(o.type, t.ne[0]);
      f32.resize(static_cast<std::size_t>(nr * t.ne[0]));
      qbuf.resize(static_cast<std::size_t>(nr * out_row));
      const std::uint32_t tin = t.type, tout = o.type;
      parallel_rows(static_cast<std::int64_t>(nr), threads, [&](std::int64_t b, std::int64_t e) {
        for (std::int64_t r = b; r < e; ++r) {
          float* row = f32.data() + r * ne0;
          const std::uint8_t* rin = raw.data() + static_cast<std::uint64_t>(r) * in_row;
          if (tin == static_cast<std::uint32_t>(quant::Type::f32)) std::memcpy(row, rin, static_cast<std::size_t>(ne0) * 4);
          else to_f32(tin, rin, row, ne0);
          (void)quant::quantize_row(static_cast<quant::Type>(tout), row, qbuf.data() + static_cast<std::uint64_t>(r) * out_row, ne0);
        }
      });
      if (!row_data_finite(o.type, qbuf.data(), qbuf.size()))
        return Status::fail(Code::corrupt, "quantized data is not finite (the input holds inf or NaN values)", 0, t.name);
      s = w.write(qbuf.data(), qbuf.size());
      if (!s) return s;
    }
    if (o.type == t.type) ++rep.copied;
    else ++rep.quantized;
    ++rep.tensors;
    rep.bytes_in += t.bytes;
    rep.bytes_out += o.bytes;
    if (o.type < 64) ++rep.by_type[o.type];
    if (progress) progress(j, order.size(), t.name, o.type);
    return Status::good();
  });
  if (!ws) return R::fail(ws);
  rep.seconds = timer.seconds();
  return R::good(rep);
}

Result<RequantReport> requantize(const os::FileMap& map, const GgufReader& in, const std::string& out_path, const RequantOptions& opt,
                                 const RequantProgress& progress) {
  using R = Result<RequantReport>;
  if (in.alignment() != 32)
    return R::fail(Status::fail(Code::unsupported, "input alignment " + std::to_string(in.alignment()) + " (the reference writes 32; other alignments are not built yet)"));
  if (in.data_start() + in.data_bytes() > map.size()) return R::fail(Status::fail(Code::corrupt, "the input's tensor data is cut off (a header sample cannot be quantized)"));

  QuantSource src;
  for (const GgufKv& e : in.kv()) {
    GgufKvBlob b(map.data() + e.file_begin, map.data() + e.file_end);
    // the reference writes booleans as 0 or 1 whatever byte the input held
    if (e.value.type == GgufValueType::boolean && !b.empty()) b.back() = b.back() != 0 ? 1 : 0;
    if (e.value.type == GgufValueType::array && e.value.array_type == GgufValueType::boolean)
      for (std::uint64_t i = e.value.array_offset - e.file_begin; i < b.size(); ++i) b[i] = b[i] != 0 ? 1 : 0;
    src.kv.push_back(std::move(b));
  }
  src.facts.arch = in.get_string("general.architecture");
  src.facts.n_expert = in.get_uint(src.facts.arch + ".expert_count");
  src.facts.n_layer = in.get_uint(src.facts.arch + ".block_count");
  const auto& ts = in.tensors();
  for (const GgufTensor& t : ts) {
    SourceTensor x;
    x.name = t.name;
    for (int k = 0; k < 4; ++k) x.ne[k] = t.ne[k];
    x.type = t.type;
    x.bytes = t.bytes;
    src.tensors.push_back(x);
  }
  src.read = [&](std::size_t i, std::uint64_t row0, std::uint64_t n, std::uint8_t* outp) -> Status {
    const GgufTensor& t = ts[i];
    const std::uint64_t rb = quant::row_bytes(t.type, t.ne[0]);
    std::memcpy(outp, map.data() + t.file_offset + row0 * rb, static_cast<std::size_t>(n * rb));
    return Status::good();
  };
  return quantize_source(src, out_path, opt, progress);
}

}  // namespace pelias::io
