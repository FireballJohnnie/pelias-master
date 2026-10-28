// provenance: original (procedure READ-SRC, see requantize.h)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/requantize.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <set>
#include <filesystem>
#include <thread>
#include <vector>

#include "io/gguf_writer.h"
#include "io/text_file.h"
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
    case quant::Type::q8_z:
      for (std::uint64_t i = 0; i < bytes; i += 264) if (!half_finite(p + i)) return false;
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

// gguf_set_val semantics: a key already present is replaced where it stands, a new key is added at the end
void set_kv(std::vector<GgufKvBlob>& kv, GgufKvBlob b) {
  const std::string key = gguf_kv_key(b);
  for (GgufKvBlob& e : kv)
    if (gguf_kv_key(e) == key) {
      e = std::move(b);
      return;
    }
  kv.push_back(std::move(b));
}

}  // namespace

Result<std::map<std::string, std::uint32_t>> read_recipe(const std::string& path) {
  using R = Result<std::map<std::string, std::uint32_t>>;
  const Result<std::string> text = read_text_file(path, std::uint64_t{64} << 20);
  if (!text.ok()) return R::fail(text.status);
  std::map<std::string, std::uint32_t> out;
  const std::string& t = text.value;
  std::size_t line_no = 0;
  for (std::size_t p = 0; p < t.size();) {
    std::size_t q = t.find('\n', p);
    if (q == std::string::npos) q = t.size();
    std::string line = t.substr(p, q - p);
    p = q + 1;
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.find_first_not_of(" \t") == std::string::npos) continue;
    const std::size_t tab = line.find('\t');
    if (tab == std::string::npos || tab == 0 || line.find('\t', tab + 1) != std::string::npos)
      return R::fail(Status::fail(Code::invalid_argument, "recipe line " + std::to_string(line_no) + " is not `name<TAB>TYPE`"));
    const std::string name = line.substr(0, tab);
    std::string ty = line.substr(tab + 1);
    for (char& c : ty) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    std::uint32_t id = 64;
    for (std::uint32_t k = 0; k < 64; ++k) {
      const quant::TypeInfo ti = quant::type_info(k);
      if (ti.known && ty == ti.name) id = k;
    }
    if (id == 64) return R::fail(Status::fail(Code::invalid_argument, "recipe line " + std::to_string(line_no) + ": unknown type '" + ty + "'", 0, name));
    if (!out.emplace(name, id).second) return R::fail(Status::fail(Code::invalid_argument, "the recipe names a tensor twice", 0, name));
  }
  if (out.empty()) return R::fail(Status::fail(Code::invalid_argument, "the recipe is empty"));
  return R::good(std::move(out));
}

bool rot32_selected(const std::string& name, const std::vector<std::string>& groups) {
  static const char* const kProducts[] = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_qkv.weight",
                                          "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight", "ffn_up_exps.weight",
                                          "ffn_gate_exps.weight", "ffn_down_exps.weight", "ffn_up_shexp.weight", "ffn_gate_shexp.weight",
                                          "ffn_down_shexp.weight"};
  bool product = name == "output.weight";
  if (name.rfind("blk.", 0) == 0) {
    const std::size_t dot = name.find('.', 4);
    if (dot != std::string::npos)
      for (const char* p : kProducts)
        if (name.compare(dot + 1, std::string::npos, p) == 0) product = true;
  }
  if (!product) return false;
  for (const std::string& g : groups)
    if (g == "all" || name.find(g) != std::string::npos) return true;
  return false;
}

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
  if (opt.imatrix) {  // the keys llama-quantize adds as overrides (quantize.cpp), in its order; the file by name only (rule 7)
    set_kv(kv, gguf_kv_str("quantize.imatrix.file", opt.imatrix_name.substr(0, 127)));
    if (!opt.imatrix->datasets.empty()) set_kv(kv, gguf_kv_str("quantize.imatrix.dataset", opt.imatrix->datasets[0].substr(0, 127)));
    set_kv(kv, gguf_kv_u32("quantize.imatrix.entries_count", static_cast<std::uint32_t>(opt.imatrix->weights.size())));
    if (opt.imatrix->chunk_count > 0) set_kv(kv, gguf_kv_u32("quantize.imatrix.chunks_count", opt.imatrix->chunk_count));
  }

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
  Result<std::vector<std::uint32_t>> types = quant::choose_types(descs, src.facts, opt.ftype);
  if (!types.ok()) return R::fail(types.status);
  if (!opt.recipe.empty()) {  // the recipe decides every type
    std::set<std::string> seen;
    for (std::size_t j = 0; j < order.size(); ++j) {
      const SourceTensor& t = st[order[j]];
      const auto it = opt.recipe.find(t.name);
      if (it == opt.recipe.end()) return R::fail(Status::fail(Code::invalid_argument, "the recipe does not name this tensor", 0, t.name));
      seen.insert(t.name);
      const std::uint32_t want = it->second;
      // a tensor the reference never quantizes may still change float width (F32 / F16 / BF16): Unsloth's MTP file keeps
      // blk.40.ffn_gate_inp in BF16 where the reference keeps F32; a block format stays refused
      if (!quant::tensor_allows_quantization(descs[j], src.facts.arch) && want != t.type && !(is_float_type(want) && is_float_type(t.type)))
        return R::fail(Status::fail(Code::unsupported, std::string("the recipe asks for ") + quant::type_info(want).name +
                                                           " but the reference never quantizes this tensor (it keeps " + quant::type_info(t.type).name + ")", 0, t.name));
      if (want != t.type && quant::row_bytes(want, t.ne[0]) == 0)
        return R::fail(Status::fail(Code::unsupported, std::string("the row length is not a whole number of ") + quant::type_info(want).name + " blocks", 0, t.name));
      types.value[j] = want;
    }
    for (const auto& [name, ty] : opt.recipe)
      if (seen.count(name) == 0) return R::fail(Status::fail(Code::invalid_argument, "the recipe names a tensor the source does not have", 0, name));
  }

  std::vector<GgufOutTensor> out;
  std::vector<char> rotate(order.size(), 0);
  std::vector<std::string> rotated_names;
  for (std::size_t j = 0; j < order.size(); ++j) {
    const SourceTensor& t = st[order[j]];
    const std::uint32_t to = types.value[j];
    if (!opt.rotate32.empty() && rot32_selected(t.name, opt.rotate32)) {
      if (t.ne[0] % 32 != 0) return R::fail(Status::fail(Code::unsupported, "a rotated tensor needs a row length that is a multiple of 32", 0, t.name));
      if (to == t.type && t.type != static_cast<std::uint32_t>(quant::Type::f32))
        return R::fail(Status::fail(Code::unsupported, "rotating a tensor that would be copied unchanged is built for F32 only", 0, t.name));
      if (!is_float_type(t.type) && !quant::can_dequantize(static_cast<quant::Type>(t.type)))
        return R::fail(Status::fail(Code::unsupported, "reading this type to rotate it is not built yet", 0, t.name));
      rotate[j] = 1;
      rotated_names.push_back(t.name);
    }
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
  // importance matrix: which tensors it weighs, and whether its rotation matches this file's
  std::vector<const std::vector<float>*> imat(order.size(), nullptr);
  if (opt.imatrix) {
    const std::set<std::string> im_rot(opt.imatrix->rot32.begin(), opt.imatrix->rot32.end());
    for (std::size_t j = 0; j < order.size(); ++j) {
      const SourceTensor& t = st[order[j]];
      const auto it = opt.imatrix->weights.find(t.name);
      if (it == opt.imatrix->weights.end()) continue;
      if (rotate[j] && im_rot.count(t.name) == 0)
        return R::fail(Status::fail(Code::invalid_argument, "this tensor is stored rotated but the importance matrix was collected on unrotated inputs "
                                                            "(collect it with `pelias imatrix` on a file whose weights are rotated the same way)", 0, t.name));
      if (!rotate[j] && im_rot.count(t.name) != 0)
        return R::fail(Status::fail(Code::invalid_argument, "the importance matrix was collected on rotated inputs but this tensor is not rotated here", 0, t.name));
      if (out[j].type == t.type) continue;  // copied: the reference uses no matrix
      if (it->second.size() != t.ne[0] * t.ne[2]) {
        if (t.name == "token_embd.weight") continue;  // the reference ignores this one mismatch
        return R::fail(Status::fail(Code::invalid_argument, "the importance matrix has " + std::to_string(it->second.size()) + " values for a tensor of " +
                                                                std::to_string(t.ne[0] * t.ne[2]), 0, t.name));
      }
      imat[j] = &it->second;
    }
  }
  if (!opt.rotate32.empty()) {
    if (rotated_names.empty()) return R::fail(Status::fail(Code::invalid_argument, "--rotate32 selected no product weight"));
    kv.push_back(gguf_kv_str("pelias.rot32.kind", "walsh-hadamard-32-orthonormal-row-blocks"));
    kv.push_back(gguf_kv_arr_str("pelias.rot32.tensors", rotated_names));
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
      if (o.type == t.type && !rotate[j]) {
        s = w.write(raw.data(), raw.size());
        if (!s) return s;
        continue;
      }
      const auto ne0 = static_cast<std::int64_t>(t.ne[0]);
      const std::uint64_t out_row = quant::row_bytes(o.type, t.ne[0]);
      f32.resize(static_cast<std::size_t>(nr * t.ne[0]));
      qbuf.resize(static_cast<std::size_t>(nr * out_row));
      const std::uint32_t tin = t.type, tout = o.type;
      const bool rot = rotate[j] != 0;
      const float* im = imat[j] ? imat[j]->data() : nullptr;
      const std::uint64_t rows_per_matrix = t.ne[1];
      parallel_rows(static_cast<std::int64_t>(nr), threads, [&](std::int64_t b, std::int64_t e) {
        for (std::int64_t r = b; r < e; ++r) {
          float* row = f32.data() + r * ne0;
          const std::uint8_t* rin = raw.data() + static_cast<std::uint64_t>(r) * in_row;
          if (tin == static_cast<std::uint32_t>(quant::Type::f32)) std::memcpy(row, rin, static_cast<std::size_t>(ne0) * 4);
          else to_f32(tin, rin, row, ne0);
          if (rot) quant::hadamard32(row, ne0);
          std::uint8_t* qo = qbuf.data() + static_cast<std::uint64_t>(r) * out_row;
          if (tout == static_cast<std::uint32_t>(quant::Type::f32)) std::memcpy(qo, row, static_cast<std::size_t>(ne0) * 4);
          else if (im) (void)quant::quantize_rows(static_cast<quant::Type>(tout), row, qo, 1, ne0, im + ((r0 + static_cast<std::uint64_t>(r)) / rows_per_matrix) * t.ne[0]);
          else (void)quant::quantize_row(static_cast<quant::Type>(tout), row, qo, ne0);
        }
      });
      if (!row_data_finite(o.type, qbuf.data(), qbuf.size()))
        return Status::fail(Code::corrupt, "quantized data is not finite (the input holds inf or NaN values)", 0, t.name);
      s = w.write(qbuf.data(), qbuf.size());
      if (!s) return s;
    }
    if (o.type == t.type && !rotate[j]) ++rep.copied;
    else ++rep.quantized;
    if (rotate[j]) ++rep.rotated;
    if (opt.imatrix && o.type != t.type) {
      if (imat[j] && quant::uses_importance(static_cast<quant::Type>(o.type))) ++rep.with_imatrix;
      else if (!imat[j] && opt.imatrix->weights.count(t.name) == 0) rep.imatrix_missing.push_back(t.name);
    }
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
