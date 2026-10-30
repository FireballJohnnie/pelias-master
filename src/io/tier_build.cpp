// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
#include "io/tier_build.h"

#include <algorithm>
#include <atomic>
#include <set>
#include <thread>

#include "io/gguf_writer.h"
#include "io/imatrix_weights.h"
#include "io/text_file.h"
#include "quant/quant_cpu.h"

namespace pelias::io {

namespace {
bool float_type(std::uint32_t t) {
  return t == static_cast<std::uint32_t>(quant::Type::f32) || t == static_cast<std::uint32_t>(quant::Type::f16) ||
         t == static_cast<std::uint32_t>(quant::Type::bf16);
}

std::uint32_t type_by_name(const std::string& name) {
  for (std::uint32_t k = 0; k < 256; ++k) {
    if (k == 64) k = 200;  // the reference library's ids (below 64) and the Pelias-only ids (200 and up)
    const quant::TypeInfo ti = quant::type_info(k);
    if (ti.known && name == ti.name) return k;
  }
  return 0xFFFFFFFFu;
}

struct Part {             // one output tensor
  std::size_t src = 0;    // index of the base tensor
  bool tiered = false;
  std::uint32_t type = 0;  // tiered: the type of this part
  std::vector<std::uint64_t> experts;  // tiered: the experts it holds, in order
};
}  // namespace

Result<std::map<std::string, std::vector<std::uint32_t>>> read_tier_alloc(const std::string& path) {
  using R = Result<std::map<std::string, std::vector<std::uint32_t>>>;
  const Result<std::string> text = read_text_file(path, std::uint64_t{64} << 20);
  if (!text.ok()) return R::fail(text.status);
  std::map<std::string, std::vector<std::uint32_t>> out;
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
    if (tab == std::string::npos || tab == 0)
      return R::fail(Status::fail(Code::invalid_argument, "tier line " + std::to_string(line_no) + " is not `name<TAB>TYPE,TYPE,...`"));
    const std::string name = line.substr(0, tab);
    std::vector<std::uint32_t> types;
    for (std::size_t a = tab + 1; a <= line.size();) {
      std::size_t b = line.find(',', a);
      if (b == std::string::npos) b = line.size();
      const std::uint32_t id = type_by_name(line.substr(a, b - a));
      if (id == 0xFFFFFFFFu)
        return R::fail(Status::fail(Code::invalid_argument, "tier line " + std::to_string(line_no) + ": unknown type '" + line.substr(a, b - a) + "'", 0, name));
      types.push_back(id);
      a = b + 1;
    }
    if (!out.emplace(name, std::move(types)).second) return R::fail(Status::fail(Code::invalid_argument, "the tier names a weight twice", 0, name));
  }
  if (out.empty()) return R::fail(Status::fail(Code::invalid_argument, "the tier allocation is empty"));
  return R::good(std::move(out));
}

Result<TierBuildReport> build_tiered(const os::FileMap& base_map, const GgufReader& base, const os::FileMap& orig_map, const GgufReader& orig,
                                     const std::map<std::string, std::vector<std::uint32_t>>& alloc, const std::string& out_path,
                                     const TierBuildOptions& opt, const TierProgress& progress) {
  using R = Result<TierBuildReport>;
  TierBuildReport rep;
  if (base.alignment() != 32) return R::fail(Status::fail(Code::unsupported, "the base file's alignment is not 32"));
  ImatrixWeights imx;
  if (!opt.imatrix.empty()) {
    Result<ImatrixWeights> r = read_imatrix_weights(opt.imatrix);
    if (!r.ok()) return R::fail(r.status);
    imx = std::move(r.value);
  }
  // metadata: the base file's entries as encoded, then the tier's own
  std::vector<GgufKvBlob> kv;
  for (const GgufKv& e : base.kv()) {
    if (e.key.rfind("pelias.tier.", 0) == 0) return R::fail(Status::fail(Code::invalid_argument, "the base file is already tiered"));
    kv.emplace_back(base_map.data() + e.file_begin, base_map.data() + e.file_end);
  }
  kv.push_back(gguf_kv_u32("pelias.tier.version", 1));
  // tensors in the base file's order; each tiered weight becomes one part per type (by type id)
  const auto& ts = base.tensors();
  std::vector<Part> parts;
  std::vector<GgufOutTensor> out;
  std::set<std::string> used;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    const GgufTensor& t = ts[i];
    const auto it = alloc.find(t.name);
    if (it == alloc.end()) {
      Part p;
      p.src = i;
      parts.push_back(p);
      GgufOutTensor o;
      o.name = t.name;
      for (int k = 0; k < 4; ++k) o.ne[k] = t.ne[k];
      o.type = t.type;
      o.bytes = t.bytes;
      o.n_dims = t.n_dims;
      out.push_back(o);
      ++rep.copied;
      continue;
    }
    used.insert(t.name);
    const std::vector<std::uint32_t>& types = it->second;
    if (t.n_dims != 3 || t.ne[3] != 1 || types.size() != t.ne[2])
      return R::fail(Status::fail(Code::invalid_argument, "the tier gives " + std::to_string(types.size()) + " types; the weight has " + std::to_string(t.ne[2]) + " experts", 0, t.name));
    const GgufTensor* src = orig.find_tensor(t.name);
    if (!src) return R::fail(Status::fail(Code::invalid_argument, "the original model has no tensor of this name", 0, t.name));
    if (src->ne[0] != t.ne[0] || src->ne[1] != t.ne[1] || src->ne[2] != t.ne[2] || src->ne[3] != 1)
      return R::fail(Status::fail(Code::invalid_argument, "the original's shape differs from the base file's", 0, t.name));
    if (!float_type(src->type)) return R::fail(Status::fail(Code::invalid_argument, "the original is not F32 / F16 / BF16", 0, t.name));
    std::map<std::uint32_t, std::vector<std::uint64_t>> by_type;
    for (std::uint64_t e = 0; e < types.size(); ++e) by_type[types[e]].push_back(e);
    std::vector<std::int32_t> ids;
    for (const std::uint32_t ty : types) ids.push_back(static_cast<std::int32_t>(ty));
    kv.push_back(gguf_kv_arr_i32(tier_types_key(t.name), ids));
    for (const auto& [ty, ex] : by_type) {
      if (!quant::can_quantize(static_cast<quant::Type>(ty)) || !quant::can_dequantize(static_cast<quant::Type>(ty)))
        return R::fail(Status::fail(Code::unsupported, std::string("quantizing to ") + quant::type_info(ty).name + " is not built", 0, t.name));
      const std::uint64_t rb = quant::row_bytes(ty, t.ne[0]);
      if (rb == 0) return R::fail(Status::fail(Code::unsupported, std::string("the row is not a whole number of ") + quant::type_info(ty).name + " blocks", 0, t.name));
      Part p;
      p.src = i;
      p.tiered = true;
      p.type = ty;
      p.experts = ex;
      parts.push_back(p);
      GgufOutTensor o;
      o.name = tier_part_name(t.name, ty);
      o.ne[0] = t.ne[0];
      o.ne[1] = t.ne[1];
      o.ne[2] = ex.size();
      o.type = ty;
      o.bytes = rb * t.ne[1] * ex.size();
      o.n_dims = 3;
      out.push_back(o);
      rep.expert_bytes_out += o.bytes;
      rep.experts_by_type[ty] += ex.size();
    }
    rep.expert_bytes_base += t.bytes;
    ++rep.tiered;
  }
  for (const auto& [name, types] : alloc)
    if (used.count(name) == 0) return R::fail(Status::fail(Code::invalid_argument, "the tier names a weight the base file does not have", 0, name));
  for (const GgufOutTensor& o : out) rep.file_bytes += o.bytes;

  GgufFileWriter wr;
  if (Status s = wr.begin(out_path, kv, out, 32); !s) return R::fail(s);
  const int threads = opt.threads > 0 ? opt.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  std::vector<std::uint8_t> buf;
  std::set<std::string> imx_used;
  for (std::size_t pi = 0; pi < parts.size(); ++pi) {
    const Part& p = parts[pi];
    const GgufTensor& t = ts[p.src];
    Status st = Status::good();
    if (!p.tiered) {  // the base file's own bytes
      constexpr std::uint64_t kChunk = std::uint64_t{64} << 20;
      for (std::uint64_t off = 0; off < t.bytes && st; off += kChunk)
        st = wr.write(base_map.data() + t.file_offset + off, std::min(kChunk, t.bytes - off));
    } else {
      const GgufTensor* src = orig.find_tensor(t.name);
      const auto n = static_cast<std::int64_t>(t.ne[0]);
      const std::uint64_t rows = t.ne[1], srb = quant::row_bytes(src->type, t.ne[0]), rb = quant::row_bytes(p.type, t.ne[0]);
      const std::vector<float>* imp = nullptr;
      if (auto it = imx.weights.find(t.name); it != imx.weights.end() && it->second.size() == t.ne[0] * t.ne[2] &&
                                              std::find(imx.rot32.begin(), imx.rot32.end(), t.name) == imx.rot32.end())
        imp = &it->second, imx_used.insert(t.name);
      for (const std::uint64_t e : p.experts) {  // one expert at a time: rows in parallel
        buf.assign(rb * rows, 0);
        std::atomic<std::uint64_t> next{0};
        std::atomic<bool> bad{false};
        auto work = [&]() {
          std::vector<float> row(static_cast<std::size_t>(n));
          for (std::uint64_t r = next.fetch_add(1); r < rows; r = next.fetch_add(1)) {
            const std::uint8_t* in = orig_map.data() + src->file_offset + (e * rows + r) * srb;
            (void)quant::dequantize_row(static_cast<quant::Type>(src->type), in, row.data(), n);
            const float* w = imp ? imp->data() + e * t.ne[0] : nullptr;
            if (!quant::quantize_rows(static_cast<quant::Type>(p.type), row.data(), buf.data() + r * rb, 1, n, w)) bad = true;
          }
        };
        std::vector<std::thread> pool;
        for (int k = 1; k < threads; ++k) pool.emplace_back(work);
        work();
        for (auto& th : pool) th.join();
        if (bad) {
          st = Status::fail(Code::unsupported, std::string("quantizing to ") + quant::type_info(p.type).name + " failed", 0, t.name);
          break;
        }
        st = wr.write(buf.data(), buf.size());
        if (!st) break;
      }
    }
    if (!st) {
      wr.abandon();  // the incomplete output stays on disk for the author to remove (nothing is deleted here)
      return R::fail(st);
    }
    if (progress) progress(pi + 1, parts.size(), p.tiered ? tier_part_name(t.name, p.type) : t.name);
  }
  if (Status s = wr.finish(); !s) return R::fail(s);
  rep.with_imatrix = imx_used.size();
  return R::good(std::move(rep));
}

}  // namespace pelias::io
