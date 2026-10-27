// provenance: original (the conversion READ-SRC, see stage_a.h)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "ingest/stage_a.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <set>

#include "ingest/hf_metadata.h"
#include "ingest/hf_vocab.h"
#include "io/text_file.h"
#include "pelias/timer.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"

namespace pelias::ingest {
namespace {

using io::GgufKvBlob;
using io::Json;
namespace fs = std::filesystem;

constexpr std::uint32_t kQuantizationVersion = 2;  // GGML_QUANT_VERSION
constexpr auto kBF16 = static_cast<std::uint32_t>(quant::Type::bf16);
constexpr auto kF32 = static_cast<std::uint32_t>(quant::Type::f32);

Status bad(const std::string& msg, const std::string& name = {}) { return Status::fail(Code::unsupported, msg, 0, name); }

// The reference's TextModel.filter_tensors: names it never converts (vision, audio and projector parts)
bool skipped_name(const std::string& n) {
  for (const char* p : {"mlp", "vit.", "vpm.", "siglip2.", "conformer.", "merger.", "resampler.", "sound_encoder.", "sound_projection.", "speech_embeddings."})
    if (n.rfind(p, 0) == 0) return true;
  for (const char* p : {"visual.", "vision.", "audio.", "talker.", "vision_", "audio_", "token2wav.", "code2wav.", "projector.", "pre_mm_projector_norm",
                        "image_newline", "view_seperator", "patch_embed", "patch_embedding", "patch_merger.", "patch_merge_mlp.", "model.connector."})
    if (n.find(p) != std::string::npos) return true;
  return false;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
  std::size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) {
    s.replace(p, from.size(), to);
    p += to.size();
  }
  return s;
}

// The renames of ModelBase.filter_tensors and _QwenMtpMixin.filter_tensors. Returns false for a skipped tensor.
bool renamed(std::string& name, std::uint32_t n_layer) {
  if (skipped_name(name)) return false;
  if (name.size() >= 23 && name.compare(name.size() - 23, 23, "e_score_correction_bias") == 0)
    name = replace_all(name, "e_score_correction_bias", "e_score_correction.bias");
  name = replace_all(name, "language_model.", "");
  if (name.rfind("model.mtp.", 0) == 0) name = name.substr(6);
  if (name.rfind("mtp.", 0) == 0) {
    // mtp.layers.N.<rest> -> model.layers.<n_layer + N>.<rest>; mtp.fc / pre_fc_norm_* / norm -> model.layers.<n_layer>.*
    std::vector<std::string> parts;
    std::size_t i = 0;
    for (int k = 0; k < 3; ++k) {
      const std::size_t j = name.find('.', i);
      if (j == std::string::npos) break;
      parts.push_back(name.substr(i, j - i));
      i = j + 1;
    }
    parts.push_back(name.substr(i));
    const auto is_dec = [](const std::string& s) { return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; }); };
    if (parts.size() == 4 && parts[1] == "layers" && is_dec(parts[2])) {
      name = "model.layers." + std::to_string(n_layer + std::stoull(parts[2])) + "." + parts[3];
    } else if (parts.size() == 3) {
      const std::string& f = parts[1];
      const char* to = f == "fc" ? "eh_proj" : f == "pre_fc_norm_embedding" ? "enorm" : f == "pre_fc_norm_hidden" ? "hnorm" : f == "norm" ? "shared_head.norm" : nullptr;
      if (to) name = "model.layers." + std::to_string(n_layer) + "." + to + "." + parts[2];
    }
  }
  return true;
}

// pathlib's Path(folder).name: the last component, "." components dropped, trailing separators ignored
std::string path_name(const std::string& folder) {
  std::vector<std::string> comps;
  std::string cur;
  for (const char c : folder) {
    if (c == '/' || c == '\\') {
      if (!cur.empty() && cur != ".") comps.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty() && cur != ".") comps.push_back(cur);
  if (comps.empty()) return {};
  const std::string& last = comps.back();
  if (last.size() == 2 && last[1] == ':') return {};  // a bare drive ("D:")
  return last;
}

struct Folder {
  std::string dir;
  Json config;     // as written
  Json hparams;    // config.json with text_config's entries on top
  std::string readme;
  bool has_readme = false;
  Json gen;
  bool has_gen = false;
  Json tokenizer, tokenizer_config, chat_template_json;
  bool has_tokenizer_config = false, has_chat_template_json = false;
  std::string chat_template_jinja;
  bool has_chat_template_jinja = false, additional_templates = false;
  std::vector<std::unique_ptr<io::SafetensorsShard>> shards;
  std::vector<std::string> shard_names;
  std::vector<SourceRef> sources;
};

Result<Json> read_json(const std::string& path, const char* what) {
  Result<std::string> t = io::read_text_file(path);
  if (!t.ok()) return Result<Json>::fail(t.status);
  Result<Json> j = Json::parse(t.value);
  if (!j.ok()) {
    j.status.name = what;
    return j;
  }
  return j;
}

Status open_folder(const std::string& dir, Folder& f) {
  f.dir = dir;
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return Status::fail(Code::not_found, "the model folder does not exist");
  const fs::path d(dir);
  Result<Json> cfg = read_json((d / "config.json").string(), "config.json");
  if (!cfg.ok()) return cfg.status;
  f.config = cfg.value;
  if (!f.config.is_object()) return bad("config.json is not an object", "config.json");
  // the reference's architecture: architectures[0], or text_config's when it names one
  std::string arch;
  if (const Json* a = f.config.get("architectures"); a && a->is_array() && a->size() > 0 && a->items()[0].is_string()) arch = a->items()[0].as_string();
  if (const Json* tc = f.config.get("text_config"); tc && tc->is_object())
    if (const Json* a = tc->get("architectures"); a && !a->is_null()) {
      if (!a->is_array() || a->size() == 0 || !a->items()[0].is_string()) return bad("text_config.architectures is not a list of names", "config.json");
      arch = a->items()[0].as_string();
    }
  if (arch != "Qwen3_5MoeForConditionalGeneration" && arch != "Qwen3_5MoeForCausalLM")
    return bad("architecture '" + arch + "' is not converted yet (Qwen3.5/3.6 MoE is)", "config.json");
  f.hparams = Json::object();
  for (const auto& [k, v] : f.config.members()) f.hparams.set(k, v);
  if (const Json* tc = f.config.get("text_config"); tc && tc->is_object())
    for (const auto& [k, v] : tc->members()) f.hparams.set(k, v);

  if (Result<std::string> r = io::read_text_file((d / "README.md").string()); r.ok()) {
    f.readme = std::move(r.value);
    f.has_readme = true;
  }
  if (Result<std::string> r = io::read_text_file((d / "generation_config.json").string()); r.ok()) {
    Result<Json> j = Json::parse(r.value);
    if (j.ok()) {  // the reference ignores a generation config it cannot parse
      f.gen = j.value;
      f.has_gen = true;
    }
  }
  Result<Json> tok = read_json((d / "tokenizer.json").string(), "tokenizer.json");
  if (!tok.ok()) return tok.status;
  f.tokenizer = std::move(tok.value);
  if (fs::is_regular_file(d / "tokenizer_config.json", ec)) {
    Result<Json> j = read_json((d / "tokenizer_config.json").string(), "tokenizer_config.json");
    if (!j.ok()) return j.status;
    f.tokenizer_config = std::move(j.value);
    f.has_tokenizer_config = true;
  }
  if (fs::is_regular_file(d / "chat_template.jinja", ec)) {
    Result<std::string> r = io::read_text_file((d / "chat_template.jinja").string());
    if (!r.ok()) return r.status;
    f.chat_template_jinja = std::move(r.value);
    f.has_chat_template_jinja = true;
    if (fs::is_directory(d / "additional_chat_templates", ec))
      for (const auto& e : fs::directory_iterator(d / "additional_chat_templates", ec))
        if (e.path().extension() == ".jinja") f.additional_templates = true;
  } else if (fs::is_regular_file(d / "chat_template.json", ec)) {
    Result<Json> j = read_json((d / "chat_template.json").string(), "chat_template.json");
    if (!j.ok()) return j.status;
    f.chat_template_json = std::move(j.value);
    f.has_chat_template_json = true;
  }

  // ---- shards: the index's files (sorted), or the folder's model*.safetensors files
  std::set<std::string> index_names;
  bool has_index = false;
  if (fs::is_regular_file(d / "model.safetensors.index.json", ec)) {
    Result<Json> j = read_json((d / "model.safetensors.index.json").string(), "model.safetensors.index.json");
    if (!j.ok()) return j.status;
    const Json* wm = j.value.get("weight_map");
    if (!wm || !wm->is_object()) return Status::fail(Code::corrupt, "no weight_map in the index", 0, "model.safetensors.index.json");
    std::set<std::string> parts;
    for (const auto& [k, v] : wm->members()) {
      if (!v.is_string()) return Status::fail(Code::corrupt, "a weight_map entry is not a file name", 0, k);
      index_names.insert(k);
      parts.insert(v.as_string());
    }
    f.shard_names.assign(parts.begin(), parts.end());
    has_index = true;
  } else {
    for (const auto& e : fs::directory_iterator(d, ec)) {
      const std::string n = e.path().filename().string();
      if (n.rfind("model", 0) == 0 && n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) f.shard_names.push_back(n);
    }
    std::sort(f.shard_names.begin(), f.shard_names.end());
  }
  if (f.shard_names.empty()) return Status::fail(Code::not_found, "no safetensors files in the model folder");

  std::uint32_t n_layer = 0;
  if (const Json* nl = f.hparams.get("num_hidden_layers"); nl && nl->is_integer() && nl->as_int() >= 0 && nl->as_uint() < 100000)
    n_layer = static_cast<std::uint32_t>(nl->as_uint());
  else
    return bad("num_hidden_layers is missing", "config.json");
  std::set<std::string> seen_parts, kept;
  for (std::size_t si = 0; si < f.shard_names.size(); ++si) {
    auto shard = std::make_unique<io::SafetensorsShard>();
    Status s = shard->open((d / f.shard_names[si]).string());
    if (!s) {
      s.name = f.shard_names[si] + (s.name.empty() ? "" : ": " + s.name);
      return s;
    }
    // names in sorted order, as the reference's shard reader lists them
    std::vector<std::size_t> idx(shard->reader().tensors().size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    const auto& ts = shard->reader().tensors();
    std::sort(idx.begin(), idx.end(), [&](std::size_t x, std::size_t y) { return ts[x].name < ts[y].name; });
    for (const std::size_t i : idx) {
      seen_parts.insert(ts[i].name);
      std::string name = ts[i].name;
      if (!renamed(name, n_layer)) continue;
      if (!kept.insert(name).second) return Status::fail(Code::corrupt, "two source tensors map to one name", 0, name);
      SourceRef r;
      r.name = name;
      r.shard = si;
      r.index = i;
      r.t = &ts[i];
      f.sources.push_back(r);
    }
    f.shards.push_back(std::move(shard));
  }
  if (has_index && seen_parts != index_names) return Status::fail(Code::corrupt, "the index and the shards list different tensors (a shard missing or incomplete?)");
  return Status::good();
}

struct Assembled {
  std::vector<GgufKvBlob> kv;
  ArchPlan arch;
};

Result<Assembled> assemble(const Folder& f) {
  using R = Result<Assembled>;
  Assembled a;
  Result<ArchPlan> arch = plan_qwen35moe(f.hparams, f.sources, static_cast<std::uint32_t>(quant::FileType::bf16));
  if (!arch.ok()) return R::fail(arch.status);
  a.arch = std::move(arch.value);
  std::vector<NamedShape> shapes;
  for (const TensorPlan& t : a.arch.tensors) shapes.push_back({t.name, t.shape});
  const ParamCount pc = count_parameters(shapes);
  Result<std::vector<GgufKvBlob>> general =
      general_metadata(folder_name(f.dir), f.has_readme ? &f.readme : nullptr, &f.config, f.has_gen ? &f.gen : nullptr, pc);
  if (!general.ok()) return R::fail(general.status);
  VocabFiles vf;
  vf.tokenizer = &f.tokenizer;
  vf.tokenizer_config = f.has_tokenizer_config ? &f.tokenizer_config : nullptr;
  vf.config = &f.config;
  vf.chat_template_jinja = f.has_chat_template_jinja ? &f.chat_template_jinja : nullptr;
  vf.chat_template_json = f.has_chat_template_json ? &f.chat_template_json : nullptr;
  vf.additional_chat_templates = f.additional_templates;
  std::uint64_t vocab_size = 0;
  if (const Json* v = f.hparams.get("vocab_size"); v && !v->is_null()) {
    if (!v->is_integer() || v->as_int() <= 0) return R::fail(bad("vocab_size is not a positive integer", "config.json"));
    vocab_size = v->as_uint();
  }
  Result<std::vector<GgufKvBlob>> vocab = gpt2_vocab_metadata(vf, vocab_size);
  if (!vocab.ok()) return R::fail(vocab.status);

  a.kv.push_back(io::gguf_kv_str("general.architecture", a.arch.arch));
  a.kv.push_back(io::gguf_kv_str("general.type", "model"));
  for (auto& b : general.value) a.kv.push_back(std::move(b));
  for (const auto& b : a.arch.kv) a.kv.push_back(b);
  a.kv.push_back(io::gguf_kv_u32("general.quantization_version", kQuantizationVersion));
  for (auto& b : vocab.value) a.kv.push_back(std::move(b));
  return R::good(std::move(a));
}

// ---- the row reader: output rows of a planned tensor, in its output type (BF16 bytes copied, F32 widened)
std::uint64_t head_index(const TensorPlan& p, std::uint64_t r) {
  if (r < p.head_start) return r;
  const std::uint64_t n = r - p.head_start;
  const std::uint64_t block = p.nk * p.nvpk * p.hd;
  if (n >= block) return r;
  const std::uint64_t j = n / (p.nk * p.hd);
  const std::uint64_t rem = n % (p.nk * p.hd);
  const std::uint64_t i = rem / p.hd;
  const std::uint64_t h = rem % p.hd;
  return p.head_start + i * p.nvpk * p.hd + j * p.hd + h;
}

std::uint64_t rows_of(const std::vector<std::uint64_t>& shape) {
  std::uint64_t r = 1;
  for (std::size_t i = 0; i + 1 < shape.size(); ++i) r *= shape[i];
  return r;
}

float transform(TensorPlan::Op op, float x) {
  switch (op) {
    case TensorPlan::Op::plus_one: return x + 1.0f;  // a float32 addition, rounded once
    case TensorPlan::Op::neg_exp: return -static_cast<float>(std::exp(static_cast<double>(x)));
    default: return x;
  }
}

Status read_rows(const Folder& f, const TensorPlan& p, std::uint64_t row0, std::uint64_t n, std::uint8_t* out) {
  const io::SafetensorsShard& sh = *f.shards[p.shard];
  const io::SafetensorsTensor& t = sh.reader().tensors()[p.source];
  const auto* src = reinterpret_cast<const std::uint8_t*>(sh.data(t));
  const std::uint64_t cols = p.shape.empty() ? 1 : p.shape.back();
  const std::uint64_t src_rows = t.bytes() / 2 / cols;
  for (std::uint64_t k = 0; k < n; ++k) {
    const std::uint64_t r = row0 + k;
    std::uint64_t sr = r;
    if (p.rows == TensorPlan::Rows::expert_part) {
      const std::uint64_t e = r / p.part_rows, kk = r % p.part_rows;
      sr = e * 2 * p.part_rows + p.part_offset + kk;
    } else if (p.rows == TensorPlan::Rows::heads && !p.head_columns) {
      sr = head_index(p, r);
    }
    if (sr >= src_rows) return Status::fail(Code::internal, "a planned row lies outside its source", 0, p.name);
    const std::uint8_t* s = src + sr * cols * 2;
    if (p.type == kBF16) {
      std::uint8_t* o = out + k * cols * 2;
      if (p.rows == TensorPlan::Rows::heads && p.head_columns)
        for (std::uint64_t c = 0; c < cols; ++c) std::memcpy(o + c * 2, s + head_index(p, c) * 2, 2);
      else
        std::memcpy(o, s, static_cast<std::size_t>(cols * 2));
      // the reference widens to f32 and rounds back to bf16, which sets the quiet bit of a NaN
      for (std::uint64_t c = 0; c < cols; ++c) {
        std::uint16_t h;
        std::memcpy(&h, o + c * 2, 2);
        if ((h & 0x7FFFu) > 0x7F80u) {
          h = static_cast<std::uint16_t>(h | 0x40u);
          std::memcpy(o + c * 2, &h, 2);
        }
      }
    } else {
      auto* o = reinterpret_cast<std::uint8_t*>(out + k * cols * 4);
      for (std::uint64_t c = 0; c < cols; ++c) {
        const std::uint64_t sc = p.rows == TensorPlan::Rows::heads && p.head_columns ? head_index(p, c) : c;
        std::uint16_t h;
        std::memcpy(&h, s + sc * 2, 2);
        const float x = transform(p.op, quant::bf16_to_fp32(h));
        std::memcpy(o + c * 4, &x, 4);
      }
    }
  }
  return Status::good();
}

std::uint64_t elem_bytes(std::uint32_t type) { return type == kF32 ? 4 : 2; }

}  // namespace

std::string folder_name(const std::string& folder) { return path_name(folder); }

struct StageASource::Impl {
  Folder f;
  Assembled a;
};

StageASource::~StageASource() { delete impl_; }

Result<std::unique_ptr<StageASource>> StageASource::open(const std::string& folder) {
  using R = Result<std::unique_ptr<StageASource>>;
  std::unique_ptr<StageASource> s(new StageASource());
  s->impl_ = new Impl();
  Status st = open_folder(folder, s->impl_->f);
  if (!st) return R::fail(st);
  Result<Assembled> a = assemble(s->impl_->f);
  if (!a.ok()) return R::fail(a.status);
  s->impl_->a = std::move(a.value);
  for (const TensorPlan& t : s->impl_->a.arch.tensors)
    if (t.shape.size() > 4) return R::fail(Status::fail(Code::unsupported, "a tensor with more than 4 dimensions", 0, t.name));
  return R::good(std::move(s));
}

const std::vector<GgufKvBlob>& StageASource::kv() const { return impl_->a.kv; }
const std::vector<TensorPlan>& StageASource::tensors() const { return impl_->a.arch.tensors; }
const quant::ModelFacts& StageASource::facts() const { return impl_->a.arch.facts; }
Status StageASource::read_rows(std::size_t i, std::uint64_t row0, std::uint64_t n, std::uint8_t* out) const {
  return ingest::read_rows(impl_->f, impl_->a.arch.tensors[i], row0, n, out);
}
std::uint64_t StageASource::rows(std::size_t i) const { return rows_of(impl_->a.arch.tensors[i].shape); }
std::uint64_t StageASource::row_bytes(std::size_t i) const {
  const TensorPlan& t = impl_->a.arch.tensors[i];
  return (t.shape.empty() ? 1 : t.shape.back()) * elem_bytes(t.type);
}

namespace {

io::GgufOutTensor out_tensor(const TensorPlan& t) {
  io::GgufOutTensor o;
  o.name = t.name;
  std::uint64_t n = 1;
  for (std::size_t i = 0; i < t.shape.size(); ++i) {
    o.ne[i] = t.shape[t.shape.size() - 1 - i];
    n *= t.shape[i];
  }
  o.n_dims = static_cast<std::uint32_t>(t.shape.size());
  o.type = t.type;
  o.bytes = n * elem_bytes(t.type);
  return o;
}

}  // namespace

Result<ConvertPreview> preview_folder(const std::string& folder) {
  using R = Result<ConvertPreview>;
  Result<std::unique_ptr<StageASource>> src = StageASource::open(folder);
  if (!src.ok()) return R::fail(src.status);
  ConvertPreview p;
  p.kv = src.value->kv();
  for (const TensorPlan& t : src.value->tensors()) p.tensors.push_back(out_tensor(t));
  return R::good(std::move(p));
}

Result<io::RequantReport> convert_folder(const std::string& folder, const std::string& out_path, const ConvertOptions& opt,
                                         const io::RequantProgress& progress) {
  using R = Result<io::RequantReport>;
  Timer timer;
  if (opt.ftype == quant::FileType::f16 || opt.ftype == quant::FileType::all_f32)
    return R::fail(Status::fail(Code::unsupported, "convert writes BF16 or a quantized mix; for F16 or F32 convert to BF16, then use 'pelias quantize'"));
  Result<std::unique_ptr<StageASource>> opened = StageASource::open(folder);
  if (!opened.ok()) return R::fail(opened.status);
  const StageASource& sa = *opened.value;
  const std::vector<TensorPlan>& plans = sa.tensors();

  if (opt.ftype == quant::FileType::bf16) {
    // the converter's own file: its order, its shapes (the array's rank), its types
    std::vector<io::GgufOutTensor> outs;
    for (const TensorPlan& t : plans) outs.push_back(out_tensor(t));
    io::RequantReport rep;
    std::vector<std::uint8_t> buf;
    constexpr std::uint64_t kSlabBytes = std::uint64_t{128} << 20;
    const Status ws = io::write_gguf_file(out_path, sa.kv(), outs, [&](std::size_t j, io::GgufFileWriter& w) -> Status {
      const TensorPlan& p = plans[j];
      const std::uint64_t row_bytes = sa.row_bytes(j), rows = sa.rows(j);
      const std::uint64_t slab = std::max<std::uint64_t>(1, kSlabBytes / row_bytes);
      for (std::uint64_t r0 = 0; r0 < rows; r0 += slab) {
        const std::uint64_t nr = std::min(slab, rows - r0);
        buf.resize(static_cast<std::size_t>(nr * row_bytes));
        Status st = sa.read_rows(j, r0, nr, buf.data());
        if (!st) return st;
        st = w.write(buf.data(), buf.size());
        if (!st) return st;
      }
      ++rep.tensors;
      if (p.type == kBF16 && p.op == TensorPlan::Op::none) ++rep.copied;
      else ++rep.quantized;  // widened to F32 (and transformed): reported as "converted"
      rep.bytes_in += outs[j].bytes / elem_bytes(p.type) * 2;  // read as BF16
      rep.bytes_out += outs[j].bytes;
      ++rep.by_type[p.type];
      if (progress) progress(j, plans.size(), p.name, p.type);
      return Status::good();
    });
    if (!ws) return R::fail(ws);
    rep.seconds = timer.seconds();
    return R::good(rep);
  }

  // a stock mix: the reference quantizer's procedure on the BF16 file this folder converts to, without writing it
  io::QuantSource src;
  src.kv = sa.kv();
  src.facts = sa.facts();
  for (const TensorPlan& t : plans) {
    const io::GgufOutTensor o = out_tensor(t);
    io::SourceTensor x;
    x.name = o.name;
    for (int i = 0; i < 4; ++i) x.ne[i] = o.ne[i];
    x.type = o.type;
    x.bytes = o.bytes;
    src.tensors.push_back(x);
  }
  src.read = [&](std::size_t i, std::uint64_t row0, std::uint64_t n, std::uint8_t* outp) { return sa.read_rows(i, row0, n, outp); };
  io::RequantOptions ro;
  ro.ftype = opt.ftype;
  ro.threads = opt.threads;
  Result<io::RequantReport> r = io::quantize_source(src, out_path, ro, progress);
  if (r.ok()) r.value.seconds = timer.seconds();
  return r;
}

}  // namespace pelias::ingest
