// provenance: original (rules READ-SRC, see hf_vocab.h)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "ingest/hf_vocab.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace pelias::ingest {
namespace {

using io::GgufKvBlob;
using io::Json;

constexpr std::int32_t kNormal = 1, kControl = 3, kUserDefined = 4, kUnused = 5;  // gguf TokenType

Status bad(const std::string& msg, const std::string& file = "tokenizer.json") { return Status::fail(Code::unsupported, msg, 0, file); }

bool starts(const std::string& s, const std::string& p) { return s.size() >= p.size() && s.compare(0, p.size(), p) == 0; }
bool ends(const std::string& s, const std::string& p) { return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0; }

// conversion/base.py does_token_look_special
bool looks_special(const std::string& t) {
  if (t == "<pad>" || t == "<mask>" || t == "<2mass>" || t == "[@BOS@]") return true;
  if (starts(t, "<|") && ends(t, "|>")) return true;
  if (starts(t, "<\xEF\xBD\x9C") && ends(t, "\xEF\xBD\x9C>")) return true;  // fullwidth vertical bar U+FF5C
  if (starts(t, "<unused") && ends(t, ">")) return true;
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

// The pre-tokenizer families Pelias recognizes, by the exact definition in tokenizer.json. Each entry was matched to the
// reference's name for it: "qwen35" is what the reference writes for the Qwen3.5 / 3.6 tokenizer (READ-SRC: its hash
// table names Qwen/Qwen3.5-9B-Instruct "qwen35"; the served model's file, converted by the reference, carries "qwen35").
struct PreTokenizer {
  const char* name;
  const char* normalizer;     // compact JSON
  const char* pre_tokenizer;  // compact JSON
};
const PreTokenizer kPreTokenizers[] = {
    {"qwen35", R"({"type":"NFC"})",
     R"({"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]})"},
};

void parse_normalizer(const Json& n, std::optional<bool>& lowercase, std::optional<bool>& strip_accents) {
  const Json* t = n.get("type");
  const std::string type = t && t->is_string() ? t->as_string() : std::string();
  if (type == "Lowercase") lowercase = true;
  else if (type == "StripAccents") strip_accents = true;
  else if (type == "BertNormalizer") {
    if (const Json* v = n.get("lowercase"); v && v->is_bool()) lowercase = v->as_bool();
    if (const Json* v = n.get("strip_accents"); v && v->is_bool()) strip_accents = v->as_bool();
  } else if (type == "Sequence") {
    if (const Json* list = n.get("normalizers"); list && list->is_array())
      for (const Json& x : list->items())
        if (x.is_object()) parse_normalizer(x, lowercase, strip_accents);
  }
}

}  // namespace

Result<std::vector<GgufKvBlob>> gpt2_vocab_metadata(const VocabFiles& f, std::uint64_t vocab_size_in) {
  using R = Result<std::vector<GgufKvBlob>>;
  if (!f.tokenizer || !f.tokenizer->is_object()) return R::fail(bad("tokenizer.json is required (other tokenizer files are not read yet)"));
  const Json& tok = *f.tokenizer;
  const Json* model = tok.get("model");
  if (!model || !model->is_object()) return R::fail(bad("no 'model' in tokenizer.json"));
  const Json* mtype = model->get("type");
  if (!mtype || !mtype->is_string() || mtype->as_string() != "BPE") return R::fail(bad("only byte-level BPE tokenizers are converted yet"));
  const Json* vocab = model->get("vocab");
  if (!vocab || !vocab->is_object()) return R::fail(bad("the BPE vocabulary is missing"));

  // ---- the library's vocabulary: the model's tokens and the added tokens
  struct Added {
    std::string content;
    bool special = false, normalized = false;
  };
  std::map<std::uint64_t, Added> added;  // by id
  std::unordered_map<std::string, std::uint64_t> added_ids;
  if (const Json* at = tok.get("added_tokens"); at && at->is_array()) {
    for (const Json& a : at->items()) {
      const Json* id = a.get("id");
      const Json* content = a.get("content");
      if (!id || !id->is_integer() || id->as_int() < 0 || !content || !content->is_string()) return R::fail(bad("an added token without an id or text"));
      Added x;
      x.content = content->as_string();
      if (const Json* s = a.get("special"); s && s->is_bool()) x.special = s->as_bool();
      if (const Json* n = a.get("normalized"); n && n->is_bool()) x.normalized = n->as_bool();
      if (added.count(id->as_uint()) || added_ids.count(x.content)) return R::fail(bad("an added token appears twice"));
      added_ids[x.content] = id->as_uint();
      added[id->as_uint()] = x;
    }
  }
  // tokenizer_config.json's added_tokens_decoder must agree with tokenizer.json (the library takes its flags from there)
  if (f.tokenizer_config && f.tokenizer_config->is_object())
    if (const Json* dec = f.tokenizer_config->get("added_tokens_decoder"); dec && dec->is_object())
      for (const auto& [key, v] : dec->members()) {
        std::uint64_t id = 0;
        for (const char c : key) {
          if (c < '0' || c > '9') return R::fail(bad("added_tokens_decoder has a key that is not an id", "tokenizer_config.json"));
          id = id * 10 + static_cast<std::uint64_t>(c - '0');
        }
        const auto it = added.find(id);
        const Json* c = v.get("content");
        const Json* s = v.get("special");
        if (it == added.end() || !c || !c->is_string() || c->as_string() != it->second.content || (s && s->is_bool() && s->as_bool() != it->second.special))
          return R::fail(bad("tokenizer_config.json and tokenizer.json disagree about the added token " + key, "tokenizer_config.json"));
      }

  std::unordered_map<std::uint64_t, std::string> reverse;
  std::uint64_t max_id = 0;
  bool any = false;
  for (const auto& [content, id] : vocab->members()) {
    if (!id.is_integer() || id.as_int() < 0) return R::fail(bad("a vocabulary id is not a non-negative integer"));
    if (added_ids.count(content)) return R::fail(bad("an added token repeats a vocabulary token's text"));
    if (!reverse.emplace(id.as_uint(), content).second) return R::fail(bad("two vocabulary tokens share an id"));
    max_id = std::max(max_id, id.as_uint());
    any = true;
  }
  for (const auto& [id, a] : added) {
    if (!reverse.emplace(id, a.content).second) return R::fail(bad("an added token reuses a vocabulary id"));
    max_id = std::max(max_id, id);
    any = true;
  }
  const std::uint64_t vocab_size = vocab_size_in ? vocab_size_in : reverse.size();
  if (!any || max_id >= vocab_size) return R::fail(bad("a token id is outside the model's vocabulary size"));

  // special tokens the library would add when they are named but missing would change the ids: refuse
  if (f.tokenizer_config && f.tokenizer_config->is_object()) {
    for (const char* k : {"bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"}) {
      const Json* e = f.tokenizer_config->get(k);
      std::string text;
      if (e && e->is_string()) text = e->as_string();
      else if (e && e->is_object() && e->get("content") && e->get("content")->is_string()) text = e->get("content")->as_string();
      else continue;
      if (!added_ids.count(text) && !vocab->get(text))
        return R::fail(bad(std::string(k) + " names a token that is not in the vocabulary", "tokenizer_config.json"));
    }
  }

  // ---- the pre-tokenizer name
  const Json* norm = tok.get("normalizer");
  const Json* pre = tok.get("pre_tokenizer");
  const std::string norm_s = norm ? norm->dump() : "null";
  const std::string pre_s = pre ? pre->dump() : "null";
  const char* tokpre = nullptr;
  for (const PreTokenizer& p : kPreTokenizers)
    if (norm_s == p.normalizer && pre_s == p.pre_tokenizer) tokpre = p.name;
  if (!tokpre) return R::fail(bad("this tokenizer's pre-tokenizer is not in Pelias's table yet (the reference names it by a hash of a tokenized text)"));

  // ---- tokens and types (get_vocab_base)
  std::vector<std::string> tokens;
  std::vector<std::int32_t> types;
  tokens.reserve(static_cast<std::size_t>(vocab_size));
  types.reserve(static_cast<std::size_t>(vocab_size));
  for (std::uint64_t i = 0; i < vocab_size; ++i) {
    const auto it = reverse.find(i);
    if (it == reverse.end()) {
      tokens.push_back("[PAD" + std::to_string(i) + "]");
      types.push_back(kUnused);
      continue;
    }
    std::string t = it->second;
    const auto ad = added.find(i);
    if (ad != added.end()) {
      // a non-normalized added token is encoded and decoded by the library: its text comes back unchanged
      if (ad->second.special || looks_special(t)) {
        types.push_back(kControl);
      } else {
        t = replace_all(t, "\xE2\x96\x81", " ");  // pre-normalized user-defined spaces
        types.push_back(kUserDefined);
      }
    } else {
      types.push_back(kNormal);
    }
    tokens.push_back(std::move(t));
  }

  // ---- SpecialVocab: merges, special ids, add flags, chat template, normalizer
  std::vector<std::string> merges;
  if (const Json* m = model->get("merges"); m && m->is_array() && m->size() > 0) {
    const Json& first = m->items()[0];
    if (first.is_string()) {
      for (const Json& x : m->items()) {
        if (!x.is_string()) return R::fail(bad("the merges mix two formats"));
        merges.push_back(x.as_string());
      }
    } else if (first.is_array() && first.size() == 2 && first.items()[0].is_string()) {
      for (const Json& x : m->items()) {
        if (!x.is_array() || x.size() != 2 || !x.items()[0].is_string() || !x.items()[1].is_string()) return R::fail(bad("a merge is not a pair of strings"));
        std::string a = replace_all(x.items()[0].as_string(), " ", "\xC4\xA0");  // ' ' encoded as chr(ord(' ') + 256)
        std::string b = replace_all(x.items()[1].as_string(), " ", "\xC4\xA0");
        merges.push_back(a + " " + b);
      }
    } else {
      return R::fail(bad("unknown merges format"));
    }
  }
  if (merges.empty()) return R::fail(bad("no merges in tokenizer.json (merges.txt is not read yet)"));
  std::optional<bool> lowercase, strip_accents;
  if (norm && norm->is_object()) parse_normalizer(*norm, lowercase, strip_accents);

  // insertion-ordered maps, as the reference's dicts
  std::vector<std::pair<std::string, std::uint64_t>> special_ids;
  std::vector<std::pair<std::string, bool>> add_flags;
  auto has_id = [&](const std::string& t) {
    for (const auto& p : special_ids)
      if (p.first == t) return true;
    return false;
  };
  auto has_flag = [&](const std::string& t) {
    for (const auto& p : add_flags)
      if (p.first == t) return true;
    return false;
  };
  auto set_special = [&](const std::string& typ, const Json* tid) -> Status {
    if (!tid || !tid->is_integer()) {
      if (tid && tid->is_bool()) return bad("a special token id is a boolean", "config.json");
      return Status::good();  // not an int: ignored (a list of ids, a string, null)
    }
    if (tid->as_int() < 0) return bad("a special token id is negative", "config.json");
    if (!has_id(typ)) special_ids.emplace_back(typ, tid->as_uint());
    return Status::good();
  };
  const Json* tc = f.tokenizer_config && f.tokenizer_config->is_object() && f.tokenizer_config->size() > 0 ? f.tokenizer_config : nullptr;
  if (const Json* pp = tok.get("post_processor"); pp && pp->is_object()) {
    std::vector<const Json*> procs;
    if (const Json* list = pp->get("processors"); list && list->is_array()) {
      for (const Json& x : list->items()) procs.push_back(&x);
    } else {
      procs.push_back(pp);
    }
    for (const Json* p : procs) {
      const Json* t = p->get("type");
      const std::string type = t && t->is_string() ? t->as_string() : std::string();
      if (type == "ByteLevel") {
        if (!has_flag("bos")) add_flags.emplace_back("bos", false);
        if (!has_flag("eos")) add_flags.emplace_back("eos", false);
      } else if (type == "RobertaProcessing" || type == "TemplateProcessing") {
        return R::fail(bad("the " + type + " post-processor is not read yet"));
      }
    }
  }
  std::optional<std::string> chat_template;
  if (tc) {
    // bos <- cls and eos <- sep fallbacks of the reference
    auto entry_text = [&](const char* key) -> std::optional<std::string> {
      const Json* e = tc->get(key);
      if (e && e->is_string()) return e->as_string().empty() ? std::nullopt : std::optional<std::string>(e->as_string());
      if (e && e->is_object()) return std::string("<object>");
      return std::nullopt;
    };
    if (!entry_text("bos_token") && entry_text("cls_token")) return R::fail(bad("a cls token standing in for bos is not read yet", "tokenizer_config.json"));
    if (!entry_text("eos_token") && entry_text("sep_token")) return R::fail(bad("a sep token standing in for eos is not read yet", "tokenizer_config.json"));
    // chat template: tokenizer_config's, else chat_template.jinja, else chat_template.json
    if (f.additional_chat_templates) return R::fail(bad("additional chat templates are not read yet", "additional_chat_templates"));
    std::optional<std::string> alt;
    bool alt_is_list = false;
    if (f.chat_template_jinja) {
      alt = *f.chat_template_jinja;
    } else if (f.chat_template_json && f.chat_template_json->is_object()) {
      if (const Json* c = f.chat_template_json->get("chat_template"); c) {
        if (c->is_string()) alt = c->as_string();
        else if (!c->is_null()) alt_is_list = true;
      }
    }
    if (const Json* c = tc->get("chat_template"); c) {
      if (c->is_string()) chat_template = c->as_string();
      else if (c->is_array()) return R::fail(bad("a list of named chat templates is not read yet", "tokenizer_config.json"));
      // null: no template; another type: the reference ignores it
    } else {
      if (alt_is_list) return R::fail(bad("a list of named chat templates is not read yet", "chat_template.json"));
      chat_template = alt;
    }
    for (const char* typ : {"bos", "eos", "unk", "sep", "pad", "cls", "mask"}) {
      const std::string t(typ);
      if (const Json* a = tc->get("add_" + t + "_token"); a && a->is_bool() && !has_flag(t)) add_flags.emplace_back(t, a->as_bool());
      const Json* e = tc->get(t + "_token");
      std::string text;
      if (e && e->is_string()) text = e->as_string();
      else if (e && e->is_object()) {
        const Json* c = e->get("content");
        if (!c || !c->is_string()) continue;
        text = c->as_string();
      } else {
        continue;
      }
      // the first added token with that text (added tokens only, as the reference looks)
      if (const auto it = added_ids.find(text); it != added_ids.end()) {
        const Json id = Json::unsigned_integer(it->second);
        const Status s = set_special(t, &id);
        if (!s) return R::fail(s);
      }
    }
  }
  if (f.config && f.config->is_object()) {
    for (const char* typ : {"bos", "eos", "unk", "sep", "pad", "cls", "mask"}) {
      const std::string key = std::string(typ) + "_token_id";
      const Json* id = f.config->get(key);
      if ((!id || id->is_null()) && f.config->get("text_config") && f.config->get("text_config")->is_object()) id = f.config->get("text_config")->get(key);
      if (id && id->is_null()) id = nullptr;
      const Status s = set_special(typ, id);
      if (!s) return R::fail(s);
    }
  }

  // ---- the entries, in _set_vocab_gpt2 / SpecialVocab.add_to_gguf order
  std::vector<GgufKvBlob> kv;
  kv.push_back(io::gguf_kv_str("tokenizer.ggml.model", "gpt2"));
  kv.push_back(io::gguf_kv_str("tokenizer.ggml.pre", tokpre));
  kv.push_back(io::gguf_kv_arr_str("tokenizer.ggml.tokens", tokens));
  kv.push_back(io::gguf_kv_arr_i32("tokenizer.ggml.token_type", types));
  kv.push_back(io::gguf_kv_arr_str("tokenizer.ggml.merges", merges));
  for (const auto& [typ, id] : special_ids) {
    const char* key = typ == "bos"    ? "tokenizer.ggml.bos_token_id"
                      : typ == "eos"  ? "tokenizer.ggml.eos_token_id"
                      : typ == "unk"  ? "tokenizer.ggml.unknown_token_id"
                      : typ == "sep"  ? "tokenizer.ggml.seperator_token_id"
                      : typ == "pad"  ? "tokenizer.ggml.padding_token_id"
                      : typ == "mask" ? "tokenizer.ggml.mask_token_id"
                                      : nullptr;  // cls: the reference's writer has no entry for it
    if (!key) continue;
    if (id > 0xFFFFFFFFull) return R::fail(bad("a special token id does not fit 32 bits", "config.json"));
    kv.push_back(io::gguf_kv_u32(key, static_cast<std::uint32_t>(id)));
  }
  for (const auto& [typ, v] : add_flags) {
    const char* key = typ == "bos" ? "tokenizer.ggml.add_bos_token" : typ == "eos" ? "tokenizer.ggml.add_eos_token" : typ == "sep" ? "tokenizer.ggml.add_sep_token" : nullptr;
    if (key) kv.push_back(io::gguf_kv_bool(key, v));
  }
  if (chat_template && !chat_template->empty()) kv.push_back(io::gguf_kv_str("tokenizer.chat_template", *chat_template));
  if (lowercase) kv.push_back(io::gguf_kv_bool("tokenizer.ggml.normalizer.lowercase", *lowercase));
  if (strip_accents) kv.push_back(io::gguf_kv_bool("tokenizer.ggml.normalizer.strip_accents", *strip_accents));
  return R::good(std::move(kv));
}

}  // namespace pelias::ingest
