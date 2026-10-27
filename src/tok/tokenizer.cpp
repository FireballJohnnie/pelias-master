// provenance: original (behaviour READ-SRC from the model's tokenizer.json and from llama.cpp src/llama-vocab.cpp and
//   src/unicode.cpp at commit 2ca15f5; the split follows the regex's own semantics, alternative by alternative)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "tok/tokenizer.h"

#include <algorithm>
#include <queue>

#include "tok/unicode.h"

namespace pelias::tok {
namespace {

// GPT-2's reversible byte -> character map: printable Latin-1 bytes map to themselves, the rest to U+0100 onwards.
void byte_level_map(char32_t out[256]) {
  bool direct[256] = {};
  for (int b = 33; b <= 126; ++b) direct[b] = true;
  for (int b = 161; b <= 172; ++b) direct[b] = true;
  for (int b = 174; b <= 255; ++b) direct[b] = true;
  char32_t next = 256;
  for (int b = 0; b < 256; ++b) out[b] = direct[b] ? static_cast<char32_t>(b) : next++;
}

bool is_rn(char32_t c) { return c == '\r' || c == '\n'; }

// The case-insensitive contraction alternative: '(?i:s|t|re|ve|m|ll|d). Simple case folding: s also matches U+017F.
std::size_t match_contraction(const std::vector<CpUnit>& u, const std::vector<CpFlags>&, std::size_t i) {
  if (u[i].cp != '\'' || i + 1 >= u.size()) return 0;
  auto fold = [](char32_t c) -> char32_t {
    if (c >= 'A' && c <= 'Z') return c + 32;
    if (c == 0x17F) return 's';
    return c;
  };
  const char32_t a = fold(u[i + 1].cp);
  if (a == 's' || a == 't') return 2;
  if (i + 2 < u.size()) {
    const char32_t b = fold(u[i + 2].cp);
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
  }
  if (a == 'm' || a == 'd') return 2;
  return 0;
}

bool is_other(const CpFlags& f) { return !(f.whitespace || f.letter || f.mark || f.number); }

}  // namespace

Result<Tokenizer> Tokenizer::from_parts(std::vector<std::string> tokens, std::vector<std::int32_t> types,
                                        const std::vector<std::string>& merges, std::string pre) {
  using R = Result<Tokenizer>;
  if (tokens.size() != types.size()) return R::fail(Status::fail(Code::corrupt, "token and type counts differ"));
  if (pre != "qwen35") return R::fail(Status::fail(Code::unsupported, "pre-tokenizer not supported: " + pre));
  Tokenizer t;
  t.tokens_ = std::move(tokens);
  t.types_ = std::move(types);
  t.pre_ = std::move(pre);
  const Status s = t.build(merges);
  if (!s) return R::fail(s);
  return R::good(std::move(t));
}

Result<Tokenizer> Tokenizer::from_gguf(const io::GgufReader& g) {
  using R = Result<Tokenizer>;
  bool found = false;
  const std::string model = g.get_string("tokenizer.ggml.model", &found);
  if (!found) return R::fail(Status::fail(Code::not_found, "no tokenizer in this file", 0, "tokenizer.ggml.model"));
  if (model != "gpt2") return R::fail(Status::fail(Code::unsupported, "tokenizer model not supported: " + model));
  const std::string pre = g.get_string("tokenizer.ggml.pre", &found);
  const io::GgufValue* tv = g.find("tokenizer.ggml.tokens");
  const io::GgufValue* yv = g.find("tokenizer.ggml.token_type");
  const io::GgufValue* mv = g.find("tokenizer.ggml.merges");
  if (!tv || !yv || !mv) return R::fail(Status::fail(Code::not_found, "tokenizer arrays missing", 0, "tokenizer.ggml.tokens"));
  std::vector<std::string> tokens, merges;
  std::vector<std::int64_t> types64;
  Status s = g.array_strings(*tv, tokens);
  if (s) s = g.array_ints(*yv, types64);
  if (s) s = g.array_strings(*mv, merges);
  if (!s) return R::fail(s);
  std::vector<std::int32_t> types(types64.begin(), types64.end());
  R r = from_parts(std::move(tokens), std::move(types), merges, pre);
  if (!r.ok()) return r;
  auto id_of = [&](const char* key) -> std::int32_t {
    bool f = false;
    const std::uint64_t v = g.get_uint(key, &f);
    return f ? static_cast<std::int32_t>(v) : -1;
  };
  r.value.eos_ = id_of("tokenizer.ggml.eos_token_id");
  r.value.pad_ = id_of("tokenizer.ggml.padding_token_id");
  r.value.bos_ = id_of("tokenizer.ggml.bos_token_id");
  if (const io::GgufValue* b = g.find("tokenizer.ggml.add_bos_token"); b && b->type == io::GgufValueType::boolean)
    r.value.add_bos_ = b->b;
  return r;
}

Status Tokenizer::build(const std::vector<std::string>& merges) {
  by_text_.reserve(tokens_.size() * 2);
  for (std::size_t i = 0; i < tokens_.size(); ++i) by_text_.emplace(tokens_[i], static_cast<std::int32_t>(i));

  char32_t map[256];
  byte_level_map(map);
  for (int b = 0; b < 256; ++b) {
    byte_char_[b].clear();
    append_utf8(byte_char_[b], map[b]);
    char_byte_[map[b]] = static_cast<std::uint8_t>(b);
    const auto it = by_text_.find(byte_char_[b]);
    if (it == by_text_.end())
      return Status::fail(Code::corrupt, "byte-level token missing for byte " + std::to_string(b), 0, "tokenizer.ggml.tokens");
    byte_token_[b] = it->second;
  }

  merges_.reserve(merges.size() * 2);
  for (std::size_t r = 0; r < merges.size(); ++r) {
    const std::string& m = merges[r];
    const std::size_t sp = m.find(' ', 1);
    if (sp == std::string::npos) return Status::fail(Code::corrupt, "merge without a space", r, "tokenizer.ggml.merges");
    const std::string left = m.substr(0, sp), right = m.substr(sp + 1);
    const auto a = by_text_.find(left), b = by_text_.find(right), c = by_text_.find(left + right);
    if (a == by_text_.end() || b == by_text_.end() || c == by_text_.end())
      return Status::fail(Code::corrupt, "merge names a token not in the vocabulary", r, "tokenizer.ggml.merges");
    const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a->second)) << 32) |
                              static_cast<std::uint32_t>(b->second);
    merges_.emplace(key, MergeInfo{static_cast<std::uint32_t>(r), c->second});  // first (lowest) rank wins
  }

  for (std::size_t i = 0; i < tokens_.size(); ++i) {
    const auto ty = static_cast<TokenType>(types_[i]);
    if (ty == TokenType::control || ty == TokenType::user_defined) {
      if (tokens_[i].empty()) continue;
      added_.push_back(Added{tokens_[i], static_cast<std::int32_t>(i), ty == TokenType::control});
    }
  }
  // longest first, so the first match at a position is the longest one
  std::sort(added_.begin(), added_.end(), [](const Added& x, const Added& y) {
    return x.text.size() != y.text.size() ? x.text.size() > y.text.size() : x.text < y.text;
  });
  return Status::good();
}

std::int32_t Tokenizer::find(std::string_view token_text) const {
  const auto it = by_text_.find(std::string(token_text));
  return it == by_text_.end() ? -1 : it->second;
}

std::vector<std::string> Tokenizer::split(std::string_view text) const {
  std::vector<std::string> out;
  const std::vector<CpUnit> u = decode_utf8(text);
  const std::size_t n = u.size();
  std::vector<CpFlags> f(n);
  for (std::size_t k = 0; k < n; ++k)
    if (u[k].valid) f[k] = flags_of(u[k].cp);
  auto letter_or_mark = [&](std::size_t k) { return k < n && (f[k].letter || f[k].mark); };
  auto emit = [&](std::size_t from, std::size_t to) {
    const std::size_t b0 = u[from].offset;
    const std::size_t b1 = to < n ? u[to].offset : text.size();
    out.emplace_back(text.substr(b0, b1 - b0));
  };

  std::size_t i = 0;
  while (i < n) {
    // 1. contractions
    if (const std::size_t len = match_contraction(u, f, i)) {
      emit(i, i + len);
      i += len;
      continue;
    }
    const char32_t c = u[i].cp;
    const bool c_valid = u[i].valid;
    // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    {
      const bool prefix_ok = !(c_valid && is_rn(c)) && !f[i].letter && !f[i].number;
      std::size_t j = 0;
      if (prefix_ok && letter_or_mark(i + 1)) {
        j = i + 1;
      } else if (letter_or_mark(i)) {
        j = i;
      }
      if (j != 0 || letter_or_mark(i)) {
        if (j == 0) j = i;
        while (letter_or_mark(j)) ++j;
        emit(i, j);
        i = j;
        continue;
      }
    }
    // 3. \p{N}
    if (f[i].number) {
      emit(i, i + 1);
      ++i;
      continue;
    }
    // 4. ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*'
    {
      std::size_t j = i;
      if (c_valid && c == ' ' && i + 1 < n && is_other(f[i + 1])) j = i + 1;
      if (is_other(f[j])) {
        while (j < n && is_other(f[j])) ++j;
        while (j < n && u[j].valid && is_rn(u[j].cp)) ++j;
        emit(i, j);
        i = j;
        continue;
      }
    }
    // 5 to 7: whitespace runs
    std::size_t end = i;
    std::size_t last_rn = 0;
    bool has_rn = false;
    while (end < n && f[end].whitespace) {
      if (u[end].valid && is_rn(u[end].cp)) {
        last_rn = end;
        has_rn = true;
      }
      ++end;
    }
    if (end == i) {  // cannot happen with the classes above; keep the split total
      emit(i, i + 1);
      ++i;
      continue;
    }
    if (has_rn) {  // \s*[\r\n]+
      emit(i, last_rn + 1);
      i = last_rn + 1;
      continue;
    }
    if (end < n && end - i >= 2) {  // \s+(?!\S)
      emit(i, end - 1);
      i = end - 1;
      continue;
    }
    emit(i, end);  // \s+
    i = end;
  }
  return out;
}

void Tokenizer::bpe(std::string_view word, std::vector<std::int32_t>& out) const {
  struct Sym {
    std::int32_t id;
    int prev;
    int next;
    bool alive;
  };
  std::vector<Sym> sym;
  sym.reserve(word.size());
  for (std::size_t k = 0; k < word.size(); ++k) {
    const auto b = static_cast<unsigned char>(word[k]);
    sym.push_back(Sym{byte_token_[b], static_cast<int>(k) - 1, k + 1 < word.size() ? static_cast<int>(k + 1) : -1, true});
  }
  if (sym.size() <= 1) {
    for (const Sym& s : sym) out.push_back(s.id);
    return;
  }
  struct Cand {
    std::uint32_t rank;
    int pos;
    std::int32_t left, right;
  };
  auto worse = [](const Cand& x, const Cand& y) { return x.rank != y.rank ? x.rank > y.rank : x.pos > y.pos; };
  std::priority_queue<Cand, std::vector<Cand>, decltype(worse)> q(worse);
  auto lookup = [&](std::int32_t a, std::int32_t b) -> const MergeInfo* {
    const auto it = merges_.find((static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) | static_cast<std::uint32_t>(b));
    return it == merges_.end() ? nullptr : &it->second;
  };
  auto push = [&](int p) {
    if (p < 0 || !sym[static_cast<std::size_t>(p)].alive) return;
    const int nx = sym[static_cast<std::size_t>(p)].next;
    if (nx < 0) return;
    const std::int32_t a = sym[static_cast<std::size_t>(p)].id, b = sym[static_cast<std::size_t>(nx)].id;
    if (const MergeInfo* m = lookup(a, b)) q.push(Cand{m->rank, p, a, b});
  };
  for (int p = 0; p + 1 < static_cast<int>(sym.size()); ++p) push(p);
  while (!q.empty()) {
    const Cand c = q.top();
    q.pop();
    Sym& l = sym[static_cast<std::size_t>(c.pos)];
    if (!l.alive || l.next < 0 || l.id != c.left) continue;
    Sym& r = sym[static_cast<std::size_t>(l.next)];
    if (r.id != c.right) continue;
    const MergeInfo* m = lookup(l.id, r.id);
    if (!m || m->rank != c.rank) continue;
    l.id = m->result;
    r.alive = false;
    l.next = r.next;
    if (r.next >= 0) sym[static_cast<std::size_t>(r.next)].prev = c.pos;
    push(l.prev);
    push(c.pos);
  }
  for (int p = 0; p >= 0; p = sym[static_cast<std::size_t>(p)].next) out.push_back(sym[static_cast<std::size_t>(p)].id);
}

void Tokenizer::encode_plain(std::string_view piece, const EncodeOptions& opt, std::vector<std::int32_t>& out) const {
  if (piece.empty()) return;
  std::string normalized;
  std::string_view text = piece;
  if (opt.nfc) {
    normalized = to_nfc(piece);
    text = normalized;
  }
  for (const std::string& w : split(text)) bpe(w, out);
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text, const EncodeOptions& opt) const {
  std::vector<std::int32_t> out;
  out.reserve(text.size() / 3 + 4);
  std::size_t start = 0, i = 0;
  while (i < text.size()) {
    const Added* hit = nullptr;
    for (const Added& a : added_) {
      if (a.control && !opt.parse_special) continue;
      if (a.text[0] == text[i] && text.compare(i, a.text.size(), a.text) == 0) {
        hit = &a;
        break;
      }
    }
    if (!hit) {
      ++i;
      continue;
    }
    encode_plain(text.substr(start, i - start), opt, out);
    out.push_back(hit->id);
    i += hit->text.size();
    start = i;
  }
  encode_plain(text.substr(start), opt, out);
  return out;
}

std::string Tokenizer::piece(std::int32_t id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= tokens_.size()) return {};
  const std::string& t = tokens_[static_cast<std::size_t>(id)];
  const auto ty = static_cast<TokenType>(types_[static_cast<std::size_t>(id)]);
  if (ty == TokenType::control || ty == TokenType::user_defined) return t;
  std::string out;
  for (const CpUnit& u : decode_utf8(t)) {
    const auto it = char_byte_.find(u.cp);
    if (u.valid && it != char_byte_.end()) {
      out.push_back(static_cast<char>(it->second));
    } else {
      out.append(t, u.offset, u.length);
    }
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<std::int32_t>& ids) const {
  std::string out;
  for (std::int32_t id : ids) out += piece(id);
  return out;
}

}  // namespace pelias::tok
