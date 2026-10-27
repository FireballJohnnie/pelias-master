// provenance: original (rules READ-SRC, see hf_metadata.h)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "ingest/hf_metadata.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <regex>
#include <set>

namespace pelias::ingest {
namespace {

using io::GgufKvBlob;
using io::Json;

bool is_ascii(std::string_view s) {
  for (const char c : s)
    if (static_cast<unsigned char>(c) >= 0x80) return false;
  return true;
}
bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool is_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool is_lower(char c) { return c >= 'a' && c <= 'z'; }
bool is_upper(char c) { return c >= 'A' && c <= 'Z'; }
char to_upper(char c) { return is_lower(c) ? static_cast<char>(c - 'a' + 'A') : c; }
char to_lower(char c) { return is_upper(c) ? static_cast<char>(c - 'A' + 'a') : c; }
bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f' || (c >= 0x1c && c <= 0x1f); }

// Python str.islower() for ASCII: at least one cased character and no upper-case one.
bool py_islower(std::string_view w) {
  bool any = false;
  for (const char c : w) {
    if (is_upper(c)) return false;
    if (is_lower(c)) any = true;
  }
  return any;
}
// Python str.title() for ASCII: a letter is upper-case after a non-letter, lower-case after a letter.
std::string py_title(std::string_view w) {
  std::string r(w);
  bool prev_cased = false;
  for (char& c : r) {
    if (is_alpha(c)) {
      c = prev_cased ? to_lower(c) : to_upper(c);
      prev_cased = true;
    } else {
      prev_cased = false;
    }
  }
  return r;
}
std::vector<std::string> split_ws(std::string_view s) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && is_space(s[i])) ++i;
    std::size_t j = i;
    while (j < s.size() && !is_space(s[j])) ++j;
    if (j > i) out.emplace_back(s.substr(i, j - i));
    i = j;
  }
  return out;
}

bool full(const std::string& s, const std::regex& re) { return std::regex_match(s, re); }

// ---- KV helpers in the reference writer's rules: add_string skips an empty string, add_array an empty list
void add_str(std::vector<GgufKvBlob>& kv, const std::string& key, const std::string& v) {
  if (!v.empty()) kv.push_back(io::gguf_kv_str(key, v));
}

// ---- the YAML subset
struct Line {
  std::size_t indent = 0;
  std::string text;  // without the indentation
  bool blank = false;
};

// Python str.splitlines() boundaries: \n, \r, \r\n, \v, \f, \x1c, \x1d, \x1e, U+0085, U+2028, U+2029.
std::vector<std::string> py_splitlines(std::string_view s) {
  std::vector<std::string> out;
  std::string cur;
  std::size_t i = 0;
  bool open = false;
  auto end_line = [&]() {
    out.push_back(cur);
    cur.clear();
    open = false;
  };
  while (i < s.size()) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c == '\r') {
      end_line();
      i += (i + 1 < s.size() && s[i + 1] == '\n') ? 2u : 1u;
      continue;
    }
    if (c == '\n' || c == 0x0b || c == 0x0c || c == 0x1c || c == 0x1d || c == 0x1e) {
      end_line();
      ++i;
      continue;
    }
    if (c == 0xC2 && i + 1 < s.size() && static_cast<unsigned char>(s[i + 1]) == 0x85) {
      end_line();
      i += 2;
      continue;
    }
    if (c == 0xE2 && i + 2 < s.size() && static_cast<unsigned char>(s[i + 1]) == 0x80 &&
        (static_cast<unsigned char>(s[i + 2]) == 0xA8 || static_cast<unsigned char>(s[i + 2]) == 0xA9)) {
      end_line();
      i += 3;
      continue;
    }
    cur.push_back(static_cast<char>(c));
    open = true;
    ++i;
  }
  if (open) out.push_back(cur);
  return out;
}

std::string replace_all(std::string s, std::string_view from, std::string_view to) {
  std::size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) {
    s.replace(p, from.size(), to);
    p += to.size();
  }
  return s;
}

// PyYAML's implicit resolvers (YAML 1.1) for a plain scalar: true when the scalar is not a string.
bool plain_is_not_string(const std::string& s) {
  static const std::regex boolean("yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF");
  static const std::regex integer("[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+");
  static const std::regex floating(
      "[-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+][0-9]+)?|\\.[0-9_]+(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*|"
      "[-+]?\\.(?:inf|Inf|INF)|\\.(?:nan|NaN|NAN)");
  static const std::regex timestamp(
      "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]|[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?(?:[Tt]|[ \\t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9]"
      "(?:\\.[0-9]*)?(?:[ \\t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?");
  return full(s, boolean) || full(s, integer) || full(s, floating) || full(s, timestamp) || s == "<<" || s == "=";
}

bool plain_is_null(const std::string& s) { return s.empty() || s == "~" || s == "null" || s == "Null" || s == "NULL"; }

// One scalar written on a single line (plain, 'single' or "double" quoted). Kind other when outside the subset.
CardValue scalar(std::string t) {
  CardValue v;
  // strip trailing spaces
  while (!t.empty() && (t.back() == ' ')) t.pop_back();
  if (t.empty()) return v;  // null
  if (t[0] == '\'') {
    std::string out;
    std::size_t i = 1;
    for (; i < t.size(); ++i) {
      if (t[i] == '\'') {
        if (i + 1 < t.size() && t[i + 1] == '\'') {
          out.push_back('\'');
          ++i;
          continue;
        }
        break;
      }
      out.push_back(t[i]);
    }
    if (i >= t.size()) {
      v.kind = CardValue::Kind::other;
      return v;
    }
    std::string rest = t.substr(i + 1);
    while (!rest.empty() && rest.front() == ' ') rest.erase(rest.begin());
    if (!rest.empty() && rest[0] != '#') {
      v.kind = CardValue::Kind::other;
      return v;
    }
    v.kind = CardValue::Kind::string;
    v.text = out;
    return v;
  }
  if (t[0] == '"') {
    std::string out;
    std::size_t i = 1;
    for (; i < t.size(); ++i) {
      const char c = t[i];
      if (c == '"') break;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (++i >= t.size()) break;
      switch (t[i]) {
        case '\\': out.push_back('\\'); break;
        case '"': out.push_back('"'); break;
        case '/': out.push_back('/'); break;
        case 'n': out.push_back('\n'); break;
        case 't': out.push_back('\t'); break;
        case 'r': out.push_back('\r'); break;
        case '0': out.push_back('\0'); break;
        case ' ': out.push_back(' '); break;
        default: v.kind = CardValue::Kind::other; return v;  // \x, \u ... outside the subset
      }
    }
    if (i >= t.size()) {
      v.kind = CardValue::Kind::other;
      return v;
    }
    std::string rest = t.substr(i + 1);
    while (!rest.empty() && rest.front() == ' ') rest.erase(rest.begin());
    if (!rest.empty() && rest[0] != '#') {
      v.kind = CardValue::Kind::other;
      return v;
    }
    v.kind = CardValue::Kind::string;
    v.text = out;
    return v;
  }
  // plain: a comment starts at " #"
  const std::size_t hash = t.find(" #");
  if (hash != std::string::npos) t = t.substr(0, hash);
  while (!t.empty() && t.back() == ' ') t.pop_back();
  const char c0 = t[0];
  // indicators that start something other than a plain scalar in this subset
  if (c0 == '[' || c0 == '{' || c0 == '|' || c0 == '>' || c0 == '&' || c0 == '*' || c0 == '!' || c0 == '%' || c0 == '@' || c0 == '`' || c0 == '#' ||
      (c0 == '-' && (t.size() == 1 || t[1] == ' ')) || (c0 == '?' && (t.size() == 1 || t[1] == ' ')) || (c0 == ':' && (t.size() == 1 || t[1] == ' ')) ||
      t.find(": ") != std::string::npos || t.back() == ':') {
    v.kind = CardValue::Kind::other;
    return v;
  }
  if (plain_is_null(t)) return v;
  if (plain_is_not_string(t)) {
    v.kind = CardValue::Kind::other;
    return v;
  }
  v.kind = CardValue::Kind::string;
  v.text = t;
  return v;
}

// A flow sequence on one line: [a, "b", 'c'].
CardValue flow_sequence(const std::string& t) {
  CardValue v;
  v.kind = CardValue::Kind::other;
  std::string s = t;
  const std::size_t hash = s.find(" #");
  if (hash != std::string::npos && s.rfind(']') < hash) s = s.substr(0, hash);
  while (!s.empty() && s.back() == ' ') s.pop_back();
  if (s.size() < 2 || s.front() != '[' || s.back() != ']') return v;
  const std::string body = s.substr(1, s.size() - 2);
  std::vector<std::string> parts;
  std::string cur;
  char quote = 0;
  for (const char c : body) {
    if (quote) {
      cur.push_back(c);
      if (c == quote) quote = 0;
      continue;
    }
    if (c == '\'' || c == '"') quote = c;
    if (c == '[' || c == ']' || c == '{' || c == '}') return v;
    if (c == ',') {
      parts.push_back(cur);
      cur.clear();
      continue;
    }
    cur.push_back(c);
  }
  if (quote) return v;
  parts.push_back(cur);
  CardValue seq;
  seq.kind = CardValue::Kind::sequence;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    std::string p = parts[i];
    while (!p.empty() && p.front() == ' ') p.erase(p.begin());
    while (!p.empty() && p.back() == ' ') p.pop_back();
    if (p.empty()) {
      if (i + 1 == parts.size() && i > 0) break;  // a trailing comma
      if (parts.size() == 1) return seq;           // []
      return v;
    }
    seq.items.push_back(scalar(p));
  }
  return seq;
}

// Python's re.match(r"https?://huggingface.co/([^/]+/[^/]+)$", s): group 1, or nothing. ('$' also matches before a
// final newline.)
std::optional<std::string> hf_url_id(const std::string& s) {
  static const std::regex re("https?://huggingface.co/([^/]+/[^/]+)");
  std::string t = s;
  if (!t.empty() && t.back() == '\n') t.pop_back();
  std::smatch m;
  if (std::regex_match(t, m, re)) {
    std::string g = m[1].str();
    if (g.find('\n') != std::string::npos) return std::nullopt;
    return g;
  }
  return std::nullopt;
}

struct Source {  // a base model or a dataset entry, in the keys the writer knows
  std::optional<std::string> name, author, version, organization, description, url, doi, uuid, repo_url;
};

struct Meta {
  std::optional<std::string> sampling_sequence;
  std::optional<std::int64_t> top_k, penalty_last_n, mirostat;
  std::optional<double> top_p, min_p, xtc_probability, xtc_threshold, temp, penalty_repeat, mirostat_tau, mirostat_eta;
  std::optional<std::string> name, author, version, organization, finetune, basename, description, quantized_by, size_label;
  std::optional<std::string> url, doi, uuid, repo_url, source_url, source_doi, source_uuid, source_repo_url;
  std::optional<std::vector<std::string>> license;  // a string is a list of one; a list is joined with ","
  bool license_is_list = false;
  std::optional<std::string> license_name, license_link;
  std::optional<std::vector<Source>> base_models, datasets;
  std::optional<std::vector<std::string>> tags, languages;
};

Status unsupported_card(const std::string& key, const std::string& why) {
  return Status::fail(Code::unsupported, "model card value for '" + key + "' " + why + " (the reference would not write it as text)", 0, "README.md");
}

}  // namespace

// ------------------------------------------------------------------------------------------------ parameter counting

ParamCount count_parameters(const std::vector<NamedShape>& tensors) {
  ParamCount p;
  std::int64_t expert_sum = 0, n_expert_tensors = 0;
  for (const NamedShape& t : tensors) {
    std::int64_t size = 1;
    for (const std::uint64_t d : t.shape) size *= static_cast<std::int64_t>(d);
    if (t.name.find("_exps.") != std::string::npos) {
      if (t.shape.size() >= 3) {
        const bool bias = t.name.find(".bias") != std::string::npos;
        const auto ec = static_cast<std::int64_t>(t.shape[t.shape.size() - (bias ? 2 : 3)]);
        p.expert += ec ? size / ec : 0;
        expert_sum += ec;
        ++n_expert_tensors;
      } else {
        p.shared += size;
      }
    } else {
      p.shared += size;
    }
    p.total += size;
  }
  p.expert_count = n_expert_tensors > 0 ? expert_sum / n_expert_tensors : 0;
  return p;
}

std::string rounded_count(std::int64_t n, int min_digits) {
  double scaled;
  char suffix;
  if (n > 1000000000000LL) {
    scaled = static_cast<double>(n) * 1e-12;
    suffix = 'T';
  } else if (n > 1000000000LL) {
    scaled = static_cast<double>(n) * 1e-9;
    suffix = 'B';
  } else if (n > 1000000LL) {
    scaled = static_cast<double>(n) * 1e-6;
    suffix = 'M';
  } else {
    scaled = static_cast<double>(n) * 1e-3;
    suffix = 'K';
  }
  // len(str(round(scaled)).lstrip('0')): Python's round() rounds half to even
  const double r = std::nearbyint(scaled);
  std::string rs = std::to_string(static_cast<long long>(r));
  std::size_t lead = 0;
  while (lead < rs.size() && rs[lead] == '0') ++lead;
  const int digits = static_cast<int>(rs.size() - lead);
  const int fix = std::max(min_digits - digits, 0);
  std::array<char, 64> buf{};
  const auto res = std::to_chars(buf.data(), buf.data() + buf.size(), scaled, std::chars_format::fixed, fix);
  return std::string(buf.data(), res.ptr) + suffix;
}

std::string size_label(const ParamCount& p) {
  if (p.expert_count > 0) return std::to_string(p.expert_count) + "x" + rounded_count(std::llabs(p.shared) + std::llabs(p.expert), 2);
  return rounded_count(std::llabs(p.total), 2);
}

// ------------------------------------------------------------------------------------------------ names

std::string id_to_title(std::string_view s) {
  // string.strip().replace('-', ' ').split()
  std::string t(s);
  for (char& c : t)
    if (c == '-') c = ' ';
  static const std::regex keep("v[0-9]+(?:\\.[0-9]+)*|[0-9].*");
  std::string out;
  for (const std::string& w : split_ws(t)) {
    if (!out.empty()) out.push_back(' ');
    out += py_islower(w) && !std::regex_match(w, keep) ? py_title(w) : w;
  }
  return out;
}

Result<ModelIdParts> model_id_components(std::string_view model_id_view, std::int64_t total_params) {
  using R = Result<ModelIdParts>;
  const std::string model_id(model_id_view);
  if (!is_ascii(model_id)) return R::fail(Status::fail(Code::unsupported, "a model or folder name outside ASCII is not read yet", 0, model_id));
  ModelIdParts r;
  if (model_id.find(' ') != std::string::npos) {
    r.full_name = model_id;
    return R::good(r);
  }
  std::optional<std::string> org;
  std::string full_name;
  const std::size_t slash = model_id.find('/');
  if (slash != std::string::npos) {
    org = model_id.substr(0, slash);
    full_name = model_id.substr(slash + 1);
  } else {
    full_name = model_id;
  }
  if (org && !org->empty() && (*org)[0] == '.') org.reset();

  std::vector<std::string> parts;
  {
    std::size_t i = 0;
    while (true) {
      const std::size_t j = full_name.find('-', i);
      const std::string p = full_name.substr(i, j == std::string::npos ? std::string::npos : j - i);
      if (!p.empty()) parts.push_back(p);
      if (j == std::string::npos) break;
      i = j + 1;
    }
  }
  enum : unsigned { kBase = 1, kSize = 2, kFine = 4, kVer = 8, kType = 16 };
  std::vector<unsigned> types(parts.size(), 0);
  static const std::regex version_re("(v|iter)?[0-9]+([.][0-9]+)*", std::regex::icase);
  static const std::regex type_re("i?q[0-9](_[A-Za-z0-9_])*|b?fp?(16|32)", std::regex::icase);
  static const std::regex size_re("(([A]|[0-9]+[x])?[0-9]+([._][0-9]+)?[KMBT][0-9]?|small|mini|medium|large|x?xl)", std::regex::icase);
  static const std::regex fine_re("chat|instruct|vision|lora", std::regex::icase);
  for (std::size_t i = 0; i < parts.size(); ++i) {
    std::string part = parts[i];
    if (full(part, version_re)) {
      types[i] |= kVer;
    } else if (full(part, type_re)) {
      types[i] |= kType;
      for (char& c : part) c = to_upper(c);
      parts[i] = part;
    } else if (i > 0 && full(part, size_re)) {
      for (char& c : part)
        if (c == '_') c = '.';
      if (is_digit(part.back())) part = part.substr(0, part.size() - 2) + "." + part.back() + part[part.size() - 2];
      if (part.size() > 1 && is_digit(part[part.size() - 2])) {
        const char last = part.back();
        if (last == 'k' || last == 'm' || last == 'b' || last == 't') part.back() = to_upper(last);
      }
      if (total_params != 0) {
        // float(part[:-1]) * pow(1000, " KMBT".find(part[-1])); a ValueError skips the check
        const std::string num = part.substr(0, part.size() - 1);
        bool numeric = !num.empty();
        int dots = 0;
        for (const char c : num) {
          if (c == '.') ++dots;
          else if (!is_digit(c)) numeric = false;
        }
        if (dots > 1 || num == ".") numeric = false;
        if (numeric) {
          const double f = std::strtod(num.c_str(), nullptr);
          const std::string kmbt = " KMBT";
          const std::size_t pos = kmbt.find(part.back());
          const double scale = pos == std::string::npos ? 0.001 : std::pow(1000.0, static_cast<double>(pos));
          const double label_params = f * scale;
          const auto tp = static_cast<double>(total_params);
          const bool lora_small = total_params < 0 && label_params < static_cast<double>(std::llabs(total_params) / 8);
          // 7 * total_params // 8 (floor division of a positive integer)
          const bool far = total_params > 0 && std::fabs(label_params - tp) > static_cast<double>(7 * total_params / 8);
          if (lora_small || far) {
            types[i] |= kFine;
            part.back() = to_lower(part.back());
          }
        }
      }
      if (types[i] == 0) types[i] |= kSize;
      parts[i] = part;
    } else if (i > 0 && full(part, fine_re)) {
      std::string low = part;
      for (char& c : low) c = to_lower(c);
      if (total_params < 0 && low == "lora") types[i] |= kType;
      else types[i] |= kFine;
    }
  }
  // word-based size labels are dropped when a number-based one is present
  bool numeric_size = false;
  for (std::size_t i = 0; i < parts.size(); ++i)
    if (types[i] & kSize)
      for (const char c : parts[i])
        if (is_digit(c)) numeric_size = true;
  if (numeric_size)
    for (std::size_t i = 0; i < parts.size(); ++i)
      if (types[i] & kSize) {
        bool all_alpha = true;
        for (const char c : parts[i])
          if (!is_alpha(c)) all_alpha = false;
        if (all_alpha) types[i] &= ~static_cast<unsigned>(kSize);
      }
  bool at_start = true;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (at_start && ((types[i] == 0 && is_alpha(parts[i][0])) || (types[i] & kVer))) {
      types[i] |= kBase;
    } else {
      at_start = false;
      if (types[i] == 0) types[i] |= kFine;
    }
  }
  for (std::size_t k = parts.size(); k-- > 0;) {
    const unsigned t = types[k];
    const bool more_than_base = (t & ~static_cast<unsigned>(kBase)) != 0;
    if ((t & kBase) && more_than_base) types[k] &= ~static_cast<unsigned>(kBase);
    else break;
  }
  auto join = [&](unsigned want, unsigned exclude, bool dedupe) -> std::optional<std::string> {
    std::string out;
    std::set<std::string> seen;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      if (!(types[i] & want) || (types[i] & exclude)) continue;
      if (dedupe && !seen.insert(parts[i]).second) continue;
      if (!out.empty()) out.push_back('-');
      out += parts[i];
    }
    if (out.empty()) return std::nullopt;
    return out;
  };
  r.full_name = full_name;
  r.org = org;
  r.basename = join(kBase, 0, false);
  r.size_label = join(kSize, 0, true);
  r.finetune = join(kFine, 0, false);
  r.version = join(kVer, kBase, false);
  if (!r.size_label && !r.finetune && !r.version) r.basename.reset();
  return R::good(r);
}

// ------------------------------------------------------------------------------------------------ model card

Result<std::map<std::string, CardValue>> parse_model_card(std::string_view readme) {
  using R = Result<std::map<std::string, CardValue>>;
  std::map<std::string, CardValue> out;
  const std::vector<std::string> lines = py_splitlines(readme);
  if (lines.empty() || lines[0] != "---") return R::good(out);
  std::string yaml;
  for (std::size_t i = 1; i < lines.size(); ++i) {
    if (lines[i] == "---") break;
    if (i > 1) yaml += "\n";
    yaml += lines[i];
  }
  yaml += "\n";
  yaml = replace_all(yaml, "- no\n", "- \"no\"\n");
  yaml = replace_all(yaml, "\t", "  ");

  // lines of the YAML text (it is joined with "\n" above, so only "\n" separates them now)
  std::vector<Line> ls;
  {
    std::size_t i = 0;
    while (i < yaml.size()) {
      const std::size_t j = yaml.find('\n', i);
      std::string l = yaml.substr(i, j - i);
      i = j + 1;
      Line x;
      while (x.indent < l.size() && l[x.indent] == ' ') ++x.indent;
      x.text = l.substr(x.indent);
      while (!x.text.empty() && x.text.back() == ' ') x.text.pop_back();
      x.blank = x.text.empty() || x.text[0] == '#';
      ls.push_back(x);
    }
  }
  std::size_t i = 0;
  bool any_entry = false;
  while (i < ls.size()) {
    const Line& l = ls[i];
    if (l.blank) {
      ++i;
      continue;
    }
    if (l.indent != 0 || l.text == "---" || l.text == "..." || l.text[0] == '-' || l.text[0] == '[' || l.text[0] == '{' || l.text[0] == '%')
      return R::fail(Status::fail(Code::unsupported, "the model card's front matter is not a plain mapping", 0, "README.md"));
    // key
    std::string key;
    std::size_t p = 0;
    if (l.text[0] == '"' || l.text[0] == '\'') {
      const char q = l.text[0];
      const std::size_t e = l.text.find(q, 1);
      if (e == std::string::npos) return R::fail(Status::fail(Code::unsupported, "the model card has a quoted key outside the subset", 0, "README.md"));
      key = l.text.substr(1, e - 1);
      p = e + 1;
      if (p >= l.text.size() || l.text[p] != ':') return R::fail(Status::fail(Code::unsupported, "the model card has a key without ':'", 0, "README.md"));
    } else {
      const std::size_t c = l.text.find(": ");
      if (c != std::string::npos) p = c;
      else if (l.text.back() == ':') p = l.text.size() - 1;
      else return R::fail(Status::fail(Code::unsupported, "the model card has a line that is not 'key: value'", 0, "README.md"));
      key = l.text.substr(0, p);
    }
    std::string inline_value = l.text.substr(p + 1);
    while (!inline_value.empty() && inline_value.front() == ' ') inline_value.erase(inline_value.begin());
    // the entry's continuation lines: indented, or a sequence item at column 0
    std::size_t j = i + 1;
    std::vector<const Line*> cont;
    while (j < ls.size()) {
      const Line& n = ls[j];
      if (!n.blank && n.indent == 0 && !(n.text[0] == '-' && (n.text.size() == 1 || n.text[1] == ' '))) break;
      if (!n.blank) cont.push_back(&n);
      ++j;
    }
    CardValue v;
    if (!inline_value.empty() && inline_value[0] != '#') {
      if (inline_value[0] == '[') v = flow_sequence(inline_value);
      else v = scalar(inline_value);
      if (!cont.empty()) v.kind = CardValue::Kind::other;  // a multi-line scalar or something nested: outside the subset
    } else if (!cont.empty()) {
      v.kind = CardValue::Kind::sequence;
      const std::size_t ind = cont[0]->indent;
      for (const Line* c : cont) {
        const bool item = c->indent == ind && c->text[0] == '-' && (c->text.size() == 1 || c->text[1] == ' ');
        if (!item) {
          v.kind = CardValue::Kind::other;  // a nested mapping, or an item that runs over several lines
          v.items.clear();
          break;
        }
        std::string t = c->text.size() > 1 ? c->text.substr(2) : std::string();
        while (!t.empty() && t.front() == ' ') t.erase(t.begin());
        CardValue it = t.empty() ? CardValue{} : (t[0] == '[' ? flow_sequence(t) : scalar(t));
        if (it.kind == CardValue::Kind::sequence) it.kind = CardValue::Kind::other;
        v.items.push_back(it);
      }
    }
    out[key] = v;  // a repeated key: the last one wins, as in the reference's YAML reader
    any_entry = true;
    i = j;
  }
  (void)any_entry;
  return R::good(out);
}

// ------------------------------------------------------------------------------------------------ general.* entries

Result<std::vector<GgufKvBlob>> general_metadata(const std::string& folder_name, const std::string* readme, const Json* config,
                                                 const Json* gen, const ParamCount& params) {
  using R = Result<std::vector<GgufKvBlob>>;
  Meta md;
  std::map<std::string, CardValue> card;
  if (readme) {
    Result<std::map<std::string, CardValue>> c = parse_model_card(*readme);
    if (!c.ok()) return R::fail(c.status);
    card = std::move(c.value);
  }
  const std::int64_t total = params.total;

  // ---- model card heuristics (metadata.py apply_metadata_heuristic, in the reference's order)
  auto use_str = [&](std::optional<std::string>& field, const char* key) -> Status {
    const auto it = card.find(key);
    if (it == card.end() || field) return Status::good();
    if (it->second.kind == CardValue::Kind::null) return Status::good();
    if (it->second.kind != CardValue::Kind::string) return unsupported_card(key, "is not a single string");
    field = it->second.text;
    return Status::good();
  };
  struct Use {
    std::optional<std::string>* field;
    const char* key;
  };
  const Use uses[] = {
      {&md.name, "name"}, {&md.author, "author"}, {&md.version, "version"}, {&md.organization, "organization"}, {&md.description, "description"},
      {&md.finetune, "finetune"}, {&md.basename, "basename"}, {&md.size_label, "size_label"}, {&md.source_url, "url"}, {&md.source_doi, "doi"},
      {&md.source_uuid, "uuid"}, {&md.source_repo_url, "repo_url"}, {&md.name, "model_name"}, {&md.author, "model_author"},
      {&md.version, "model_version"}, {&md.organization, "model_organization"}, {&md.description, "model_description"},
      {&md.finetune, "model_finetune"}, {&md.basename, "model_basename"}, {&md.size_label, "model_size_label"}, {&md.source_url, "model_url"},
      {&md.source_doi, "model_doi"}, {&md.source_uuid, "model_uuid"}, {&md.source_repo_url, "model_repo_url"}, {&md.name, "model_name"},
      {&md.author, "model_creator"}, {&md.basename, "model_type"},
  };
  for (const Use& u : uses) {
    const Status s = use_str(*u.field, u.key);
    if (!s) return R::fail(s);
  }

  // base models and datasets: a string, or a list of strings (a list of mappings is outside the subset)
  auto sources = [&](const char* k1, const char* k2, const char* k3, std::optional<std::vector<Source>>& field) -> Status {
    const bool present = card.count(k1) || card.count(k2) || card.count(k3);
    if (!present) return Status::good();
    const char* used = card.count(k1) ? k1 : card.count(k2) ? k2 : k3;
    const CardValue& v = card.at(used);
    std::vector<std::string> ids;
    if (v.kind == CardValue::Kind::string) ids.push_back(v.text);
    else if (v.kind == CardValue::Kind::sequence) {
      for (const CardValue& it : v.items) {
        if (it.kind != CardValue::Kind::string) return unsupported_card(used, "holds an entry that is not a string");
        ids.push_back(it.text);
      }
    } else if (v.kind == CardValue::Kind::other) {
      return unsupported_card(used, "is neither a string nor a list of strings");
    }
    if (!field) field = std::vector<Source>{};
    for (const std::string& id : ids) {
      Source src;
      if (id.rfind("http://", 0) == 0 || id.rfind("https://", 0) == 0 || id.rfind("ssh://", 0) == 0) {
        src.repo_url = id;
        if (id.find("huggingface.co") != std::string::npos) {
          if (const std::optional<std::string> mid = hf_url_id(id)) {
            const Result<ModelIdParts> c = model_id_components(*mid, total);
            if (!c.ok()) return c.status;
            if (c.value.full_name) src.name = id_to_title(*c.value.full_name);
            if (c.value.org) src.organization = id_to_title(*c.value.org);
            if (c.value.version) src.version = c.value.version;
          }
        }
      } else {
        const Result<ModelIdParts> c = model_id_components(id, total);
        if (!c.ok()) return c.status;
        if (c.value.full_name) src.name = id_to_title(*c.value.full_name);
        if (c.value.org) src.organization = id_to_title(*c.value.org);
        if (c.value.version) src.version = c.value.version;
        if (c.value.org && c.value.full_name) src.repo_url = "https://huggingface.co/" + *c.value.org + "/" + *c.value.full_name;
      }
      field->push_back(src);
    }
    return Status::good();
  };
  Status s = sources("base_model", "base_models", "base_model_sources", md.base_models);
  if (!s) return R::fail(s);
  s = sources("datasets", "dataset", "dataset_sources", md.datasets);
  if (!s) return R::fail(s);

  // license: a string or a list of strings (joined with ",")
  if (const auto it = card.find("license"); it != card.end() && it->second.kind != CardValue::Kind::null) {
    if (it->second.kind == CardValue::Kind::string) md.license = std::vector<std::string>{it->second.text};
    else if (it->second.kind == CardValue::Kind::sequence) {
      std::vector<std::string> l;
      for (const CardValue& x : it->second.items) {
        if (x.kind != CardValue::Kind::string) return R::fail(unsupported_card("license", "holds an entry that is not a string"));
        l.push_back(x.text);
      }
      md.license = l;
      md.license_is_list = true;
    } else {
      return R::fail(unsupported_card("license", "is neither a string nor a list of strings"));
    }
  }
  s = use_str(md.license_name, "license_name");
  if (!s) return R::fail(s);
  s = use_str(md.license_link, "license_link");
  if (!s) return R::fail(s);
  auto use_array = [&](std::optional<std::vector<std::string>>& field, const char* key) -> Status {
    const auto it = card.find(key);
    if (it == card.end() || it->second.kind == CardValue::Kind::null) return Status::good();
    std::vector<std::string> cur = field ? *field : std::vector<std::string>{};
    if (it->second.kind == CardValue::Kind::string) cur.push_back(it->second.text);
    else if (it->second.kind == CardValue::Kind::sequence) {
      for (const CardValue& x : it->second.items) {
        if (x.kind != CardValue::Kind::string) return unsupported_card(key, "holds an entry that is not a string");
        cur.push_back(x.text);
      }
    } else {
      return Status::good();  // the reference ignores a value that is neither a string nor a list
    }
    field = cur;
    return Status::good();
  };
  for (const auto& [field, key] : {std::pair{&md.tags, "tags"}, std::pair{&md.tags, "pipeline_tag"}, std::pair{&md.languages, "languages"},
                                   std::pair{&md.languages, "language"}}) {
    s = use_array(*field, key);
    if (!s) return R::fail(s);
  }

  // ---- config.json: _name_or_path when it is a model name (at most one '/')
  auto apply_id = [&](const std::string& id) -> Status {
    const Result<ModelIdParts> c = model_id_components(id, total);
    if (!c.ok()) return c.status;
    if (!md.name && c.value.full_name) md.name = id_to_title(*c.value.full_name);
    if (!md.organization && c.value.org) md.organization = id_to_title(*c.value.org);
    if (!md.basename && c.value.basename) md.basename = c.value.basename;
    if (!md.finetune && c.value.finetune) md.finetune = c.value.finetune;
    if (!md.version && c.value.version) md.version = c.value.version;
    if (!md.size_label && c.value.size_label) md.size_label = c.value.size_label;
    return Status::good();
  };
  if (config) {
    if (const Json* np = config->get("_name_or_path"); np && !np->is_null()) {
      if (!np->is_string()) return R::fail(Status::fail(Code::unsupported, "_name_or_path is not a string", 0, "config.json"));
      const std::string& v = np->as_string();
      std::size_t slashes = 0;
      for (const char c : v) slashes += c == '/';
      if (slashes <= 1) {
        s = apply_id(v);
        if (!s) return R::fail(s);
      }
    }
  }
  // ---- the folder's name
  s = apply_id(folder_name);
  if (!s) return R::fail(s);

  // ---- generation_config.json (sampling)
  if (gen && gen->is_object()) {
    auto int_of = [&](const char* key, std::optional<std::int64_t>& field) -> Status {
      const Json* v = gen->get(key);
      if (!v || v->is_null()) return Status::good();
      if (!v->is_integer() || v->as_int() < INT32_MIN || v->as_int() > INT32_MAX)
        return Status::fail(Code::unsupported, std::string("'") + key + "' is not a 32-bit integer", 0, "generation_config.json");
      field = v->as_int();
      return Status::good();
    };
    auto float_of = [&](const char* key, std::optional<double>& field) -> Status {
      const Json* v = gen->get(key);
      if (!v || v->is_null()) return Status::good();
      if (!v->is_number() || !(std::fabs(v->as_double()) <= 3.4028234663852886e38))
        return Status::fail(Code::unsupported, std::string("'") + key + "' is not a number a 32-bit float holds", 0, "generation_config.json");
      field = v->as_double();
      return Status::good();
    };
    if (const Json* v = gen->get("sequence"); v && !v->is_null()) {
      if (!v->is_string()) return R::fail(Status::fail(Code::unsupported, "'sequence' is not a string", 0, "generation_config.json"));
      md.sampling_sequence = v->as_string();
    }
    for (const Status& st : {int_of("top_k", md.top_k), float_of("top_p", md.top_p), float_of("min_p", md.min_p),
                             float_of("xtc_probability", md.xtc_probability), float_of("xtc_threshold", md.xtc_threshold),
                             float_of("temperature", md.temp), int_of("penalty_last_n", md.penalty_last_n)})
      if (!st) return R::fail(st);
    // penalty_repeat, else repetition_penalty
    if (const Json* v = gen->get("penalty_repeat"); v) {
      s = float_of("penalty_repeat", md.penalty_repeat);
    } else {
      s = float_of("repetition_penalty", md.penalty_repeat);
    }
    if (!s) return R::fail(s);
    for (const Status& st : {int_of("mirostat", md.mirostat), float_of("mirostat_tau", md.mirostat_tau), float_of("mirostat_eta", md.mirostat_eta)})
      if (!st) return R::fail(st);
  }

  // ---- what prepare_metadata adds: the folder's name when nothing named the model, the size label from the tensors
  if (!md.name) md.name = folder_name;
  if (!md.size_label && total > 0) md.size_label = size_label(params);

  // ---- the entries, in set_gguf_meta_model's order
  std::vector<GgufKvBlob> kv;
  if (md.sampling_sequence) add_str(kv, "general.sampling.sequence", *md.sampling_sequence);
  if (md.top_k) kv.push_back(io::gguf_kv_i32("general.sampling.top_k", static_cast<std::int32_t>(*md.top_k)));
  if (md.top_p) kv.push_back(io::gguf_kv_f32("general.sampling.top_p", static_cast<float>(*md.top_p)));
  if (md.min_p) kv.push_back(io::gguf_kv_f32("general.sampling.min_p", static_cast<float>(*md.min_p)));
  if (md.xtc_probability) kv.push_back(io::gguf_kv_f32("general.sampling.xtc_probability", static_cast<float>(*md.xtc_probability)));
  if (md.xtc_threshold) kv.push_back(io::gguf_kv_f32("general.sampling.xtc_threshold", static_cast<float>(*md.xtc_threshold)));
  if (md.temp) kv.push_back(io::gguf_kv_f32("general.sampling.temp", static_cast<float>(*md.temp)));
  if (md.penalty_last_n) kv.push_back(io::gguf_kv_i32("general.sampling.penalty_last_n", static_cast<std::int32_t>(*md.penalty_last_n)));
  if (md.penalty_repeat) kv.push_back(io::gguf_kv_f32("general.sampling.penalty_repeat", static_cast<float>(*md.penalty_repeat)));
  if (md.mirostat) kv.push_back(io::gguf_kv_i32("general.sampling.mirostat", static_cast<std::int32_t>(*md.mirostat)));
  if (md.mirostat_tau) kv.push_back(io::gguf_kv_f32("general.sampling.mirostat_tau", static_cast<float>(*md.mirostat_tau)));
  if (md.mirostat_eta) kv.push_back(io::gguf_kv_f32("general.sampling.mirostat_eta", static_cast<float>(*md.mirostat_eta)));
  add_str(kv, "general.name", *md.name);
  if (md.author) add_str(kv, "general.author", *md.author);
  if (md.version) add_str(kv, "general.version", *md.version);
  if (md.organization) add_str(kv, "general.organization", *md.organization);
  if (md.finetune) add_str(kv, "general.finetune", *md.finetune);
  if (md.basename) add_str(kv, "general.basename", *md.basename);
  if (md.description) add_str(kv, "general.description", *md.description);
  if (md.quantized_by) add_str(kv, "general.quantized_by", *md.quantized_by);
  if (md.size_label) add_str(kv, "general.size_label", *md.size_label);
  if (md.license) {
    std::string joined;
    for (std::size_t i = 0; i < md.license->size(); ++i) joined += (i ? "," : "") + (*md.license)[i];
    add_str(kv, "general.license", joined);
  }
  if (md.license_name) add_str(kv, "general.license.name", *md.license_name);
  if (md.license_link) add_str(kv, "general.license.link", *md.license_link);
  if (md.url) add_str(kv, "general.url", *md.url);
  if (md.doi) add_str(kv, "general.doi", *md.doi);
  if (md.uuid) add_str(kv, "general.uuid", *md.uuid);
  if (md.repo_url) add_str(kv, "general.repo_url", *md.repo_url);
  if (md.source_url) add_str(kv, "general.source.url", *md.source_url);
  if (md.source_doi) add_str(kv, "general.source.doi", *md.source_doi);
  if (md.source_uuid) add_str(kv, "general.source.uuid", *md.source_uuid);
  if (md.source_repo_url) add_str(kv, "general.source.repo_url", *md.source_repo_url);
  auto write_sources = [&](const std::vector<Source>& list, const std::string& prefix) {
    kv.push_back(io::gguf_kv_u32("general." + prefix + ".count", static_cast<std::uint32_t>(list.size())));
    for (std::size_t k = 0; k < list.size(); ++k) {
      const Source& e = list[k];
      const std::string p = "general." + prefix + "." + std::to_string(k) + ".";
      if (e.name) add_str(kv, p + "name", *e.name);
      if (e.author) add_str(kv, p + "author", *e.author);
      if (e.version) add_str(kv, p + "version", *e.version);
      if (e.organization) add_str(kv, p + "organization", *e.organization);
      if (e.description) add_str(kv, p + "description", *e.description);
      if (e.url) add_str(kv, p + "url", *e.url);
      if (e.doi) add_str(kv, p + "doi", *e.doi);
      if (e.uuid) add_str(kv, p + "uuid", *e.uuid);
      if (e.repo_url) add_str(kv, p + "repo_url", *e.repo_url);
    }
  };
  if (md.base_models) write_sources(*md.base_models, "base_model");
  if (md.datasets) write_sources(*md.datasets, "dataset");
  if (md.tags && !md.tags->empty()) kv.push_back(io::gguf_kv_arr_str("general.tags", *md.tags));
  if (md.languages && !md.languages->empty()) kv.push_back(io::gguf_kv_arr_str("general.languages", *md.languages));
  return R::good(std::move(kv));
}

}  // namespace pelias::ingest
