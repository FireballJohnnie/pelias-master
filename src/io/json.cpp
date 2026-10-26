// provenance: original (RFC 8259)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/json.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace pelias::io {
namespace {
constexpr int kMaxDepth = 512;
}  // namespace

namespace detail {

class JsonParser {
 public:
  explicit JsonParser(std::string_view t) : t_(t) {}
  Status run(Json& out) {
    skip_ws();
    Status s = value(out, 0);
    if (!s) return s;
    skip_ws();
    if (pos_ != t_.size()) return fail("unexpected text after the document");
    return Status::good();
  }

 private:
  Status fail(const std::string& m) const { return Status::fail(Code::corrupt, "JSON: " + m, pos_); }
  [[nodiscard]] bool eof() const { return pos_ >= t_.size(); }
  [[nodiscard]] char peek() const { return eof() ? '\0' : t_[pos_]; }
  void skip_ws() {
    while (!eof() && (t_[pos_] == ' ' || t_[pos_] == '\t' || t_[pos_] == '\n' || t_[pos_] == '\r')) ++pos_;
  }
  bool consume(std::string_view lit) {
    if (t_.substr(pos_, lit.size()) == lit) {
      pos_ += lit.size();
      return true;
    }
    return false;
  }

  Status value(Json& out, int depth) {
    if (depth > kMaxDepth) return fail("nesting deeper than 512");
    if (eof()) return fail("unexpected end of document");
    const char c = peek();
    if (c == '{') return object(out, depth);
    if (c == '[') return array(out, depth);
    if (c == '"') {
      std::string s;
      Status st = string(s);
      if (!st) return st;
      out = Json::string(std::move(s));
      return Status::good();
    }
    if (consume("true")) { out = Json::boolean(true); return Status::good(); }
    if (consume("false")) { out = Json::boolean(false); return Status::good(); }
    if (consume("null")) { out = Json::null(); return Status::good(); }
    if (c == '-' || (c >= '0' && c <= '9')) return number(out);
    return fail(std::string("unexpected character '") + c + "'");
  }

  Status number(Json& out) {
    const std::size_t start = pos_;
    bool neg = false;
    if (peek() == '-') { neg = true; ++pos_; }
    if (eof() || !(peek() >= '0' && peek() <= '9')) return fail("digit expected");
    if (peek() == '0') {
      ++pos_;
      if (!eof() && peek() >= '0' && peek() <= '9') return fail("leading zero");
    } else {
      while (!eof() && peek() >= '0' && peek() <= '9') ++pos_;
    }
    bool is_int = true;
    if (!eof() && peek() == '.') {
      is_int = false;
      ++pos_;
      if (eof() || !(peek() >= '0' && peek() <= '9')) return fail("digit expected after the decimal point");
      while (!eof() && peek() >= '0' && peek() <= '9') ++pos_;
    }
    if (!eof() && (peek() == 'e' || peek() == 'E')) {
      is_int = false;
      ++pos_;
      if (!eof() && (peek() == '+' || peek() == '-')) ++pos_;
      if (eof() || !(peek() >= '0' && peek() <= '9')) return fail("digit expected in the exponent");
      while (!eof() && peek() >= '0' && peek() <= '9') ++pos_;
    }
    const std::string text(t_.substr(start, pos_ - start));
    if (is_int) {
      // exact integer when it fits in 64 bits
      std::uint64_t u = 0;
      bool fits = true;
      for (std::size_t k = neg ? 1 : 0; k < text.size(); ++k) {
        const unsigned d = static_cast<unsigned>(text[k] - '0');
        if (u > (std::numeric_limits<std::uint64_t>::max() - d) / 10) { fits = false; break; }
        u = u * 10 + d;
      }
      if (fits && !neg) { out = Json::unsigned_integer(u); return Status::good(); }
      if (fits && neg && u <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1) {
        out = Json::integer(u == static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1 ? std::numeric_limits<std::int64_t>::min() : -static_cast<std::int64_t>(u));
        return Status::good();
      }
    }
    char* end = nullptr;
    const double d = std::strtod(text.c_str(), &end);
    if (!end || *end != '\0' || !std::isfinite(d)) return fail("number out of range");
    out = Json::number(d);
    return Status::good();
  }

  static void put_utf8(std::string& s, std::uint32_t cp) {
    if (cp < 0x80) s += static_cast<char>(cp);
    else if (cp < 0x800) { s += static_cast<char>(0xC0 | (cp >> 6)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += static_cast<char>(0xE0 | (cp >> 12)); s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
    else { s += static_cast<char>(0xF0 | (cp >> 18)); s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
  }
  Status hex4(std::uint32_t& out) {
    if (pos_ + 4 > t_.size()) return fail("truncated \\u escape");
    out = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = t_[pos_++];
      out <<= 4;
      if (c >= '0' && c <= '9') out |= static_cast<std::uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') out |= static_cast<std::uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') out |= static_cast<std::uint32_t>(c - 'A' + 10);
      else return fail("bad hex digit in \\u escape");
    }
    return Status::good();
  }
  Status string(std::string& out) {
    ++pos_;  // opening quote
    out.clear();
    while (true) {
      if (eof()) return fail("unterminated string");
      const unsigned char c = static_cast<unsigned char>(t_[pos_++]);
      if (c == '"') return Status::good();
      if (c < 0x20) return fail("control character inside a string");
      if (c != '\\') { out += static_cast<char>(c); continue; }
      if (eof()) return fail("unterminated escape");
      const char e = t_[pos_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          std::uint32_t cp = 0;
          Status s = hex4(cp);
          if (!s) return s;
          if (cp >= 0xD800 && cp <= 0xDBFF) {  // high surrogate: a low one must follow
            if (!consume("\\u")) return fail("lone high surrogate");
            std::uint32_t lo = 0;
            s = hex4(lo);
            if (!s) return s;
            if (lo < 0xDC00 || lo > 0xDFFF) return fail("bad low surrogate");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            return fail("lone low surrogate");
          }
          put_utf8(out, cp);
          break;
        }
        default: return fail("unknown escape");
      }
    }
  }

  Status array(Json& out, int depth) {
    ++pos_;
    out = Json::array();
    skip_ws();
    if (peek() == ']') { ++pos_; return Status::good(); }
    while (true) {
      Json v;
      Status s = value(v, depth + 1);
      if (!s) return s;
      out.push(std::move(v));
      skip_ws();
      if (peek() == ',') { ++pos_; skip_ws(); continue; }
      if (peek() == ']') { ++pos_; return Status::good(); }
      return fail("',' or ']' expected");
    }
  }

  Status object(Json& out, int depth) {
    ++pos_;
    out = Json::object();
    skip_ws();
    if (peek() == '}') { ++pos_; return Status::good(); }
    std::unordered_set<std::string> keys;
    while (true) {
      skip_ws();
      if (peek() != '"') return fail("object key expected");
      const std::size_t key_at = pos_;
      std::string key;
      Status s = string(key);
      if (!s) return s;
      if (!keys.insert(key).second) return Status::fail(Code::corrupt, "JSON: duplicate key", key_at, key);
      skip_ws();
      if (peek() != ':') return fail("':' expected");
      ++pos_;
      skip_ws();
      Json v;
      s = value(v, depth + 1);
      if (!s) return s;
      out.members_.emplace_back(std::move(key), std::move(v));  // unique: checked above, so no scan
      skip_ws();
      if (peek() == ',') { ++pos_; continue; }
      if (peek() == '}') { ++pos_; return Status::good(); }
      return fail("',' or '}' expected");
    }
  }

  std::string_view t_;
  std::size_t pos_ = 0;
};

}  // namespace detail

namespace {

void escape_into(std::string& out, const std::string& s) {
  out += '"';
  for (const char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
}

}  // namespace

const Json* Json::get(std::string_view key) const {
  if (kind_ != Kind::object) return nullptr;
  for (const auto& m : members_)
    if (m.first == key) return &m.second;
  return nullptr;
}

const Json* Json::at(std::size_t i) const { return kind_ == Kind::array && i < items_.size() ? &items_[i] : nullptr; }

Json& Json::push(Json v) {
  items_.push_back(std::move(v));
  return *this;
}

Json& Json::set(std::string key, Json v) {
  for (auto& m : members_)
    if (m.first == key) {
      m.second = std::move(v);
      return *this;
    }
  members_.emplace_back(std::move(key), std::move(v));
  return *this;
}

Result<Json> Json::parse(std::string_view text) {
  Json j;
  detail::JsonParser p(text);
  const Status s = p.run(j);
  if (!s) return Result<Json>::fail(s);
  return Result<Json>::good(std::move(j));
}

std::string Json::dump(int indent) const {
  std::string out;
  dump_to(out, indent, 0);
  return out;
}

void Json::dump_to(std::string& out, int indent, int depth) const {
  auto newline = [&](int d) {
    if (indent <= 0) return;
    out += '\n';
    out.append(static_cast<std::size_t>(d * indent), ' ');
  };
  switch (kind_) {
    case Kind::null: out += "null"; break;
    case Kind::boolean: out += b_ ? "true" : "false"; break;
    case Kind::number: {
      if (is_int_) {
        out += (i_ < 0 && u_ == 0) ? std::to_string(i_) : std::to_string(u_);
      } else if (std::isfinite(d_)) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.17g", d_);
        out += buf;
      } else {
        out += "null";  // JSON has no representation for infinities or NaN
      }
      break;
    }
    case Kind::string: escape_into(out, s_); break;
    case Kind::array:
      out += '[';
      for (std::size_t i = 0; i < items_.size(); ++i) {
        if (i) out += ',';
        newline(depth + 1);
        items_[i].dump_to(out, indent, depth + 1);
      }
      if (!items_.empty()) newline(depth);
      out += ']';
      break;
    case Kind::object:
      out += '{';
      for (std::size_t i = 0; i < members_.size(); ++i) {
        if (i) out += ',';
        newline(depth + 1);
        escape_into(out, members_[i].first);
        out += indent > 0 ? ": " : ":";
        members_[i].second.dump_to(out, indent, depth + 1);
      }
      if (!members_.empty()) newline(depth);
      out += '}';
      break;
  }
}

}  // namespace pelias::io
