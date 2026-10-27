// provenance: original (the composition algorithm follows the Unicode Standard Annex #15 description; the Hangul
//   arithmetic is the one in the Unicode Standard, chapter 3.12)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "tok/unicode.h"

#include <algorithm>

namespace pelias::tok {
namespace {

bool in_ranges(const CpRange* r, std::size_t n, char32_t cp) {
  std::size_t lo = 0, hi = n;
  while (lo < hi) {
    const std::size_t mid = (lo + hi) / 2;
    if (cp < r[mid].lo) {
      hi = mid;
    } else if (cp > r[mid].hi) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

void decompose(char32_t cp, std::u32string& out) {
  if (cp >= kSBase && cp < kSBase + kSCount) {
    const char32_t s = cp - kSBase;
    out.push_back(kLBase + s / kNCount);
    out.push_back(kVBase + (s % kNCount) / kTCount);
    if (s % kTCount != 0) out.push_back(kTBase + s % kTCount);
    return;
  }
  std::size_t lo = 0, hi = tables::kDecomp_count;
  while (lo < hi) {
    const std::size_t mid = (lo + hi) / 2;
    const DecompEntry& e = tables::kDecomp[mid];
    if (cp < e.cp) {
      hi = mid;
    } else if (cp > e.cp) {
      lo = mid + 1;
    } else {
      out.append(tables::kDecompPool + e.offset, e.length);
      return;
    }
  }
  out.push_back(cp);
}

char32_t compose(char32_t a, char32_t b) {
  if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount)
    return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
  if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase && b < kTBase + kTCount)
    return a + (b - kTBase);
  const ComposePair* first = tables::kCompose;
  const ComposePair* last = tables::kCompose + tables::kCompose_count;
  const ComposePair* it = std::lower_bound(first, last, ComposePair{a, b, 0}, [](const ComposePair& x, const ComposePair& y) {
    return x.first != y.first ? x.first < y.first : x.second < y.second;
  });
  if (it != last && it->first == a && it->second == b) return it->composite;
  return 0;
}

}  // namespace

const char* unicode_version() { return tables::kUnicodeVersion; }

CpFlags flags_of(char32_t cp) {
  CpFlags f;
  if (cp < 0x80) {
    f.letter = (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    f.number = cp >= '0' && cp <= '9';
    f.whitespace = cp == ' ' || (cp >= 0x09 && cp <= 0x0D);
    return f;
  }
  f.letter = in_ranges(tables::kLetter, tables::kLetter_count, cp);
  if (!f.letter) f.mark = in_ranges(tables::kMark, tables::kMark_count, cp);
  if (!f.letter && !f.mark) f.number = in_ranges(tables::kNumber, tables::kNumber_count, cp);
  f.whitespace = in_ranges(tables::kWhiteSpace, tables::kWhiteSpace_count, cp);
  return f;
}

std::uint8_t combining_class(char32_t cp) {
  if (cp < 0x300) return 0;
  std::size_t lo = 0, hi = tables::kCcc_count;
  while (lo < hi) {
    const std::size_t mid = (lo + hi) / 2;
    if (cp < tables::kCcc[mid].lo) {
      hi = mid;
    } else if (cp > tables::kCcc[mid].hi) {
      lo = mid + 1;
    } else {
      return tables::kCcc[mid].ccc;
    }
  }
  return 0;
}

std::vector<CpUnit> decode_utf8(std::string_view text) {
  std::vector<CpUnit> out;
  out.reserve(text.size());
  const auto* p = reinterpret_cast<const unsigned char*>(text.data());
  const std::size_t n = text.size();
  std::size_t i = 0;
  while (i < n) {
    const unsigned char c = p[i];
    CpUnit u;
    u.offset = static_cast<std::uint32_t>(i);
    std::size_t len = 0;
    char32_t cp = 0;
    char32_t min = 0;
    if (c < 0x80) {
      len = 1;
      cp = c;
    } else if ((c & 0xE0) == 0xC0) {
      len = 2;
      cp = c & 0x1F;
      min = 0x80;
    } else if ((c & 0xF0) == 0xE0) {
      len = 3;
      cp = c & 0x0F;
      min = 0x800;
    } else if ((c & 0xF8) == 0xF0) {
      len = 4;
      cp = c & 0x07;
      min = 0x10000;
    }
    bool ok = len != 0 && i + len <= n;
    for (std::size_t k = 1; ok && k < len; ++k) {
      if ((p[i + k] & 0xC0) != 0x80) {
        ok = false;
      } else {
        cp = (cp << 6) | (p[i + k] & 0x3F);
      }
    }
    if (ok && len > 1 && (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))) ok = false;
    if (!ok) {
      u.cp = c;
      u.length = 1;
      u.valid = false;
      out.push_back(u);
      ++i;
      continue;
    }
    u.cp = cp;
    u.length = static_cast<std::uint8_t>(len);
    out.push_back(u);
    i += len;
  }
  return out;
}

void append_utf8(std::string& out, char32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

bool is_valid_utf8(std::string_view text) {
  for (const CpUnit& u : decode_utf8(text))
    if (!u.valid) return false;
  return true;
}

std::string to_nfc(std::string_view text) {
  // Fast path: below U+0300 nothing decomposes into, or composes with, anything else.
  bool simple = true;
  for (const char ch : text) {
    if (static_cast<unsigned char>(ch) >= 0xCC) {  // lead bytes of U+0300 and above
      simple = false;
      break;
    }
  }
  if (simple) return std::string(text);
  const std::vector<CpUnit> units = decode_utf8(text);
  std::u32string buf;
  buf.reserve(units.size() + 8);
  for (const CpUnit& u : units) {
    if (!u.valid) return std::string(text);
    decompose(u.cp, buf);
  }
  // canonical ordering: a stable sort of each run of non-starters by combining class
  for (std::size_t i = 0; i < buf.size();) {
    if (combining_class(buf[i]) == 0) {
      ++i;
      continue;
    }
    std::size_t j = i;
    while (j < buf.size() && combining_class(buf[j]) != 0) ++j;
    std::stable_sort(buf.begin() + static_cast<std::ptrdiff_t>(i), buf.begin() + static_cast<std::ptrdiff_t>(j),
                     [](char32_t a, char32_t b) { return combining_class(a) < combining_class(b); });
    i = j;
  }
  // canonical composition
  if (!buf.empty()) {
    std::size_t starter_pos = 0;
    char32_t starter = buf[0];
    int last_class = combining_class(starter);
    if (last_class != 0) last_class = 256;  // a leading non-starter blocks everything after it
    std::size_t comp_pos = 1;
    for (std::size_t k = 1; k < buf.size(); ++k) {
      const char32_t ch = buf[k];
      const int ch_class = combining_class(ch);
      const char32_t composite = compose(starter, ch);
      if (composite != 0 && (last_class < ch_class || last_class == 0)) {
        buf[starter_pos] = composite;
        starter = composite;
        continue;
      }
      if (ch_class == 0) {
        starter_pos = comp_pos;
        starter = ch;
      }
      last_class = ch_class;
      buf[comp_pos++] = ch;
    }
    buf.resize(comp_pos);
  }
  std::string out;
  out.reserve(text.size());
  for (char32_t cp : buf) append_utf8(out, cp);
  return out;
}

}  // namespace pelias::tok
