// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// unicode.h - the few Unicode facts the tokenizer needs: the category class of a code point (letter, mark, number),
// the White_Space property, UTF-8 decoding that keeps byte positions, and canonical composition (NFC). The tables are
// generated from a stated Unicode version (tools/gen_unicode_tables.py) and looked up by binary search.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace pelias::tok {

struct CpRange {
  char32_t lo;
  char32_t hi;
};
struct CccRange {
  char32_t lo;
  char32_t hi;
  std::uint8_t ccc;
};
struct DecompEntry {
  char32_t cp;
  std::uint32_t offset;  // into kDecompPool
  std::uint16_t length;
};
struct ComposePair {
  char32_t first;
  char32_t second;
  char32_t composite;
};

namespace tables {
extern const char* const kUnicodeVersion;
extern const CpRange kLetter[];
extern const std::size_t kLetter_count;
extern const CpRange kMark[];
extern const std::size_t kMark_count;
extern const CpRange kNumber[];
extern const std::size_t kNumber_count;
extern const CpRange kWhiteSpace[];
extern const std::size_t kWhiteSpace_count;
extern const CccRange kCcc[];
extern const std::size_t kCcc_count;
extern const DecompEntry kDecomp[];
extern const std::size_t kDecomp_count;
extern const char32_t kDecompPool[];
extern const std::size_t kDecompPool_count;
extern const ComposePair kCompose[];
extern const std::size_t kCompose_count;
}  // namespace tables

// Category flags of one code point; a code point that is none of these has all flags false.
struct CpFlags {
  bool letter = false;
  bool mark = false;
  bool number = false;
  bool whitespace = false;
};
CpFlags flags_of(char32_t cp);
std::uint8_t combining_class(char32_t cp);
const char* unicode_version();

// One decoded code point with its place in the source bytes. An invalid byte is returned as its own unit with
// valid = false (cp = the byte value); it carries no category flags, so it falls into the "other" class.
struct CpUnit {
  char32_t cp = 0;
  std::uint32_t offset = 0;
  std::uint8_t length = 0;
  bool valid = true;
};
std::vector<CpUnit> decode_utf8(std::string_view text);
void append_utf8(std::string& out, char32_t cp);

// True when the text is valid UTF-8.
bool is_valid_utf8(std::string_view text);
// Canonical composition (Unicode NFC). Input must be valid UTF-8; invalid input is returned unchanged.
std::string to_nfc(std::string_view text);

}  // namespace pelias::tok
