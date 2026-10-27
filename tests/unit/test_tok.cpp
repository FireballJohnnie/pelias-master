// provenance: original (expected splits derived by hand from the qwen35 pattern's alternatives; NFC expectations from
//   Python's unicodedata.normalize('NFC', ...), Unicode 15.1)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <string>
#include <vector>

#include "pelias_test.h"
#include "tok/tokenizer.h"
#include "tok/unicode.h"

using pelias::tok::EncodeOptions;
using pelias::tok::Tokenizer;

namespace {

// A tiny byte-level vocabulary: the 256 byte characters (ids 0..255 in byte order), a few merges, two added tokens.
Tokenizer tiny() {
  std::vector<std::string> tokens;
  std::vector<std::int32_t> types;
  // the GPT-2 byte map, written out the same way the tokenizer builds it
  bool direct[256] = {};
  for (int b = 33; b <= 126; ++b) direct[b] = true;
  for (int b = 161; b <= 172; ++b) direct[b] = true;
  for (int b = 174; b <= 255; ++b) direct[b] = true;
  char32_t next = 256;
  for (int b = 0; b < 256; ++b) {
    std::string s;
    pelias::tok::append_utf8(s, direct[b] ? static_cast<char32_t>(b) : next++);
    tokens.push_back(s);
    types.push_back(1);
  }
  const std::string sp = "\xC4\xA0";  // the byte-level character of ' ' (U+0120)
  for (const std::string& t : {std::string("ab"), std::string("bc"), std::string("abc"), std::string("aa"), sp + "a", sp + "ab"}) {
    tokens.push_back(t);
    types.push_back(1);
  }
  tokens.push_back("<|im_start|>");  // 262
  types.push_back(3);
  tokens.push_back("<think>");  // 263
  types.push_back(4);
  tokens.push_back("<|im_start|>x");  // 264, longer control token sharing a prefix
  types.push_back(3);
  const std::vector<std::string> merges = {"a b", "b c", "ab c", "a a", sp + " a", sp + "a b"};
  auto r = Tokenizer::from_parts(tokens, types, merges, "qwen35");
  CHECK(r.ok());
  return std::move(r.value);
}

std::vector<std::string> V(std::initializer_list<const char*> xs) {
  std::vector<std::string> v;
  for (const char* x : xs) v.emplace_back(x);
  return v;
}

}  // namespace

TEST(tok_split_qwen35_pattern) {
  const Tokenizer t = tiny();
  CHECK(t.split("Hello world") == V({"Hello", " world"}));
  CHECK(t.split("it's IT'S we're") == V({"it", "'s", " IT", "'S", " we", "'re"}));
  CHECK(t.split("x'\xC5\xBF") == V({"x", "'\xC5\xBF"}));  // ' + U+017F folds to 's
  CHECK(t.split("a  b") == V({"a", " ", " b"}));
  CHECK(t.split("a \n\n b") == V({"a", " \n\n", " b"}));
  CHECK(t.split("end   ") == V({"end", "   "}));
  CHECK(t.split("x!!\n\ny") == V({"x", "!!\n\n", "y"}));
  CHECK(t.split("123") == V({"1", "2", "3"}));
  CHECK(t.split(" \xC2\xBFQu\xC3\xA9?") == V({" \xC2\xBF", "Qu\xC3\xA9", "?"}));
  CHECK(t.split("e\xCC\x81x") == V({"e\xCC\x81x"}));             // letter, combining mark, letter: one run
  CHECK(t.split("\xCC\x81" "abc") == V({"\xCC\x81" "abc"}));       // a leading mark joins the run
  CHECK(t.split("\xE4\xBD\xA0\xE5\xA5\xBD!") == V({"\xE4\xBD\xA0\xE5\xA5\xBD", "!"}));
  CHECK(t.split("\tfoo") == V({"\tfoo"}));
  CHECK(t.split("a\r\n\r\nb") == V({"a", "\r\n\r\n", "b"}));
  CHECK(t.split(" \t x") == V({" \t", " x"}));
  CHECK(t.split("") == V({}));
}

TEST(tok_nfc) {
  using pelias::tok::to_nfc;
  CHECK_EQ(to_nfc("e\xCC\x81"), std::string("\xC3\xA9"));                 // e + U+0301 -> U+00E9
  CHECK_EQ(to_nfc("\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8"), std::string("\xEA\xB0\x81"));  // Hangul L V T -> U+AC01
  CHECK_EQ(to_nfc("a\xCC\x82\xCC\xA3"), std::string("\xE1\xBA\xAD"));       // reorder 0323 before 0302 -> U+1EAD
  CHECK_EQ(to_nfc("\xE0\xA4\x95\xE0\xA4\xBC"), std::string("\xE0\xA4\x95\xE0\xA4\xBC"));  // exclusion U+0958 stays apart
  CHECK_EQ(to_nfc("\xE2\x84\xAB"), std::string("\xC3\x85"));               // singleton U+212B -> U+00C5
  CHECK_EQ(to_nfc("\xCC\x81" "e"), std::string("\xCC\x81" "e"));           // a leading mark composes with nothing
  CHECK_EQ(to_nfc("A\xCC\x8A\xCC\x81"), std::string("\xC7\xBA"));          // -> U+01FA
  CHECK_EQ(to_nfc("Ha\xCC\x88\xCC\x84llo"), std::string("H\xC7\x9Fllo"));   // -> U+01DF
  CHECK_EQ(to_nfc("plain ascii"), std::string("plain ascii"));
}

TEST(tok_bpe_rank_and_ties) {
  const Tokenizer t = tiny();
  EncodeOptions o;
  CHECK(t.encode("abc", o) == std::vector<std::int32_t>({258}));       // a b -> ab (rank 0), ab c -> abc (rank 2)
  CHECK(t.encode("bc", o) == std::vector<std::int32_t>({257}));
  CHECK(t.encode("aaa", o) == std::vector<std::int32_t>({259, 'a'}));  // equal ranks: leftmost first
  CHECK(t.encode(" ab", o) == std::vector<std::int32_t>({' ', 256}));  // a b (rank 0) first; then no merge (Ġ, ab) exists
}

TEST(tok_added_tokens) {
  const Tokenizer t = tiny();
  EncodeOptions o;
  CHECK(t.encode("<|im_start|>ab<think>", o) == std::vector<std::int32_t>({262, 256, 263}));
  CHECK(t.encode("<|im_start|>xab", o) == std::vector<std::int32_t>({264, 256}));  // leftmost-longest
  o.parse_special = false;
  const auto ids = t.encode("<|im_start|>", o);
  CHECK(ids.size() > 1);  // control token left as text
  CHECK(t.encode("<think>", o) == std::vector<std::int32_t>({263}));  // user-defined still matched
}

TEST(tok_decode_round_trip) {
  const Tokenizer t = tiny();
  for (const char* s : {"Hello  world\n\n", "x'S \xE4\xBD\xA0 e\xCC\x81", "<|im_start|>ab<think> 12"}) {
    const auto ids = t.encode(s);
    CHECK_EQ(t.decode(ids), pelias::tok::to_nfc(s));
  }
}
