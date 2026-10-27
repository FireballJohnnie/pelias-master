// provenance: original (behaviour READ-SRC from the model's tokenizer.json (BPE, byte-level, NFC normalizer, the
//   qwen35 split pattern) and llama.cpp src/llama-vocab.cpp and src/unicode.cpp at commit 2ca15f5)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// tokenizer.h - the one tokenizer of the project (master 6.13): byte-level BPE as the model was trained with, read
// from the GGUF vocabulary. Used by the trainer's data pipeline and by the engine's server; gate P-D1 compares it
// with the reference tokenizers (zero mismatches).
//
// Encoding, in order:
//   1. the text is split on added tokens (control and user-defined), leftmost-longest match;
//   2. each remaining piece is normalised (NFC; off to match llama.cpp, which does not normalise);
//   3. the piece is split by the model's pre-tokenizer pattern (qwen35);
//   4. each split is mapped byte by byte to the byte-level alphabet and merged by BPE rank.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "io/gguf_reader.h"
#include "pelias/status.h"

namespace pelias::tok {

enum class TokenType : std::int32_t { undefined = 0, normal = 1, unknown = 2, control = 3, user_defined = 4, unused = 5, byte = 6 };

struct EncodeOptions {
  // Match control tokens such as <|im_start|> in the text. User-defined added tokens (<think>, <tool_call> ...) are
  // always matched, as both references do. With false, control tokens are tokenized as plain text (llama.cpp
  // --no-parse-special).
  bool parse_special = true;
  // NFC before splitting, as the model's tokenizer.json asks (the Hugging Face reference). llama.cpp skips it.
  bool nfc = true;
};

class Tokenizer {
 public:
  // Reads tokenizer.ggml.* from a parsed GGUF. Supports model "gpt2" with pre-tokenizer "qwen35".
  static Result<Tokenizer> from_gguf(const io::GgufReader& gguf);
  // Builds from parts (tests and other sources): tokens, their types, and merges as "left right" strings.
  static Result<Tokenizer> from_parts(std::vector<std::string> tokens, std::vector<std::int32_t> types,
                                      const std::vector<std::string>& merges, std::string pre);

  [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text, const EncodeOptions& opt = {}) const;
  // Pre-tokenizer split only (byte ranges of the input); exposed for tests and audits.
  [[nodiscard]] std::vector<std::string> split(std::string_view piece) const;
  [[nodiscard]] std::string decode(const std::vector<std::int32_t>& ids) const;
  [[nodiscard]] std::string piece(std::int32_t id) const;  // the text of one token (byte-level decoded)

  [[nodiscard]] std::size_t vocab_size() const { return tokens_.size(); }
  [[nodiscard]] const std::string& token(std::int32_t id) const { return tokens_[static_cast<std::size_t>(id)]; }
  [[nodiscard]] TokenType type(std::int32_t id) const { return static_cast<TokenType>(types_[static_cast<std::size_t>(id)]); }
  [[nodiscard]] std::int32_t find(std::string_view token_text) const;  // -1 when absent
  [[nodiscard]] std::int32_t eos_id() const { return eos_; }
  [[nodiscard]] std::int32_t pad_id() const { return pad_; }
  [[nodiscard]] std::int32_t bos_id() const { return bos_; }
  [[nodiscard]] bool add_bos() const { return add_bos_; }

 private:
  struct Added {
    std::string text;
    std::int32_t id;
    bool control;
  };
  struct MergeInfo {
    std::uint32_t rank;
    std::int32_t result;
  };

  Status build(const std::vector<std::string>& merges);
  void bpe(std::string_view word, std::vector<std::int32_t>& out) const;
  void encode_plain(std::string_view piece, const EncodeOptions& opt, std::vector<std::int32_t>& out) const;

  std::vector<std::string> tokens_;
  std::vector<std::int32_t> types_;
  std::string pre_;
  std::unordered_map<std::string, std::int32_t> by_text_;
  std::unordered_map<std::uint64_t, MergeInfo> merges_;
  std::int32_t byte_token_[256] = {};
  std::string byte_char_[256];                  // byte -> its byte-level character (UTF-8)
  std::unordered_map<char32_t, std::uint8_t> char_byte_;  // byte-level character -> byte
  std::vector<Added> added_;                    // sorted by text for matching
  std::int32_t eos_ = -1, pad_ = -1, bos_ = -1;
  bool add_bos_ = false;
};

}  // namespace pelias::tok
