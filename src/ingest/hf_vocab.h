// provenance: original (the rules READ-SRC from llama.cpp convert_hf_to_gguf.py, conversion/base.py (get_vocab_base,
//   _set_vocab_gpt2, does_token_look_special) and gguf-py/gguf/vocab.py (SpecialVocab) at commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a, used as a specification; no code copied)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// hf_vocab.h - the tokenizer.* metadata of a converted model, from a model folder's tokenizer.json,
// tokenizer_config.json, config.json and chat template files, exactly as the reference converter writes it for a GPT-2
// style (byte-level BPE) tokenizer: the token list padded to the model's vocabulary size, the token types, the merges,
// the special token ids, the add-BOS/EOS flags, the chat template and the normalizer flags.
//
// Two steps of the reference run the tokenizer library itself, and Pelias does them without it:
//   - the pre-tokenizer name ("qwen35", ...) comes from a hash of the tokens the library makes of a fixed text; Pelias
//     recognizes the tokenizer by its pre-tokenizer definition instead (a table below, one entry per tokenizer family
//     checked against the reference) and refuses a tokenizer that is not in it;
//   - an added token that is not "normalized" is encoded and decoded once by the library; for an added token the library
//     returns its text unchanged, which Pelias assumes (checked on the reference model's 33 added tokens by P-X6).
// Every other tokenizer layout the reference handles (SentencePiece, WordPiece, template post-processors ...) is
// refused as not built yet.
#pragma once

#include <string>
#include <vector>

#include "io/gguf_writer.h"
#include "io/json.h"
#include "pelias/status.h"

namespace pelias::ingest {

struct VocabFiles {
  const io::Json* tokenizer = nullptr;         // tokenizer.json (required)
  const io::Json* tokenizer_config = nullptr;  // tokenizer_config.json (optional)
  const io::Json* config = nullptr;            // config.json as written (special token ids)
  const std::string* chat_template_jinja = nullptr;  // chat_template.jinja (optional)
  const io::Json* chat_template_json = nullptr;      // chat_template.json (optional)
  bool additional_chat_templates = false;            // an additional_chat_templates/*.jinja file exists (not built yet)
};

// The tokenizer.* entries in the reference's order. `vocab_size` is the model's (config "vocab_size"; 0 = the
// tokenizer's own size).
Result<std::vector<io::GgufKvBlob>> gpt2_vocab_metadata(const VocabFiles& f, std::uint64_t vocab_size);

}  // namespace pelias::ingest
