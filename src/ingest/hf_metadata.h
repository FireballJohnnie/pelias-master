// provenance: original (the rules READ-SRC from llama.cpp gguf-py/gguf/metadata.py, utility.py and gguf_writer.py at
//   commit 2ca15f5404760548c39e7b92bd43116a09414a1a, used as a specification: which files are read, the model card
//   heuristics, the name parsing, the size label and the order of the general.* entries; no code copied)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// hf_metadata.h - the general.* metadata a converted model carries, worked out exactly as the reference converter works
// it out from a model folder: the model card's front matter (README.md), config.json, generation_config.json and the
// folder's name, plus the parameter count of the tensors written. Only the YAML the front matter of real model cards
// uses is read (top-level keys with plain or quoted scalars, block or flow sequences); a key the reference would read
// with a value outside that subset, or a value the reference itself could not write, is refused rather than guessed.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "io/gguf_writer.h"
#include "io/json.h"
#include "pelias/status.h"

namespace pelias::ingest {

// A tensor as the reference converter records it: its GGUF name and its array shape (outermost dimension first).
struct NamedShape {
  std::string name;
  std::vector<std::uint64_t> shape;
};

// The reference's parameter count (gguf_writer get_total_parameter_count): expert tensors are those whose name holds
// "_exps." and have at least 3 dimensions; their per-expert size counts once.
struct ParamCount {
  std::int64_t total = 0, shared = 0, expert = 0, expert_count = 0;
};
ParamCount count_parameters(const std::vector<NamedShape>& tensors);

// "256x2.6B", "35B" ... (utility.py size_label and model_weight_count_rounded_notation).
std::string rounded_count(std::int64_t n, int min_digits);
std::string size_label(const ParamCount& p);

// Name helpers (metadata.py id_to_title and get_model_id_components), ASCII names only.
std::string id_to_title(std::string_view s);
struct ModelIdParts {
  std::optional<std::string> full_name, org, basename, finetune, version, size_label;
};
Result<ModelIdParts> model_id_components(std::string_view model_id, std::int64_t total_params);

// One value of the model card's front matter, as far as the reference's rules need it.
struct CardValue {
  enum class Kind { null, string, sequence, other };  // other: a value outside the subset (number, boolean, mapping ...)
  Kind kind = Kind::null;
  std::string text;                        // string
  std::vector<CardValue> items;            // sequence (each item null, string or other)
};
// Parses the front matter of a README.md as the reference does (the lines between a first line "---" and the next
// "---"); an empty map when there is none. Fails only when the top-level structure cannot be read at all.
Result<std::map<std::string, CardValue>> parse_model_card(std::string_view readme);

// The general.* entries in the reference's order, from the folder's name, its files (nullptr when absent) and the
// parameter count of the tensors being written.
Result<std::vector<io::GgufKvBlob>> general_metadata(const std::string& folder_name, const std::string* readme, const io::Json* config,
                                                     const io::Json* generation_config, const ParamCount& params);

}  // namespace pelias::ingest
