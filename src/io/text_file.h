// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// text_file.h - reads a small whole file (a config, tokenizer or model card file of a model folder) into memory, with a
// size limit so a wrong path cannot pull a weight file into RAM. The bytes are returned as they are (no newline or
// encoding changes); the caller interprets them.
#pragma once

#include <cstdint>
#include <string>

#include "pelias/status.h"

namespace pelias::io {

// Code::not_found when the file does not exist; Code::invalid_argument when it is larger than `limit` bytes.
Result<std::string> read_text_file(const std::string& path, std::uint64_t limit = std::uint64_t{256} << 20);

}  // namespace pelias::io
