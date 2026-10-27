// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// synth_model.h - a scaled-down qwen35moe model for the quantizer's end-to-end test: the reference model's metadata
// (from the 16 MiB header sample of tests/data, with block_count 41 and nextn_predict_layers 1 as in the served file)
// and the served file's 753 tensor names and types in its export order, with small shapes and data made by an integer
// generator, so every platform builds the same bytes. The reference quantizer's output on this file is pinned by its
// SHA-256 in test_requantize.cpp.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pelias/status.h"
#include "quant/type_rules.h"

namespace ptest {

// The served model's tensor set in the order its bf16 export stores it, with the export's types (BF16 / F32).
std::vector<pelias::quant::TensorDesc> qwen35moe_served_tensors();

// Writes the synthetic model to `out_path` from the header sample at `header_path`. Returns the status.
pelias::Status write_synthetic_qwen35moe(const std::string& header_path, const std::string& out_path);

}  // namespace ptest
