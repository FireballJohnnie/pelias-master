// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
#pragma once
// planes_build.h - writes the residual sign planes of a model file against its original weights (`pelias planes`;
// quant/planes.h has the format and the fit). Any model file works, ours or anyone's: for every product weight it stores
// in a quantized type, the original weight of the same name and shape (F32 / F16 / BF16, from the converted original) is
// rotated the way the model file stores it (pelias.rot32.tensors) and the residual original - stored is fitted with K
// planes, weighted per column by an importance matrix when one is given. The side file is a GGUF: one I8 tensor per weight
// (same name; rows of plane_row_bytes(n, K)), the model file's type of each in pelias.planes.types (so the side file is
// refused against another file), and optionally output.weight.exact, the original output rows in BF16 (unrotated; the
// embedding table's rows when the model ties the output layer to it).
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "io/gguf_reader.h"
#include "os/file_map.h"
#include "pelias/status.h"

namespace pelias::io {

struct PlaneBuildOptions {
  int k = 2;                 // planes per row (1..4)
  int refit_rounds = 8;      // joint refit rounds for k >= 2
  int threads = 0;           // 0: every hardware thread
  bool exact_head = false;   // also store the original output rows (BF16)
  std::string imatrix;       // optional importance matrix (GGUF)
  bool skip_mtp = true;      // leave out the MTP blocks (the last nextn_predict_layers blocks; the reference does not run them)
};

struct PlaneBuildReport {
  std::size_t tensors = 0;                 // weights given planes
  std::vector<std::string> skipped;        // quantized weights left out, with why
  std::uint64_t bytes = 0;                 // plane bytes written (without the exact head)
  double err_before = 0.0, err_after = 0.0;  // sum over every planed weight of imp * residual^2, before and after
  std::size_t imatrix_basis_skips = 0;     // weights fitted without importance: the matrix was collected in the other basis
};

using PlaneProgress = std::function<void(std::size_t done, std::size_t total, const std::string& name)>;

Result<PlaneBuildReport> build_planes(const os::FileMap& model_map, const GgufReader& model, const os::FileMap& base_map, const GgufReader& base,
                                      const std::string& out_path, const PlaneBuildOptions& opt, const PlaneProgress& progress = nullptr);

}  // namespace pelias::io
