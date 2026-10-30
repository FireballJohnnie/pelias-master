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
#include <map>
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
  // Planes per tensor: the first rule whose text is part of the tensor's name sets its plane count (0 = no planes for it);
  // a tensor no rule names gets k. Example: {"output.weight", 3}, {"_exps", 1}, {"", 1}.
  std::vector<std::pair<std::string, int>> k_rules;
  // Block selection (Option A, docs/PELIAS_RESIDUAL_PLANES.md 6.7): the tensors whose name contains `select` keep planes only
  // on the 256-blocks that remove the most relative error (the block's importance-weighted error removed / the importance-
  // weighted energy of its matrix, times block_weights[name] when given), budget_bytes of plane data in all, in the side file
  // as block-selective tensors (<name> + <name>.planemask + <name>.planeoff). Those tensors must share one plane count.
  std::string select;
  std::uint64_t budget_bytes = 0;
  std::map<std::string, double> block_weights;
};

struct PlaneBuildReport {
  std::size_t tensors = 0;                 // weights given planes
  std::vector<std::string> skipped;        // quantized weights left out, with why
  std::uint64_t bytes = 0;                 // plane bytes written (without the exact head)
  double err_before = 0.0, err_after = 0.0;  // sum over every planed weight of imp * residual^2, before and after
  std::size_t imatrix_basis_skips = 0;     // weights fitted without importance: the matrix was collected in the other basis
  std::size_t tiered = 0;                  // usage-tiered weights given planes (fitted expert by expert in each expert's type)
  std::uint64_t blocks_candidate = 0, blocks_selected = 0;  // block selection: blocks scored, blocks kept
  double select_err_before = 0.0, select_err_after = 0.0;   // the selected tensors' importance-weighted error, before / after
};

using PlaneProgress = std::function<void(std::size_t done, std::size_t total, const std::string& name)>;

Result<PlaneBuildReport> build_planes(const os::FileMap& model_map, const GgufReader& model, const os::FileMap& base_map, const GgufReader& base,
                                      const std::string& out_path, const PlaneBuildOptions& opt, const PlaneProgress& progress = nullptr);

}  // namespace pelias::io
