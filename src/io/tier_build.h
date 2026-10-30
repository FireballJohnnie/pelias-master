// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// Usage-tiered expert files (docs/PELIAS_RESIDUAL_PLANES.md 6.3-6.7; author 2026-10-30 06:41): every routed expert of a
// layer keeps its own storage type, so the experts the router picks most can be stored finer and the rest coarser at the
// same total size. Stock GGUF stores all experts of a layer in one tensor of one type; a tiered file instead stores
//   pelias.tier.version           u32 = 1
//   pelias.tier.types.<name>      i32 array, one type id per expert (expert order)
//   <name>.<type name, lower>     one tensor per type used: [ne0, ne1, count] holding those experts in expert order
// for each tiered weight <name> (blk.L.ffn_{gate,up,down}_exps.weight). Every other tensor is the base file's own, copied
// byte for byte. Only Pelias reads this layout.
#pragma once

#include <cctype>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "io/gguf_reader.h"
#include "os/file_map.h"
#include "pelias/status.h"
#include "quant/block_types.h"

namespace pelias::io {

inline std::string tier_types_key(const std::string& name) { return "pelias.tier.types." + name; }

inline std::string tier_part_name(const std::string& name, std::uint32_t type) {
  std::string t = quant::type_info(type).name;
  for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return name + "." + t;
}

// The allocation (tools/tier_alloc.py): weight name -> one type id per expert.
Result<std::map<std::string, std::vector<std::uint32_t>>> read_tier_alloc(const std::string& path);

struct TierBuildOptions {
  int threads = 0;          // 0: every hardware thread
  std::string imatrix;      // importance matrix (GGUF); each expert quantized with its own column weights when present
};

struct TierBuildReport {
  std::size_t tiered = 0, copied = 0;     // tiered weights; tensors copied from the base file
  std::uint64_t expert_bytes_base = 0;    // the tiered weights' bytes in the base file
  std::uint64_t expert_bytes_out = 0;     // and in the output
  std::uint64_t file_bytes = 0;
  std::map<std::uint32_t, std::uint64_t> experts_by_type;  // expert matrices per type
  std::size_t with_imatrix = 0;           // tiered weights quantized with importance
};

using TierProgress = std::function<void(std::size_t done, std::size_t total, const std::string& name)>;

// Writes `out_path` (refused if it exists): the base file's metadata and tensors, except that each weight named in
// `alloc` is quantized again, expert by expert, from the original model's rows (`orig`, F32 / F16 / BF16, same name and
// shape) into the type the allocation gives that expert.
Result<TierBuildReport> build_tiered(const os::FileMap& base_map, const GgufReader& base, const os::FileMap& orig_map, const GgufReader& orig,
                                     const std::map<std::string, std::vector<std::uint32_t>>& alloc, const std::string& out_path,
                                     const TierBuildOptions& opt, const TierProgress& progress = nullptr);

}  // namespace pelias::io
