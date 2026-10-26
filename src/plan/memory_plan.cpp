// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "plan/memory_plan.h"

#include <cstdio>
#include <map>

#include "plan/probe_constants.h"

namespace pelias::plan {

const char* class_name(TensorClass c) {
  switch (c) {
    case TensorClass::embedding: return "embedding";
    case TensorClass::head: return "head";
    case TensorClass::spine: return "spine";
    case TensorClass::expert: return "expert";
  }
  return "?";
}

TensorClass classify(const std::string& n) {
  if (n.rfind("token_embd", 0) == 0) return TensorClass::embedding;
  if (n == "output.weight" || n.rfind("output_norm", 0) == 0 || n == "output.bias") return TensorClass::head;
  // routed experts: ffn_gate_exps, ffn_up_exps, ffn_down_exps (and their biases); shared experts (_shexp) and the
  // router (ffn_gate_inp) stay on the spine
  if (n.find("_exps.") != std::string::npos || n.find("_exps_") != std::string::npos) return TensorClass::expert;
  return TensorClass::spine;
}

ByteMap byte_map(const io::GgufReader& r) {
  ByteMap m;
  std::map<std::string, std::uint64_t> per_layer;
  bool has_output = false;
  for (const io::GgufTensor& t : r.tensors()) {
    ++m.tensors;
    switch (classify(t.name)) {
      case TensorClass::embedding: m.embedding += t.bytes; break;
      case TensorClass::head:
        m.head += t.bytes;
        if (t.name == "output.weight") has_output = true;
        break;
      case TensorClass::spine: m.spine += t.bytes; break;
      case TensorClass::expert: {
        m.experts += t.bytes;
        const std::size_t dot = t.name.find('.', 4);  // "blk.N."
        per_layer[t.name.substr(0, dot)] += t.bytes;
        break;
      }
    }
  }
  m.head_tied = !has_output;
  m.expert_layers = per_layer.size();
  for (const auto& [k, v] : per_layer) m.per_layer_experts = v > m.per_layer_experts ? v : m.per_layer_experts;
  return m;
}

std::string format_bytes(std::uint64_t b) {
  char buf[64];
  if (b >= kGiB) std::snprintf(buf, sizeof buf, "%.2f GiB", static_cast<double>(b) / static_cast<double>(kGiB));
  else if (b >= (std::uint64_t{1} << 20)) std::snprintf(buf, sizeof buf, "%.1f MiB", static_cast<double>(b) / 1048576.0);
  else std::snprintf(buf, sizeof buf, "%llu bytes", static_cast<unsigned long long>(b));
  return buf;
}

Result<Plan> make_plan(const io::GgufReader& r, const PlanRequest& req) {
  Plan p;
  p.map = byte_map(r);
  p.ceiling = req.ceiling_bytes;
  p.ceiling_basis = req.ceiling_basis;
  p.working_set = req.working_set_bytes ? req.working_set_bytes : kWorkingSetEstimateBytes;
  p.working_set_label = req.working_set_bytes ? "stated" : kWorkingSetLabel;
  p.resident_fixed = p.map.always_resident();
  if (p.ceiling == 0) return Result<Plan>::fail(Status::fail(Code::invalid_argument, "no VRAM ceiling: no device was found and none was stated"));
  if (req.bank_bytes > p.map.experts)
    return Result<Plan>::fail(Status::fail(Code::invalid_argument, "requested bank " + format_bytes(req.bank_bytes) + " exceeds the experts' " + format_bytes(p.map.experts)));
  const std::uint64_t fixed = p.resident_fixed + p.working_set;
  p.largest_bank_that_fits = p.ceiling > fixed ? (p.ceiling - fixed < p.map.experts ? p.ceiling - fixed : p.map.experts) : 0;
  p.bank = req.bank_bytes ? req.bank_bytes : p.largest_bank_that_fits;
  p.peak = fixed + p.bank;
  p.fits = p.peak <= p.ceiling;
  p.headroom = p.fits ? p.ceiling - p.peak : 0;
  p.expert_bytes_resident_fraction = p.map.experts ? static_cast<double>(p.bank) / static_cast<double>(p.map.experts) : 0.0;
  if (!p.fits) {
    p.verdict = "REFUSED: peak " + format_bytes(p.peak) + " is over the ceiling " + format_bytes(p.ceiling) + " by " + format_bytes(p.peak - p.ceiling) +
                (p.largest_bank_that_fits ? "; largest bank that fits: " + format_bytes(p.largest_bank_that_fits) : "; nothing fits: the always-resident part alone is " + format_bytes(fixed));
    Result<Plan> res = Result<Plan>::fail(Status::fail(Code::over_ceiling, p.verdict));
    res.value = p;
    return res;
  }
  p.verdict = "fits: peak " + format_bytes(p.peak) + " under the ceiling " + format_bytes(p.ceiling) + " with " + format_bytes(p.headroom) + " to spare";
  return Result<Plan>::good(p);
}

std::string plan_text(const Plan& p) {
  char buf[256];
  std::string s;
  s += "model bytes: embedding " + format_bytes(p.map.embedding) + ", head " + format_bytes(p.map.head) + (p.map.head_tied ? " (tied to the embedding)" : "") +
       ", spine " + format_bytes(p.map.spine) + ", experts " + format_bytes(p.map.experts) + " in " + std::to_string(p.map.expert_layers) + " layers (largest layer " +
       format_bytes(p.map.per_layer_experts) + "); " + std::to_string(p.map.tensors) + " tensors, " + format_bytes(p.map.total()) + " in all (DERIVED from the tensor table)\n";
  s += "ceiling: " + format_bytes(p.ceiling) + " (" + p.ceiling_basis + ")\n";
  s += "always resident: " + format_bytes(p.resident_fixed) + "; working set " + format_bytes(p.working_set) + " (" + p.working_set_label + ")\n";
  std::snprintf(buf, sizeof buf, "hot bank: %s (%.1f %% of the expert bytes)\n", format_bytes(p.bank).c_str(), 100.0 * p.expert_bytes_resident_fraction);
  s += buf;
  s += "peak: " + format_bytes(p.peak) + "\n";
  s += p.verdict + "\n";
  return s;
}

}  // namespace pelias::plan
