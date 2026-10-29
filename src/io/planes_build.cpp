// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
#include "io/planes_build.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <thread>

#include "io/gguf_writer.h"
#include "io/imatrix_weights.h"
#include "quant/block_types.h"
#include "quant/planes.h"
#include "quant/quant_cpu.h"

namespace pelias::io {

namespace {
bool float_type(std::uint32_t t) {
  return t == static_cast<std::uint32_t>(quant::Type::f32) || t == static_cast<std::uint32_t>(quant::Type::f16) ||
         t == static_cast<std::uint32_t>(quant::Type::bf16);
}
bool raw_type(std::uint32_t t) {  // types with no rows to plane: floats, f64 and the integer tensors
  return float_type(t) || t == static_cast<std::uint32_t>(quant::Type::f64) ||
         (t >= static_cast<std::uint32_t>(quant::Type::i8) && t <= static_cast<std::uint32_t>(quant::Type::i64));
}
struct Job {
  const GgufTensor* m = nullptr;
  const GgufTensor* b = nullptr;
  bool rot = false;
  const std::vector<float>* imp = nullptr;
  std::uint64_t prb = 0, rows = 0;
};
}  // namespace

Result<PlaneBuildReport> build_planes(const os::FileMap& model_map, const GgufReader& model, const os::FileMap& base_map, const GgufReader& base,
                                      const std::string& out_path, const PlaneBuildOptions& opt, const PlaneProgress& progress) {
  using R = Result<PlaneBuildReport>;
  if (opt.k < 1 || opt.k > quant::kPlaneMaxK) return R::fail(Status::fail(Code::invalid_argument, "planes per row are 1 to 4"));
  if (opt.refit_rounds < 0) return R::fail(Status::fail(Code::invalid_argument, "refit rounds are 0 or more"));
  PlaneBuildReport rep;
  // the weights the model file stores rotated
  std::vector<std::string> rot;
  for (const GgufKv& kv : model.kv())
    if (kv.key == "pelias.rot32.tensors" && kv.value.type == GgufValueType::array && kv.value.array_type == GgufValueType::string)
      if (Status s = model.array_strings(kv.value, rot); !s) return R::fail(s);
  const std::string arch = model.get_string("general.architecture");
  bool found = false;
  const std::uint64_t blocks = model.get_uint(arch + ".block_count", &found);
  bool has_nextn = false;
  const std::uint64_t nextn = model.get_uint(arch + ".nextn_predict_layers", &has_nextn);
  if (!found || nextn >= blocks) has_nextn = false;  // the trunk is block_count - nextn_predict_layers blocks; the MTP blocks follow it
  ImatrixWeights imx;
  if (!opt.imatrix.empty()) {
    Result<ImatrixWeights> r = read_imatrix_weights(opt.imatrix);
    if (!r.ok()) return R::fail(r.status);
    imx = std::move(r.value);
  }
  std::vector<Job> jobs;
  for (const GgufTensor& t : model.tensors()) {
    if (raw_type(t.type) || t.n_dims < 2) continue;
    const auto skip = [&](const std::string& why) { rep.skipped.push_back(t.name + ": " + why); };
    if (opt.skip_mtp && has_nextn && nextn > 0) {
      bool in_mtp = false;
      for (std::uint64_t b = blocks - nextn; b < blocks; ++b) in_mtp = in_mtp || t.name.rfind("blk." + std::to_string(b) + ".", 0) == 0;
      if (in_mtp) continue;
    }
    if (t.ne[0] % quant::kPlaneBlock != 0) { skip("rows not a whole number of 256-value blocks"); continue; }
    if (t.ne[3] != 1) { skip("4-D"); continue; }
    if (!quant::can_dequantize(static_cast<quant::Type>(t.type))) { skip("type not readable"); continue; }
    const GgufTensor* b = base.find_tensor(t.name);
    if (!b) { skip("not in the original"); continue; }
    if (b->ne[0] != t.ne[0] || b->ne[1] != t.ne[1] || b->ne[2] != t.ne[2] || b->ne[3] != 1) { skip("another shape in the original"); continue; }
    if (!float_type(b->type)) { skip("the original is not F32 / F16 / BF16"); continue; }
    Job j;
    j.m = &t;
    j.b = b;
    j.rot = std::find(rot.begin(), rot.end(), t.name) != rot.end();
    // importance per column, only when it was collected in the basis the file stores (rotated or not)
    const bool imx_rot = std::find(imx.rot32.begin(), imx.rot32.end(), t.name) != imx.rot32.end();
    if (auto it = imx.weights.find(t.name); it != imx.weights.end() && it->second.size() == t.ne[0] * t.ne[2]) {
      if (imx_rot == j.rot) j.imp = &it->second;
      else ++rep.imatrix_basis_skips;
    }
    j.rows = t.ne[1] * t.ne[2];
    j.prb = static_cast<std::uint64_t>(quant::plane_row_bytes(static_cast<std::int64_t>(t.ne[0]), opt.k));
    jobs.push_back(j);
  }
  if (jobs.empty()) return R::fail(Status::fail(Code::invalid_argument, "no quantized weight of the model file has an original to fit planes to"));
  std::vector<GgufOutTensor> out;
  std::vector<std::int32_t> types;
  for (const Job& j : jobs) {
    GgufOutTensor o;
    o.name = j.m->name;
    o.ne[0] = j.prb;
    o.ne[1] = j.rows;
    o.type = static_cast<std::uint32_t>(quant::Type::i8);
    o.bytes = j.prb * j.rows;
    o.n_dims = 2;
    out.push_back(o);
    types.push_back(static_cast<std::int32_t>(j.m->type));
  }
  // the output layer's original rows: output.weight, or the embedding table when the model ties the two
  const GgufTensor* head = nullptr;
  if (opt.exact_head) head = base.find_tensor("output.weight") ? base.find_tensor("output.weight") : base.find_tensor("token_embd.weight");
  if (opt.exact_head && (!head || !float_type(head->type) || head->ne[2] != 1))
    return R::fail(Status::fail(Code::invalid_argument, "--exact-head needs output.weight (or a tied token_embd.weight) in the original as F32 / F16 / BF16"));
  if (head) {
    GgufOutTensor o;
    o.name = "output.weight.exact";
    o.ne[0] = head->ne[0];
    o.ne[1] = head->ne[1];
    o.type = static_cast<std::uint32_t>(quant::Type::bf16);
    o.bytes = head->ne[0] * head->ne[1] * 2;
    o.n_dims = 2;
    out.push_back(o);
  }
  std::vector<GgufKvBlob> kv{gguf_kv_str("general.architecture", "pelias-planes"), gguf_kv_u32("pelias.planes.k", static_cast<std::uint32_t>(opt.k)),
                             gguf_kv_u32("pelias.planes.block", static_cast<std::uint32_t>(quant::kPlaneBlock)),
                             gguf_kv_u32("pelias.planes.refit_rounds", static_cast<std::uint32_t>(opt.refit_rounds)),
                             gguf_kv_bool("pelias.planes.imatrix", !opt.imatrix.empty()), gguf_kv_arr_i32("pelias.planes.types", types)};
  GgufFileWriter wr;
  if (Status s = wr.begin(out_path, kv, out, 32); !s) return R::fail(s);
  const int threads = opt.threads > 0 ? opt.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  std::vector<std::uint8_t> buf;
  for (std::size_t ji = 0; ji < jobs.size(); ++ji) {
    const Job& j = jobs[ji];
    const std::int64_t n = static_cast<std::int64_t>(j.m->ne[0]);
    const std::uint64_t mrb = quant::row_bytes(j.m->type, j.m->ne[0]), brb = quant::row_bytes(j.b->type, j.b->ne[0]);
    buf.assign(j.prb * j.rows, 0);
    std::atomic<std::uint64_t> next{0};
    std::vector<double> eb(static_cast<std::size_t>(threads), 0.0), ea(static_cast<std::size_t>(threads), 0.0);
    auto work = [&](int tid) {
      std::vector<float> q(static_cast<std::size_t>(n)), w(static_cast<std::size_t>(n)), wt(static_cast<std::size_t>(n), 1.0f);
      for (std::uint64_t r = next.fetch_add(1); r < j.rows; r = next.fetch_add(1)) {
        (void)quant::dequantize_row(static_cast<quant::Type>(j.m->type), model_map.data() + j.m->file_offset + r * mrb, q.data(), n);
        (void)quant::dequantize_row(static_cast<quant::Type>(j.b->type), base_map.data() + j.b->file_offset + r * brb, w.data(), n);
        if (j.rot) quant::hadamard32(w.data(), n);
        const float* imp = j.imp ? j.imp->data() + (r / j.m->ne[1]) * static_cast<std::uint64_t>(n) : nullptr;
        for (std::int64_t i = 0; i < n; ++i) {
          w[static_cast<std::size_t>(i)] -= q[static_cast<std::size_t>(i)];
          const double iw = imp ? imp[i] : 1.0;
          eb[static_cast<std::size_t>(tid)] += iw * static_cast<double>(w[static_cast<std::size_t>(i)]) * w[static_cast<std::size_t>(i)];
        }
        ea[static_cast<std::size_t>(tid)] += quant::plane_fit_row(w.data(), imp, n, opt.k, opt.refit_rounds, buf.data() + r * j.prb);
      }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(work, t);
    work(0);
    for (auto& th : pool) th.join();
    for (int t = 0; t < threads; ++t) rep.err_before += eb[static_cast<std::size_t>(t)], rep.err_after += ea[static_cast<std::size_t>(t)];
    if (Status s = wr.write(buf.data(), buf.size()); !s) {
      wr.abandon();
      return R::fail(s);
    }
    rep.bytes += buf.size();
    ++rep.tensors;
    if (progress) progress(ji + 1, jobs.size(), j.m->name);
  }
  if (head) {
    const std::uint64_t rb = quant::row_bytes(head->type, head->ne[0]);
    std::vector<float> row(head->ne[0]);
    std::vector<std::uint8_t> hb(head->ne[0] * 2);
    for (std::uint64_t r = 0; r < head->ne[1]; ++r) {
      (void)quant::dequantize_row(static_cast<quant::Type>(head->type), base_map.data() + head->file_offset + r * rb, row.data(),
                                  static_cast<std::int64_t>(head->ne[0]));
      for (std::uint64_t i = 0; i < head->ne[0]; ++i) {
        const std::uint16_t h = quant::fp32_to_bf16(row[i]);
        hb[2 * i] = static_cast<std::uint8_t>(h & 0xff);
        hb[2 * i + 1] = static_cast<std::uint8_t>(h >> 8);
      }
      if (Status s = wr.write(hb.data(), hb.size()); !s) {
        wr.abandon();
        return R::fail(s);
      }
    }
  }
  if (Status s = wr.finish(); !s) return R::fail(s);
  return R::good(std::move(rep));
}

}  // namespace pelias::io
