// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
#include "io/planes_build.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <thread>

#include "io/gguf_writer.h"
#include "io/imatrix_weights.h"
#include "io/tier_build.h"
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
struct Part {  // the rows of one expert (tiered weight) or of the whole tensor (stock weight)
  const std::uint8_t* data = nullptr;
  std::uint32_t type = 0;
  std::uint64_t row_bytes = 0;
};
struct Job {
  std::string name;
  std::uint64_t ne0 = 0, ne1 = 0, ne2 = 1;
  std::vector<Part> parts;     // one per expert (tiered) or one for all rows
  bool tiered = false;
  std::int64_t signature = 0;  // pelias.planes.types entry
  const GgufTensor* b = nullptr;
  bool rot = false;
  const std::vector<float>* imp = nullptr;
  int k = 1;
  bool select = false;
  double weight = 1.0;
  std::uint64_t rows() const { return ne1 * ne2; }
  std::uint64_t nb() const { return ne0 / static_cast<std::uint64_t>(quant::kPlaneBlock); }
  const std::uint8_t* row(std::uint64_t r, std::uint32_t& type) const {
    if (!tiered) {
      type = parts[0].type;
      return parts[0].data + r * parts[0].row_bytes;
    }
    const Part& p = parts[r / ne1];
    type = p.type;
    return p.data + (r % ne1) * p.row_bytes;
  }
};

// the residual (original - stored) of row r, rotated as the file stores it, and the row's importance
void residual(const Job& j, const os::FileMap& base_map, std::uint64_t r, std::vector<float>& q, std::vector<float>& w) {
  const auto n = static_cast<std::int64_t>(j.ne0);
  std::uint32_t ty = 0;
  const std::uint8_t* src = j.row(r, ty);
  (void)quant::dequantize_row(static_cast<quant::Type>(ty), src, q.data(), n);
  const std::uint64_t brb = quant::row_bytes(j.b->type, j.b->ne[0]);
  (void)quant::dequantize_row(static_cast<quant::Type>(j.b->type), base_map.data() + j.b->file_offset + r * brb, w.data(), n);
  if (j.rot) quant::hadamard32(w.data(), n);
}

template <typename F>
void parallel_rows(std::uint64_t rows, int threads, F fn) {
  std::atomic<std::uint64_t> next{0};
  auto work = [&](int tid) {
    for (std::uint64_t r = next.fetch_add(1); r < rows; r = next.fetch_add(1)) fn(r, tid);
  };
  std::vector<std::thread> pool;
  for (int t = 1; t < threads; ++t) pool.emplace_back(work, t);
  work(0);
  for (auto& th : pool) th.join();
}
}  // namespace

Result<PlaneBuildReport> build_planes(const os::FileMap& model_map, const GgufReader& model, const os::FileMap& base_map, const GgufReader& base,
                                      const std::string& out_path, const PlaneBuildOptions& opt, const PlaneProgress& progress) {
  using R = Result<PlaneBuildReport>;
  if (opt.k < 1 || opt.k > quant::kPlaneMaxK) return R::fail(Status::fail(Code::invalid_argument, "planes per row are 1 to 4"));
  if (opt.refit_rounds < 0) return R::fail(Status::fail(Code::invalid_argument, "refit rounds are 0 or more"));
  for (const auto& [pat, kk] : opt.k_rules)
    if (kk < 0 || kk > quant::kPlaneMaxK) return R::fail(Status::fail(Code::invalid_argument, "a plane-count rule gives " + std::to_string(kk) + " (0 to 4)", 0, pat));
  if (!opt.select.empty() && opt.budget_bytes == 0) return R::fail(Status::fail(Code::invalid_argument, "block selection needs a byte budget"));
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
  auto k_of = [&](const std::string& name) {
    for (const auto& [pat, kk] : opt.k_rules)
      if (name.find(pat) != std::string::npos) return kk;
    return opt.k;
  };
  auto in_mtp = [&](const std::string& name) {
    if (!(opt.skip_mtp && has_nextn && nextn > 0)) return false;
    for (std::uint64_t b = blocks - nextn; b < blocks; ++b)
      if (name.rfind("blk." + std::to_string(b) + ".", 0) == 0) return true;
    return false;
  };
  // usage-tiered weights (io/tier_build.h): the weight name, its experts' types and parts
  std::map<std::string, std::vector<std::uint32_t>> tiered;
  for (const GgufKv& kv : model.kv()) {
    const std::string pre = "pelias.tier.types.";
    if (kv.key.rfind(pre, 0) != 0) continue;
    std::vector<std::int64_t> t;
    if (Status s = model.array_ints(kv.value, t); !s) return R::fail(s);
    std::vector<std::uint32_t>& v = tiered[kv.key.substr(pre.size())];
    for (const std::int64_t x : t) v.push_back(static_cast<std::uint32_t>(x));
  }
  std::vector<Job> jobs;
  auto finish_job = [&](Job& j) -> bool {  // shared checks; false: skipped (reason recorded)
    const auto skip = [&](const std::string& why) {
      rep.skipped.push_back(j.name + ": " + why);
      return false;
    };
    if (j.ne0 % quant::kPlaneBlock != 0) return skip("rows not a whole number of 256-value blocks");
    const GgufTensor* b = base.find_tensor(j.name);
    if (!b) return skip("not in the original");
    if (b->ne[0] != j.ne0 || b->ne[1] != j.ne1 || b->ne[2] != j.ne2 || b->ne[3] != 1) return skip("another shape in the original");
    if (!float_type(b->type)) return skip("the original is not F32 / F16 / BF16");
    j.b = b;
    j.rot = std::find(rot.begin(), rot.end(), j.name) != rot.end();
    // importance per column, only when it was collected in the basis the file stores (rotated or not)
    const bool imx_rot = std::find(imx.rot32.begin(), imx.rot32.end(), j.name) != imx.rot32.end();
    if (auto it = imx.weights.find(j.name); it != imx.weights.end() && it->second.size() == j.ne0 * j.ne2) {
      if (imx_rot == j.rot) j.imp = &it->second;
      else ++rep.imatrix_basis_skips;
    }
    j.select = !opt.select.empty() && j.name.find(opt.select) != std::string::npos;
    if (auto it = opt.block_weights.find(j.name); it != opt.block_weights.end()) j.weight = it->second;
    return true;
  };
  for (const GgufTensor& t : model.tensors()) {
    if (raw_type(t.type) || t.n_dims < 2 || in_mtp(t.name)) continue;
    bool part = false;  // a tiered weight's per-type part: planed through its weight below
    for (const auto& [nm, types] : tiered)
      if (t.name.rfind(nm + ".", 0) == 0) part = true;
    if (part) continue;
    Job j;
    j.name = t.name;
    j.k = k_of(t.name);
    if (j.k == 0) continue;
    if (t.ne[3] != 1) {
      rep.skipped.push_back(t.name + ": 4-D");
      continue;
    }
    if (!quant::can_dequantize(static_cast<quant::Type>(t.type))) {
      rep.skipped.push_back(t.name + ": type not readable");
      continue;
    }
    j.ne0 = t.ne[0];
    j.ne1 = t.ne[1];
    j.ne2 = t.ne[2];
    j.parts.push_back(Part{model_map.data() + t.file_offset, t.type, quant::row_bytes(t.type, t.ne[0])});
    j.signature = static_cast<std::int64_t>(t.type);
    if (finish_job(j)) jobs.push_back(std::move(j));
  }
  for (const auto& [nm, types] : tiered) {
    if (in_mtp(nm)) continue;
    Job j;
    j.name = nm;
    j.k = k_of(nm);
    if (j.k == 0) continue;
    j.tiered = true;
    j.signature = tier_signature(types);
    std::map<std::uint32_t, std::uint64_t> next;
    for (const std::uint32_t ty : types) {
      const GgufTensor* pt = model.find_tensor(tier_part_name(nm, ty));
      if (!pt) return R::fail(Status::fail(Code::corrupt, "a tiered weight's part is missing", 0, tier_part_name(nm, ty)));
      if (j.ne0 == 0) j.ne0 = pt->ne[0], j.ne1 = pt->ne[1];
      if (pt->ne[0] != j.ne0 || pt->ne[1] != j.ne1) return R::fail(Status::fail(Code::corrupt, "a tiered weight's parts differ in shape", 0, nm));
      const std::uint64_t rb = quant::row_bytes(ty, j.ne0);
      j.parts.push_back(Part{model_map.data() + pt->file_offset + next[ty]++ * j.ne1 * rb, ty, rb});
    }
    j.ne2 = types.size();
    if (finish_job(j)) {
      jobs.push_back(std::move(j));
      ++rep.tiered;
    }
  }
  if (jobs.empty()) return R::fail(Status::fail(Code::invalid_argument, "no quantized weight of the model file has an original to fit planes to"));
  const int threads = opt.threads > 0 ? opt.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  const auto nthr = static_cast<std::uint64_t>(threads);
  constexpr std::int64_t B = quant::kPlaneBlock;

  // ---- block selection, pass 1: every block of the selected tensors scored (relative error removed per byte)
  int ksel = 0;
  for (const Job& j : jobs)
    if (j.select) {
      if (ksel != 0 && ksel != j.k) return R::fail(Status::fail(Code::invalid_argument, "the selected tensors have different plane counts", 0, j.name));
      ksel = j.k;
    }
  std::vector<std::vector<std::uint8_t>> keep(jobs.size());  // per selected job: one bit per block, rows x nb
  if (ksel > 0) {
    std::vector<std::vector<float>> score(jobs.size());
    std::size_t done = 0, nsel = 0;
    for (const Job& j : jobs) nsel += j.select ? 1 : 0;
    for (std::size_t ji = 0; ji < jobs.size(); ++ji) {
      const Job& j = jobs[ji];
      if (!j.select) continue;
      const std::uint64_t nb = j.nb(), rows = j.rows();
      std::vector<float>& sc = score[ji];
      sc.assign(rows * nb, 0.0f);
      std::vector<double> energy(static_cast<std::size_t>(j.ne2 * nthr), 0.0);  // per matrix, per thread: sum imp w^2
      parallel_rows(rows, threads, [&](std::uint64_t r, int tid) {
        thread_local std::vector<float> q, w;
        thread_local std::vector<std::uint8_t> pb;
        q.resize(j.ne0), w.resize(j.ne0), pb.resize(static_cast<std::size_t>(34 * ksel));
        residual(j, base_map, r, q, w);
        const float* imp = j.imp ? j.imp->data() + (r / j.ne1) * j.ne0 : nullptr;
        double en = 0.0;
        for (std::uint64_t i = 0; i < j.ne0; ++i) en += (imp ? imp[i] : 1.0) * static_cast<double>(w[i]) * w[i];
        energy[static_cast<std::size_t>((r / j.ne1) * nthr + static_cast<std::uint64_t>(tid))] += en;
        for (std::uint64_t i = 0; i < j.ne0; ++i) w[i] -= q[i];
        for (std::uint64_t b = 0; b < nb; ++b) {
          const float* rb = w.data() + b * B;
          const float* ib = imp ? imp + b * B : nullptr;
          double before = 0.0;
          for (std::int64_t i = 0; i < B; ++i) before += (ib ? ib[i] : 1.0) * static_cast<double>(rb[i]) * rb[i];
          const double after = quant::plane_fit_row(rb, ib, B, ksel, opt.refit_rounds, pb.data());
          sc[r * nb + b] = static_cast<float>(before - after);  // normalised below by the matrix's energy
        }
      });
      for (std::uint64_t r = 0; r < rows; ++r) {
        double en = 0.0;
        for (std::uint64_t t = 0; t < nthr; ++t) en += energy[static_cast<std::size_t>((r / j.ne1) * nthr + t)];
        const double f = en > 0.0 ? j.weight / en : 0.0;
        for (std::uint64_t b = 0; b < nb; ++b) sc[r * nb + b] = static_cast<float>(sc[r * nb + b] * f);
      }
      rep.blocks_candidate += rows * nb;
      if (progress) progress(++done, nsel, j.name + " (scored)");
    }
    // the threshold: the budget's worth of the highest scores
    const std::uint64_t want = opt.budget_bytes / static_cast<std::uint64_t>(34 * ksel);
    std::vector<float> all;
    all.reserve(rep.blocks_candidate);
    for (const auto& v : score) all.insert(all.end(), v.begin(), v.end());
    float tau = 0.0f;
    std::uint64_t at_tau = 0;  // blocks equal to tau that still fit, taken in table order
    if (want == 0) {
      tau = 1e30f;
    } else if (want < all.size()) {
      std::nth_element(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(want - 1), all.end(), std::greater<float>());
      tau = all[want - 1];
      std::uint64_t above = 0;
      for (const float x : all) above += x > tau ? 1 : 0;
      at_tau = want - above;
    } else {
      tau = -1e30f;
    }
    std::vector<float>().swap(all);
    for (std::size_t ji = 0; ji < jobs.size(); ++ji) {
      if (!jobs[ji].select) continue;
      const std::uint64_t nb = jobs[ji].nb(), rows = jobs[ji].rows(), mb = (nb + 7) / 8;
      keep[ji].assign(rows * mb, 0);
      const std::vector<float>& sc = score[ji];
      for (std::uint64_t r = 0; r < rows; ++r)
        for (std::uint64_t b = 0; b < nb; ++b) {
          const float x = sc[r * nb + b];
          bool k = x > tau;
          if (!k && x == tau && at_tau > 0) k = true, --at_tau;
          if (k) {
            keep[ji][r * mb + b / 8] |= static_cast<std::uint8_t>(1u << (b % 8));
            ++rep.blocks_selected;
          }
        }
      std::vector<float>().swap(score[ji]);
    }
  }

  // ---- the side file's layout
  std::vector<GgufOutTensor> out;
  std::vector<std::int32_t> types, ks, sparse;
  std::vector<std::vector<std::uint32_t>> offs(jobs.size());
  for (std::size_t ji = 0; ji < jobs.size(); ++ji) {
    const Job& j = jobs[ji];
    const std::uint64_t rows = j.rows();
    GgufOutTensor o;
    o.name = j.name;
    o.type = static_cast<std::uint32_t>(quant::Type::i8);
    o.n_dims = 2;
    if (!j.select) {
      o.ne[0] = static_cast<std::uint64_t>(quant::plane_row_bytes(static_cast<std::int64_t>(j.ne0), j.k));
      o.ne[1] = rows;
      o.bytes = o.ne[0] * rows;
      out.push_back(o);
    } else {
      const std::uint64_t nb = j.nb(), mb = (nb + 7) / 8;
      std::vector<std::uint32_t>& off = offs[ji];
      off.resize(rows);
      std::uint64_t cnt = 0;
      for (std::uint64_t r = 0; r < rows; ++r) {
        off[r] = static_cast<std::uint32_t>(cnt);
        for (std::uint64_t b = 0; b < nb; ++b) cnt += static_cast<std::uint64_t>((static_cast<unsigned>(keep[ji][r * mb + b / 8]) >> (b % 8)) & 1u);
      }
      if (cnt > 0xFFFFFFFFull) return R::fail(Status::fail(Code::unsupported, "more than 2^32 selected blocks in one weight", 0, j.name));
      o.ne[0] = static_cast<std::uint64_t>(34 * j.k);
      o.ne[1] = std::max<std::uint64_t>(cnt, 1);  // an empty selection keeps one zero block (a tensor needs data)
      o.bytes = o.ne[0] * o.ne[1];
      out.push_back(o);
      GgufOutTensor m;
      m.name = j.name + ".planemask";
      m.type = static_cast<std::uint32_t>(quant::Type::i8);
      m.n_dims = 2;
      m.ne[0] = mb;
      m.ne[1] = rows;
      m.bytes = mb * rows;
      out.push_back(m);
      GgufOutTensor f;
      f.name = j.name + ".planeoff";
      f.type = static_cast<std::uint32_t>(quant::Type::i32);
      f.n_dims = 1;
      f.ne[0] = rows;
      f.bytes = rows * 4;
      out.push_back(f);
    }
    types.push_back(static_cast<std::int32_t>(j.signature));
    ks.push_back(j.k);
    sparse.push_back(j.select ? 1 : 0);
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
  int kmax = 1;
  for (const std::int32_t x : ks) kmax = std::max(kmax, static_cast<int>(x));
  std::vector<GgufKvBlob> kv{gguf_kv_str("general.architecture", "pelias-planes"), gguf_kv_u32("pelias.planes.k", static_cast<std::uint32_t>(kmax)),
                             gguf_kv_u32("pelias.planes.block", static_cast<std::uint32_t>(quant::kPlaneBlock)),
                             gguf_kv_u32("pelias.planes.refit_rounds", static_cast<std::uint32_t>(opt.refit_rounds)),
                             gguf_kv_bool("pelias.planes.imatrix", !opt.imatrix.empty()), gguf_kv_arr_i32("pelias.planes.types", types),
                             gguf_kv_arr_i32("pelias.planes.ks", ks), gguf_kv_arr_i32("pelias.planes.sparse", sparse)};
  if (ksel > 0) kv.push_back(gguf_kv_str("pelias.planes.select", opt.select + " / " + std::to_string(opt.budget_bytes) + " bytes"));
  GgufFileWriter wr;
  if (Status s = wr.begin(out_path, kv, out, 32); !s) return R::fail(s);

  // ---- pass 2: fit and write
  std::vector<std::uint8_t> buf;
  for (std::size_t ji = 0; ji < jobs.size(); ++ji) {
    const Job& j = jobs[ji];
    const std::uint64_t rows = j.rows(), nb = j.nb(), mb = (nb + 7) / 8;
    const std::uint64_t blk = static_cast<std::uint64_t>(34 * j.k);
    const std::uint64_t prb = j.select ? 0 : static_cast<std::uint64_t>(quant::plane_row_bytes(static_cast<std::int64_t>(j.ne0), j.k));
    buf.assign(j.select ? 0 : prb * rows, 0);
    if (j.select) {  // the selected blocks' total (the last row's offset + its own blocks)
      std::uint64_t cnt = rows ? offs[ji].back() : 0;
      for (std::uint64_t b = 0; rows && b < nb; ++b) cnt += static_cast<std::uint64_t>((static_cast<unsigned>(keep[ji][(rows - 1) * mb + b / 8]) >> (b % 8)) & 1u);
      buf.assign(std::max<std::uint64_t>(cnt, 1) * blk, 0);
    }
    std::vector<double> eb(static_cast<std::size_t>(threads), 0.0), ea(static_cast<std::size_t>(threads), 0.0);
    parallel_rows(rows, threads, [&](std::uint64_t r, int tid) {
      thread_local std::vector<float> q, w;
      q.resize(j.ne0), w.resize(j.ne0);
      residual(j, base_map, r, q, w);
      const float* imp = j.imp ? j.imp->data() + (r / j.ne1) * j.ne0 : nullptr;
      for (std::uint64_t i = 0; i < j.ne0; ++i) w[i] -= q[i];
      double before = 0.0;
      for (std::uint64_t i = 0; i < j.ne0; ++i) before += (imp ? imp[i] : 1.0) * static_cast<double>(w[i]) * w[i];
      eb[static_cast<std::size_t>(tid)] += before;
      if (!j.select) {
        ea[static_cast<std::size_t>(tid)] += quant::plane_fit_row(w.data(), imp, static_cast<std::int64_t>(j.ne0), j.k, opt.refit_rounds, buf.data() + r * prb);
        return;
      }
      std::uint64_t at = offs[ji][r];
      double after = before;
      for (std::uint64_t b = 0; b < nb; ++b) {
        if (!((keep[ji][r * mb + b / 8] >> (b % 8)) & 1)) continue;
        const float* rb = w.data() + b * B;
        const float* ib = imp ? imp + b * B : nullptr;
        double bb = 0.0;
        for (std::int64_t i = 0; i < B; ++i) bb += (ib ? ib[i] : 1.0) * static_cast<double>(rb[i]) * rb[i];
        after += quant::plane_fit_row(rb, ib, B, j.k, opt.refit_rounds, buf.data() + at * blk) - bb;
        ++at;
      }
      ea[static_cast<std::size_t>(tid)] += after;
    });
    double sb = 0.0, sa = 0.0;
    for (int t = 0; t < threads; ++t) sb += eb[static_cast<std::size_t>(t)], sa += ea[static_cast<std::size_t>(t)];
    rep.err_before += sb;
    rep.err_after += sa;
    if (j.select) rep.select_err_before += sb, rep.select_err_after += sa;
    Status s = wr.write(buf.data(), buf.size());
    if (s && j.select) {
      s = wr.write(keep[ji].data(), keep[ji].size());
      if (s) s = wr.write(reinterpret_cast<const std::uint8_t*>(offs[ji].data()), offs[ji].size() * 4);
    }
    if (!s) {
      wr.abandon();
      return R::fail(s);
    }
    rep.bytes += buf.size();
    ++rep.tensors;
    if (progress) progress(ji + 1, jobs.size(), j.name);
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
