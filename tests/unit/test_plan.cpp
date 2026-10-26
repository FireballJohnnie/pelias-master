// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/gguf_reader.h"
#include "os/file_map.h"
#include "pelias_test.h"
#include "plan/memory_plan.h"
#include "plan/probe_constants.h"
#include "unit/gguf_builder.h"

using namespace pelias;
using namespace pelias::plan;

namespace {
// a model with 1 MiB of spine, 2 MiB embedding, 2 MiB head and 8 MiB of experts in 2 layers (F32 so bytes are plain)
ptest::GgufBuilder model() {
  ptest::GgufBuilder b;
  b.kv_string("general.architecture", "qwen35moe");
  b.tensor("token_embd.weight", {256, 2048}, 0);           // 2 MiB
  b.tensor("output.weight", {256, 2048}, 0);               // 2 MiB
  b.tensor("output_norm.weight", {256}, 0);
  b.tensor("blk.0.attn_norm.weight", {256, 512}, 0);       // 512 KiB spine
  b.tensor("blk.1.ffn_gate_inp.weight", {256, 512}, 0);    // 512 KiB spine (router)
  b.tensor("blk.0.ffn_gate_exps.weight", {256, 4096}, 0);  // 4 MiB experts
  b.tensor("blk.1.ffn_up_exps.weight", {256, 4096}, 0);    // 4 MiB experts
  return b;
}
constexpr std::uint64_t MiB = 1u << 20;
}  // namespace

TEST(plan_classifies_and_sizes) {
  os::FileMap m;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/plan.gguf", model().build(), m));
  io::GgufReader r;
  CHECK(r.parse(m).ok());
  const ByteMap bm = byte_map(r);
  CHECK_EQ(bm.embedding, 2 * MiB);
  CHECK_EQ(bm.head, 2 * MiB + 1024);
  CHECK_EQ(bm.spine, 1 * MiB);
  CHECK_EQ(bm.experts, 8 * MiB);
  CHECK_EQ(bm.expert_layers, 2u);
  CHECK_EQ(bm.per_layer_experts, 4 * MiB);
  CHECK(!bm.head_tied);
  CHECK(classify("blk.3.ffn_down_shexp.weight") == TensorClass::spine);
  CHECK(classify("blk.3.ffn_down_exps.weight") == TensorClass::expert);
  CHECK(classify("blk.3.ssm_a") == TensorClass::spine);
}

TEST(plan_refuses_over_the_ceiling_and_proposes) {
  os::FileMap m;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/plan.gguf", model().build(), m));
  io::GgufReader r;
  CHECK(r.parse(m).ok());
  PlanRequest q;
  q.ceiling_basis = "stated";
  q.working_set_bytes = 1 * MiB;
  // fixed = 5 MiB + 1024 + 1 MiB working set = 6 MiB + 1024; experts 8 MiB
  q.ceiling_bytes = 10 * MiB;
  q.bank_bytes = 2 * MiB;
  Result<Plan> p = make_plan(r, q);
  CHECK(p.ok());
  CHECK(p.value.fits);
  CHECK_EQ(p.value.peak, 8 * MiB + 1024);
  CHECK_EQ(p.value.headroom, 2 * MiB - 1024);
  // bank 0 = largest that fits: 10 MiB - (6 MiB + 1024) = 4 MiB - 1024
  q.bank_bytes = 0;
  p = make_plan(r, q);
  CHECK(p.ok());
  CHECK_EQ(p.value.bank, 4 * MiB - 1024);
  // bank 6 MiB does not fit: refused with the proposal
  q.bank_bytes = 6 * MiB;
  p = make_plan(r, q);
  CHECK(!p.ok());
  CHECK(p.status.code == Code::over_ceiling);
  CHECK(!p.value.fits);
  CHECK_EQ(p.value.largest_bank_that_fits, 4 * MiB - 1024);
  // a ceiling smaller than the always-resident part: nothing fits
  q.ceiling_bytes = 3 * MiB;
  q.bank_bytes = 0;
  p = make_plan(r, q);
  CHECK(!p.ok());
  CHECK_EQ(p.value.largest_bank_that_fits, 0u);
  // bank over the experts: invalid
  q.ceiling_bytes = 100 * MiB;
  q.bank_bytes = 9 * MiB;
  p = make_plan(r, q);
  CHECK(p.status.code == Code::invalid_argument);
  // no ceiling: invalid
  q.ceiling_bytes = 0;
  q.bank_bytes = 0;
  CHECK(make_plan(r, q).status.code == Code::invalid_argument);
  // the default working set is the labelled serving figure (not the training 2.50 GiB)
  q.ceiling_bytes = 100 * MiB;
  q.working_set_bytes = 0;
  p = make_plan(r, q);
  CHECK_EQ(p.value.working_set, kServingWorkingSetBytes);
  CHECK_EQ(kServingWorkingSetBytes, kGiB);
  CHECK(p.value.working_set_label.find("DERIVED") != std::string::npos);
  CHECK(!plan_text(p.value).empty());
}

TEST(plan_counts_the_mtp_head) {
  // two trunk blocks and one MTP block (block_count 3, nextn_predict_layers 1), as the served file stores it
  ptest::GgufBuilder b = model();
  b.kv_u32("qwen35moe.block_count", 3).kv_u32("qwen35moe.nextn_predict_layers", 1);
  b.tensor("blk.2.attn_q.weight", {256, 1024}, 0);          // 1 MiB, mtp
  b.tensor("blk.2.nextn.eh_proj.weight", {256, 1024}, 0);   // 1 MiB, mtp
  b.tensor("blk.2.ffn_down_exps.weight", {256, 2048}, 0);   // 2 MiB, expert (the head's routed experts)
  os::FileMap m;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/plan_mtp.gguf", b.build(), m));
  io::GgufReader r;
  CHECK(r.parse(m).ok());
  CHECK_EQ(first_mtp_block(r), 2u);
  CHECK(classify("blk.2.attn_q.weight", 2) == TensorClass::mtp);
  CHECK(classify("blk.2.nextn.enorm.weight", 2) == TensorClass::mtp);
  CHECK(classify("blk.2.ffn_down_exps.weight", 2) == TensorClass::expert);
  CHECK(classify("blk.1.attn_q.weight", 2) == TensorClass::spine);
  CHECK(classify("blk.12.attn_q.weight", 2) == TensorClass::mtp);   // 12 >= 2: index read as a number, not a prefix
  CHECK(classify("blk.2.attn_q.weight") == TensorClass::spine);     // no MTP in the file: ordinary block
  const ByteMap bm = byte_map(r);
  CHECK_EQ(bm.mtp_blocks, 1u);
  CHECK_EQ(bm.mtp, 2 * MiB);
  CHECK_EQ(bm.mtp_experts, 2 * MiB);
  CHECK_EQ(bm.experts, 10 * MiB);
  CHECK_EQ(bm.expert_layers, 3u);
  CHECK_EQ(bm.always_resident(), 7 * MiB + 1024);   // 5 MiB + 1024 as before, plus the 2 MiB head
  // the head shrinks the largest bank by exactly its resident bytes
  PlanRequest q;
  q.ceiling_basis = "stated";
  q.working_set_bytes = 1 * MiB;
  q.ceiling_bytes = 12 * MiB;
  Result<Plan> p = make_plan(r, q);
  CHECK(p.ok());
  CHECK_EQ(p.value.largest_bank_that_fits, 4 * MiB - 1024);
  CHECK(plan_text(p.value).find("MTP head: 1 block(s)") != std::string::npos);
  // a file without the keys has no head
  os::FileMap m2;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/plan.gguf", model().build(), m2));
  io::GgufReader r2;
  CHECK(r2.parse(m2).ok());
  CHECK_EQ(first_mtp_block(r2), kNoMtp);
  CHECK(plan_text(make_plan(r2, q).value).find("MTP head: none") != std::string::npos);
}
