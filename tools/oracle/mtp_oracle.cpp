// provenance: original (test-only oracle; links the reference library llama.cpp, built outside this repository, and
//   drives its MTP draft context the way its common/speculative.cpp draft-mtp driver does, READ-SRC as a specification)
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
//
// GP3-08 gate: the reference library's MTP draft logits for a prompt. The target context runs the prompt (every
// position's post-norm hidden kept, unmasked); the MTP context runs token p paired with the target's hidden of p - 1
// (zeros before the first token), as the driver's catch-up step does; every position's logits and the MTP block's own
// post-norm hidden are written. Output: int32 n, int32 n_vocab, int32 n_embd, then n x n_vocab logits, then n x n_embd
// hidden (fp32, little-endian). Nothing in the product links this (boundary check BC2).
//   mtp_oracle MODEL.gguf OUT.bin ID ID ID ...
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common.h"
#include "llama.h"
#include "llama-ext.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: mtp_oracle MODEL.gguf OUT.bin ID ID ...\n");
    return 1;
  }
  std::vector<llama_token> ids;
  for (int i = 3; i < argc; ++i) ids.push_back(static_cast<llama_token>(std::atoi(argv[i])));
  const int n = static_cast<int>(ids.size());
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  mp.load_mtp = true;  // the driver sets this whenever MTP drafting is on (common.cpp common_model_params_to_llama)
  llama_model* model = llama_model_load_from_file(argv[1], mp);
  if (!model) return 2;
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_batch = 512;
  cp.n_ubatch = 512;
  cp.n_threads = 2;
  cp.n_threads_batch = 2;
  cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  cp.type_k = GGML_TYPE_F32;
  cp.type_v = GGML_TYPE_F32;
  llama_context* tgt = llama_init_from_model(model, cp);
  if (!tgt) return 3;
  llama_set_embeddings_nextn(tgt, true, /*masked*/ false);
  llama_context_params dp = cp;
  dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
  dp.n_rs_seq = 0;
  dp.ctx_other = tgt;
  llama_context* dft = llama_init_from_model(model, dp);
  if (!dft) return 4;
  llama_set_embeddings_nextn(dft, true, /*masked*/ true);
  const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
  const int ne = llama_model_n_embd(model);
  common_batch bt(tgt);
  for (int i = 0; i < n; ++i) bt.add(ids[static_cast<std::size_t>(i)], i, 0, true);
  if (llama_process(tgt, LLAMA_PROCESS_TYPE_DECODE, bt.get()) != 0) return 5;
  std::vector<float> h(static_cast<std::size_t>(n) * ne);
  for (int i = 0; i < n; ++i) std::memcpy(h.data() + static_cast<std::size_t>(i) * ne, llama_get_embeddings_nextn_ith(tgt, i), sizeof(float) * ne);
  std::vector<float> zero(static_cast<std::size_t>(ne), 0.0f);
  common_batch bd(dft);
  for (int i = 0; i < n; ++i) {
    const int idx = bd.add(ids[static_cast<std::size_t>(i)], i, 0, true);
    const float* row = i == 0 ? zero.data() : h.data() + static_cast<std::size_t>(i - 1) * ne;
    bd.set_embd(idx, {row, 1, static_cast<std::size_t>(ne)});
  }
  if (llama_process(dft, LLAMA_PROCESS_TYPE_DECODE, bd.get()) != 0) return 6;
  FILE* f = std::fopen(argv[2], "wb");
  if (!f) return 7;
  const int hdr[3] = {n, nv, ne};
  std::fwrite(hdr, sizeof hdr, 1, f);
  for (int i = 0; i < n; ++i) std::fwrite(llama_get_logits_ith(dft, i), sizeof(float), static_cast<std::size_t>(nv), f);
  for (int i = 0; i < n; ++i) std::fwrite(llama_get_embeddings_nextn_ith(dft, i), sizeof(float), static_cast<std::size_t>(ne), f);
  std::fclose(f);
  llama_free(dft);
  llama_free(tgt);
  llama_model_free(model);
  return 0;
}
