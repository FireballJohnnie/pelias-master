// provenance: original (test-only oracle; links the reference library llama.cpp, built outside this repository)
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
//
// GP3-09 gate: the reference library's logits for a prompt, every position, one batch, fp32 KV cache, CPU only.
// Output: int32 n, int32 n_vocab, then n x n_vocab logits (fp32, little-endian). Nothing in the product links this (BC2).
//   logits_oracle MODEL.gguf OUT.bin ID ID ID ...
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "llama.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: logits_oracle MODEL.gguf OUT.bin ID ID ...\n");
    return 1;
  }
  std::vector<llama_token> ids;
  for (int i = 3; i < argc; ++i) ids.push_back(static_cast<llama_token>(std::atoi(argv[i])));
  const int n = static_cast<int>(ids.size());
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
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
  llama_context* ctx = llama_init_from_model(model, cp);
  if (!ctx) return 3;
  llama_batch b = llama_batch_init(n, 0, 1);
  for (int i = 0; i < n; ++i) {
    b.token[i] = ids[static_cast<std::size_t>(i)];
    b.pos[i] = i;
    b.n_seq_id[i] = 1;
    b.seq_id[i][0] = 0;
    b.logits[i] = 1;
  }
  b.n_tokens = n;
  if (llama_decode(ctx, b) != 0) return 4;
  const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
  FILE* f = std::fopen(argv[2], "wb");
  if (!f) return 5;
  const int hdr[2] = {n, nv};
  std::fwrite(hdr, sizeof hdr, 1, f);
  for (int i = 0; i < n; ++i) std::fwrite(llama_get_logits_ith(ctx, i), sizeof(float), static_cast<std::size_t>(nv), f);
  std::fclose(f);
  llama_batch_free(b);
  llama_free(ctx);
  llama_model_free(model);
  return 0;
}
