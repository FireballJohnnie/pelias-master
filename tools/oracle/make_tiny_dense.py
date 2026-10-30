# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# Test-only (GP3-09 gate): a tiny dense decoder (qwen3 / qwen2 / llama) with small deterministic random F32 weights and the
# real vocabulary of llama.cpp's vocab-only test file for that family (models/ggml-vocab-<pre>.gguf), written with the
# gguf-py package that ships with llama.cpp (its writer only; the weights and shapes are Pelias's own). Both engines run
# the same file: Pelias's reference and fast paths against the reference library's logits (logits_oracle.cpp).
#   PYTHONPATH=<llama.cpp>/gguf-py python3 make_tiny_dense.py FAMILY VOCAB.gguf OUT.gguf [--rope-freqs]
import sys
import numpy as np
import gguf

fam, vocab_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
rope_freqs = "--rope-freqs" in sys.argv
assert fam in ("qwen3", "qwen2", "llama")
E, H, HKV, HD, FF, L = 128, 4, 2, 32, 192, 3  # head_dim 32 = E / H, so llama-style models need no key_length

vr = gguf.GGUFReader(vocab_path)
def field(name):
    f = vr.fields.get(name)
    return f
tokens = [bytes(field("tokenizer.ggml.tokens").parts[i]).decode("utf-8") for i in field("tokenizer.ggml.tokens").data]
types = [int(field("tokenizer.ggml.token_type").parts[i][0]) for i in field("tokenizer.ggml.token_type").data]
merges = [bytes(field("tokenizer.ggml.merges").parts[i]).decode("utf-8") for i in field("tokenizer.ggml.merges").data]
pre = bytes(field("tokenizer.ggml.pre").parts[-1]).decode("utf-8")
V = len(tokens)

w = gguf.GGUFWriter(out, fam)
w.add_block_count(L)
w.add_context_length(4096)
w.add_embedding_length(E)
w.add_feed_forward_length(FF)
w.add_head_count(H)
w.add_head_count_kv(HKV)
w.add_rope_freq_base(500000.0 if fam == "llama" else 1000000.0)
w.add_layer_norm_rms_eps(1e-6)
if fam == "qwen3":
    w.add_key_length(HD)
    w.add_value_length(HD)
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre(pre)
w.add_token_list(tokens)
w.add_token_types(types)
w.add_token_merges(merges)
w.add_eos_token_id(V - 1)
w.add_add_bos_token(False)

rng = np.random.default_rng(20261030)
def mat(rows, cols, scale=None):  # gguf shape [cols(ne0), rows(ne1)] -> numpy (rows, cols)
    s = scale if scale is not None else 1.0 / np.sqrt(cols)
    return (rng.uniform(-1.0, 1.0, size=(rows, cols)) * s).astype(np.float32)
def vec(n, base=1.0, spread=0.1):
    return (base + spread * rng.uniform(-1.0, 1.0, size=n)).astype(np.float32)

w.add_tensor("token_embd.weight", mat(V, E, 0.5))
w.add_tensor("output_norm.weight", vec(E))
if fam != "qwen3":  # qwen3 ties the output layer to the embedding (no output.weight), as Qwen3-1.7B does
    w.add_tensor("output.weight", mat(V, E))
if rope_freqs:
    w.add_tensor("rope_freqs.weight", (1.0 + 7.0 * rng.uniform(0.0, 1.0, size=HD // 2)).astype(np.float32))
for il in range(L):
    p = f"blk.{il}."
    w.add_tensor(p + "attn_norm.weight", vec(E))
    w.add_tensor(p + "attn_q.weight", mat(H * HD, E))
    w.add_tensor(p + "attn_k.weight", mat(HKV * HD, E))
    w.add_tensor(p + "attn_v.weight", mat(HKV * HD, E))
    w.add_tensor(p + "attn_output.weight", mat(E, H * HD))
    if fam == "qwen3":
        w.add_tensor(p + "attn_q_norm.weight", vec(HD))
        w.add_tensor(p + "attn_k_norm.weight", vec(HD))
    if fam == "qwen2":
        w.add_tensor(p + "attn_q.bias", vec(H * HD, 0.0, 0.3))
        w.add_tensor(p + "attn_k.bias", vec(HKV * HD, 0.0, 0.3))
        w.add_tensor(p + "attn_v.bias", vec(HKV * HD, 0.0, 0.3))
    w.add_tensor(p + "ffn_norm.weight", vec(E))
    w.add_tensor(p + "ffn_gate.weight", mat(FF, E))
    w.add_tensor(p + "ffn_up.weight", mat(FF, E))
    w.add_tensor(p + "ffn_down.weight", mat(E, FF))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"{out}: {fam}, pre {pre}, vocab {V}, {L} layers, n_embd {E}")
