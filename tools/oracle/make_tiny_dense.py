# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# Test-only (GP3-09 gate): a tiny dense decoder (qwen3 / qwen2 / llama) with small deterministic random F32 weights and the
# real vocabulary of llama.cpp's vocab-only test file for that family (models/ggml-vocab-<pre>.gguf), written with the
# gguf-py package that ships with llama.cpp (its writer only; the weights and shapes are Pelias's own). Both engines run
# the same file: Pelias's reference and fast paths against the reference library's logits (logits_oracle.cpp).
#   PYTHONPATH=<llama.cpp>/gguf-py python3 make_tiny_dense.py FAMILY VOCAB.gguf OUT.gguf [--rope-freqs] [--type NVFP4|Q8_0 [--deq]] [--scales]
# --type: the projection weights stored in that type: Q8_0 by gguf-py's own quantizer; NVFP4 (which gguf-py does not
# quantize) by a plain encoder here: per 16 values the UE4M3 scale nearest amax / 6 (doubled grid: x 2), codes the nearest
# E2M1 value; any valid block will do, the gate is the two engines reading the same bytes.
import sys
import numpy as np
import gguf

fam, vocab_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
rope_freqs = "--rope-freqs" in sys.argv
qtype = None
if "--type" in sys.argv:
    qtype = getattr(gguf.GGMLQuantizationType, sys.argv[sys.argv.index("--type") + 1])
assert fam in ("qwen3", "qwen2", "llama")
E, H, HKV, HD, FF, L = 256, 4, 2, 64, 512, 3  # every row a whole number of 256; head_dim = E / H (llama-style needs no key_length)

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
UE4M3 = []
for x in range(256):  # the decode of every UE4M3 byte, halved (as llama.cpp's ggml_ue4m3_to_fp32)
    if x in (0, 0x7F) or x > 0x7F:
        UE4M3.append(0.0 if x in (0, 0x7F) else -1.0)
        continue
    e, m = (x >> 3) & 0xF, x & 7
    UE4M3.append((m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)) * 0.5)
UE4M3 = np.array(UE4M3)
GRID = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float64)
def nvfp4(arr):
    rows, cols = arr.shape
    out = np.zeros((rows, cols // 64 * 36), np.uint8)
    for r in range(rows):
        for b in range(cols // 64):
            blk = arr[r, 64 * b : 64 * b + 64].astype(np.float64)
            for s in range(4):
                v = blk[16 * s : 16 * s + 16]
                want = max(float(np.abs(v).max()) / 12.0, 1e-12)  # scale for the doubled grid (max 12)
                ok = np.where(UE4M3 > 0)[0]
                sb = int(ok[np.argmin(np.abs(UE4M3[ok] - want))])
                d = UE4M3[sb]
                codes = np.array([int(np.argmin(np.abs(GRID * d - x))) for x in v], np.uint8)
                out[r, 36 * b + s] = sb
                out[r, 36 * b + 4 + 8 * s : 36 * b + 12 + 8 * s] = codes[:8] | (codes[8:] << 4)
    return out
deq = "--deq" in sys.argv  # store the reference library's own dequantization (gguf-py) of the --type bytes, as F32
scales = "--scales" in sys.argv  # a <name>.scale F32 [1] tensor per projection (as converted NVFP4 files carry)
def proj(name, arr):
    if scales:
        w.add_tensor(name[: -len(".weight")] + ".scale", np.array([0.25 + 1.5 * rng.uniform()], np.float32))
    if qtype is not None and deq:
        q = nvfp4(arr) if qtype == gguf.GGMLQuantizationType.NVFP4 else gguf.quants.quantize(arr, qtype)
        w.add_tensor(name, gguf.quants.dequantize(q, qtype).astype(np.float32).reshape(arr.shape))
        return
    if qtype is None:
        w.add_tensor(name, arr)
    elif qtype == gguf.GGMLQuantizationType.NVFP4:
        w.add_tensor(name, nvfp4(arr), raw_dtype=qtype)
    else:
        w.add_tensor(name, gguf.quants.quantize(arr, qtype), raw_dtype=qtype)
for il in range(L):
    p = f"blk.{il}."
    w.add_tensor(p + "attn_norm.weight", vec(E))
    proj(p + "attn_q.weight", mat(H * HD, E))
    proj(p + "attn_k.weight", mat(HKV * HD, E))
    proj(p + "attn_v.weight", mat(HKV * HD, E))
    proj(p + "attn_output.weight", mat(E, H * HD))
    if fam == "qwen3":
        w.add_tensor(p + "attn_q_norm.weight", vec(HD))
        w.add_tensor(p + "attn_k_norm.weight", vec(HD))
    if fam == "qwen2":
        w.add_tensor(p + "attn_q.bias", vec(H * HD, 0.0, 0.3))
        w.add_tensor(p + "attn_k.bias", vec(HKV * HD, 0.0, 0.3))
        w.add_tensor(p + "attn_v.bias", vec(HKV * HD, 0.0, 0.3))
    w.add_tensor(p + "ffn_norm.weight", vec(E))
    proj(p + "ffn_gate.weight", mat(FF, E))
    proj(p + "ffn_up.weight", mat(FF, E))
    proj(p + "ffn_down.weight", mat(E, FF))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"{out}: {fam}, pre {pre}, vocab {V}, {L} layers, n_embd {E}")
