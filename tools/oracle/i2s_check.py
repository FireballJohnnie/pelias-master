# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# Test-only (BitNet b1.58, pre-build list item 5): settles how an I2_S tensor (type 36, Microsoft's BitNet llama.cpp fork)
# stores its ternary weights, against the BF16 original (microsoft/bitnet-b1.58-2B-4T-bf16, safetensors). The original
# is quantized as BitNet's own weight quantizer does it (scale = mean |w| over the tensor, q = round(clip(w / scale, -1,
# 1))) and compared value for value with the I2_S decode under each code mapping and each scale reading; q/k rows are also
# checked under the llama-style row permutation a converter may apply. Nothing is written.
#   python3 i2s_check.py MODEL-I2_S.gguf BF16_FOLDER
import json
import struct
import sys
import numpy as np

gguf_path, hf_dir = sys.argv[1], sys.argv[2]

# ---- a minimal GGUF reader (header, metadata skipped by type, tensor infos)
def read_gguf(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    ver, n_t, n_kv = struct.unpack("<IQQ", f.read(20))
    def rstr():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    def skip(t):
        if t == 8:
            rstr()
        elif t == 9:
            at, n = struct.unpack("<IQ", f.read(12))
            for _ in range(n):
                skip(at)
        else:
            f.read(sizes[t])
    kv = {}
    for _ in range(n_kv):
        k = rstr()
        (t,) = struct.unpack("<I", f.read(4))
        if t in (4, 5, 6) and k.startswith(("general.", "bitnet")):
            kv[k] = struct.unpack({4: "<I", 5: "<i", 6: "<f"}[t], f.read(4))[0]
        elif t == 8 and k.startswith("general."):
            kv[k] = rstr()
        else:
            skip(t)
    tens = {}
    for _ in range(n_t):
        name = rstr()
        (nd,) = struct.unpack("<I", f.read(4))
        ne = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        ty, off = struct.unpack("<IQ", f.read(12))
        tens[name] = (ne, ty, off)
    align = 32
    pos = f.tell()
    data0 = (pos + align - 1) // align * align
    return f, kv, tens, data0

f, kv, tens, data0 = read_gguf(gguf_path)
print("architecture", kv.get("general.architecture"), "tensors", len(tens))

# ---- safetensors (several shards possible)
import glob, os
shards = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
index = {}
for sp in shards:
    with open(sp, "rb") as sf:
        (hl,) = struct.unpack("<Q", sf.read(8))
        hdr = json.loads(sf.read(hl))
    for k, v in hdr.items():
        if k != "__metadata__":
            index[k] = (sp, 8 + hl, v)

def hf_tensor(name):
    sp, base, v = index[name]
    a, b = v["data_offsets"]
    with open(sp, "rb") as sf:
        sf.seek(base + a)
        raw = np.frombuffer(sf.read(b - a), dtype=np.uint16)
    assert v["dtype"] == "BF16", v["dtype"]
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(v["shape"])

def i2s_codes(name):
    ne, ty, off = tens[name]
    assert ty == 36, (name, ty)
    n = int(np.prod(ne))
    f.seek(data0 + off)
    raw = np.frombuffer(f.read(n // 4 + 32), dtype=np.uint8)
    packed, tail = raw[: n // 4], raw[n // 4 :]
    blocks = packed.reshape(-1, 32)  # 128 values per 32 bytes
    codes = np.empty((blocks.shape[0], 128), dtype=np.uint8)
    for g, sh in enumerate((6, 4, 2, 0)):  # byte gp holds values gp, 32+gp, 64+gp, 96+gp at shifts 6, 4, 2, 0
        codes[:, 32 * g : 32 * g + 32] = (blocks >> sh) & 3
    return codes.reshape(-1), tail, ne

def permute_llama(w, n_head):  # the converter's q/k row permutation for adjacent-pair rope (llama.cpp convert LlamaModel.permute)
    out, inp = w.shape
    return w.reshape(n_head, 2, out // n_head // 2, inp).swapaxes(1, 2).reshape(out, inp)

maps = {"00=-1,01=0,10=+1": np.array([-1, 0, 1, 0], np.int8), "00=0,01=+1,10=-1": np.array([0, 1, -1, 0], np.int8)}
pairs = [("blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight", None),
         ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight", 20),
         ("blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight", 5),
         ("blk.0.ffn_down.weight", "model.layers.0.mlp.down_proj.weight", None),
         ("blk.17.ffn_gate.weight", "model.layers.17.mlp.gate_proj.weight", None)]
for gname, hname, heads in pairs:
    if gname not in tens or hname not in index:
        print("missing", gname if gname not in tens else hname)
        continue
    codes, tail, ne = i2s_codes(gname)
    w = hf_tensor(hname)
    scale = float(np.mean(np.abs(w)))
    tq = np.clip(np.rint(w / max(scale, 1e-5)), -1, 1).astype(np.int8)
    tail_f = np.frombuffer(tail.tobytes(), dtype=np.float32)
    print(f"{gname}: ne {ne}, codes used {np.bincount(codes, minlength=4).tolist()}, tail floats {tail_f[:8].tolist()}, mean|w| {scale:.6g}, 1/mean|w| {1/scale:.6g}")
    cands = {"as stored": tq.reshape(-1)}
    if heads:
        cands["llama-permuted"] = permute_llama(tq.astype(np.float32), heads).astype(np.int8).reshape(-1)
    for mname, m in maps.items():
        dec = m[codes]
        for cname, ref in cands.items():
            print(f"   mapping {mname:18s} rows {cname:15s}: agree {np.mean(dec == ref) * 100:.4f} %")
