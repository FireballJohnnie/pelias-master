# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# Test-only (BitNet b1.58 gate): the Hugging Face transformers implementation of BitNetForCausalLM (the model's own
# reference) run in fp32 on the CPU, with each BitLinear weight replaced so that the model's own weight quantizer gives
# exactly the I2_S file's ternary values and scale (w = t * s / mean|t|: mean|w| = s, round(w / s) = t), so the two engines
# see the same weights and differ only in their arithmetic. Writes every position's logits (int32 n, int32 n_vocab,
# n x n_vocab fp32), the format of `pelias logits`.
#   python hf_bitnet_logits.py BF16_FOLDER MODEL-I2_S.gguf OUT.bin ID ID ID ...
import struct
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM

hf_dir, gguf_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [int(x) for x in sys.argv[4:]]

def read_gguf(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    _, n_t, n_kv = struct.unpack("<IQQ", f.read(20))
    def rstr():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n)
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
    for _ in range(n_kv):
        rstr()
        (t,) = struct.unpack("<I", f.read(4))
        skip(t)
    tens = {}
    for _ in range(n_t):
        name = rstr().decode()
        (nd,) = struct.unpack("<I", f.read(4))
        ne = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        ty, off = struct.unpack("<IQ", f.read(12))
        tens[name] = (ne, ty, off)
    data0 = (f.tell() + 31) // 32 * 32
    return f, tens, data0

f, tens, data0 = read_gguf(gguf_path)
lut = np.array([-1, 0, 1, 1], np.float32)  # code 3 does not occur (MEASURED); read as code - 1 like Pelias
def ternary(name):
    ne, ty, off = tens[name]
    assert ty == 36
    n = int(np.prod(ne))
    f.seek(data0 + off)
    raw = np.frombuffer(f.read(n // 4 + 4), dtype=np.uint8)
    blocks = raw[: n // 4].reshape(-1, 32)
    codes = np.empty((blocks.shape[0], 128), dtype=np.uint8)
    for g, sh in enumerate((6, 4, 2, 0)):
        codes[:, 32 * g : 32 * g + 32] = (blocks >> sh) & 3
    t = (codes.reshape(-1).astype(np.int16) - 1).astype(np.float32)
    scale = float(np.frombuffer(raw[n // 4 : n // 4 + 4].tobytes(), dtype=np.float32)[0])
    return t.reshape(ne[1], ne[0]), scale

model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32)
model.eval()
names = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output",
         "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}
replaced = 0
with torch.no_grad():
    for name, mod in model.named_modules():
        last = name.split(".")[-1]
        if last in names and hasattr(mod, "weight"):
            il = int(name.split(".")[2])
            t, s = ternary(f"blk.{il}.{names[last]}.weight")
            mt = float(np.mean(np.abs(t)))
            w = torch.from_numpy(t * (s / mt))
            assert tuple(mod.weight.shape) == tuple(w.shape), (name, mod.weight.shape, w.shape)
            mod.weight.copy_(w)
            replaced += 1
    logits = model(torch.tensor([ids])).logits[0].float().numpy()
with open(out, "wb") as fo:
    fo.write(struct.pack("<ii", logits.shape[0], logits.shape[1]))
    fo.write(logits.astype(np.float32).tobytes())
print(f"{replaced} BitLinear weights replaced; {logits.shape[0]} positions x {logits.shape[1]} logits written to {out}")
