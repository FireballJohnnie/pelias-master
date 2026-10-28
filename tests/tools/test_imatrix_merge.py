#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""Unit test of tools/imatrix_merge.py on two synthetic GGUF importance matrices: rotated entries come from the rotated
file, every other entry from the other file, byte for byte; refusals (overwrite, rotated other file, missing rotated
entry). Exit 0 on success."""
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "..", "tools", "imatrix_merge.py")


def ws(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write(path, entries, rot=None, chunks=10, size=512, ds="d.txt"):
    kvs = [ws("general.type") + struct.pack("<I", 8) + ws("imatrix"),
           ws("imatrix.datasets") + struct.pack("<IIQ", 9, 8, 1) + ws(ds),
           ws("imatrix.chunk_count") + struct.pack("<II", 4, chunks), ws("imatrix.chunk_size") + struct.pack("<II", 4, size)]
    if rot is not None:
        kvs.append(ws("pelias.rot32.tensors") + struct.pack("<IIQ", 9, 8, len(rot)) + b"".join(ws(r) for r in rot))
    infos, data, off = [], b"", 0
    for name, vals in entries:
        for suffix, v in ((".in_sum2", vals), (".counts", [4.0])):
            infos.append(ws(name + suffix) + struct.pack("<IQ", 1, len(v)) + struct.pack("<IQ", 0, off))
            blob = struct.pack("<%df" % len(v), *v)
            blob += b"\0" * ((32 - len(blob) % 32) % 32)
            data += blob
            off += len(blob)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(kvs)) + b"".join(kvs) + b"".join(infos)
    head += b"\0" * ((32 - len(head) % 32) % 32)
    with open(path, "wb") as f:
        f.write(head + data)


def read_sum2(path):
    sys.path.insert(0, os.path.dirname(TOOL))
    import imatrix_merge
    g = imatrix_merge.Gguf(path)
    e = g.entries()
    return {n: g.read(p[".in_sum2"][2], p[".in_sum2"][3]) for n, p in e.items()}, g.strings("pelias.rot32.tensors")


def run(*a):
    return subprocess.run([sys.executable, TOOL] + list(a), capture_output=True, text=True)


def main():
    fails = 0
    with tempfile.TemporaryDirectory() as d:
        ours, theirs, out = os.path.join(d, "ours.gguf"), os.path.join(d, "theirs.gguf"), os.path.join(d, "m.gguf")
        write(ours, [("blk.0.attn_qkv.weight", [1.0] * 8), ("blk.0.ffn_up_exps.weight", [2.0] * 8)], rot=["blk.0.attn_qkv.weight"], ds="calib.txt")
        write(theirs, [("blk.0.attn_qkv.weight", [3.0] * 8), ("blk.0.ffn_up_exps.weight", [5.0] * 8)], chunks=76, size=11008)
        r = run("--rotated", ours, "--other", theirs, "--out", out)
        if r.returncode != 0:
            print("FAIL merge:", r.stderr); return 1
        sums, rot = read_sum2(out)
        if struct.unpack("<f", sums["blk.0.attn_qkv.weight"][:4])[0] != 1.0: print("FAIL: rotated entry not from ours"); fails += 1
        if struct.unpack("<f", sums["blk.0.ffn_up_exps.weight"][:4])[0] != 5.0: print("FAIL: other entry not from theirs"); fails += 1
        if rot != ["blk.0.attn_qkv.weight"]: print("FAIL: rot32 list", rot); fails += 1
        if run("--rotated", ours, "--other", theirs, "--out", out).returncode == 0: print("FAIL: overwrote"); fails += 1
        if run("--rotated", ours, "--other", ours, "--out", out + "2").returncode == 0: print("FAIL: accepted a rotated other file"); fails += 1
        bad = os.path.join(d, "bad.gguf")
        write(bad, [("blk.0.ffn_up_exps.weight", [2.0] * 8)], rot=["blk.0.attn_qkv.weight"])
        if run("--rotated", bad, "--other", theirs, "--out", out + "3").returncode == 0: print("FAIL: accepted a missing rotated entry"); fails += 1
    print("imatrix_merge: %s" % ("ok" if fails == 0 else "%d failures" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
