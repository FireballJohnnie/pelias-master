#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# tg2_tags.py - read-only count of the Q4_TG2 grid tags in a GGUF (TG2 probe, 2026-10-30 22:2x, author 22:17).
#   python3 tg2_tags.py FILE.gguf OUT.json
# Q4_TG2 (type 204, 144 bytes per 256): fp16 d, six bytes of eight 6-bit levels, eight tag bytes (byte j = sub-block j of
# 32 values; bits 2g..2g+1 = the grid tag of values 8g..8g+7), 128 bytes of 4-bit codes. Grids (v2): 0 = (i - 7) / 8,
# 1 = 1.3 log grid with a 0, 2 = 1.45 log grid, 3 = 0.6 x grid 0. Tags 0 and 3 are integer codes (i - 7) times a constant;
# 1 and 2 are not. Counts per tensor kind: groups of 8 per tag, and sub-blocks of 32 whose four tags are all 0, all 3,
# all in {0, 3} (integer-only), or any log tag. The file is opened read only; nothing is written except OUT.json (new
# name; the script refuses an existing one).
import json
import os
import struct
import sys
import time

import numpy as np

Q4_TG2 = 204


def rd(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack("<" + fmt, f.read(n))


def rd_str(f):
    (n,) = rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def rd_val(f, t):
    if t in SCALAR:
        return rd(f, SCALAR[t])[0]
    if t == 8:
        return rd_str(f)
    if t == 9:
        (it,) = rd(f, "I")
        (n,) = rd(f, "Q")
        if it in SCALAR:
            sz = struct.calcsize(SCALAR[it])
            f.seek(n * sz, os.SEEK_CUR)
            return None
        for _ in range(n):
            rd_val(f, it)
        return None
    raise ValueError("unknown GGUF value type %d" % t)


def kind(name):
    for k in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"):
        if k in name:
            return k
    return "other"


def main():
    if len(sys.argv) != 3:
        print("usage: tg2_tags.py FILE.gguf OUT.json")
        return 1
    path, out = sys.argv[1], sys.argv[2]
    if os.path.exists(out):
        print("refusing: %s exists" % out)
        return 1
    t0 = time.time()
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            print("not a GGUF file")
            return 1
        (ver,) = rd(f, "I")
        n_t, n_kv = rd(f, "QQ")
        align = 32
        for _ in range(n_kv):
            k = rd_str(f)
            (t,) = rd(f, "I")
            v = rd_val(f, t)
            if k == "general.alignment" and v:
                align = int(v)
        infos = []
        for _ in range(n_t):
            name = rd_str(f)
            (nd,) = rd(f, "I")
            dims = rd(f, "Q" * nd)
            (typ,) = rd(f, "I")
            (off,) = rd(f, "Q")
            infos.append((name, dims, typ, off))
        base = (f.tell() + align - 1) // align * align
    tg = [(n, d, o) for (n, d, t, o) in infos if t == Q4_TG2]
    tot = {}
    nbytes = 0
    with open(path, "rb") as f:
        for name, dims, off in tg:
            ne = 1
            for d in dims:
                ne *= d
            nb = ne // 256
            acc = tot.setdefault(kind(name), {"tensors": 0, "blocks": 0, "groups_per_tag": [0, 0, 0, 0], "subblocks": 0,
                                              "sub_all_tag0": 0, "sub_all_tag3": 0, "sub_int_only": 0, "sub_any_log": 0})
            acc["tensors"] += 1
            acc["blocks"] += nb
            f.seek(base + off)
            done = 0
            while done < nb:
                m = min(nb - done, 1 << 18)
                raw = np.frombuffer(f.read(m * 144), dtype=np.uint8).reshape(m, 144)
                tb = raw[:, 8:16].reshape(-1)  # one byte per sub-block of 32
                tags = np.stack([(tb >> (2 * g)) & 3 for g in range(4)], axis=1)  # (sub-blocks, 4)
                acc["groups_per_tag"] = [a + int(c) for a, c in zip(acc["groups_per_tag"], np.bincount(tags.reshape(-1), minlength=4))]
                acc["subblocks"] += int(tags.shape[0])
                acc["sub_all_tag0"] += int(np.count_nonzero((tags == 0).all(axis=1)))
                acc["sub_all_tag3"] += int(np.count_nonzero((tags == 3).all(axis=1)))
                isint = (tags == 0) | (tags == 3)
                acc["sub_int_only"] += int(np.count_nonzero(isint.all(axis=1)))
                acc["sub_any_log"] += int(np.count_nonzero(~isint.all(axis=1)))
                done += m
                nbytes += m * 144
    for k, a in tot.items():
        g = sum(a["groups_per_tag"])
        a["group_share_per_tag"] = [c / g if g else 0.0 for c in a["groups_per_tag"]]
        s = a["subblocks"]
        for key in ("sub_all_tag0", "sub_all_tag3", "sub_int_only", "sub_any_log"):
            a[key + "_share"] = a[key] / s if s else 0.0
    res = {"probe": "TG2-TAGS", "label": "MEASURED", "file_bytes": os.path.getsize(path), "gguf_version": ver,
           "q4_tg2_tensors": len(tg), "bytes_read": nbytes, "seconds": time.time() - t0, "by_kind": tot}
    with open(out, "x") as o:
        json.dump(res, o, indent=1)
    for k, a in sorted(tot.items()):
        print("%-16s %5d tensors %10d blocks: groups per tag %s; sub-blocks all-0 %.4f all-3 %.4f integer-only %.4f any-log %.4f"
              % (k, a["tensors"], a["blocks"], " ".join("%.4f" % x for x in a["group_share_per_tag"]), a["sub_all_tag0_share"],
                 a["sub_all_tag3_share"], a["sub_int_only_share"], a["sub_any_log_share"]))
    print("read %.2f GB in %.1f s" % (nbytes / 1e9, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
