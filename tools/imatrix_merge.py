#!/usr/bin/env python3
# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# provenance: original (GGUF layout per the GGUF v3 specification; importance-matrix keys as llama.cpp's imatrix writes them)
"""Merge two GGUF importance matrices for a file whose dense weights are stored rotated (game plan: urm).

    python3 tools/imatrix_merge.py --rotated OURS.gguf --other THEIRS.gguf --out MERGED.gguf

--rotated is a Pelias importance matrix with pelias.rot32.tensors (entries collected in the rotated basis): every
listed entry is taken from it. Every other entry is taken from --other (collected unrotated), which must not list any
rotated tensors itself. The data of each entry (in_sum2, counts and in_sum when present) is copied byte for byte with
its own shape. Metadata: imatrix.datasets is the union (rotated first), imatrix.chunk_count and chunk_size are the
--other file's (most entries come from it), pelias.rot32.* is carried, and pelias.imatrix.merge records both sources.
Refuses to overwrite. Standard library only.
"""
import os
import struct
import sys

ALIGN = 32


class Gguf:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                die(path + ": not a GGUF file")
            ver, nt, nkv = self.rd(f, "IQQ")
            if ver != 3:
                die("%s: GGUF version %d (3 expected)" % (path, ver))
            self.kv = {}
            for _ in range(nkv):
                k = self.rs(f)
                t, = self.rd(f, "I")
                self.kv[k] = (t, self.val(f, t))
            align = self.kv.get("general.alignment", (4, ALIGN))[1]
            self.tensors = []
            for _ in range(nt):
                n = self.rs(f)
                nd, = self.rd(f, "I")
                ne = self.rd(f, "Q" * nd)
                ty, off = self.rd(f, "IQ")
                if ty != 0:
                    die("%s: %s is not F32" % (path, n))
                count = 1
                for x in ne:
                    count *= x
                self.tensors.append((n, ne, off, count * 4))
            self.data0 = (f.tell() + align - 1) // align * align

    @staticmethod
    def rd(f, fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

    def rs(self, f):
        n, = self.rd(f, "Q")
        return f.read(n).decode("utf-8")

    def val(self, f, t):
        simple = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
        if t in simple:
            return self.rd(f, simple[t])[0]
        if t == 8:
            return self.rs(f)
        if t == 9:
            at, n = self.rd(f, "IQ")
            return (at, [self.val(f, at) for _ in range(n)])
        die("%s: unknown metadata type %d" % (self.path, t))

    def entries(self):
        e = {}
        for n, ne, off, nb in self.tensors:
            for suffix in (".in_sum2", ".counts", ".in_sum"):
                if n.endswith(suffix):
                    e.setdefault(n[: -len(suffix)], {})[suffix] = (n, ne, off, nb)
        for name, parts in e.items():
            if ".in_sum2" not in parts or ".counts" not in parts:
                die("%s: entry %s lacks in_sum2 or counts" % (self.path, name))
        return e

    def strings(self, key):
        t, v = self.kv.get(key, (None, None))
        if t is None:
            return None
        if t != 9 or v[0] != 8:
            die("%s: %s is not a list of strings" % (self.path, key))
        return v[1]

    def read(self, off, nb):
        with open(self.path, "rb") as f:
            f.seek(self.data0 + off)
            b = f.read(nb)
        if len(b) != nb:
            die(self.path + ": truncated tensor data")
        return b


def die(msg):
    print("imatrix_merge: " + msg, file=sys.stderr)
    sys.exit(2)


def ws(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(k, v):
    return ws(k) + struct.pack("<I", 8) + ws(v)


def kv_u32(k, v):
    return ws(k) + struct.pack("<II", 4, v)


def kv_strings(k, vs):
    return ws(k) + struct.pack("<IIQ", 9, 8, len(vs)) + b"".join(ws(v) for v in vs)


def main(argv):
    args = {}
    i = 1
    while i < len(argv):
        if argv[i] in ("--rotated", "--other", "--out") and i + 1 < len(argv):
            args[argv[i]] = argv[i + 1]
            i += 2
        else:
            die("usage: imatrix_merge.py --rotated OURS.gguf --other THEIRS.gguf --out MERGED.gguf")
    if len(args) != 3:
        die("usage: imatrix_merge.py --rotated OURS.gguf --other THEIRS.gguf --out MERGED.gguf")
    out = args["--out"]
    if os.path.exists(out):
        die(out + " exists (nothing is overwritten)")
    a, b = Gguf(args["--rotated"]), Gguf(args["--other"])
    rot = a.strings("pelias.rot32.tensors")
    if not rot:
        die(a.path + " lists no rotated tensors (pelias.rot32.tensors)")
    if b.strings("pelias.rot32.tensors"):
        die(b.path + " lists rotated tensors; the other matrix must be collected unrotated")
    ea, eb = a.entries(), b.entries()
    missing = [n for n in rot if n not in ea]
    if missing:
        die("rotated tensors without an entry in %s: %s" % (a.path, ", ".join(missing[:5])))
    rotset = set(rot)
    pick = {}
    for n in sorted(set(ea) | set(eb)):
        if n in rotset:
            pick[n] = (a, ea[n])
        elif n in eb:
            pick[n] = (b, eb[n])
    only_a = sorted(n for n in ea if n not in rotset and n not in eb)
    for k in ("imatrix.chunk_count", "imatrix.chunk_size"):
        if k not in b.kv:
            die("%s lacks %s" % (b.path, k))
    datasets = (a.strings("imatrix.datasets") or []) + (b.strings("imatrix.datasets") or [])
    note = "rotated entries (%d) from %s (chunk_count %d, chunk_size %d); other entries (%d) from %s (chunk_count %d, chunk_size %d)" % (
        len(rot), os.path.basename(a.path), a.kv["imatrix.chunk_count"][1], a.kv["imatrix.chunk_size"][1],
        len(pick) - len(rot), os.path.basename(b.path), b.kv["imatrix.chunk_count"][1], b.kv["imatrix.chunk_size"][1])
    kvs = [kv_string("general.type", "imatrix"), kv_strings("imatrix.datasets", datasets),
           kv_u32("imatrix.chunk_count", b.kv["imatrix.chunk_count"][1]), kv_u32("imatrix.chunk_size", b.kv["imatrix.chunk_size"][1]),
           kv_string("pelias.rot32.kind", a.kv.get("pelias.rot32.kind", (8, "walsh-hadamard-32-orthonormal-row-blocks"))[1]),
           kv_strings("pelias.rot32.tensors", rot), kv_string("pelias.imatrix.merge", note)]
    infos, blobs, off = [], [], 0
    for n in sorted(pick):
        src, parts = pick[n]
        for suffix in (".in_sum2", ".counts", ".in_sum"):
            if suffix not in parts:
                continue
            tname, ne, toff, nb = parts[suffix]
            infos.append(ws(tname) + struct.pack("<I", len(ne)) + struct.pack("<" + "Q" * len(ne), *ne) + struct.pack("<IQ", 0, off))
            blobs.append((src, toff, nb))
            off += (nb + ALIGN - 1) // ALIGN * ALIGN
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(kvs)) + b"".join(kvs) + b"".join(infos)
    head += b"\0" * ((ALIGN - len(head) % ALIGN) % ALIGN)
    with open(out + ".partial", "xb") as f:
        f.write(head)
        for src, toff, nb in blobs:
            f.write(src.read(toff, nb))
            f.write(b"\0" * ((ALIGN - nb % ALIGN) % ALIGN))
    os.rename(out + ".partial", out)
    print("merged: %d entries (%d rotated from %s, %d from %s); %d unrotated entries of the rotated file not used%s -> %s" % (
        len(pick), len(rot), os.path.basename(a.path), len(pick) - len(rot), os.path.basename(b.path), len(only_a) + sum(
            1 for n in ea if n not in rotset and n in eb), "" if not only_a else " (%d had no counterpart)" % len(only_a), out))


if __name__ == "__main__":
    main(sys.argv)
