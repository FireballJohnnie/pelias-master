# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# Tokenizer parity (test-only): llama.cpp ships vocab-only GGUFs with test strings (models/ggml-vocab-<pre>.gguf.inp,
# cases split by "\n__ggml_vocab_test__\n") and the reference ids (.out, one case per line). Runs `pelias tokenize
# --no-nfc --no-special` on every case and counts mismatches.
#   python3 tools/oracle/vocab_check.py PELIAS VOCAB.gguf TMPDIR
import subprocess, sys, json
pel, vocab = sys.argv[1], sys.argv[2]
inp = open(vocab + ".inp", encoding="utf-8").read().split("\n__ggml_vocab_test__\n")
out = open(vocab + ".out").read().split("\n")
ok = bad = 0
for i, (t, o) in enumerate(zip(inp, out)):
    want = [int(x) for x in o.split()]
    f = f"{sys.argv[3]}/case.txt"
    open(f, "w", encoding="utf-8", newline="").write(t)
    r = subprocess.run([pel, "tokenize", "--model", vocab, "--file", f, "--no-nfc", "--no-special"], capture_output=True, text=True)
    got = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else None
    if got == want: ok += 1
    else:
        bad += 1
        print("MISMATCH", i, repr(t[:60]), want[:12], got[:12] if got else r.stderr[:200])
print(vocab.split("/")[-1], "cases", ok + bad, "match", ok, "mismatch", bad)
