# Game plan runs on the reference machine (2026-10-27 to 10-28)

Raw output of the game plan blocks run on the reference machine (RTX 4070, WSL2): logs, the per-token KL dumps (`.kld`, format PELKLD01, `src/verify/kl.cpp`) that every paired result is re-computed from, recipes, and the TALLY / RUNBOOK lines each block wrote. The reading of each result is in `docs/PELIAS_GAME_PLAN.md` and `logs/BUILD_LOG.md`; proven methods are in `docs/PELIAS_NEW_METHODS.md`. Every number is MEASURED.

| Folder | Block | What |
|---|---|---|
| `GP1-01` | A | Stock Q4_K_M through Pelias, fp32 reference (`stock.kld`, per-expert statistics) |
| `GP1-02` | B | llama.cpp with the GPU path off |
| `GP1-03` | B | 8-bit activation rounding: all CPU-style (a), all GPU-style (b), experts only (c), dense only (d) |
| `J` | J | Rotation: R1 (dense inputs rotated), R2 (dense weights rotated, fp32 and q8_1), R2b (experts rotated), the unrotated control |
| `K` | K | Routing records: flips stock vs bf16, hot-set statistics at 84 / 120 / 161 experts per layer (the 7 MB `.rt` route files stay on the author's drive) |
| `L` | L | Recipe check, rotated importance matrix, Unsloth non-MTP vs t8r vs t9r, the MTP download log, recipes (t8, t9, mixes A/B) |
| `N` | N | Unsloth's MTP UD-Q4_K_M: verify against the non-MTP file, fp32 and 8-bit runs |
| `M` | M | Price list of speed knobs on t8r: expert deferral, near-tie swap, neuron skipping, 8-bit and 4-bit activations |
| `POWER` | - | Dual-bus power test (GPU clocks and power while reading RAM) |

Not here: the calibration text (private; used only through a link), the model files, and the bundles (backup paths shown as `<backup>/`).
