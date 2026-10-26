# P-G0 / P-G0b results (graphics probe)

Raw JSON written by `tools/probes/gfx_probe` on the reference machine (RTX 4070, Ryzen 7 5800X3D, Windows build 10.0.26300, NVIDIA driver 621.01, Resizable BAR on). Every file is kept, failures included (ruling 2026-10-26 13:21). Summary and reading: masterwork 21.2.1.

| File (UTC start) | Probe version | Note |
|---|---|---|
| `gfx_probe_20261026_171413Z.json` | 0.1.0 | First run. System-RAM doorbell did not complete on either API (O-G1) |
| `gfx_probe_20261026_172937Z.json` | 0.2.0 | P-G0b variants. The summary line printed a 1-round Vulkan system-RAM run as 0.80 us; the JSON shows `rounds_done: 1`, so that variant failed |
| `gfx_probe_20261026_173341Z.json` | 0.2.1 | P-G0b variants; a variant counts only when all 1,000 rounds complete |

An earlier attempt with a web browser open (author, 13:12) wrote no file and is not part of the record.
