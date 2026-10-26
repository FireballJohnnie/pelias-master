# P-E16 results (CUDA doorbell flag in VRAM)

Raw output of `tools/probes/e16` on the reference machine, 2026-10-26. Reading: masterwork 21.2 (H-G1) and 17.2.

| Folder | Local time | Platform | Result |
|---|---|---|---|
| `windows_r2_no_flush` | 17:12 | Windows native, ReBAR on | Probe bug, kept as a record: no ring answered in any placement because Windows held the doorbell kernel's launch until a flush (fixed in r3). Launch and synchronize 30.7 us median |
| `windows` | 17:15 | Windows native, ReBAR on (CPU-visible VRAM heap 12,010 MiB) | All placements complete 20,000 of 20,000. Round trip median / p99: system RAM both 2.40 / 2.70 us; request in VRAM 2.30 / 2.60; reply in VRAM 2.30 / 2.50; both in VRAM 2.30 / 2.60. Launch and synchronize 30.5 / 45 |
| `wsl` | 17:17 | WSL2 | System RAM both 2.40 / 2.73 us; launch and synchronize 34.0 / 50.7; VRAM placements unavailable (no Vulkan device for the GPU under WSL2) |

All numbers MEASURED; CPU timer `std::chrono::steady_clock` (0.1 us steps on Windows).
