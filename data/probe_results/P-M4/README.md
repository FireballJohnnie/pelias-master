# P-M4 results (reading the model file)

Raw output of the Wave 0 probe `pelias_probe m4` on the reference machine, 2026-10-26, reading the 22,134,528,992-byte Unsloth Q4_K_M model file from the C: drive (Samsung NVMe, NTFS). The probe never records the file's name or path. Reading: masterwork 17.2 (P-M4 row).

| File | Local time | Platform | Note |
|---|---|---|---|
| `wsl/m4_modelfile.json` | 16:55 | WSL2, through `/mnt/c` (the mount reports virtiofs) | Run as root so the Linux page cache could be dropped; the Windows file cache is outside WSL's control, so "cold" here is Windows-warm-ish (cold and warm match) |
| `windows/m4_modelfile.json` | 16:58 | Windows native | "Cold" = unbuffered reads (bypass the Windows file cache); the memory-mapped case cannot bypass it |

Not kept as a result file: an earlier WSL attempt (16:2x) under memory pressure, in which the four read cases failed and the memory-mapped case crashed with a bus error before writing anything; `dd` then read 21.36 of 22.13 GB and stopped with an I/O error. After a restart and `wsl --shutdown`, `dd` read the whole file at 1.39 GB/s and the probe ran clean. Owed probe fixes: record a failed read's error code, survive a bus error in the memory-mapped case, and stop comparing checksums across read styles that sample the file differently (`checksum_variants_across_cases_expected_1` is 2 in both files for that reason, not corruption).

The author's third-party WSL speed daemon was running on the machine during these runs; its WSL-side state for each run was not recorded.
