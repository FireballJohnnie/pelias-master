# P-M4 results (reading the model file)

Raw output of the Wave 0 probe `pelias_probe m4` on the reference machine, 2026-10-26, reading the 22,134,528,992-byte Unsloth Q4_K_M model file from the C: drive (Samsung NVMe, NTFS). The probe never records the file's name or path. Reading: masterwork 17.2 (P-M4 row).

| File | Local time | Platform | Note |
|---|---|---|---|
| `wsl/m4_modelfile.json` | 16:55 | WSL2, through `/mnt/c` (the mount reports virtiofs) | Run as root so the Linux page cache could be dropped; the Windows file cache is outside WSL's control, so "cold" here is Windows-warm-ish (cold and warm match) |
| `windows/m4_modelfile.json` | 16:58 | Windows native | "Cold" = unbuffered reads (bypass the Windows file cache); the memory-mapped case cannot bypass it |

| `windows_fix/m4_modelfile.json` | 17:37 | Windows native | Rerun after the probe fixes: all six cases ok, one checksum variant (was two); unbuffered 2.55 / 3.13 GB/s, file cache 12.27 / 12.67, memory-mapped 4.68 |
| `wsl_fix/m4_modelfile.json` | 17:40 | WSL2 through `/mnt/c` | Rerun after the probe fixes: all six cases ok, one checksum variant; 1.52 to 1.58 GB/s, memory-mapped 1.21 / 1.28 |

Not kept as a result file: an earlier WSL attempt (16:2x) under memory pressure, in which the four read cases failed and the memory-mapped case crashed with a bus error before writing anything; `dd` then read 21.36 of 22.13 GB and stopped with an I/O error. After a restart and `wsl --shutdown`, `dd` read the whole file at 1.39 GB/s and the probe ran clean. The probe fixes (made 17:2x, verified by the `*_fix` runs): a failed read records its error code and how far it got; a bus error (Linux) or in-page error (Windows) in the memory-mapped case is caught and recorded instead of ending the program; every case uses one checksum (the 8-byte word at every 4 KiB offset), so the first two files' `checksum_variants_across_cases_expected_1` = 2 came from the old sampling, not corruption.

The author's third-party WSL speed daemon was running on the machine during these runs; its WSL-side state for each run was not recorded.
