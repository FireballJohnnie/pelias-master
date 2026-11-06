# PELIAS RAM WALL — MASTER (host RAM, the WSL2 layer, pinning, PCIe, VRAM residency; what is measured, what is theory, what is wrong)

Copyright (c) 2026 John Lipka and the Pelias contributors. Confidential, not for distribution.

Written 2026-11-06 from two research documents the author commissioned on 2026-11-01 (the `v28`–`v32` era, when the CPU still computed light experts): `pelias_memory_virtualization_and_bus_architecture_analysis.md` ("doc A", 8 sections) and `pelias_ram_wall_systems_and_architecture_research_dossier.md` ("doc B", 9 sections), combined with what Pelias has MEASURED on the reference PC since (BUILD_LOG v30–v60g, the game plan, this week's runs). Where the two documents disagree, or where a measurement here contradicts them, the measurement wins and the disagreement is kept visible. Labels: MEASURED (on this PC, log cited) / DERIVED (formula from measured inputs) / READ-SRC (read in source or config on this PC) / PUBLISHED (vendor doc, kernel source, paper — cited by the research docs, not re-verified here) / ESTIMATE / DECISION / UNVERIFIED / CONTRADICTED / SUPERSEDED.

Reference PC: Ryzen 7 5800X3D (8 cores / 16 threads, 96 MB L3), RTX 4070 12 GB (46 SMs, PCIe 4.0 x16), 64 GB DDR4-3200 dual channel (51.2 GB/s peak), Windows 11, WSL 3.0.4, kernel 6.18.54.1-microsoft-standard-WSL2, `.wslconfig` memory=40GB (was 48), swap=48GB, autoMemoryReclaim=disabled, kernelCommandLine `transparent_hugepage=always swiotlb=65536 sysctl.vm.swappiness=10`, virtiofs=true.

---

## 0. What changed since the two documents were written (so their remedies are read against today's engine)

1. **THE PLAN (2026-11-01 17:54): the GPU does all of the decoding; the CPU only does work that is free and never on the token's path.** Both documents are built around the `v28`–`v32` split (CPU computes light experts from the mapping, GPU reads heavy experts over PCIe, 40 doorbells per token). That path is retired (commit A, v60). Everything in the documents about CPU thread placement, L3 prefetch of light experts, SMT collapse and the 12.4 ms GPU idle gap describes a path that no longer runs. It is kept here as recorded theory (B7: parked ≠ dead), not as action.
2. **Already done in Pelias, before or since the documents:** model file on `ext4` (`~/models`, v31); small-pages advise before every `cudaHostRegister` of a file mapping (v34, and again v60c); the doorbell as a mapped-memory user-space spin (2.40 µs round trip MEASURED, P-E16) — doc B's "Fix 1B" is what Pelias already does; the 2 MB `O_DIRECT` arena probe (36× faster registration, MEASURED 11-01); the expert planes as a packed pinned copy in host RAM (v54 → v60 B); the three-way pinned-read probe at open (v60 C2/C3).
3. **New since the documents (this week, MEASURED):** the pinned-read lottery — same binary, fresh VM, some processes stream the pinned planes 2–3× slower through SM loads while the copy engine reads 26 GB/s regardless (v60d/e/g). The documents' "hidden 4 KB penalty on the GPU side" (doc B §3.2) is the only mechanism in either document that matches it; see §4.

---

## 1. The bus physics, stated once and correctly (MEASURED on this PC unless marked)

| Path | Number | Label / source |
|---|---|---|
| DDR4-3200 dual channel, peak | 51.2 GB/s | PUBLISHED (spec) |
| CPU plain read of host RAM | 37 GB/s one thread, 43.7–45.8 four threads | MEASURED, Wave 0 floors; `pelias probe` 45.9 (11-04) |
| CPU decoding Q4_TG2 in place (the retired split) | 12.1–15.1 GB/s bench at 8 threads; 8.3–8.5 in-engine, flat 6–12 threads, collapse at 16 | MEASURED v35–v37 |
| CPU math over anonymous RAM vs file-mapped RAM | 10.4–11.1 vs 8.1–8.4 GB/s | MEASURED v31/v32 (the gap both documents explain; see §3 claim 4) |
| GPU zero-copy read of pinned host RAM over PCIe (copy engine / streaming) | 26.2–26.6 GB/s | MEASURED bus_bench 10/28; 26.1–26.2 in every v60e/g arm; registered file read 25.7–26.2 at any chunk ≥ 256 KiB |
| GPU SM loads of pinned host RAM, one 128 B line per 4 KB page | 23.4–24.7 GB/s of lines when healthy | MEASURED v60g (C3 probe) |
| H2D uploads (cudaMemcpy) | 12.5 → 22.5 GB/s | MEASURED (speedup rev5) |
| **Dual bus** | **26.56 (GPU over PCIe) + 21.06 (CPU reading DDR4) = 47.62 GB/s, 93 % of 51.2** | MEASURED bus_bench 10/28 |
| VRAM read | 523–534 GB/s plain; R3 ceiling 494; lean GEMV 503 | MEASURED |
| Doorbell (mapped-memory spin) | 2.40 µs round trip; 2.0 µs release store | MEASURED P-E16 |

**What "dual bus" means, exactly (DERIVED from the bus_bench setup, and a correction to a line I gave the author on 11-05):** the 47.6 GB/s is the DDR4 bus serving two consumers at once — the GPU pulling bytes over PCIe (26.6) and the CPU reading other bytes for itself (21.1). It is *not* PCIe's two directions. On PCIe, every way of moving weights into the GPU — the GPU's own reads of host memory (completions), the copy engine's H2D DMA, a CPU write into BAR1 (Windows only) — carries data in the same host→GPU direction and shares that direction's ~26 GB/s practical ceiling (Gen4 x16, 32 GB/s raw). The GPU→host direction carries requests and our few kilobytes of results. So: **PCIe into the GPU is one lane of ~26 GB/s; the only second lane for weight bytes is the CPU consuming them from DDR4 itself.** That is what the retired v28 split was, and its limit was the CPU decoder (12–15 GB/s on Q4_TG2 against a 27 GB/s gate), not the bus. Any C4 "second lane" design has to say which of these it means; the texture path is a different *fetch route inside the GPU* and reads through the same PCIe link when the data is in RAM (see §6).

---

## 2. The memory layers on this PC, as they actually are today

1. **The model file** (22.66 GB tier file) lives on `~/models` (ext4 inside `ext4.vhdx`). Hyper-V opens the vhdx unbuffered on the host (PUBLISHED, both docs; consistent with RAMMap never showing it — UNVERIFIED here, see §5 check 5). Inside the VM it sits in the Linux page cache as file-backed pages, **and on this kernel those pages are 2 MB where they can be: `FileHugePages: 5.1 GB`, `FilePmdMapped: 770 MB` right now (READ-SRC `/proc/meminfo`, 11-06 12:19)**, even though `# CONFIG_READ_ONLY_THP_FOR_FS is not set` (READ-SRC `/proc/config.gz`). Both documents say file mappings are forced to 4 KB on the stock WSL kernel; **that is CONTRADICTED on this kernel (6.18.54)** — doc B itself notes mainline commit a1a8bab74176 (June 2026) removed the gate; this kernel has it. Consequence: the v33 register stall (cudaHostRegister on 2 MB file folios) is exactly what you get here without v34's `MADV_NOHUGEPAGE` first, and v60b reproduced it on the embedding range.
2. **The pinned planes** (9.66–9.83 GB at 3 bits; 14.76 at 5): anonymous, driver-allocated (`cudaHostAlloc` mapped), packed at open from the file mapping, one copy; the GPU reads them in place. Guest THP is `always` (cmdline), so guest-side 2 MB; **the host-side backing (Hyper-V SLAT) is whatever Windows gives the VM at that moment — this is the variable behind the lottery (§4).**
3. **Registered file ranges** (the unplaned experts outside the bank: 0.02–0.32 GB) and the token / MTP embedding ranges: `cudaHostRegister` on the page-cache pages after the small-pages advise.
4. **The VM**: 40 GB (`MemTotal` 39 GB), THP always, swiotlb 64 MB pool reserved by `hv_pci` (READ-SRC dmesg: `reserved swiotlb pool [0x04000000..0x08000000)`) — used by VirtIO devices (virtiofs, console), **not** by the GPU path (`/dev/dxg` has no PCIe device in the guest: `lspci` shows only two VirtIO devices; READ-SRC).
5. **VRAM**: 10.78–10.80 GB engine (dense 2.01–2.06, bank 4.6–4.9, cache 3.5–4.0) + context; 11.7–11.8 GB used of 12.28 in every arm; WDDM owns residency (§3 claims 7–8).
6. **Windows host**: 64 GB physical; during a run 51–57 GB used, 8.5–14 GB available; page file 0.9 → 4–14 GB → 0.9 at every process start/stop (MEASURED HWiNFO v60c/d/e: the churn is the same with the VM at 48 or 40 GB).

---

## 3. Claim ledger — every claim in the two documents against this PC

| # | Claim (doc) | Status here | Evidence / note |
|---|---|---|---|
| 1 | `/mnt/c`, `/mnt/d` through virtiofs without DAX: double cache + SWIOTLB bounce → 0.52 GB/s (A §1, B §6.1) | **CONSISTENT, MEASURED** | v30 0.52 GB/s pre-load, 21–27 major faults/token → v31 ext4 0.2–0.6 s (BUILD_LOG L286+). The 64 MB swiotlb pool exists (dmesg). Never put the model on `/mnt/*` — standing rule since v31. |
| 2 | Windows does not double-cache `ext4.vhdx` (vhdmp.sys unbuffered) (A §1.3, B §6.2) | **PUBLISHED, UNVERIFIED here** | Cheap check: RAMMap → File Summary → `ext4.vhdx` during a run (§5 check 5). Nothing in our logs contradicts it. |
| 3 | WSL `cudaHostRegister` costs 0.8–1.0 s/GiB (dxg VMBus GPADL per 4 KB page) vs 0.07 on Windows; 2 MB backing cuts it ~10× (A §2, B §3.1) | **CONSISTENT, MEASURED** | 11-01 arena probe: a 2 MB `O_DIRECT` arena registers **36× faster** (BUILD_LOG L316–345). Our planes avoid the issue (driver allocation, ~18 s is the pack, not the pin); registered ranges are small (0.02–0.32 GB). |
| 4 | CPU math 8.1–8.4 GB/s on file mapping vs 10.4–11.1 on anonymous = 4 KB pages → 24-step (A) / 15-step (B) nested walks + prefetcher stops at 4 KB (B) | **MEASURED gap; mechanism PARTLY CONTRADICTED** | The gap is real (v31/v32). But file mappings here *are* partly 2 MB (claim 12), so "forced 4 KB" is not the whole story; SMT, DRAM row conflicts and the decoder's own cost were all in play (v35–v37). **SUPERSEDED in practice:** the CPU no longer decodes experts (THE PLAN). Matters again only if a C4 lane puts work on the CPU. |
| 5 | Zero-copy GPU reads are true DMA after registration, no bounce (A §2.3, B §3.2) | **CONSISTENT, MEASURED** | 26.2–26.6 GB/s = the link; "regardless of page size" for streaming reads (11-01 arena probe; v60e/g copy engine in every arm). |
| 6 | Hidden PCIe penalty on 4 KB host pages: GPU MMU / IOMMU TLB miss every 4 KB, scattered pages (B §3.2 only) | **THE LIVE HYPOTHESIS for the lottery — PARTLY MEASURED** | Streaming/copy-engine reads do *not* show it (26 GB/s in slow arms too). SM loads do: 2–3× slower whole-engine in some processes (v60d 3/5, v60e 1/8). The SM one-line-per-page probe reads 23.4–24.7 healthy; a slow arm's number is still UNMEASURED (none has appeared since the probe went in). See §4. |
| 7 | WDDM pinned ("shared GPU memory") cap ≈ 50 % of host RAM; `SetProcessWorkingSetSizeEx` does **not** raise it (B §3.3) vs. `SetProcessWorkingSetSizeEx` expands the locked budget (A §4.2) | **A and B CONTRADICT each other; B better sourced; our number fits B** | Wave 0 floors: "32 GiB pinnable" on a 64 GB host = 50 %. Working-set calls lock CPU pages; WDDM's `SharedSystemMemory` is a separate ceiling (B cites forum #228235 post 4). Treat the cap as 32 GB, fixed. |
| 8 | NVIDIA "CUDA – Sysmem Fallback Policy": WSL2 **ignores** it, silently spills VRAM to host RAM (A §4.1, WSL#11050) vs. setting it on Windows **governs** WSL2 too (B §7) | **A and B CONTRADICT each other; UNVERIFIED here** | Our data: VRAM used 11.7–11.8 of 12.28 GB in *every* arm, slow and fast alike (v60d/e/g sampler), so VRAM spillover is not the lottery. The policy's current setting on this PC is UNVERIFIED (registry/control panel — the author's call to look). Cheap test in §5 check 6. |
| 9 | Hyper-V backs VM RAM with 2 MB SLAT pages unless ballooning/dynamic memory fragments it ("never shatter") (B §2 Q10) | **PUBLISHED; fits our symptom** | `autoMemoryReclaim=disabled` already; but Windows pages 4–14 GB of *something* at every arm boundary (HWiNFO). If that is vmmem's backing, SLAT shattering is the lottery's cause. UNVERIFIED which memory the page file is taking; §5 check 4. |
| 10 | 16-thread collapse = SMT siblings sharing L1D/DTLB + DRAM row conflicts (A §5.1, B §5.2); pin 8 threads 1/core | **MEASURED collapse; mechanism PUBLISHED; SUPERSEDED** | v35–v37: flat 6–12 threads, collapse at 16. Topology here: siblings are **(0,1) (2,3) … (14,15)** (READ-SRC `lscpu -e`), not 0/8 as A guessed. No CPU worker pool runs on the token path today (THE PLAN); the pack threads at open use 16 (18 s, off the token path). |
| 11 | 5800X3D TjMax 90 °C; 99.7 W runs throttle (A §5.1) | **MEASURED thermals (v32 Tctl 91.1 °C at 99.7 W); SUPERSEDED** | CPU draws 35–42 W during decode now (HWiNFO v60d–g). The watchdog's 87 °C abort stays. |
| 12 | Stock WSL kernel: `# CONFIG_READ_ONLY_THP_FOR_FS is not set` → file mappings cannot get 2 MB pages; needs a custom kernel or `O_DIRECT` into anonymous THP (A §3.3, B §4 Option 3B) | **Config half CONFIRMED (READ-SRC), conclusion CONTRADICTED** | `/proc/config.gz`: not set. `/proc/meminfo`: `FileHugePages 5.1 GB, FilePmdMapped 770 MB` — this 6.18 kernel maps files with PMD huge pages anyway (mainline a1a8bab74176, which B cites, is in it). **No custom kernel needed; and this is why v34's small-pages advise before registering is mandatory.** |
| 13 | `O_DIRECT` read into `MADV_HUGEPAGE` anonymous memory = one copy in RAM, 2 MB pages, +35 % CPU read (B §4 Option 3A, A §3.3 Option B) | **MEASURED in part; the planes already are this** | The 11-01 arena probe is the mechanism (36× registration). The packed planes are a one-copy anonymous pinned image of the planed experts; the page cache behind them is evictable. For the CPU-read gain: SUPERSEDED (no CPU decode). |
| 14 | Windows native: `VirtualAlloc(MEM_LARGE_PAGES)` + `SeLockMemoryPrivilege` + `FILE_FLAG_NO_BUFFERING` read = one copy, 2 MB pages, 10× faster register, no SLAT (B §4 Workaround 2, A §3.3 Option B) | **PUBLISHED; the Windows build exists (wincheck57d, v58b 87.1 tok/s); UNVERIFIED on our pinned path** | This is the 11-02 handoff's "Windows large pages (Lock pages privilege)" item. It is also the only remedy in either document that removes the SLAT layer the lottery lives in. §6 action 2. |
| 15 | Resident pin host via `dlopen` to keep registrations across runs (B Workaround 1) | **Harness convenience; not product** | The product opens once per session. For the test harness it would also *stop the free/re-pin churn that rolls the lottery dice* — a real diagnostic use. ESTIMATE 50–100 lines; parked until the lottery's cause is confirmed. |
| 16 | Hyper-V DDA / bare-metal Linux remove WDDM and the VMBus pin path (B Workarounds 4–5) | **PUBLISHED; out of scope** | The product targets Windows; DDA needs an iGPU for the desktop and a Server SKU or third-party unlock. Recorded, not planned. |
| 17 | 12.4 ms GPU idle per 32.2 ms token; four fixes (L3 lookahead prefetch, spin doorbells, MTP overlap, GPU takes cold experts over PCIe) (A §5.2, B §8) | **SUPERSEDED by THE PLAN; two of the four are in Pelias in another form** | Spin doorbells: done (2.40 µs). "GPU takes the cold experts over PCIe": that *is* the R3 expert stage reading the planes. MTP overlap: the draft chain on the device (commit C). L3 lookahead prefetch for a CPU decoder: no CPU decoder. |
| 18 | Pre-gated / lookahead routing predicts next layer's experts 70–85 % (B §8.1, arXiv 2308.12066) | **PUBLISHED; our own number is better-grounded** | Pelias measured pre-gating 83.8 % (step 2 replay) and causal 64–82 %; it feeds P1 "bank that follows the text", not a CPU prefetch. |
| 19 | 2D nested page walk = 24 accesses (A) vs 15 with 2 MB host pages (B) | **Both DERIVED, textbook; immaterial to the token path today** | Only the GPU's translation of host pages matters now (claim 6). |
| 20 | Pin only the heavy ranges, not the whole file (A §4.2, B §9 step 3) | **Done since v27 / v54** | 0.02–0.32 GB registered; the rest is the packed planes. |

---

## 4. The pinned-read lottery, read through both documents (the one place they help today)

MEASURED (v60d/e/g): same binary, fresh VM, GPU 99–100 % util at **84 W** in a slow arm vs 112–122 W in a fast one; prefill 55 vs 154 tok/s, verify-2 49 vs 17.4 ms; one-token decode (bank hits) unaffected; VRAM used identical; CPU 9–19 %; planes the same size; **copy engine 26 GB/s in slow arms too**. The v59e "aged VM" (52.6 → 19.5 ms after `wsl --shutdown`) is the same signature with age instead of chance.

Mechanism that fits every observation (doc B §3.2 + §2 Q10, UNVERIFIED as a whole, each part labelled):
- The planes are guest-2 MB anonymous pages (THP always). Their **host** backing is Hyper-V's SLAT: 2 MB entries when Windows can give the VM contiguous physical memory, 4 KB ("shattered") when it cannot — after ballooning, paging of vmmem, or plain host fragmentation (PUBLISHED, B).
- The GPU's SM loads over PCIe translate host pages through the IOMMU / GPU MMU path; a 4 KB-shattered range costs a translation per 4 KB of scattered host frames — small, latency-bound expert loads (32–128 B) pay it on every line; the copy engine's long bursts hide it (PUBLISHED mechanism, B §3.2; MEASURED consequence: copy engine fine, SM loads slow).
- Windows pages 4–14 GB at every arm boundary (MEASURED HWiNFO) — a 9.8 GB pin/free cycle per process on a host with 8–14 GB available is exactly the condition that fragments backing. The dice are rolled per process; the product rolls them once per session.

What this says about the remedies, in order of certainty:
1. **Measure it first (in hand):** the C3 probe's SM one-line-per-page reader. A slow arm must show it under the 23.4–24.7 healthy band for this mechanism to stand; if it doesn't drop, the mechanism is wrong and the next instrument is the stage counters' staged-task cycles per arm.
2. **Re-pin at open** (C2, floor to be set from that measurement): a fresh `cudaHostAlloc` may land on unshattered backing. Cheap, product-safe, measured by its own success rate.
3. **Native Windows with `MEM_LARGE_PAGES`** (claim 14): removes SLAT from the GPU's path entirely and makes the host backing 2 MB by construction. The strongest fix, and the Windows build exists; the pinned-planes allocation on Windows would need `VirtualAlloc(MEM_LARGE_PAGES)` + `cudaHostRegister` instead of `cudaHostAlloc` (ESTIMATE: 40 lines in `hal/`). This is also the "WSL doesn't like to split lanes" item the author named for C4.
4. **Less host pressure**: 48 → 40 GB VM moved the odds (2/5 → 7/8 fast) without ending the churn; the page file still swings. Going lower starves the file cache (the model is 22.7 GB; planes 9.8). Not a fix, a dial.
5. **Sysmem fallback policy** (claim 8): settle A vs B with one test (§5 check 6); it is not the lottery (VRAM equal in slow/fast arms) but it decides whether a VRAM overrun fails loudly or silently.
6. The `dlopen` pin host (claim 15) as a *harness* fix: stops the dice from rolling between arms; would also hide the problem — use only after 1–3 are measured.

---

## 5. Cheap checks that turn UNVERIFIED into MEASURED (read-only unless marked; the author runs anything on Windows)

1. ✓ `zcat /proc/config.gz | grep READ_ONLY_THP_FOR_FS` → not set (11-06). ✓ `/proc/meminfo` FileHugePages 5.1 GB, FilePmdMapped 770 MB. ✓ `lscpu -e` sibling pairs (0,1)…(14,15). ✓ swiotlb pool 64 MB at 0x04000000 (dmesg). ✓ `lspci`: no GPU in the guest; `/dev/dxg` present.
2. `grep -E 'AnonHugePages|FilePmdMapped' /proc/<pelias pid>/smaps_rollup` during a run: how much of the planes and of the file mapping is 2 MB **in the guest** (does not show the host side).
3. The host side: Windows `RAMMap` (Sysinternals) → *Use Counts* / *Physical Pages* during a run: how fragmented physical memory is; → *File Summary*: whether `ext4.vhdx` appears (claim 2). Author-run, read-only.
4. HWiNFO columns to add for the next runs: *Page File Usage* is logged; add *Memory Compressed* and *Standby List* if available — tells whether the 4–14 GB swings are vmmem being paged (claim 9).
5. ✓/partly: the page-file churn exists with 48 and 40 GB (MEASURED). Try one run with the page file fixed-size (author's call; UNVERIFIED effect) — it cannot shrink the churn's cause, only change where it lands.
6. Sysmem fallback (claims 8): read the current setting in NVIDIA Control Panel → Manage 3D Settings → Global (author). Test: a tiny WSL program `cudaMalloc`s 1 GB past free VRAM — fails (policy honoured) or succeeds and runs slowly (ignored). 5 minutes, no engine change.
7. The lottery's own number: keep the C3 probe in every arm until a slow arm prints its three readers (nothing to do but run).

---

## 6. What this means for the plan

- **Standing rules confirmed by the documents' mechanisms:** model on ext4 (v31); small-pages advise before registering any file mapping (v34/v60c — now explained: this kernel gives files 2 MB pages); pin only what the GPU reads (v27/v54); `wsl --shutdown` before speed sessions; a fresh build dir per attempt; the probe line read first in every arm.
- **For the second lane (C4 addendum):** the physics in §1 fixes the vocabulary. PCIe into the GPU is one ~26 GB/s lane whatever reads it (SM loads, copy engine, texture fetches, or BAR1 writes from the CPU on Windows); the DDR4 bus has room for a second *consumer* (the CPU, ~21 GB/s left under the GPU's 26.6). A texture-unit fetch of RAM-resident planes therefore buys no bandwidth; what it can buy is a cheaper decode (8-bit texels → float for free) and a different cache path *for VRAM-resident data* — a decode/occupancy question, not a bytes question. A CPU lane buys bytes only if the CPU's work per byte is cheap (the v28 decoder was not: 12–15 GB/s). The author's RAM research (these two documents) does not settle the lane question; the sandbox's two probe runs do — and the first of them is which route the second lane's bytes take.
- **Order (DECISION proposed, author's call):** (1) finish the lottery instrument (C3 probe → floor → re-pin), (2) the Windows-native run of today's engine with `MEM_LARGE_PAGES` for the planes — it tests claim 14 and gives the native-Windows baseline the C4 rule asks for in one run, (3) the sysmem-fallback 5-minute test, (4) then C4 with §1's physics on the table.

---

## 7. Sources named by the two documents (not re-verified here; kept for the record)
dxgkrnl (LKML 2022 patch series; `create_existing_sysmem`, `vmbus_establish_gpadl`); NVIDIA forums #255472, #279429, #228235 (post 4), #32980; microsoft/WSL issues #11050, #11095, #40284, #40426 (PR), #40719, #40732, #40773; Box of Cables (Hayden Barnes, 2026-05-31) on virtiofs/SWIOTLB; Linux THP admin guide; mainline a1a8bab74176 (2026-06-23); Microsoft Learn: SLAT requirements, CSV cache (vhdx unbuffered I/O), `CreateFileMapping2`/`SEC_LARGE_PAGES`, `SetProcessWorkingSetSizeEx`, DDA; TechPowerUp 5800X3D; arXiv 2308.12066 (Pre-gated MoE), 2402.07033 (Fiddler), 2411.11217 (MoE-Lightning), 2412.19437 (KTransformers), 2604.10152 (SpecMoE), 2605.17889 (CoX-MoE), 2607.14647 (D-cut), 2610.28620 (barrier scaling), 2505.19645, 2512.01278; FlashML-org/FreeToken #443, #214; ExHyperV / HyperV-DDA tools. Two documents cite the same llama.cpp discussion #29347 for mmap/THP behaviour — llama.cpp is an oracle for numbers, never a design source (B1).
