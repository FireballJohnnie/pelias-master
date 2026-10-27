# Decisions log

Every ruling that shapes this repository, with its date and source. AUTHOR-STATED rulings are the author's words; DECISION (proposed) items wait for the author.

| Date and time | Ruling | Label | Source |
|---|---|---|---|
| 2026-10-18 23:59 | "if we have to write our own dang program from the ground up and call it Pelias, i'm down for it." | AUTHOR-STATED | Long master 8.3 |
| 2026-10-22 | Float spread governs the accuracy gate (H6) | AUTHOR-STATED | Engine blueprint |
| 2026-10-23 | VRAM ceiling 10.7 GiB (D26) | AUTHOR-STATED | Trainer master |
| 2026-10-24 06:07 | Engine on hold | AUTHOR-STATED | Long master 8.10 (lifted 2026-10-26) |
| 2026-10-24 06:21 | Draft built for WSL2 and Windows; native Linux held | AUTHOR-STATED | Long master 8.11 (native Linux un-held 2026-10-26) |
| 2026-10-24 07:39 | Micro-batch is an engine setting; no Q8_0 core by default; Mode Q default (D31, D32, D34) | AUTHOR-STATED | Trainer master |
| 2026-10-24 08:15 | "when i say i wanted things from the ground up, i meant the ground up." (D37) | AUTHOR-STATED | Long master 8.11 |
| 2026-10-26 11:44 | Build the whole system now; one masterwork markdown; installers for Linux (WSL first) and Windows 10/11; ground up in C++20/CUDA; universal detection; three paths; server on 8062; web tools with FlareSolverr as the only outside piece | AUTHOR-STATED | Masterwork 0.2 (A1 to A9) |
| 2026-10-26 11:52 | GPU: the system accounts for other vendors itself; import files for the author's machine, model and parameters; recommend on a higher ceiling; web interface yes; OS-provided HTTPS yes; search the author's proposals folder and use all attachments; everything in pelias-master, ready for git | AUTHOR-STATED | Masterwork 0.2 (A4 to A11) |
| 2026-10-26 11:55 | Pause for a Fable review of the markdowns before the build is queued | AUTHOR-STATED | Masterwork 0.2 (A12) |
| 2026-10-26 12:05 | The first build is benchmarked against the official and Unsloth 4-bit files and the unquantized model, and its speed against the ceiling | AUTHOR-STATED | Masterwork 16.4 (A13) |
| 2026-10-26 12:19 | Record future hypotheses: the bus outside LLMs, kept tuned by a tiny CPU model or by code alone; field applications with on-paper gains; consumer PCs doing data-centre work | AUTHOR-STATED | Masterwork 21 (A14) |
| 2026-10-26 12:23 | Have Fable review it | AUTHOR-STATED | docs/reviews/FABLE_REVIEW_2026-10-26.md |
| 2026-10-26 12:24, 12:29 | Add gaming (players and developers) to Section 21; give ballpark end-to-end gains | AUTHOR-STATED | Masterwork 21.5, 21.6 |
| 2026-10-26 12:26 | Test Direct3D 12 and Vulkan with and without while the review runs; files approved | AUTHOR-STATED | Probe P-G0 |
| 2026-10-26 12:52 | Run the Direct3D 12 / Vulkan probe now; each other Section 21 area gets its own scripted probe, run against a stock baseline with unchanged output, after pelias-master is built and running | AUTHOR-STATED | Masterwork 21.7 (to be updated) |
| 2026-10-26 13:21 | Factor all probe data into the Pelias build whether it works or not: probes and every result (failures included) kept in pelias-master; the engine chooses each path (doorbell flag location, submit style, upload path) by startup measurement, with both paths kept behind a setting for testing | AUTHOR-STATED | Masterwork 21.7 |
| 2026-10-26 13:25 | Publish open source for developers, enthusiasts and end users (repositories stay private until the author says push) | AUTHOR-STATED | Author |
| 2026-10-26 13:48 | Go: build step 1, close the measurement gaps | AUTHOR-STATED | Masterwork 15.2 |
| 2026-10-26 14:3x to 16:00 | Fix the weak spots, redesign as deep as needed, leave no stone unturned and understand the why; build v5, v7, v8, v9, v10, v11 and v12; correct the M=8192 open item; split-K for the low-tile shapes and the layer-order L2 test | AUTHOR-STATED | Masterwork 17.5 |
| 2026-10-26 16:00 | Split-K is wrong for expert down (enough tiles already) and M=64 experts need their own test | AUTHOR-STATED (challenge, accepted) | Masterwork 17.5, O13 |
| 2026-10-26 16:1x | Dense layers: int8 v10 (3 stages, fp32 output); MoE experts at small M: grouped 64x128 launch; activation quantization fused into the RMSNorm; no split-K | DECISION (from the measurements, author's "excellent") | Masterwork 17.5 |
| 2026-10-26 16:14 | Update everything on git with these results | AUTHOR-STATED | This commit (not pushed) |
| 2026-10-26 17:20 | Proceed in the proposed order: O15 fence variant, P-M4 probe fixes, then the P-M5 route | AUTHOR-STATED | Masterwork O15 |
| 2026-10-26 17:26 | P-M5 deferred to step 10 (engine byte counters, no third-party counter tool) | AUTHOR-STATED | Masterwork O16 |
| 2026-10-26 17:29 to 17:30 | ReBAR on/off pair optional: build ReBAR support in, let the startup probe choose; account for the ReBAR-off path on another build day (an adjustment, not a rewrite). Step 1 closes | AUTHOR-STATED | Masterwork 15.2, O17 |
| 2026-10-26 17:34 | Push | AUTHOR-STATED | All three repositories, main branches |
| 2026-10-26 17:41 to 17:43 | From all the probe data update Section 21 and 21.5 (everything mapped beyond the hot bank), push it and anything left unfinished | AUTHOR-STATED | Masterwork 21 |
| 2026-10-26 17:53 | Complete step 2 stages A and B (write, build host-only, test, boundary and privacy checks, back up, commit locally, no push, no deletions, nothing in the author's PowerShell, predictions before runs); stop and hand over the stage C gate commands; hard stop before step 3 | AUTHOR-STATED | BUILD_LOG |
| 2026-10-26 18:0x | plan -> io edge in the module graph; `require_data=false` header-only reading; the 2.50 GiB working-set ESTIMATE labelled in every plan until P-M2 | DECISION (step 2) | Masterwork 14.3; BUILD_NOTES |
| 2026-10-26 | Vulkan compute as the second GPU backend; pelias-master as the code repository; own installer; tighter kernel tolerance until measured; reference profile 7.5 GiB bank at `-ub 2048`; poll level 100 | DECISION (proposed) | Masterwork 3.1, 14.1, 13.3, 16.2, 18.1 |
| 2026-10-26 18:38 | Fix the four review findings (strict CLI numbers, NVML card match, JSON duplicate keys, safetensors zero-length order) | AUTHOR-STATED | `src/cli`, `src/hal/nvml`, `src/io` |
| 2026-10-26 18:51 | Push | AUTHOR-STATED | pelias-master main 94a0478..d879212 |
| 2026-10-26 18:55 to 18:58 | Pelias is built for accuracy and speed; MTP must be included (it adds speed and leaves the chosen tokens to the trunk) | AUTHOR-STATED | Masterwork 15.2 step 10, O18 |
| 2026-10-26 18:5x | Planner: the MTP head's routed experts stay in the expert pool, the rest of the head is always resident; MTP blocks found from `<arch>.nextn_predict_layers` and `<arch>.block_count` | DECISION (step 2, O18) | `src/plan/memory_plan` |
| 2026-10-26 19:07 | Fix all mistakes found: the serving planner used the training working set (2.50 GiB, 8.8); serving now uses 1.00 GiB, DERIVED upper bound from 17.4 and P-H8 (steps in `src/plan/probe_constants.h`); O18 records the tier build's measured MTP gain | AUTHOR-STATED (fix) / DECISION (figure) | `src/plan`, masterwork 4 and O18, BUILD_NOTES |
| 2026-10-26 19:22 | Start step 3 | AUTHOR-STATED | Masterwork 15.2 step 3 |
| 2026-10-26 19:2x | Step 3 in three stages, each stopping for the author: A quantizers (sandbox, oracle), B type rule + GGUF writer + `pelias quantize` with the author's P-E6 run, C safetensors conversion with the author's P-X6 run; quantizer module compiled without floating-point contraction; the oracle is opt-in and built from a checkout outside the repository; golden hashes from the reference carry the check to Windows | DECISION (step 3) | `src/quant`, `tests/oracle`, `tests/unit/quant_golden.h`, BUILD_NOTES |
| 2026-10-26 19:40 | Add 2-bit and 3-bit support (logged for the future) | AUTHOR-STATED | Masterwork 5.3, O19 |
| 2026-10-26 19:56 | Go on stage B | AUTHOR-STATED | Step 3 |
| 2026-10-26 20:1x | Driver for GGUF-to-GGUF quantization lives in `io/` (io may depend on quant, 14.3); offered mixes Q4_K_M, Q6_K, Q8_0, BF16, F16, F32 (the types this build can write); inputs with an alignment other than 32 refused (the reference writes 32 whatever the input said); exit codes 5 (verify differs) and 6 (write failed) | DECISION (step 3) | `src/io/requantize`, `src/cli/main.cc` |
| 2026-10-26 20:30 | Add a "Pelias mix" (more accurate 4-bit variant) to the plan; the benchmark's question is accuracy | AUTHOR-STATED | Masterwork O20 |
| 2026-10-26 20:31 | P-E6 passed (byte-identical on the real model); push; add a "Pelias mix" (more accurate 4-bit variant, judged by KL against bf16) to the plan | AUTHOR-STATED | Masterwork O20; RUNBOOK |
| 2026-10-26 22:4x | Stage A computes ssm_a = -exp(A_log) correctly rounded (the reference's MKL exponential is not reproducible and differs on 13 of 960 values by 1 ulp); P-X6 expects byte identity everywhere else and at most 1 ulp on the 30 ssm_a tensors | DECISION (proposed, the author confirms at P-X6) | `src/ingest/stage_a.cpp`, BUILD_NOTES |
| 2026-10-26 22:4x | Stage A reads config.json as written (the pinned transformers 4.57.6 has no qwen3_5_moe, so the reference falls back to the file too, READ-SRC); the pre-tokenizer name comes from a table keyed by the tokenizer's pre-tokenizer definition (the reference hashes a tokenized text); layouts outside what this model uses are refused, not guessed | DECISION (step 3) | `src/ingest/` |
| 2026-10-26 22:4x | `pelias convert --type Q4_K_M` quantizes in one pass from the shards (no 67 GB BF16 file on disk); its output must equal the reference's two passes (unit test: one pass equals convert-then-quantize, byte for byte) | DECISION (step 3) | `src/io/requantize`, `src/ingest/stage_a.cpp` |
| 2026-10-27 01:40 | ssm_a stays correctly rounded (no copy of the reference's values): it is not worse, the 0.00016 KLD gap only reflects that the bf16 reference carries the reference's own rounding | AUTHOR-STATED | P-X6 record, BUILD_LOG |
| 2026-10-27 02:01 | The Pelias quantization family is named PEL (PEL4, PEL3 ..., versioned, recipe in the GGUF metadata) | AUTHOR-STATED | Masterwork O20 |
