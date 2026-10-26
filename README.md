# PELIAS (system)

**One ground-up program that probes the machine, takes a local GGUF or safetensors model, coaches and runs training or quantization, writes the resulting GGUF, and serves it on port 8062 through its own engine and server.**

> **PRE-RELEASE, PRIVATE. Specification stage, 2026-10-26.** No product code exists yet. The design is complete in [`docs/PELIAS_SYSTEM_MASTERWORK.md`](docs/PELIAS_SYSTEM_MASTERWORK.md), reviewed by Claude Fable on 2026-10-26 (`docs/reviews/`), and waits for the author's go before the build starts. Nothing in this repository is a measured result of this program; the measured numbers it cites come from the sibling repositories' ledgers and probe runs, each with its label.

**Author and maintainer:** John Lipka. Developed with AI assistance under his direction. Sibling repositories (records of the design and measurements): pelias-trainer, pelias-engine.

## Origin

In the author's words: "this started as an effort for me to better create my own llm, when i found this... accidentally. everything said we couldn't train an moe using my type of system, we found a way. then we saw the hardware ceiling limit when running that quantized llm, and we found a way to unify the bus to make it run even better from the way we found to train it... and then even more!" The full timeline, from the first training folder of September 2026 to this repository, is Section 20 of the masterwork.

## What it will be

| Piece | One line |
|---|---|
| probe | Detects CPU, GPU, RAM, drives, OS and ReBAR; measures what matters; uses published values only until it has measured |
| plan | Fits the work under the measured VRAM ceiling and offers two to four plans; recommends a bigger plan when it finds more room than an imported profile assumed (and waits for the user's OK) |
| intake / coach | Imports parameters from json, txt, log, csv, md (and `.py`, parsed, never run); coaches the training set for the clients that will call the model |
| quantize / train / export | Own bit-exact quantizers; layer-major training with the core trained through the exact quantizer; frozen bytes proven unchanged |
| engine + server | The unified bus (GPU hot bank, CPU cold experts, doorbell handoff, dual DRAM readers) behind an OpenAI-compatible server on 127.0.0.1:**8062** |
| web interface | Served by Pelias itself: lithic style, the Pelias logo, the spear progress bar |
| web tools | Search, fetch and read for the model; FlareSolverr as the one optional outside process |
| installers | Windows 10/11, and Linux (WSL first, native capable) |

## Repository map

| Path | Holds |
|---|---|
| `docs/PELIAS_SYSTEM_MASTERWORK.md` | The specification: requirements, product, boundary, detection, models, flow, parameters, training, quantizing, engine, server, web tools, installers, scaffold, order of operations, gates, numbers, conflicts, sources, history, and future hypotheses beyond LLMs (Section 21) |
| `docs/PELIAS_SYSTEM_MASTER.md` | The one-page scaffold and audited numbers of 2026-10-25 (two toolchain rows corrected) |
| `docs/BUILD_NOTES.md` | How to build, per platform (filled as the build proceeds) |
| `docs/inputs/` | The author's input files, unchanged, and `REVIEW.md` |
| `docs/reviews/` | Independent reviews and what was done with each finding |
| `THIRD_PARTY_NOTICES.md` | Every outside work the product or its build relies on, with its licence |
| `docs/scaffold/` | `REPO_INVENTORY.tsv` and `verify_inventory.py` |
| `profiles/reference_pelias/` | Import files for the author's machine and model |
| `profiles/local_only/` | The author's private training profile (ignored by git) |
| `resources/ui/` | Logo, spear stages, baked spear textures |
| `support/ui_reference/` | The lithic interface header of 2026-10-24 (specification for the web interface) |
| `tools/ui/` | The spear texture baker (offline tool) |
| `logs/` | `BUILD_LOG.md`, `DECISIONS_LOG.md` |

## Rules

Every number carries a label (MEASURED, READ-SRC, PUBLISHED, DERIVED, ESTIMATE, UNVERIFIED, DECISION, AUTHOR-STATED). No llama.cpp or other outside code in the product; llama.cpp is an external, test-only comparison. Never truncate training data. The VRAM ceiling is measured at start and enforced. No model downloads. Dataset text, persona and private equations never enter this repository. Nothing is pushed without the author's word.

## Licence

Apache-2.0 (`LICENSE`). Copyright (c) 2026 John Lipka and the Pelias contributors.
