### 1. System Mode, Startup Probe & Roofline Planner (`system.*`, `probe.*`, `plan.*`)

> **Redacted for publication (2026-10-26):** the private model alias and the author's drive paths are replaced by `<model alias>` and `<drive path>`; everything else is word for word. The unredacted copy is kept local-only.

| Canonical Parameter Key | Accepted Aliases (JSON / Legacy CLI) | Type | Allowed Values / Range | Default on Reference (`PELIAS`) | Definition & Architectural Effect |
| --- | --- | --- | --- | --- | --- |
| `system.mode` | `--mode` | Enum | `standard`, `expert` | `standard` | Controls UI/CLI safeguard level; `expert` (`U11`) unlocks the Dynamic Variable Injection Drawer (`+ Add Variable`) and badges manual overrides as `USER-FORCED`. |
| `plan.goal_preset` | `system.preset`, `--preset` | Enum | `balanced`, `max_decode_speed`, `max_context`, `low_vram`, `deterministic` | `balanced` | High-level objective weighting for `plan/roofline_solver` when generating the 2–4 feasible options on the Plan Card (`U3`). |
| `plan.auto_select_default` | `--yes`, `--auto-plan` | Bool | `true`, `false` | `false` | When `false`, pauses at the Plan Card (`U3`) for user confirmation; when `true`, automatically selects Plan 1 if all pre-flight checks pass. |
| `probe.startup_mode` | `--probe-mode` | Enum | `quick_every_start`, `cached_only`, `full_suite`, `skip_forced` | `quick_every_start` | Controls `startup/quick_pass` (`U1`): runs the 5-second check (free VRAM, pinned BW, flush need, burst MMA) on every start and reuses slow probes cached by `machine_state_sha`. |
| `probe.settle_wait` | `SETTLE` | Bool | `true`, `false` | `true` | Waits between probes or stages until hardware free VRAM settles within `probe.vram_tol_mib` of the pre-probe baseline (`U5`). |
| `probe.settle_max_s` | `--settle-timeout` | Int | `5` to `300` s | `60` | Maximum seconds to wait for VRAM to settle before logging a settle warning and proceeding. |
| `probe.vram_tol_mib` | `--vram-tol-mib` | Int | `16` to `512` MiB | `64` | Allowed variance in MiB from baseline free VRAM when checking memory settlement (`P-M1`). |
| `probe.retry_on_fail` | `RETRY_ON_FAIL` | Bool | `true`, `false` | `true` | If a short probe fails from a transient desktop VRAM spike, automatically re-runs once plain and once settled (`U5`). |
| `probe.temp_abort_c` | `--temp-abort` | Int | `70` to `95` C | `87` | Hardware thermal tripwire (`15.3`): immediately pauses/aborts and writes an atomic checkpoint if GPU temperature reaches this value. |

### 2. Hardware Abstraction, OS Isolation & VRAM Governor (`hal.*`, `os.*`, `vram.*`, `host.*`)

| Canonical Parameter Key | Accepted Aliases (JSON / Legacy CLI) | Type | Allowed Values / Range | Default on Reference (`PELIAS`) | Definition & Architectural Effect |
| --- | --- | --- | --- | --- | --- |
| `hal.backend` | `hal.gpu_backend`, `--backend` | Enum | `auto`, `cuda`, `hip`, `sycl`, `vulkan`, `cpu_only` | `cuda` (`auto`) | Selects the Layer 0 GPU compute backend; `auto` selects the highest-capability native backend compiled into the binary. |
| `hal.device_id` | `--device`, `-dev` | Int | `0` to `15` | `0` | Selects the physical GPU device index on multi-GPU hosts. |
| `hal.gpu_arch_override` | `--gpu-arch` | String | `auto`, `sm_89`, `sm_86`, `gfx1100`, `xe_hpg`, etc. | `sm_89` (`auto`) | Overrides the queried GPU compute capability for `kern/` tile-size selection. |
| `vram.ceiling_mode` | `--ceiling-mode` | Enum | `probe_auto`, `whole_card`, `process_only`, `user_forced` | `probe_auto` | Determines how `hal/vram_governor` computes the hard VRAM cap (`probe_auto` uses `min(CUDA_free, NVML_free) - vram.margin_gib` after context creation, `D28`). |
| `vram.ceiling_override_gib` | `vram.ceiling_gib`, `PELIAS_TIER_MAX_MIB` | Float | `2.00` to `128.00` GiB | `10.70` | Hard VRAM ceiling in GiB (`D26`); any memory plan exceeding this limit is refused before the first allocation (`PG3`). |
| `vram.margin_gib` | `--vram-margin` | Float | `0.00` to `4.00` GiB | `0.13` | Safety buffer subtracted from post-context free VRAM (`10.83 - 0.13 = 10.70 GiB`) to absorb desktop/WDDM fluctuations. |
| `vram.governor_check_every_steps` | `vram.governor_check_steps` | Int | `1` to `1000` | `10` | Interval (in training steps or decode requests) at which the governor polls NVML/driver memory and clock telemetry. |
| `vram.scan_propose_close` | `vram.close_apps_prompt` | Bool / Enum | `true` (`ask`), `false` (`off`) | `true` | Scans per-process GPU memory at startup (`startup/vram_scan`, `U6`) and lists non-system apps the user can close to reclaim VRAM. |
| `host.pinned_pool_gib` | `--pinned-gib` | Float | `0.00` (`auto`) to `128.00` GiB | `0.00` (`auto`: `18.65` serve / `28.0` train) | Capacity of the page-locked (`cudaHostAlloc`) host pool holding frozen routed experts, doorbell mailboxes, and layer boundaries. |
| `host.alloc_backend` | `--load-mode` | Enum | `auto`, `host_alloc`, `host_register`, `pageable_staging` | `host_alloc` | Host pinning API; `host_alloc` (`cudaHostAllocMapped`) guarantees identical host and device pointers on WSL2 (`07_pin_modes.cu`). |
| `host.fallback_chain_enabled` | `host.fallback_policy` | Bool / Enum | `true` (`bf16_boundaries`), `false` | `true` | Automatically steps down (`fp32 -> bf16` boundary store `->` smaller step `->` pageable staging ring) on machines with `< 32 GiB` RAM. |
| `os.large_pages` | `--large-pages` | Enum | `off`, `auto`, `madvise_thp`, `win32_large_pages`, `linux_hugetlbfs` | `off` | Requests `2 MiB` huge/large pages for host stores (`off` by default until `P-E11` demonstrates a `>= 5%` end-to-end gain). |
| `os.high_res_timer_win32` | `--win32-hires-timer` | Bool | `true`, `false` | `true` | Calls `timeBeginPeriod(1)` on Windows 10/11 to eliminate the default `15.6 ms` OS timer tick overshoot seen in `P-M3`. |
| `os.linux_dma_latency_zero` | `--dma-latency-zero` | Bool | `true`, `false` | `false` | Holds `/dev/cpu_dma_latency = 0` on Native Linux so CPU cores do not enter deep C-states between layer doorbell handoffs (`G11`). |
| `hal.stream_flush_policy` | `train.stream_flush_query` | Enum | `auto`, `batch_query`, `every_enqueue`, `off` | `batch_query` | Enforces `D27`: calls `cudaStreamQuery` once per multi-ms batch so WDDM dispatches H2D, D2H, and compute in parallel (`0.92` overlap in `P-H12`). |
| `hal.cuda_graphs` | `engine.cuda_graphs` | Enum / Bool | `on` (`true`), `off` (`false`), `auto` | `on` | Captures and replays static spine kernel sequences (`68–85%` kernel launch overhead reduction measured in `P-H9`). |
| `hal.l2_persisting_mib` | `--l2-persist-mib` | Float | `0.0` to `24.75` MiB | `0.0` | Pins a window of the GPU L2 cache (`cudaAccessPropertyPersisting`) for spine activations and DeltaNet recurrent states (`E-A4`). |
| `hal.sync_audit_debug` | `--sync-audit` | Bool | `true`, `false` | `false` | Enables `hal/sync_audit` and `P-M5` hooks to assert stream ownership and zero heap/device allocations inside hot loops. |

### 3. Model I/O, Layout & Hot-Bank Parameters (`model.*`, `io.*`, `bank.*`)

| Canonical Parameter Key | Accepted Aliases (JSON / Legacy CLI) | Type | Allowed Values / Range | Default on Reference (`PELIAS`) | Definition & Architectural Effect |
| --- | --- | --- | --- | --- | --- |
| `model.path` | `-m`, `--model`, `PELIAS_GGUF` | Path | Local `.gguf` file or `.safetensors` directory | *(Required at launch)* | Local path to the model weights; validated by `io/gguf_reader` or `io/safetensors`; never logged to public telemetry. |
| `model.alias` | `--alias`, `model.custom.alias` | String | Any valid identifier | `<model alias>` | Model name returned by `/v1/models` and accepted in `/v1/chat/completions` payloads. |
| `model.arch_plugin` | `--arch` | Enum | `auto`, `qwen35moe`, `qwen3moe`, `gemma4moe`, `deepseek2`, `dense_llm` | `auto` (`qwen35moe`) | Selects the `model/arch_moe` plugin (`E13`); `auto` inspects `general.architecture` in the GGUF header. |
| `model.enable_mtp_blk40` | `--mtp` | Bool | `true`, `false` | `false` (serve) / `true` (ingest) | Loads the 41st multi-token prediction block (`blk.40`, `498 MiB` experts) for speculative decoding (`E12`) or MTP co-training (`D12`). |
| `io.verify_sha256_on_load` | `--verify-hash` | Bool | `true`, `false` | `false` (serve) / `true` (export) | Computes and verifies the full-file SHA-256 model key (`U4`), caching the result by `(path, byte_size, mtime)`. |
| `io.require_expert_2mib_align` | `--require-2mib` | Bool | `true`, `false` | `false` | Requires all 123 expert tensors to start on exact `2 MiB` boundaries (`pelias.pad` layout). |
| `io.warn_slow_mount_gbps` | `--warn-mount-gbps` | Float | `0.10` to `5.00` GB/s | `0.80` | Emits a Plan Card warning if sequential model read speed falls below this floor (catches `0.17–0.52 GB/s` WSL2 `/mnt/c` `9p` mounts). |
| `bank.size_gib` | `--bank-gib`, `PELIAS_TIER_MAX_MIB` | Float | `0.00` (`auto`) to `24.00` GiB | `7.50` | Static VRAM hot-expert bank budget (`7.50 GiB` holds `4,170` experts across 40 layers + `40` all-zero slots = `72.6 MiB`, `h = 0.8273`). |
| `bank.hot_set_file` | `PELIAS_HOT_SET`, `--hot-set` | Path | Path to `PELIAS_HOT_SET` `.txt` profile or `auto` | `profiles/pelias_hot_set_b7.5.txt` | Per-layer list of expert IDs locked in VRAM (`bank/hot_set_io`, byte-identical to `export_hot_set.py` via `P-E14`). |
| `bank.objective` | `--bank-objective` | Enum | `byte_weighted_hit`, `zero_miss_share`, `hybrid` | `byte_weighted_hit` | Optimization target for `bank/hot_set_solver` (`h` vs. zero-miss layer share $Z_l$, informed by `P-E2` regression of $c_0$ vs. $c_1$). |
| `bank.policy` | `--bank-policy` | Enum | `static_waterfill`, `lru_dynamic`, `hybrid_70_30` | `static_waterfill` | Hot-bank residency policy; `static_waterfill` incurs zero runtime H2D copy overhead during token decode. |
| `bank.zero_slot_enabled` | `--zero-slot` | Bool | `true`, `false` | `true` | Reserves one all-zero slot per layer (`72.6 MiB` total) so padded/unrouted tile entries map to a valid index without `-1` sentinel crashes. |

### 4. Engine Dual-Bus, Doorbell Handoff, KV Cache & Prefill Parameters (`handoff.*`, `cpu.*`, `gpu.*`, `kv.*`, `prefill.*`, `engine.*`)

| Canonical Parameter Key | Accepted Aliases (JSON / Legacy CLI) | Type | Allowed Values / Range | Default on Reference (`PELIAS`) | Definition & Architectural Effect |
| --- | --- | --- | --- | --- | --- |
| `handoff.mode` | `--handoff` | Enum | `auto`, `doorbell`, `stream_sync` | `doorbell` | Per-layer GPU<->CPU coordination mechanism: `doorbell` uses shared-memory flags (`13.34 us` median); `stream_sync` uses OS stream waits (`94.4 us`). |
| `handoff.atomic_order` | `--handoff-order` | Enum | `acquire_release` (`acq_rel`), `seq_cst` | `acquire_release` | Memory ordering on doorbell flags (`acquire_release` compiles to `MOV` on x86, avoiding post-`LOCK` FP stalls per AMD SOG §2.15, `G7`). |
| `handoff.timeout_us` | `--handoff-timeout-us` | Int | `100` to `5000000` us | `50000` (`10000`–`50000`) | Microseconds the resident GPU doorbell waits for CPU cold workers before exiting cleanly to host `stream_sync` fallback. |
| `handoff.depth_per_layer` | `--handoff-depth` | Int | `1` to `4` | `1` | Number of cache-line-isolated mailbox slots allocated per layer in `host_pinned`. |
| `cpu.isa_level` | `cpu.simd_isa` | Enum | `auto`, `avx2`, `avx512`, `avx_vnni`, `amx` | `avx2` | Selects the CPU cold-expert SIMD kernel dispatch path (`src/cpu/`). |
| `cpu.threads` | `cpu.threads_decode`, `-t` | Int | `1` to `128` (`0` = auto) | `6` | Number of pinned CPU worker threads executing cold experts during token decode (`5–7` is the measured sweet spot on 5800X3D). |
| `cpu.threads_batch` | `cpu.threads_prefill`, `-tb` | Int | `1` to `128` (`0` = auto) | `8` | Number of CPU worker threads active during prompt prefill batches. |
| `cpu.pin_mode` | `cpu.pin_policy` | Enum | `physical_1t`, `smt_2t`, `pcores_only`, `os_default` | `physical_1t` | Thread affinity policy; `physical_1t` binds one worker per physical core so Zen 3 Load/Store/Retire queues are not split 50/50 (`G6`). |
| `cpu.use_ecores` | `--use-ecores` | Bool | `true`, `false` | `false` | On Intel hybrid CPUs, controls whether efficiency cores (E-cores) are included in the cold-expert worker pool. |
| `cpu.prefetch_nta` | `--prefetch-nta` | Bool | `true`, `false` | `false` | Issues non-temporal prefetch hints (`_mm_prefetch(..., _MM_HINT_NTA)`) on cold expert rows to protect the 96 MB victim L3 cache (`E-A3`). |
| `cpu.poll_level` | `--poll` | Int | `0` to `100` | `50` | Percentage of inter-layer wait time that CPU workers actively spin-poll the doorbell vs. yield. |
| `gpu.cold_zerocopy_branch` | `--zerocopy-cold` | Enum | `off`, `on`, `auto_probe` | `auto_probe` | Enables the third cold branch (`gpu/zerocopy.cu`, `E12`) where the GPU reads a share of cold experts directly over PCIe in parallel with CPU AVX2. |
| `gpu.cold_share_fraction` | `cold.gpu_zerocopy_share` | Float | `0.00` to `1.00` | `0.44` (when `on`) / `0.00` (2-way) | Fraction of cold-expert rows assigned to GPU zero-copy vs. CPU AVX2 (`0.44 = 21.06 / 47.62 GB/s` from `04_dual_bus.cu`). |
| `gpu.q6k_wide_staging` | `--q6k-staging` | Enum | `shared_mem_uint4`, `padded_432b`, `byte_fallback` | `shared_mem_uint4` | Stages `210 B` `Q6_K` blocks into `__shared__` memory via `16-byte` (`uint4`) loads (`P-E10`) to overcome narrow-load PCIe throttling. |
| `kv.context_tokens` | `-c`, `--ctx-size` | Int | `512` to `262144` | `32768` (up to `131072`) | Maximum session context window in tokens allocated in the `session` arena. |
| `kv.type_k` | `kv.type`, `--cache-type-k` | Enum | `q8_0`, `f16`, `bf16`, `q4_0` | `q8_0` | Quantization format for full-attention Key cache (`q8_0` saves `~1.0 GiB` at 131k context with `12/12` needle recall). |
| `kv.type_v` | `kv.type`, `--cache-type-v` | Enum | `q8_0`, `f16`, `bf16`, `q4_0` | `q8_0` | Quantization format for full-attention Value cache. |
| `kv.deltanet_snapshots` | `cache.recurrent_snapshots` | Int | `0` to `16` | `4` | Number of 30-layer DeltaNet recurrent state checkpoints (`62.9 MiB` each) retained in `engine/state` for multi-turn prefix reuse (`~0.5 ms` restore). |
| `prefill.micro_batch` | `-ub`, `--ubatch-size` | Int | `128`, `256`, `512`, `1024`, `2048`, `4096` | `2048` | Physical token chunk size during prompt prefill (`2048` achieves `1,667–1,680 tok/s` at 12k while keeping peak VRAM `<= 10.69 GiB`). |
| `prefill.batch_cap` | `prefill.batch`, `-b` | Int | `512` to `16384` | `4096` | Maximum logical prompt batch size accepted per prefill step. |
| `prefill.bypass_tokens` | `PELIAS_TIER_BYPASS_N` | Int | `1` to `64` | `32` | Batch-size crossover threshold: chunks `< 32` tokens use the dual hot/cold bank path; chunks `>= 32` use the bulk prefill path. |
| `engine.slots` | `-np`, `--parallel` | Int | `1` to `16` | `1` | Number of concurrent session decode slots (`1` dedicates 100% of VRAM and bus bandwidth to the active user). |
| `engine.context_overflow` | `--no-context-shift` | Enum | `stop_error` (`stop`), `truncate_oldest` | `stop_error` | Behaviour when prompt + output reaches `kv.context_tokens`; `stop_error` returns a clean error and never silently evicts prefix tokens. |
| `engine.mtp_draft_max` | `engine.mtp_draft_tokens` | Int | `0` to `4` | `0` (`2` when MTP active) | Number of speculative tokens drafted per step by `blk.40` (`2` lifted 12k decode from `45.4` to `51.5–52.9 tok/s`). |
| `generation.max_tokens` | `-n`, `--n-predict` | Int | `-1` (unlimited) or `1` to `262144` | `-1` | Default maximum generated tokens per request when `max_tokens` is omitted by the client. |

### 5. Native HTTP/SSE Server, OpenAI API, Sampling & Tokenizer (`server.*`, `api.*`, `sampling.*`, `tok.*`)

| Canonical Parameter Key | Accepted Aliases (JSON / Legacy CLI) | Type | Allowed Values / Range | Default on Reference (`PELIAS`) | Definition & Architectural Effect |
| --- | --- | --- | --- | --- | --- |
| `server.bind_address` | `--host` | IP String | Local IPv4/IPv6 address | `127.0.0.1` | Socket bind address in `os/net` (`127.0.0.1` restricts access to local processes). |
| `server.port` | `--port` | Int | `1024` to `65535` | `8012` | Primary TCP port for `/v1/chat/completions` and `/v1/models` (`SO_EXCLUSIVEADDRUSE` enforced on Windows). |
| `server.ollama_port_alias` | `--ollama-port` | Int | `0` (off) or `11434` | `0` | Optional secondary listener translating Ollama API payloads into Pelias `/v1/chat/completions`. |
| `server.max_connections` | `--max-conn` | Int | `1` to `64` | `8` (`16` max) | Maximum simultaneous TCP client connections accepted by `os/net`. |
| `server.queue_max` | `--queue-max` | Int | `1` to `32` | `4` | Maximum pending requests queued while the decode slot is active before returning `HTTP 503`. |
| `server.max_header_bytes` | `--max-header-bytes` | Int | `4096` to `65536` | `16384` | Maximum byte length of HTTP request headers before rejecting with `HTTP 431`. |
| `server.max_body_bytes` | `--max-body-bytes` | Int | `65536` to `134217728` | `33554432` (`32 MiB`) | Maximum byte length of an HTTP request body (sized for 131k-token conversations). |
| `server.read_timeout_s` | `--timeout` | Int | `5` to `600` s | `60` | Socket read timeout in seconds while receiving a request body. |
| `server.idle_timeout_s` | `--idle-timeout` | Int | `10` to `3600` s | `300` | Keep-alive idle timeout in seconds for persistent HTTP/1.1 connections. |
| `server.sse_heartbeat_s` | `--sse-heartbeat` | Int | `0` (off) to `60` s | `15` | Emits an SSE comment (`: ping\n\n`) every `N` seconds during long prompt prefills so IDE clients (VS 2026 Copilot) never time out. |
| `server.api_key` | `--api-key` | String | `""` (none) or secret token | `""` | Optional Bearer token required in the `Authorization` header. |
| `server.cors_allow_origin` | `--cors` | String | `""` (off), `*`, or origin URL | `""` | Configures `Access-Control-Allow-Origin` headers for browser-based local UIs. |
| `api.reasoning_format` | `--reasoning-format` | Enum | `deepseek`, `inline_think`, `strip` | `deepseek` | Controls `<think>...</think>` handling: `deepseek` routes reasoning tokens into a separate JSON field instead of `content`. |
| `api.reasoning_field` | `--reasoning-field` | String | `reasoning_content`, `reasoning` | `reasoning_content` | Exact JSON field name used in streaming deltas and non-streaming responses when `api.reasoning_format = deepseek`. |
| `api.enable_thinking_default` | `template.enable_thinking` | Bool | `true`, `false` | `true` | Default state of `chat_template_kwargs.enable_thinking` when not overridden by the client request. |
| `api.tool_call_parser` | `--tool-parser` | Enum | `qwen_xml`, `json_strict`, `off` | `qwen_xml` | Parses model-emitted `<tool_call><function=NAME><parameter=KEY>VAL</parameter></function></tool_call>` blocks into OpenAI `tool_calls` JSON deltas. |
| `api.special_in_user_content` | `server.special_in_user` | Enum | `pass_counted` (`pass_and_count`), `escape`, `refuse` | `pass_counted` | Policy when untrusted user message text contains literal special token strings like `< |
| `api.log_structure_only` | `--log-structure-only` | Bool | `true`, `false` | `true` | Privacy rail (`15.6`): logs only request structure (roles, tool count, unknown field names, token counts, latencies) and never logs message text. |
| `sampling.temperature` | `--temp` | Float | `0.00` (greedy) to `2.00` | `1.00` | Logit temperature scaling (`1.00` is the Qwen3.6 thinking-mode vendor default; overridable per request). |
| `sampling.top_p` | `--top-p` | Float | `0.01` to `1.00` | `0.95` | Nucleus sampling cumulative probability threshold. |
| `sampling.top_k` | `--top-k` | Int | `0` (off) to `1000` | `20` | Top-K sampling cutoff (`20` is the Qwen3.6 vendor default). |
| `sampling.min_p` | `--min-p` | Float | `0.00` to `1.00` | `0.00` | Minimum probability threshold relative to the highest-probability token. |
| `sampling.repeat_penalty` | `--repeat-penalty` | Float | `1.00` (off) to `2.00` | `1.00` | Multiplicative repetition penalty on recently sampled tokens. |
| `sampling.presence_penalty` | `--presence-penalty` | Float | `-2.00` to `2.00` | `0.00` | Additive presence penalty for tokens appearing at least once in the context. |
| `sampling.frequency_penalty` | `--frequency-penalty` | Float | `-2.00` to `2.00` | `0.00` | Additive frequency penalty proportional to token occurrence count. |
| `sampling.seed` | `-s`, `--seed` | Int64 | `-1` (random) or `0` to `2^63-1` | `20261019` | Counter-based Philox RNG seed for deterministic token sampling. |
| `tok.apply_nfc` | `--tok-nfc` | Enum | `auto_probe`, `on`, `off` | `auto_probe` | Controls Unicode NFC normalization prior to pre-tokenization (`P-D1` locks parity against reference). |
| `tok.bpe_cache_entries` | `--bpe-cache` | Int | `0` to `262144` | `65536` | Capacity of the fast piece cache for frequent pre-tokenized words in `tok/bpe`. |

### 6. Stage A Ingest, Dataset Packer, Layer-Major Trainer, Mode Q/L/B & Exporter (`ingest.*`, `data.*`, `train.*`, `lora.*`, `core.*`, `mode_l.*`, `mode_b.*`, `export.*`, `warn.*`)

| Canonical Parameter Key | Accepted Aliases (JSON / Legacy CLI) | Type | Allowed Values / Range | Default on Reference (`PELIAS`) | Definition & Architectural Effect |
| --- | --- | --- | --- | --- | --- |
| `ingest.source_safetensors_dir` | `--safetensors-dir` | Path | Local directory of 26 `bf16` shards | *(User-supplied)* | Stage A input directory (`71.90 GB`, `1,045` tensors, `D33`); read via `mmap` one tensor at a time. |
| `ingest.output_0step_gguf` | `--base-0step-out` | Path | Local `.gguf` output path | `pelias_base_0step.gguf` | Stage A output file (`753` tensors, `20.22 GiB`), quantized with `quant/quant_cpu` to the stock `Q4_K_M` type map. |
| `ingest.keep_mtp_blk40` | `--keep-mtp` | Bool | `true`, `false` | `true` | Retains `mtp.*` tensors as `blk.40` (`41` blocks total) during Stage A conversion. |
| `ingest.drop_vision_tower` | `--drop-vision` | Bool | `true`, `false` | `true` | Drops the `333` `model.visual.*` tensors (`0.832 GiB`) during Stage A conversion (`3.2.2`). |
| `data.train_jsonl` | `--train-data` | Path | Local `.jsonl` file | *(User-supplied)* | Training conversation dataset (`messages[]` format); audited by `A-LEN` through `A-FAMILY`. |
| `data.eval_jsonl` | `--eval-data` | Path | Local `.jsonl` file or `auto_split` | `auto_split` (`7%`) | Held-out evaluation dataset used for eval loss, overfitting detection (`U9`), and `K2` KL verification. |
| `data.max_seq_len` | `data.max_sample_tokens`, `train.window_bucket_max` | Int | `512` to `16384` (multiple of 64) | `8192` | Hard sequence-length ceiling (Standing Rule 2: **Never truncate**; any row `> 8192` tokens aborts pre-flight audit with row IDs). |
| `data.min_step_tokens` | `--min-step-tokens` | Int | `512` to `32768` | `3072` | Minimum packed tokens per optimizer step enforced by `data/packer` so the fixed `1.49 s` PCIe expert stream is 100% hidden (`F7`). |
| `data.target_step_tokens` | `train.step_tokens_target` | Int | `2048` to `65536` | `16384` (`4900` v0.2 parity) | Target processed tokens packed across $K$ windows ($T \le 8,192$, aligned to 64-token DeltaNet chunks) per optimizer step. |
| `data.train_on_turns` | `--train-turns` | Enum | `all_assistant`, `last_assistant_only` | `all_assistant` | Determines which assistant spans receive loss weight $w_t = 1$ (verified by independent marker parser `P-D5`). |
| `data.blank_param_policy` | `--blank-policy` | Enum | `knowledge_only`, `base_model_draft` | `knowledge_only` | Policy for blank answers in `parameters.csv` (`O3`: builds specification/cross-reference rows without inventing opinions). |
| `train.forward_mode` | `--forward-mode` | Enum | `auto_probe` (`auto`), `mode_f_int8`, `mode_r_bf16` | `auto_probe` | Selects `mode_f_int8` (serve-faithful `q8_1` + `int8` MMA) vs. `mode_r_bf16` (`D6`, decided by Batch C probes `P-K2` & `P-Q6`). |
| `train.deterministic` | `train.determinism` | Bool / Enum | `true` (`deterministic`), `false` | `true` | Enforces fixed-order tree reductions (`R3`, `O2`, `O3`, zero float atomics) and repeats step 1 to verify bitwise reproducibility (`P-T4`). |
| `train.epochs` | `--epochs` | Int | `1` to `20` | `2` | Number of full training passes over `data.train_jsonl`. |
| `train.seed` | `--train-seed` | Int64 | `0` to `2^63-1` | `3407` | Master RNG seed for Kaiming LoRA initialization and deterministic dataset window shuffling. |
| `train.lr` | `optim.lr`, `--lr` | Float | `1e-6` to `1e-3` | `0.0001` (`0.00005` phase 2) | Peak learning rate for fused `fp32` AdamW (`O1`). |
| `train.warmup_frac` | `optim.warmup_ratio` | Float | `0.00` to `0.50` | `0.05` | Fraction of total optimizer steps spent in linear learning-rate warmup before cosine decay. |
| `train.adam_beta1` | `optim.betas[0]` | Float | `0.80` to `0.99` | `0.90` | AdamW first-moment exponential decay rate $\beta_1$. |
| `train.adam_beta2` | `optim.betas[1]` | Float | `0.90` to `0.9999` | `0.999` | AdamW second-moment exponential decay rate $\beta_2$. |
| `train.adam_eps` | `optim.eps` | Float | `1e-10` to `1e-5` | `1e-8` | AdamW denominator stability epsilon $\epsilon$. |
| `train.weight_decay` | `optim.weight_decay` | Float | `0.00` to `0.10` | `0.00` | Decoupled weight decay (automatically forced to `0.00` on norms, gates, and biases). |
| `train.grad_clip_norm` | `optim.grad_clip` | Float | `0.10` to `10.00` | `1.00` | Global `fp32` gradient L2-norm clipping threshold (`O2`). |
| `train.ce_chunk_positions` | `--ce-chunk` | Int | `64`, `128`, `256`, `512` | `256` | Active-loss token positions materialized per chunk in `kern/loss/ce_chunk.cu` (`256 x 248,320 x 4 B = 242.5 MiB`). |
| `train.boundary_dtype` | `--boundary-dtype` | Enum | `fp32`, `bf16` | `fp32` | Data type of the pinned host layer-boundary activation store (`5.0 GiB` at 16k step tokens in `fp32`; verified by `P-Q5`). |
| `train.expert_vram_slots` | `train.expert_slots_vram` | Int | `2`, `3`, `4` | `2` (`2 x 498 MiB`) | Number of device VRAM staging slots used by `bus/stage_streamer` to prefetch expert layers on stream `S_w`. |
| `train.canary_every_steps` | `train.canary_interval_steps` | Int | `10` to `500` | `50` | Executes `verify/canary` every `N` optimizer steps to catch silent overclock bit-flips on non-ECC GDDR6X (`15.3`). |
| `lora.rank` | `-r`, `--lora-rank` | Int | `4`, `8`, `16`, `32`, `64` | `16` | LoRA rank $r$ across the 310 core modules (`r = 16` -> `21,166,080` trainable params, `323.0 MiB` `fp32` state). |
| `lora.alpha` | `--lora-alpha` | Float | `1.0` to `128.0` | `32.0` | LoRA scaling numerator ($\text{scale} = \alpha / r = 2.0$, or $\alpha / \sqrt{r}$ when `lora.rslora = true`). |
| `lora.rslora` | `--rslora` | Bool | `true`, `false` | `false` | Enables rank-stabilized LoRA scaling $\gamma_r = \alpha / \sqrt{r}$ (`arXiv:2312.03732`, ablation `T-A3`). |
| `lora.targets` | `--lora-targets` | String | Comma list of `attn,deltanet,shexp` | `attn,deltanet,shexp` | Target module families receiving LoRA adapters (`40` attn + `150` DeltaNet + `120` shared expert = `310` modules). |
| `lora.train_shared_gates` | `train.trainable_gates_norms` | Bool | `true`, `false` | `false` | Unfreezes the 40 `F32` shared-expert scalar gate vectors (`ffn_gate_inp_shexp`, `81,920` params, ablation `T-A5`). |
| `lora.train_norms` | `train.trainable_gates_norms` | Bool | `true`, `false` | `false` | Unfreezes the `F32` RMSNorm weights in GGUF `+1`-folded form (`~175k` params, ablation `T-A5`). |
| `lora.train_lm_head` | `--train-lm-head` | Bool | `true`, `false` | `false` | Attaches a LoRA adapter to `output.weight` (`Q6_K`, ablation `T-A8`). |
| `lora.mtp_lambda` | `--mtp-lambda` | Float | `0.00` (off) to `1.00` | `0.00` (`0.10` when active) | Co-training loss weight $\lambda_{\text{mtp}}$ on `blk.40` (`D12`, gated on `P-S5` and `P-T8`). |
| `core.export_mode` | `train.core_mode` | Enum | `mode_q_served` (`mode_q`), `byte_budget_solver`, `adapter_only` | `mode_q_served` | **Default `mode_q_served` (`D31`):** trains through the exact `ggml` quantizer (`GQ`) and exports at native served types (`+0 B`). |
| `core.delta_max_mib` | `export.delta_max_mib` | Float | `0.0` to `2048.0` MiB | `0.0` | Extra VRAM budget ($\Delta_{\max}$) permitted when `core.export_mode = byte_budget_solver` (`D22`/`D32` Expert override). |
| `core.promote_class_overrides` | `--promote-classes` | String | `shexp=Q6_K,attn=Q6_K,ssm_out=Q6_K,...` | `""` (`+0 MiB`) | Manual per-class precision promotion in Expert mode (`shexp +25.7`, `attn +65.5`, `ssm_out +61.9`, `ssm_in +127.8`, `alpha_beta +1.0 MiB` for `Q6_K`). |
| `mode_l.enabled` | `--mode-l` | Bool | `true`, `false` | `false` | Unfreezes the 41 `F32` routers (`21.5 M` params, `+0.32 GiB`) for locality-aware router adaptation (`D21`, gated on `G-L1`–`G-L4`). |
| `mode_l.lr_router` | `--lr-router` | Float | `1e-6` to `1e-4` | `0.00002` (`2e-5`) | Router learning rate during Mode L continuation. |
| `mode_l.lambda_res` | `--lambda-res` | Float | `0.00` to `0.50` | `0.02` | Loss weight on bank-residency log-mass $L_{\text{res}}$. |
| `mode_l.lambda_trust` | `--lambda-trust` | Float | `0.01` to `5.00` | `0.45` | Loss weight on the 256-way router KL trust anchor $\text{KL}(p \parallel p^0)$. |
| `mode_l.zero_miss_hinge` | `--mode-lz` | Bool | `true`, `false` | `false` | Enables the zero-miss margin hinge loss $L_{\text{zm}}$ (`Mode L-Z`, `16.5 B1.2`). |
| `mode_b.bimodal_forward` | `--mode-b` | Bool | `true`, `false` | `false` | Applies `q8_1` (32-elem) activation quantization to hot-bank experts and `Q8_K` (256-elem) to cold experts during training (`D24`, gated on `P-E12`). |
| `export.layout` | `--layout` | Enum | `expert_last_2mib`, `natural_base` | `expert_last_2mib` | Places all 123 expert tensors after `pelias.pad` so every expert starts on an exact `2 MiB` boundary at zero byte overhead (`F12`). |
| `export.write_adapter_gguf` | `--write-adapter` | Bool | `true`, `false` | `true` | Emits unmerged `adapter.gguf` alongside `merged-core.gguf`. |
| `export.write_hot_set` | `--write-hot-set` | Bool | `true`, `false` | `true` | Emits a re-profiled `PELIAS_HOT_SET` and `layout_manifest.json` from the trained model's routing tape (`D25`). |
| `warn.overfit_eval_gap_ratio` | `--warn-overfit-ratio` | Float | `1.05` to `2.00` | `1.25` | Triggers a mid-training overfitting caution (`warn/fit`, `U9`) if eval loss exceeds `ratio * best_eval_loss` while train loss falls. |
| `warn.loop_ngram_max_repeat` | `--warn-loop-repeats` | Int | `3` to `20` | `5` | Flags a post-training test generation as `LOOP` (`warn/loops`, `U9`) if any 8-gram repeats more than `N` times. |

### 7. Dynamic Per-Layer & Per-Class Template Overrides (`[ + Add Variable ]` Drawer, `U11`)

In Expert Mode (`system.mode = expert`), the Dynamic Variable Drawer also accepts indexed pattern keys for fine-grained hardware and layer tuning:

* **`bank.layer_capacity.<0..39>` (`Int`, `0` to `256`):** Forces the exact number of resident hot-bank slots allocated to trunk layer `l`.
* **`mode_l.layer_weight.<0..39>` (`Float`, `0.0` to `10.0`):** Sets the per-layer locality weight $w_l$ in Mode L / Mode L-Z from `P-E2` regression results.
* **`core.class_type.<shexp|attn|ssm_in|ssm_out|alpha_beta>` (`Enum`, `served|Q4_K|Q6_K|Q8_0`):** Overrides the exported quantization type of an individual core tensor class.
* **`lora.layer_rank.<0..39>` (`Int`, `0, 4, 8, 16, 32, 64`):** Overrides the LoRA rank $r$ on a specific trunk layer (`0` freezes that layer's core projections).