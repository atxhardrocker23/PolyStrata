# Recording checklist and additional provenance

This crosswalk follows [What to record](../../../docs/COMMUNITY_BENCHMARKS.md#what-to-record). The original harness JSON and numerical measurements are unchanged. Missing fields in those original files remain null; supplementary observations below do not retroactively replace them.

| Guide item | Recorded evidence and limits |
| --- | --- |
| Hardware | Overview README: one RX 6700 XT, physical 64 GiB DDR4-3200, Ryzen 5 3600 (6 cores / 12 host logical threads), 56 GiB LXC / 10 visible CPU threads, Intel 660p NVMe, PCIe 16 GT/s x16, 211 W cap, other GPU containers stopped and lightweight resource sampler. |
| Software | Exact poly.9 source commit and native gfx1031 engine hash in every provenance.json; source-built HIP, no engine patches. environment-provenance.json adds the actual ROCm wheel and compiler versions, kernel amdgpu module source version and build metadata. An independent amdgpu driver release number was not available. |
| Model | Main/draft filenames, byte sizes and full SHA256 in every provenance.json; tokenizer file hashes there too. Immutable installer-catalog repository/revision mappings below. Files were reused and hash-verified; a separate original historical download revision was not recorded. No vision encoder was loaded. |
| Packs and initial state | Each official stage used a distinct initially absent expert-profile path, checked before startup in initial-expert-profile.json. The adaptive profile produced after answers is not an initial seed. Tokenizer paths/settings are in each published-config.json; no custom weight pack was introduced. |
| Settings | Exact sanitized configs plus full official engine logs. All use context 65536, --kv-whole, 8 CPU workers, 1024 MiB VRAM reserve, conversation cache 0, one sequential client. Prompt batch is the tested native default 1024, with no override. Expert VRAM cache is selected automatically subject to the 1024 MiB reserve; low-RAM mode was not forced and the native RAM-budget policy was retained. HIP KV format is fp16 by the tested source default; no --kv override was supplied. No KV streaming/resident-window override was supplied; architectural attention behavior remains the model's. |
| Drafting and approximation | Drafting is enabled by the engine default and confirmed by offered/accepted draft counters. Qwen and Tiel use their integrated draft block; Gemma uses the separately hashed Q8 draft GGUF. No --window override: the exact resolved speculation window was not exposed in captured API metadata. The tested native Options default tail_skip is 0; Q4_K_M explicitly passes 0, the other configs omit it, and none of the five full startup logs prints a positive tail-skip or skip-weight setting. No engine calibration or experimental speed-projection stage/flag was requested; the upstream client does its own two word/token-ratio calibration probes. No batch-MTP was requested. |
| Sampling/reasoning | The unmodified quick client requests temperature 0 and thinking off. Configs record reasoning effort none and reasoning budget 512; the suite's thinking-off request policy applies. Other sampler values were not overridden by this campaign. |
| Workload | Unmodified tools/strata_bench.py at the pinned source/harness hash: --long, default 3 repetitions, 9 stored chat + 9 fresh-prompt + 18 agent requests per configuration. The pinned source supplies the prompts and repository-file agent corpus. All actual input/output/reused/read counts, engine durations, throughput, client TTFT/total latency, draft counts and errors are in results.json. |
| Warm-up/cache/timing | A fresh engine per configuration, upstream discarded chat/1K warm-ups plus two word/token calibration probes, then evolving expert placement. Fresh-prompt reuse is 0; agent prefixes deliberately reuse tokens. This is not a cold-expert-cache test. Loading is outside throughput timings; sampled resource windows include loading/warm-up/shutdown. Full timing lines are attached below. |
| Memory/failures | sampled-official-resources.json gives per-suite sampled VRAM ranges and host resource data at nominal 10-second cadence plus measurement overhead. These are sampled maxima, not continuous peaks; host available RAM is not process RSS. Inference process RAM peak was not measured. All 180 stored official requests succeeded, zero skipped. Guest cgroup counters inspected after the official suites and later private work were oom=0 / oom_kill=0. Host swap usage in each official window stayed at 12288 bytes; per-process paging was not measured. |

## Software versions

The installed ROCm packages were **7.13.0a20260515** (`rocm`, `rocm-sdk-core`, `rocm-sdk-devel`, `rocm-sdk-libraries-gfx103X-all`). Compiler: AMD clang 23.0.0git; its full build identity is in environment-provenance.json. Driver: kernel amdgpu, kernel 7.0.14-26-pve, module source version B1D135B579C294611EB792D. These package/source observations were collected read-only after the official suites from the unchanged installed environment; the benchmark harness's original null fields are retained.

Native build: STRATA_ENABLE_HIP=ON, STRATA_ENABLE_CUDA=OFF, STRATA_BUILD_TESTS=OFF, CMAKE_HIP_ARCHITECTURES=gfx1031, existing ROCm SDK compiler/libraries; no changed engine source. Build parallelism is not an inference setting.

## Immutable model catalog pins

- `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` at `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`.
- `unsloth/gemma-4-26B-A4B-it-GGUF` at `c099eb48e663fd284577b04978a94ffccb261841`.
- `peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP` at `bbe9e566f39e4fc9652ac66b71968289a03c520a`.

The Qwen repository covers all three Qwen quants; the Gemma repository covers the main and separate Q8 draft. The actual file hashes in each per-configuration provenance.json are the artifact identity used in this report.

## Full official timing logs

- [qwen36-ud-q3_k_m](qwen36-ud-q3_k_m-full-official-engine.log): 40 timing lines (36 retained suite requests plus 2 upstream warm-ups and 2 token-ratio calibration probes).
- [qwen36-ud-q4_k_m](qwen36-ud-q4_k_m-full-official-engine.log): 40 timing lines (36 retained suite requests plus 2 upstream warm-ups and 2 token-ratio calibration probes).
- [gemma4-ud-q4_k_m](gemma4-ud-q4_k_m-full-official-engine.log): 40 timing lines (36 retained suite requests plus 2 upstream warm-ups and 2 token-ratio calibration probes).
- [tiel-ud-q3_k_xl](tiel-ud-q3_k_xl-full-official-engine.log): 40 timing lines (36 retained suite requests plus 2 upstream warm-ups and 2 token-ratio calibration probes).
- [qwen36-q4xl-cross](qwen36-q4xl-cross-full-official-engine.log): 40 timing lines (36 retained suite requests plus 2 upstream warm-ups and 2 token-ratio calibration probes).

## Post-suite cache snapshots

The original harness collects machine/API facts again after the suite. These are post-suite snapshots, not startup snapshots or inference peaks.

| Configuration | Expert slots | Expert cache MiB | Reported free VRAM MiB |
| --- | ---: | ---: | ---: |
| qwen36-ud-q3_k_m | 4764 | 6355 | 1023 |
| qwen36-ud-q4_k_m | 3479 | 6340 | 1023 |
| gemma4-ud-q4_k_m | 1509 | 5379 | 1013 |
| tiel-ud-q3_k_xl | 4789 | 6389 | 1023 |
| qwen36-q4xl-cross | 3367 | 6157 | 1023 |
