# Community benchmark on RX 6700 XT 12 GiB

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: one RX 6700 XT, nominal 12 GiB VRAM, GPU index 0.
- CPU: Ryzen 5 3600, 6 cores / 12 host logical threads; physical RAM: 64 GiB DDR4-3200. The LXC allocation was 56 GiB with 10 visible CPU threads. Storage: Intel 660p 1 TB NVMe.
- OS: Debian 13 (trixie), kernel 7.0.14-26-pve; driver: in-kernel amdgpu, module source version B1D135B579C294611EB792D; ROCm SDK 7.13.0a20260515. CUDA was not used.
- PolyStrata v0.1.45-poly.9, source 83a7a33272413bf87b8d2742dc557c165e2cb2d2; native gfx1031 HIP build, embedded engine version 0.1.45.
- Other GPU containers and inference services were stopped; a lightweight resource sampler ran outside the guest. GPU power cap: 211 W; measured PCIe link: 16 GT/s x16.

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q3_k_m
- Context: 65536 tokens; parallel slots: 1
- Settings: the attached benchmark configuration; thinking off and temperature 0 for the suite. Conversation cache 0, 8 CPU workers and 1024 MiB VRAM reserve were deliberately selected. Source defaults apply where flags are omitted; these are not claimed as untouched installation defaults.

```text
strata-poly --serve --model Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --kv-whole --max-context 65536 --expert-profile qwen36-ud-q3_k_m-official.txt --conversation-cache-mib 0 --threads 8 --vram-reserve-mib 1024
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a PolyStrata server (started by the benchmark from the attached benchmark configuration); model loading is not timed. One warm-up chat, two word/token-ratio calibration probes and a 1K warm-up prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 90.4 (75.9-101.8) | 0.34 (0.32-0.37) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1236.4 (1211.4-1239.7) | not measured | 3.27 (3.26-3.33) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1109.7 (1101.9-1111.5) | not measured | 14.51 (14.46-14.61) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 858.8 (858.6-859.1) | not measured | 37.37 (37.35-37.38) |
| agent session, follow-up turns | 14999 (14995-15002) at the last turn | see below | up to 64 | 3 | see below | 90.2 (73.4-98.4) | 4.05 (2.70-4.68) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.6); earlier-turn tokens read again per session: 242 (238-245) (earlier turns that the cache did not hold; the 14183 (14183-14183) new tokens of the files are read once); first turn (nothing cached) first token: 1.13 s (1.11-1.13).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 555.2 | 1.132 | 98.4 |
| 1 | 4058 | 602 | 3456 | 960.0 | 3.615 | 82.3 |
| 2 | 5693 | 4051 | 1642 | 610.4 | 2.713 | 77.4 |
| 3 | 9110 | 5686 | 3424 | 783.9 | 4.39 | 89.7 |
| 4 | 12072 | 9103 | 2969 | 736.3 | 4.055 | 93.2 |
| 5 | 15002 | 12065 | 2937 | 629.9 | 4.684 | 81.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. The raw harness machine fields are preserved in results.json. The hardware/software facts above supplement those fields with verified host and package provenance; guest memory is distinct from installed physical RAM. Total time of the suite: 265 s.

Verified settings, immutable model identities, full logs and resource timing limits: [recording checklist](../../WHAT_TO_RECORD.md). The descriptive corrections above do not alter results.json or the original retained harness output.

Sanitized replay from the pinned source checkout (replace the configuration path placeholders with local paths; choose a new output directory):

```text
python tools/strata_bench.py --config bench/results/2026-11-10-community-rx6700xt-poly9/qwen36-ud-q3_k_m/published-config.json --long --out <new-output-directory>
```
