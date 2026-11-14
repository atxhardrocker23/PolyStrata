# Community benchmark on NVIDIA GeForce RTX 2080

Measured on 2026-11-14 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 2080, 8 GiB VRAM
- CPU: Intel(R) Core(TM) i3-10100 CPU @ 3.60GHz (4 cores, 8 threads); installed RAM: 31.2 GiB; storage: not measured
- OS: Linux 7.3.4-arch1-1; GPU driver: 620.70.06; CUDA: not read
- PolyStrata: project 0.1.45, commit not a git checkout; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: gemma-4-26b-a4b-it-ud-q4_k_xl
- Context: 17000 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf --max-context 17000 --expert-profile expert-profile.txt --draft-model mtp-gemma-4-26B-A4B-it-Q8_0.gguf --vram-reserve-mib 500
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a PolyStrata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 48 (45-53) | 0 | 100 (100-100) | 9 | not measured | 46.6 (43.7-63.2) | 0.57 (0.49-0.63) |
| 4K prompt, fresh | 4101 (4101-4101) | 0 (0-0) | 24 cap | 3 | 1244.5 (1237.4-1249.0) | not measured | 3.30 (3.29-3.39) |
| 16K prompt, fresh | 16390 (16390-16390) | 0 (0-0) | 24 cap | 3 | 1878.1 (1875.1-1891.6) | not measured | 8.76 (8.74-9.09) |
| agent session, follow-up turns | 15278 (15276-15281) at the last turn | see below | up to 64 | 3 | see below | 48.7 (38.3-61.2) | 3.36 (2.72-3.55) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 67.9 % (67.9-67.9); earlier-turn tokens read again per session: 259 (257-262) (earlier turns that the cache did not hold; the 14658 (14658-14658) new tokens of the files are read once); first turn (nothing cached) first token: 1.14 s (1.12-1.15).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 349.1 | 1.148 | 39.0 |
| 1 | 4014 | 389 | 3625 | 1076.9 | 3.389 | 38.3 |
| 2 | 5722 | 4007 | 1715 | 628.9 | 2.744 | 46.7 |
| 3 | 9217 | 5715 | 3502 | 999.4 | 3.538 | 46.1 |
| 4 | 12235 | 9210 | 3025 | 930.0 | 3.292 | 55.3 |
| 5 | 15281 | 12228 | 3053 | 866.5 | 3.551 | 45.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 139 s.
