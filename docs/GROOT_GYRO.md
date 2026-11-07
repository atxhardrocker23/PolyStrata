# Groot-65B-A6B GyroM (TQ2_T) on PolyStrata's Strata engine

Branch `turbo-gyro`: PolyStrata (with `--kv turbo` for strata-poly) plus the Gyro trellis expert types ported from
[Strata](https://github.com/Niko1221/Strata), so the Qwen engine (`strata`) runs
[BoxyFit/Groot-65B-A6B-GyroM](https://huggingface.co/BoxyFit/Groot-65B-A6B-GyroM) (`general.architecture qwen4exp`,
256 routed experts, TQ2_T = ggml type 144 experts).

`strata-poly` does not run qwen4exp models ("is a 'qwen4exp' model, which this engine does not run"); Groot is a
256-expert Qwen3.8-Flash-Next, which is what the `strata` engine is built for (pruned variants since 8bef4161).

## What is ported

From Strata, cherry-picked with `-x` (the original commit is named in each message):

- APR port stages 1-4: the trellis type tables (TQ2_T / TQK6 / TQK7), Hadamard-folded experts on the CPU and the
  GPU paths, TQ row bytes, decode-once verify kernels, the MMQ prompt path for TQ types, the published Gyro files'
  Q6_K embedding and Q8_0 PLE table, the multi-GPU prompt offload for folded packs.
- The CPU trellis kernels (AVX2 / AVX-512 / scalar) and their parity tests.
- The trellis MMQ overlay (`cmake/strata_ggml_tq_overlay.cmake`, `third_party/ggml-tq/`), so the prompt kernels
  build on an upstream llama.cpp.

Not ported: Strata 0.2.2's faster trellis decode kernels (5f81c5dc4: 8-lane rows, codebook in shared memory). They
depend on kernel changes PolyStrata does not have (SplitK, the IQ3_XXS staging, a 3-parameter down kernel).

## Groot's MTP head

GrootTeam ships the MTP layer as a llama.cpp GGUF
(`mtp-Groot-65B-A6B-261023-BF16.gguf` in GrootTeam/Groot-65B-A6B-261023-GGUF), not as checkpoint tensors
`tools/mtp_fetch.py` can read. `tools/mtp_rt_gguf.py` undoes that converter's merges (`eh_proj` -> `fc_embedding` +
`fc_hidden`, `indexer.q_proj` + `k_proj` -> `index_qk_proj`, gate + up -> `gate_up_proj`) and writes the engine's
`--mtp` directory with the head's own 256 experts. The engine takes the draft layer's expert count from
`experts.bin`'s size (`src/program/generate.cpp`); a 512-expert file keeps the canonical geometry as before.

Check: the nine RMSNorm weights of the converted head equal the ones `tools/mtp_rt.py` writes for Qwen3.8-Flash-Next's
own head (correlation 1.000, same means), so the GGUF's `1 + w` norms are the engine's convention.

## Running

```
python tools/iq_pack.py --gguf Groot-65B-A6B-GyroM-Q2_0.gguf --out packs/groot --compat-bf16
python tools/mtp_rt_gguf.py --gguf mtp-Groot-65B-A6B-261023-BF16.gguf --out mtp/groot/rt
strata --serve --pack packs/groot --native Groot-65B-A6B-GyroM-Q2_0.gguf --ple-gguf Groot-65B-A6B-GyroM-Q2_0.gguf \
       --expert-profile data/expert-profile-groot.bin --mtp mtp/groot/rt --spec 4 --spec-min-p 0.5 \
       --prefill auto --expert-cache auto --mmap-experts --kv k8v4 --max-context 16384
```

A native pack needs `--spec 2` or more. Verify needs an expert profile: `data/expert-profile-groot.bin` comes from
Groot's own routing traces (`tools/collect_profile_traces.py --mode routing`, 12 prompts of
`bench/profile_corpus.json`, then `tools/make_profile.py --no-base --reorder --n-expert 256`). The 512-expert
`data/expert-profile.bin` does not fit a 256-expert model.

## Measured

RTX 5050 Laptop (8 GB, sm_120), Core with AVX2 (no AVX-512), 32 GB RAM, Windows 11, NVMe, 2026-11-06.
`--expert-profile data/expert-profile-groot.bin --prefill auto --expert-cache auto --mmap-experts --spec 4
--kv k8v4 --max-context 16384`; one-shot runs (a 64-token warm-up run, then the measured run: a 2,273-token prompt,
200 tokens greedy). "upstream" is the Strata 0.1.52 engine package (Windows, CUDA 13) on its own iq_pack.py pack;
upstream cannot load a 256-expert MTP head, so it drafts with Qwen3.8-Flash-Next's 512-expert head.

| engine | MTP head | decode tok/s | prefill tok/s | drafts accepted |
|---|---|---|---|---|
| upstream 0.1.52 | Flash-Next (512) | 8.82, 9.68 | 209, 232 | 0.86, 0.83 |
| this branch | Flash-Next (512) | 6.78 | 200 | 0.78 |
| this branch | Groot (256, mtp_rt_gguf.py) | 9.43, 9.45 | 283, 306 | 0.76, 0.72 |
| upstream 0.1.52 | none | 6.49 | 349 | - |
| this branch | none | 6.21 | 329 | - |

Two numbers: two runs in opposite orders (the OS file cache warms over a session). With the same draft head this
branch decodes 23-30% slower than upstream 0.1.52, which has the 0.2.2 trellis decode kernels (5f81c5dc4, not
ported); with Groot's own 256-expert head, whose experts take half the VRAM of the 512-expert one (338 vs 708 MiB),
the cache keeps more experts and this branch matches upstream's decode and reads prompts faster. The greedy text
with Groot's head matched the run without MTP.

Server (`serve/server.py --engine strata`, Groot's head, 4 requests of ~30 prompt tokens and 400 generated,
temperature 0): 13.3, 13.5, 12.1, 14.0 tok/s decode; decode expert cache hit rate 54-58% (1189 slots, 1.45 GiB);
416 MiB of VRAM free with everything loaded (the engine warns about WDDM paging below 1 GiB).

For scale, Qwen3.8-Flash-Next Gyro-S on upstream 0.1.52 with this PC's tuned config (`--kv q4_0 --expert-cache 1000
--spec 2`, STRATA_KVA=1) in the same one-shot protocol: 11.6 tok/s decode, 96 tok/s prefill.

Not tested: other GPUs, contexts above 16K, images (GrootTeam ships an mmproj; not wired here), a layer split,
tuning (`--kv q4_0`, a fixed `--expert-cache`, `--vram-reserve-mib`).
