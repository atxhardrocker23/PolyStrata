# PolyStrata: RTX 3060 12GB (Unlocked) & Hybrid CPU-GPU Tuning Guide

This repository contains custom performance-engineered branches and compiled binaries of **PolyStrata** specifically optimized for **NVIDIA GeForce RTX 3060 12GB** (both stock and power/clock-unlocked) paired with modern multi-core CPUs (e.g. Intel Core i5-12400 6 P-Cores) and 32GB+ system RAM, maintaining a full **128K (`131072`) context window**.

---

## 🚀 Benchmark Highlights (RTX 3060 12GB Unlocked + i5-12400 + 32GB RAM)

| Model | Quantization | Context Window | Prefill Speed | Sustained Decode | VRAM Cache Hit Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Flint 40B-A2.5B** | `UD-Q3_K_XL` (18.5 GB) | **128,000 (131072)** | **1,200+ tok/s** | **60–85+ tok/s** | **75.0%** (5,498 / 7,332 hits) |
| **Cyber-Tiel-Coder 35B-A3B** | `UD-Q4_K_M` (22.3 GB) | **128,000 (131072)** | **900–1,100 tok/s** | **45–65+ tok/s** | **80.8%** (8,139 / 10,076 hits) |
| **Qwen 3.6 35B-A3B Heretic** | `i1-Q6_K` (25.1 GB) | **128,000 (131072)** | **850–1,050 tok/s** | **40–55+ tok/s** | **78.5%** |
| **Gemma 4 26B-A4B IT Abliterated** | `Q6_K` (21.4 GB) | **128,000 (131072)** | **950–1,150 tok/s** | **45–60+ tok/s** | **76.2%** |

---

## 🛠️ The 6 Custom Optimizations Integrated

1. **Cache-Aware Top-k Expert Routing (`Issue #81`, `--cache-bias 0.35`)**:
   - Adds an adjustable selection bonus (`+0.350`) to VRAM-resident experts during top-$k$ gating while calculating combine weights from the **unbiased** original logits.
   - Result: MoE routing heavily prefers experts already warm in your 12GB VRAM, jumping VRAM cache hit rates to **75–81%** even though only 42–52% of total experts physically fit in VRAM!
2. **Warp-per-Row-Group 32-Expert Batch Launch (`PR #94`, `STRATA_SLOTS_WARP=1`)**:
   - Custom CUDA kernel (`native_mmvq_slots_warp_kernel`) replaces 8-slot limit with 32 slots (`kMmvqSlotsMax = 32`). Cuts kernel dispatch overhead by $4\times$.
3. **Non-Temporal Streaming CPU Stores (`PR #99`, `STRATA_POLY_STREAM_STAGE=1`)**:
   - Uses `_mm_stream_si128` + `_mm_sfence()` when staging CPU experts into pinned RAM. Bypasses the CPU cache hierarchy, keeping the Intel L3 cache clean for execution threads.
4. **AVX2 Row Prefetching & Coarser Task Boundaries (`PR #36`)**:
   - Prefetches 2 rows ahead (`_MM_HINT_T0`) across all CPU AVX2 expert loops and sets `kRowsPerDownTask = 32` to halve atomic worker queue contention.
5. **Zero-Copy File-View Expert Pinning (`PR #35` + Custom WDDM Page-Merge Fix)**:
   - Registers memory-mapped GGUF file views directly with CUDA (`cudaHostRegister`), eliminating redundant copies. Includes a custom fix merging adjacent 32-byte-aligned tensor intervals on 4096-byte boundaries and a 2048 MiB Windows WDDM aperture safety cap.
6. **Per-Model Thinking Effort & Anti-Starvation Switching (`KonnectSuite` + `knightb8`)**:
   - Exposes dynamic reasoning effort levels (`off`, `low`, `medium`, `high`) in OpenAI/Anthropic APIs and drains up to 20 queued requests before triggering a model switch.

---

## ⚡ Hardware & GPU Unlock Configuration

### GPU Profile (RTX 3060 12GB)
- **VRAM**: 12,288 MB GDDR6 (192-bit bus)
- **Power Limit / VBIOS**: GPU Unlock enabled (expanded power headroom, stable sustained boost clocks above 1950–2050 MHz without thermal throttling).
- **Driver**: NVIDIA Game Ready / Studio 550+ with CUDA 13.0 / 12.x support.
- **VRAM Allocation Scheme**:
  - KV Cache: `q8_0` (or `int8` for Gemma 4) at 128k context consumes ~4.2–4.5 GB.
  - Model Static/Dense Layers: ~1.8–2.2 GB.
  - Dynamic MoE Expert Cache: ~5.5–6.0 GB in VRAM (holds ~4,300 to 6,200 active routed experts).
  - VRAM Reserve: 450 MiB safety margin.

### CPU & RAM Profile
- **CPU**: Intel Core i5-12400 (6 Performance Cores / 12 Threads). Thread affinity locked to 6 physical cores (`OMP_NUM_THREADS=6`, `OMP_PROC_BIND=TRUE`, `OMP_PLACES=cores`).
- **RAM**: 32 GB DDR4.
- **Large Pages (`SeLockMemoryPrivilege`)**: Enabled via `ENABLE_LARGE_PAGES.bat` to lock pinned staging buffers in physical RAM without OS pagefile swapping.

---

## 🚀 How to Run

### Direct Launch
Run any of the pre-configured batch launchers:
- `run-flint-ud-q3_k_xl.bat` (Flint 40B-A2.5B, 128k context)
- `run-cybertiel-q4_k_m.bat` (Cyber-Tiel-Coder 35B-A3B, 128k context)
- `run-qwen36abl-q4_k_m.bat` (Qwen 3.6 35B-A3B Heretic, 128k context)
- `run-gemma4-q6_k.bat` (Gemma 4 26B-A4B Abliterated, 128k context)

All launchers expose an OpenAI-compatible API at:
`http://127.0.0.1:8080/v1/chat/completions`
