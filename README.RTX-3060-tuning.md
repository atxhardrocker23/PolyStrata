# PolyStrata: RTX 3060 12GB (Unlocked) & Hybrid CPU-GPU Tuning Guide

This repository contains custom performance-engineered branches and compiled binaries of **PolyStrata** specifically optimized for **NVIDIA GeForce RTX 3060 12GB** (both stock and power/clock-unlocked) paired with modern multi-core CPUs (e.g. Intel Core i5-12400 6 P-Cores) and 32GB+ system RAM, maintaining a full **128K (`131072`) context window**.

---

## 🚀 Benchmark Highlights (RTX 3060 12GB Unlocked + i5-12400 + 32GB RAM)

| Model | Quantization | Context Window | Prefill Speed | Sustained Decode | VRAM Cache Hit Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Flint 40B-A2.5B** | `UD-Q3_K_XL` (18.5 GB) | **128,000 (131072)** | **1,200+ tok/s** | **60–85+ tok/s** | **84.4%** (6,208 / 12,032 in VRAM) |
| **Cyber-Tiel-Coder 35B-A3B** | `UD-Q4_K_M` (22.3 GB) | **128,000 (131072)** | **900–1,100 tok/s** | **45–65+ tok/s** | **80.8%** (4,141 / 10,240 in VRAM) |
| **Qwen 3.6 35B-A3B Heretic** | `i1-Q6_K` (25.1 GB) | **128,000 (131072)** | **850–1,050 tok/s** | **40–55+ tok/s** | **63.6%** (3,170 / 10,240 in VRAM) |
| **Gemma 4 26B-A4B IT Abliterated** | `Q6_K` (21.4 GB) | **128,000 (131072)** | **950–1,150 tok/s** | **45–60+ tok/s** | **97.8%** (1,528 / 3,840 in VRAM) |

---

## 🛠️ Performance Architecture & Optimizations Integrated

1. **Hardware-Native AVX-VNNI `VPDPBUSD` Vector Dot Product (`quants.c`, `CMakeLists.txt`)**:
   - Enables `-DGGML_AVX_VNNI=ON` and implements 256-bit VEX-encoded `vex vpdpbusd` for CPU integer dot products (`IQ3_S`, `IQ4_NL`, `IQ4_XS`, `MXFP4`) with full sign extraction (`_mm256_sign_epi8`).
   - Cuts CPU integer matrix instructions by ~3–4× on Golden Cove P-cores.
2. **Layer-Graded MoE Cache Biasing (`--cache-bias 0.42`, `--cache-bias-outer 0.15`)**:
   - Router logit bonus gives middle feature-extraction layers strong affinity to VRAM-resident experts (`+0.420`), while tapering early input and deep output layers to `+0.150` to safeguard semantic quality and protect deep representations.
   - Combine weights remain completely **unbiased**.
3. **Active Worker Spin-Waiting & Extended Prefetch (`STRATA_CPU_SPIN=200000`, `kPrefetchRows = 4`)**:
   - Replaces premature OS sleep transitions with a ~3–5 ms active pause spin loop in `cpu_experts.cpp`. CPU worker threads remain awake while GPU attention runs, eliminating `WaitOnAddress` context-switch latency.
4. **Warp-per-Row-Group 32-Expert Batch Launch (`STRATA_SLOTS_WARP=1`)**:
   - Custom CUDA kernel (`native_mmvq_slots_warp_kernel`) replaces 8-slot limit with 32 slots (`kMmvqSlotsMax = 32`). Cuts kernel dispatch overhead by $4\times$.
5. **Non-Temporal Streaming CPU Stores (`STRATA_POLY_STREAM_STAGE=1`)**:
   - Uses `_mm_stream_si128` + `_mm_sfence()` when staging CPU experts into pinned RAM. Bypasses the CPU cache hierarchy, keeping the Intel L3 cache clean for execution threads.
6. **Zero-Copy File-View Expert Pinning (Custom WDDM Page-Merge Fix)**:
   - Registers memory-mapped GGUF file views directly with CUDA (`cudaHostRegister`), eliminating redundant copies. Includes a custom fix merging adjacent 32-byte-aligned tensor intervals on 4096-byte boundaries and a 2048 MiB Windows WDDM aperture safety cap.
7. **Per-Model Thinking Effort & Anti-Starvation Switching**:
   - Exposes dynamic reasoning effort levels (`off`, `low`, `medium`, `high`) in OpenAI/Anthropic APIs and drains queued requests cleanly before triggering model transitions.

---

## ⚡ System & Driver Tuning (Windows + RTX 3060 12GB)

### 1. Windows Graphics TDR Timeout (60 Seconds)
- Under heavy 128K context prefill, Windows WDDM can falsely detect a GPU hang if a kernel runs longer than 2 seconds.
- Double-click [`enable-tdr-60s.reg`](enable-tdr-60s.reg) to set `HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrDelay = 60` (DWORD).

### 2. Full Resizable BAR (16GB BAR1) & CUDA Memory P0 State
- By default, NVIDIA drivers downclock GDDR6 memory by 200–500 MHz during CUDA compute (Force P2 State) and restrict Resizable BAR to whitelisted games.
- Import [`rtx3060-p0-rebar.nip`](rtx3060-p0-rebar.nip) via **NVIDIA Profile Inspector** (or CLI `nvpi -silentImport rtx3060-p0-rebar.nip`):
  - Sets **CUDA - Force P2 State = Off** (unlocking maximum GDDR6 memory bandwidth).
  - Sets **rBAR - Feature = Enabled (1)**, **rBAR - Options = 1**, and **rBAR - Size Limit = 0** (full 16GB BAR1 aperture across all apps).

### 3. CPU & Memory Profile (Intel i5-12400)
- **Cores**: 6 Performance Cores / 12 Threads. Thread affinity locked to 6 physical cores (`OMP_NUM_THREADS=6`, `OMP_PROC_BIND=TRUE`, `OMP_PLACES=cores`).
- **RAM**: 32 GB DDR4 in Gear 1 mode.
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
