# AI_README.md — Autonomous Agent & LLM Operational Guide for PolyStrata (RTX 3060 12GB Rig)

> **Audience:** Any autonomous AI assistant, agentic coding model (Hermes, Claude, Antigravity, OpenAI/Coder), or script operating inside or querying this repository. Read this file first before executing or modifying scripts.

A second engine, `strata-poly` (`src/poly/`), runs GGUF models on Linux or Windows with custom MoE routing and hardware optimizations: [docs/GLM.md](docs/GLM.md), [docs/ORNITH.md](docs/ORNITH.md), [docs/DEEPSEEK.md](docs/DEEPSEEK.md), [docs/BACKENDS.md](docs/BACKENDS.md). The server frontend, model router, and web workspace are powered by `am2strata` (`G:\Strata-3060\am2strata\am2strata.exe`) on port 8088.

---

## 1. System Hardware & Topology

When planning batch sizes, context limits, thread counts, and memory allocations, assume these exact specifications:
- **GPU:** NVIDIA GeForce RTX 3060 12GB VRAM (Compute Capability `sm_86`, unlocked power limit, CUDA Toolkit v13.0, driver 621.14, Resizable BAR 16GB active).
- **CPU:** Intel Core i5-12400 (6 Physical Performance Cores, 12 Threads). Never allocate more than 6 worker threads (`OMP_NUM_THREADS=6`) to avoid hyperthread AVX2 execution contention.
- **RAM:** 32 GB DDR4.
- **OS:** Windows 11 64-bit with `SeLockMemoryPrivilege` (Large Pages) enabled and `TdrDelay=60`.
- **Root Directory:** `G:\PolyStrata` (with NTFS mirror junctions at `G:\Strata-3060`: `.venv`, `models`, `packs`, `serve`, `tools`, and `engine-poly`).

---

## 2. Directory Layout & Key Components

```
G:\PolyStrata (and G:\Strata-3060)
├── engine-poly/              # NTFS junction to PolyStrata\engine-poly
│   ├── strata-poly.exe       # Custom compiled C++/CUDA release binary (sm_86 + AVX2/AVX-VNNI + LTCG)
│   ├── strata-vision.exe     # Multimodal vision encoder binary
│   └── BUILD.json            # Version and source hash fingerprint (do not overwrite)
├── models/                   # GGUF model weights storage (NanoPlus models)
│   ├── gemma4-nanoplus/      # Gemma 4 26B-A4B IT Abliterated (i1-IQ3_S, 11.38 GB)
│   ├── tiel-coder-nanoplus/  # Tiel-Coder-35B-A3B APEX-I NanoPlus (11.69 GB)
│   ├── flint-apex-nanoplus/  # Flint 40B-A2.5B APEX-I NanoPlus V2.1 (12.33 GB)
│   ├── ornith-nanoplus/      # Ornith 1.5 35B-A3B APEX-I NanoPlus (11.69 GB)
│   └── lfm2.5-nanoplus/      # LFM 2.5 24B-A2B APEX-I NanoPlus (7.59 GB)
├── packs/                    # Model tokenizers and expert-profile.txt activation frequency caches
├── serve/                    # Python engine harness & test suites
│   ├── server.py             # Server harness & engine process management
│   ├── frontend.py           # Jinja chat templating and thinking effort injection
│   ├── model_gate.py         # Multi-model queue and zero-downtime switching gate
│   └── models.py             # Model discovery and capabilities registry
├── am2strata/                # G:\Strata-3060\am2strata: High-performance Rust web UI & server
│   ├── am2strata.exe         # Native Windows server binary (port 8088)
│   ├── configs/              # Active model configs (strata-*.json)
│   └── web/                  # Web workspace assets
└── strata-*.json             # Declarative run configurations for each model
```

---

## 3. Mandatory Environment Variables

Whenever spawning the engine or running benchmark scripts, the following environment flags **MUST** be active:

```bat
set "CUDA_MODULE_LOADING=LAZY"
set "STRATA_GDN_MMA=1"             :: Chunked GDN MMA tensor core prefill
set "STRATA_FATTN=1"               :: FlashAttention kernel for prompt prefill
set "STRATA_FDEC=1"                :: Tensor-core decode attention kernel
set "STRATA_SLOTS_WARP=1"          :: 32-slot warp-per-row-group MoE batch launch (PR #94)
set "STRATA_POLY_STREAM_STAGE=1"   :: Non-temporal streaming CPU stores for staging (PR #99)
set "STRATA_POLY_PIN_EXPERTS=2048" :: Pin out-of-VRAM expert layers into locked physical RAM (WDDM page-merge fix)
set "STRATA_CACHE_BIAS=0.42"       :: Cache-aware VRAM top-k expert routing bonus
set "STRATA_CACHE_BIAS_OUTER=0.15" :: Tapered routing bonus for boundary layers
set "STRATA_CPU_SPIN=200000"       :: Active CPU worker pause-spin loop preventing sleep context switches
set "OMP_NUM_THREADS=6"            :: Lock CPU workers to 6 physical P-cores
set "OMP_PROC_BIND=TRUE"
set "OMP_PLACES=cores"
```

---

## 4. Model Registry & Execution Constraints

### Strict Constraints:
1. **Context Window:** The user requires a full context window (up to `131072`) on all models.
2. **Active Server Port & Endpoint:** The active model server runs via `am2strata.exe` at `http://127.0.0.1:8088/v1` (or standalone `serve/server.py` at `http://127.0.0.1:8080/v1`).
3. **Available Models:**
   - `gemma-4-26b-a4b-it-abliterated-nanoplus`: Vision enabled, uncensored MoE.
   - `tiel-coder-35b-apex-nanoplus`: Vision enabled, coding MoE.
   - `flint-40b-apex-nanoplus`: Recursive coding agent MoE.
   - `ornith-nanoplus`: Vision enabled, agentic reasoning MoE.
   - `lfm2.5-24b-nanoplus`: Liquid hybrid MoE (100% in 12GB VRAM).

---

## 5. API Endpoints for Automation

- `POST /v1/chat/completions`: Standard OpenAI chat API. Supports streaming (`"stream": true`), thinking tags (`<think>...</think>`), and reasoning effort (`"reasoning_effort": "low"|"medium"|"high"`).
- `GET /v1/models`: Discovers available models, active loaded status, and thinking effort capabilities.
- `GET /health`: Health and hardware status.
- `POST /api/models/load`: Hot-swap active model in VRAM without server restart.

---

## 6. How to Recompile the Engine (if modifying C++/CUDA code)

To recompile `strata-poly.exe` with all hardware optimizations enabled for RTX 3060 (`sm_86`):
```cmd
G:\PolyStrata\build-poly.bat
```
After building, the binary is compiled to `build-poly\strata-poly.exe` and copied to `engine-poly\strata-poly.exe`.
