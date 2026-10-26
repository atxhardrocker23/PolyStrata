# AI_README.md — Autonomous Agent & LLM Operational Guide for PolyStrata (RTX 3060 12GB Rig)

> **Audience:** Any autonomous AI assistant, agentic coding model (Hermes, Claude, Antigravity, OpenAI/Coder), or script operating inside or querying this repository. Read this file first before executing or modifying scripts.

---

## 1. System Hardware & Topology

When planning batch sizes, context limits, thread counts, and memory allocations, assume these exact specifications:
- **GPU:** NVIDIA GeForce RTX 3060 12GB VRAM (Compute Capability `sm_86`, unlocked power limit, CUDA Toolkit v13.0).
- **CPU:** Intel Core i5-12400 (6 Physical Performance Cores, 12 Threads). Never allocate more than 6 worker threads to avoid hyperthread scheduling penalties.
- **RAM:** 32 GB DDR4.
- **OS:** Windows 11 64-bit with `SeLockMemoryPrivilege` (Large Pages) enabled.
- **Root Directory:** `G:\PolyStrata` (with NTFS mirror junction at `G:\Strata-3060`).

---

## 2. Directory Layout & Key Components

```
G:\PolyStrata (and G:\Strata-3060)
├── engine-poly/
│   ├── strata-poly.exe       # Custom compiled C++/CUDA release binary (sm_86 + AVX2 + LTCG)
│   └── BUILD.json            # Version and source hash fingerprint (do not overwrite)
├── models/                   # GGUF model weights storage
│   ├── flint-UD-Q3_K_XL/
│   ├── cyber-tiel-coder-35b-a3b/
│   ├── qwen36abl-Q4_K_M/
│   └── gemma4-26b-a4b-it-abliterated/
├── packs/                    # Model tokenizers and expert-profile.txt activation frequency caches
├── serve/
│   ├── server.py             # FastAPI/HTTP server (port 8080) with OpenAI/Anthropic endpoints
│   ├── frontend.py           # Jinja chat templating and thinking effort injection
│   ├── model_gate.py         # Multi-model queue and zero-downtime switching gate
│   └── models.py             # Model discovery and capabilities registry
├── src/poly/                 # C++ engine core (MoE dispatch, routing, CPU AVX2 kernels)
├── strata-*.json             # Declarative run configurations for each model
└── run-*.bat                 # Windows shell launchers with locked environment variables
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
set "STRATA_POLY_PIN_EXPERTS=1536" :: Zero-copy pinned GGUF layer views (PR #35 + WDDM fix)
set "STRATA_CACHE_BIAS=0.35"       :: Cache-aware VRAM top-k expert routing bonus (Issue #81)
set "OMP_NUM_THREADS=6"            :: Lock CPU workers to 6 physical P-cores
set "OMP_PROC_BIND=TRUE"
set "OMP_PLACES=cores"
```

---

## 4. Model Registry & Execution Constraints

### Strict Constraints:
1. **Context Window:** The user requires a **128K (`131072`) context window** on all models. Never reduce `--max-context` below `131072` unless explicitly ordered by the user for an isolated test.
2. **Port & Endpoint:** The server listens on `http://127.0.0.1:8080/v1`.
3. **Available Models:**
   - `flint-40b-a2.5b-ud-q3_k_xl`: Thinking effort levels `['low', 'medium', 'high']` (Flint always thinks; no `off`).
   - `cyber-tiel-coder-35b-a3b`: Thinking effort levels `['off', 'low', 'medium', 'high']`.
   - `qwen3.6-35b-a3b-abliterated-q4_k_m`: Thinking effort levels `['off', 'low', 'medium', 'high']`.
   - `gemma-4-26b-a4b-it-abliterated-q6_k`: Thinking effort levels `['off', 'low', 'medium', 'high']`.

---

## 5. API Endpoints for Automation

- `POST /v1/chat/completions`: Standard OpenAI chat API. Supports streaming (`"stream": true`), thinking tags (`<think>...</think>`), and reasoning effort (`"reasoning_effort": "low"|"medium"|"high"`).
- `GET /v1/models`: Discovers available models, active loaded status, and thinking effort capabilities.
- `GET /props`: Engine runtime facts (VRAM free, cache hit rates, active context).
- `GET /slots`: Real-time batch slot monitoring.

---

## 6. How to Recompile the Engine (if modifying C++/CUDA code)

To recompile `strata-poly.exe` with all hardware optimizations enabled:
```cmd
G:\PolyStrata\build-strata-poly.bat
```
After building, stamp `BUILD.json` with `setup.source_hash(setup.ENGINE_SOURCES)` so `setup.py` does not attempt to overwrite your binary.
