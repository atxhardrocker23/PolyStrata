# PolyStrata Self-Audit Report

**Audit session:** 2026-10-29
**Scope:** `G:\PolyStrata` — local PolyStrata inference server (`http://127.0.0.1:8080/v1`, model `gamma4`)
**Constraint:** Read-only audit. No source, config, script, or binary files were modified. Only this report (`G:\PolyStrata\POLYSTRATA_AUDIT_REPORT.md`) was created / written / appended.

## Step 1 — Map & Initialize

### Repository File Inventory

**Top-level entries:**
- `CMakeLists.txt` — 155,591 bytes (C++ engine build)
- `setup.py` — 557,420 bytes (~10k lines; Python orchestrator/setup monolith)
- `setup_abliterated.py` — 2,230 bytes
- `conftest.py`, `requirements.txt`, `SECURITY.md`, `LICENSE`, `AGENTS.md`, `AI_README.md`
- `README.md`, `README.de.md`, `README.es.md`, `README.fr.md`, `README.ja.md`, `README.pt-BR.md`, `README.zh-CN.md`, `README.RTX-3060-tuning.md`
- `UPDATE.bat`, `START-HERE.bat`, `SETUP.bat`, `docker-entrypoint.sh*`, `BENCH.bat`, `BENCH.sh*`
- `build-*.bat`, `build-*.sh*`, `run-*.bat`, `enable-tdr-60s.reg`, `flake.nix`, `flake.lock`, `.dockerignore`, `.gitattributes`, `.gitignore`
- `.github/` — ISSUE_TEMPLATE/, pull_request_template.md

**Source & runtime trees:**
- `src/` — subdirectories only: `artifact/`, `core/`, `kernels/`, `ngram/`, `plan/`, `platform/`, `poly/`, `prefill/`, `program/`, `qwen35/`, `spec/`
- `include/strata/` — public headers
- `serve/` — server modules (`server.py` 476,580 bytes; `frontend.py` 98,420; `responses.py` 37,855; `replicas.py` 17,274; `telemetry.py` 30,948; `mcp.py` 32,397; `models.py` 10,879; `runconfig.py` 9,252; `structured.py` 9,641; `logit_bias.py` 4,753; `cpu_affinity.py` 3,281; `cache_drop.py` 2,718; `keepawake.py` 3,250; `prometheus.py` 9,758; `session_save_retry.py` 3,376; `test_*.py many; `web/`, `fixtures/`, `images.py`, `chat_template.jinja`, `chat_golden.json`)
- `tools/` — `ab_engine.py`, `strata_bench.py`, `strata_mcp.py`, `strata_inspect.py`, `strata_pack.py`, `strata_tokenizer.py`, quant/pack tools (`iq_pack.py`, `iq_fixture.py`, `quant_quality.py`, `quantscope.py`, `quant_formats.py`, `quant_bound.py`, `embd_bf16_pack.py`, `ple_fp8_pack.py`, `h2d_bench.cu`, `sm60_dp4a_check.cu`), `conversation_cache_*.py`, `test_setup_*.py`, `test_strata_*.py`, etc.
- `tests/` — `core/`, `cuda/`, `data/`, `hip/`, `platform/`, `prefill/`, `program/`
- `bench/`, `ref/`, `docs/`, `patches/`, `packs/`, `models/`, `sycl/`, `metal/`, `data/`, `third_party/`, `cmake/`

**Build / engine artifacts present:**
- `build-poly/`, `build-vision-poly/`, `engine-poly/`

## Step 2 — Audit Log

(To be appended batch-by-batch: File & Exact Line Numbers, Old/Dead/Missing-Target Code to Remove, Performance Bottlenecks & Exact Replacement Code, Bugs & Cleanup Recommendations)

## Step 3 — Master Action Plan

(To be appended after all files are audited: Safe Purge Checklist, Ranked Performance Overhaul Plan)

## Step 2 — Audit Log (batch 1)

### `setup.py` — lines 1–500 audited (8,767 lines total, 557,420 bytes)

**File & Exact Line Numbers:**
- L30–L39: CLI options string (`--family`, `--model`, `--context`, `--rope-scaling`, `--vision`, `--port`, `--yes`, `--setup`, `--no-start`, `--host`, `--api-key`, `--experimental-speed-projection`, `--models-dir`, `--gguf-dir`, `--build`, `--check`, `--resident-budget-gib`, `--kv-streaming`).
- L68–L69: `ROOT = Path(__file__).resolve().parent`, `WIN = os.name == "nt".
- L73–L95: `HF_REVISIONS` pinned Hugging Face commit map.
- L101–L160: HF endpoint / source helpers (`hf_endpoint`, `hf`, `hf_unpinned`, `ms_endpoint`, `reachable`, `model_source`, `source_hint`, `ms_file`, `ms_url`, `ms_meta`).
- L134–L141: `reachable()` — `except OSError: # HTTPError too: an error page is not the file / return False`. Swallow pattern.
- L224–L229: `MIN_ENGINE = (0, 1, 45)` version-history comment block; `KV_CELL_BYTES = {"q4_0": 576, "k8v4": 816}`; `PY_PACKAGES`.
- L262: `UNSLOTH_ENGINE = (0, 1, 32)`.

**Old / Dead / Missing-Target Code to Remove:**
- (pending deeper reads)

**Performance Bottlenecks & Exact Replacement Code:**
- (pending deeper reads)

**Bugs & Cleanup Recommendations:**
- **L140 — swallowed exception / bug:** `reachable()` catches `OSError` (the parent class of `HTTPError`) and silently returns `False`; no logging, no distinct handling.

---

Next batch: `setup.py` lines 501–900.


## Step 2 — Audit Log (batch 2)

### `setup.py` — lines 901–1300 audited

**File & Exact Line Numbers:**
- L902–L907: `fail()` — `sys.exit(1)`; L906 message.
- L910–L925: `flush_typed_ahead()` — #841; L914–L925: try/except with `except (ImportError, OSError, ValueError, AttributeError): pass` (L924–L925). Swallow pattern.
- L928–L943: `ask()` — input loop; L935–L937 EOFError → fail; L940–L941 choice match.
- L945–L954: `run()` — subprocess.run; L950–L953 check/quiet/fail.
- L957–L961: `out()` — L960–L961: `except (OSError, subprocess.TimeoutExpired): return ""` (swallow).
- L978–L1000: `_memory_status()`, `container_ram_gb()`.
- L1002–L1024: `ram_gb()`, `_mm_multi_sz` — L1022: `except (ImportError, OSError): return None` (swallow).
- L1027–L1041: `parse_paging_file`.
- L1044–1064: `PAGE_FILE_FIXED_MB`, `_drive_mb`, `page_file_setting`, `page_file_grows`, `page_file_gb`, `page_file_advice`.
- L1128–L1160: `cpu_cores()` — L1155–L1156: `except (OSError, ValueError, AttributeError): return None` (swallow).
- L1163–1185: `linux_core_classes()`.
- L1188–1195: `cpulist()`.
- L1198–1206: `linux_hybrid_pmu()`.
- L1209–1221: `hybrid_pool_workers()`.
- L1224–1233: `recommend_pool_workers()` — L1227–L1233: hybrid CPU recommendation; L1228: `if n is None or "--pool-workers" in args`.
- L1236–1245: `linux_sockets()`.
- L1248–1274: `cpu_sockets()` — L1273–L1274: `except Exception: return None` (broad swallow).
- L1277–1285: `two_socket_note()`.
- L1288–1300: `cpu_info()`.

**Old / Dead / Missing-Target Code to Remove:**
- L845: `PREVIEW_FAMILIES = OWN_PATH_FAMILIES             # the earlier name (tools and tests that import it)` — deprecated alias.
- L871–L872: `OLD_POLY_EXE = "strata-glm.exe"`, `OLD_ENGINE_POLY_DIR = "engine-glm"` — legacy/ghost references.
- L806: `MMPROJ = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"` — possibly stale model name.

**Performance Bottlenecks & Exact Replacement Code:**
- L1224–L1233: `recommend_pool_workers` / `hybrid_pool_workers` — thread pool contention (#642, #674 #707); recommendation logic.

**Bugs & Cleanup Recommendations:**
- L924–L925, L960–L961, L1022, L1155–L1156, L1273–L1274: swallowed exceptions / broad excepts — should log or re-raise.

---

Next batch: `setup.py` lines 1301–1700.

