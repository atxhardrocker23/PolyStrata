# Architectural & Performance Audit — PolyStrata (RTX 3060 12GB / i5-12400)

**Scope:** read-only review of `src/poly/kernels.cu`, `src/kernels/cuda/`, `src/prefill/`, `src/poly/cpu_experts.cpp`, `src/poly/engine.cpp`, `src/core/expert_source.cpp`, `serve/server.py`. No files were modified.

**Hardware:** RTX 3060 12GB (sm_86, Ampere, 360 GB/s), i5-12400 (Golden Cove, AVX2 + **AVX-VNNI** + FMA, **no AVX-512**), 32 GB DDR4, Windows 11 WDDM 3.x.

**Overall:** The code is mature and well-engineered. The tensor-core paths (fattn.cu m16n8k16 MMA, MMQ pointer table, exchange-buffer rotation, WDDM aperture cap) are correct and idiomatic. The biggest missed opportunity is that the **CPU MoE expert matmul runs AVX2 int8 emulation instead of the AVX-VNNI hardware MMA that the i5-12400 has**. Below, ranked.

---

## 1. CPU: AVX-VNNI hardware MMA for the Q8_K expert dot products — High / CPU AVX

**Location:** `src/poly/cpu_experts.cpp`, `q4k_gate_up_1` (L167), `q4k_dot_multi` (L243), `q6k_dot_multi` (L310), and the `tg->vec_dot`/`tu->vec_dot` fallbacks at L936, L1038.

**Observation:** The Q4_K/Q6_K × Q8_K products accumulate int8 with `_mm256_maddubs_epi16` + `_mm256_madd_epi16` (unpack-then-add chains, e.g. L215-L219, L289-L295). This is the scalar-equivalent emulation of ggml's `vec_dot`. The i5-12400 (Golden Cove) has **AVX2-VNNI** (`_mm256_dpbusd_epi32`), which does the int8×int8→int32 MAC in one instruction. The AVX-512-VNNI path in `src/kernels/cpu/expert.cpp` (`_mm512_dpbusd_epi32`, L183/L212) is compiled out on this CPU — it needs AVX-512, which the 12400 lacks — so the hot path never touches a hardware int8 MMA.

**Proposed optimization:** In the Q8_K inner loop, load the Q8_K codes as `__m256i` and accumulate with `_mm256_dpbusd_epi32(acc, unpacked_a, q8)`, keeping the existing scale-shuffle post-processing. Gate it on `__AVX2__ && !__AVX512__` (AVX2-VNNI) so the AVX-512 and scalar paths are untouched. Keep the bit-exactness test (`attend_test`/`kv_int8_test`) — `_mm256_dpbusd_epi32` is bit-exact with the madd chain.

**Gain & trade-offs:** ~1.5–2× throughput on the gate/down projection for the dominant Q4_K/Q6_K models at the latency that matters (decode). Purely additive; no quality change. Requires a separate AVX2-VNNI compile variant (the CMake build already branches on ISA).

---

## 2. CPU: non-temporal (streaming) stores for the down-project output — Medium / CPU AVX

**Location:** `src/poly/cpu_experts.cpp`, the `d_tasks` loop (L1013–L1070), specifically the `s[]` scratch writes (L1044–L1047) and the final `out[]` accumulation (L1058–L1062).

**Observation:** The down-project writes go through a `std::vector<float> one` scratch (`s`) and then into `out`. These are large, one-shot writes that pollute L3. The file uses `_mm_prefetch(..., _MM_HINT_T0)` for the *reads* (prefetch_row) but no `_mm_stream_si128`/`_mm256_stream_ps` for the *writes*. On a 6-core Golden Cove core the write-allocate for a streaming store wastes cache lines and memory bandwidth.

**Proposed optimization:** For the `out[]` store (the expert down-project result that is read once by the next layer), use `_mm256_stream_ps` (non-temporal). The `s[]` scratch is reused within the task, so leave it as regular stores; only the final `out[]` write is a clean streaming candidate. Guard with `#if defined(__AVX2__)`.

**Gain & trade-offs:** ~5–10% on the down-projection memory-system efficiency under multi-token batches; small, stable. No correctness risk if the store is strictly non-temporal (the value is consumed before the cache line is evicted).

---

## 3. CPU: spin-loop policy under GPU-wait — Medium / CPU AVX

**Location:** `src/poly/cpu_experts.cpp`, `spins_before_sleep()` (L41–L47) and the wait at L753–L791.

**Observation:** `STRATA_CPU_SPIN` defaults to 200,000 spins (~3–5 ms on Alder Lake) to bridge a GPU attention wait. `_mm_pause()` is used (good — it's a spin-loop hint, not a busy spin). But the default is a single fixed threshold: short GPU waits spin (fine), but anything between the spin ceiling and a full sleep still costs a `sched_yield`/park round-trip. Under the 128K-context hybrid offload, the CPU frequently waits on the GPU.

**Proposed optimization:** Add a tiered policy: spin with `_mm_pause()` up to the current `STRATA_CPU_SPIN`, then a short `sched_yield` burst, then park the thread. Also expose `STRATA_CPU_SPIN_MIN` so short waits don't burn a core. The i5-12400 has 6 P-cores with no E-cores, so a parked thread is the right move for waits > 5 ms.

**Gain & trade-offs:** Lower idle latency and power under offload; avoids a core spinning for 5 ms waiting on a kernel that's 20 ms away. Stability-neutral; purely a scheduling heuristic.

---

## 4. Decode attention: INT8 keys scored via warp shuffles, not tensor cores — Medium / GPU Kernel

**Location:** `src/kernels/cuda/qsa_decode_attn.cu`, the chunk scoring kernel (L137–L260) and the gfx906 swizzle variant (L333–L416).

**Observation:** When the tensor-core `fdec_q8` path (fattn.cu, m16n8k16) is *not* taken — e.g. a head shape or scratch budget that `fdec_fits()` rejects — the decode falls to `qsa_decode_attn`, which scores INT8 keys with a `warp_sum` butterfly (L240) and per-lane 8-dimension slices. The keys are already int8 (quantized KV), but the reduction is done in float lanes, not on the tensor cores. fattn.cu already proves the m16n8k16 int8-key path works (L233–L242).

**Proposed optimization:** For the INT8-key decode case, feed the quantized key codes into an `mma.sync.m16n8k16` tile (as fattn.cu does for fdec_q8) instead of the warp-shuffle reduce. Keep the warp-shuffle path for the fp16/`query_swizzle` cases where it is already optimal. Gate on `__CUDA_ARCH__ >= 800`.

**Gain & trade-offs:** Could lift decode throughput on the fallback path by ~1.5–3× for INT8 KV. Risk: the qsa path is bit-exact against a reference; the MMA path must be verified bitwise (the existing `qsa_decode_attn` parity tests cover this). Only active when fattn doesn't fit, so no regression to the primary path.

---

## 5. WDDM aperture: cap is correct, but verify the pinned-exchange path too — Low / PCIe Streaming

**Location:** `src/poly/engine.cpp`, `pin_experts()` (L1350–L1421) and `src/core/expert_source.cpp` `reserve_exchanges()` (L3329–L3376).

**Observation:** `cudaHostRegister` is correctly capped at 2048 MiB on Windows to avoid WDDM aperture exhaustion (L1356–L1358) — this is the right mitigation for a 12 GB card on 32 GB RAM. However, the STRATA_EXCHANGE_ROTATE path uses `cudaHostAlloc` with `cudaHostAllocMapped` (L3339) to get a *device pointer* for the ownership-transfer exchange buffers. `cudaHostAllocMapped` also consumes WDDM aperture (it maps host RAM into the GART). When rotation is on, the pinned exchange pool (`xstage_`) and the mapped complement (`complement_host_`) both draw from the same aperture budget that `cudaHostRegister` is already capped against.

**Proposed optimization:** When STRATA_EXCHANGE_ROTATE is active, fold the mapped-exchange allocation into the same aperture budget the pinned-experts path already enforces (report `xstage_` + mapped complement against the 2048 MiB ceiling, or reserve it up front). Log the total aperture reservation at startup.

**Gain & trade-offs:** Prevents a rare but real WDDM exhaustion crash when both expert pinning and exchange rotation are enabled. Stability-only; no throughput change.

---

## 6. MMQ pointer table: sound, but watch the select-chain branch cost at runtime — Low / GPU Kernel

**Location:** `src/prefill/moe_mmq_tab.cuh`, `STRATA_MMQ_TAB_PICK` (L77–L80) and the per-expert base selection at L187, L282, L367.

**Observation:** The design is correct — 16 `const char * __restrict__` kernel parameters (not a dynamically-indexed array, which would go to local memory) and a ternary select chain that the compiler keeps in registers/constant cache. The comment at L66–L69 documents the measured 1.2–2.5× penalty of the array alternative, so this is already the right call. The only residual: the select chain is a compile-time-known constant per tile, so it should be branch-predictor-friendly, but under heavy expert-group divergence (different experts per row) the chain can mispredict.

**Proposed optimization:** If profiling shows branch-mispredict stalls on the pick, convert the per-tile pick to a `__ldcg`-friendly pointer load from a small constant-memory pointer array (one cache line for 16 pointers) rather than the ternary. Low priority — only if a profiler confirms it.

**Gain & trade-offs:** Potentially a few % on the prompt MMQ under high group divergence. No correctness risk; keep the bitwise-identical guarantee (the test at `tests/cuda/prefill_mmq_tab_test.cu` checks it).

---

## 7. Server: context cutoff, queuing, streaming, error propagation — all sound (no change)

**Location:** `serve/server.py` — context check (L4537–L4565), model gate (`serve/model_gate.py`), SSE streaming (L6611–L6635), error map (L6244–L6280).

**Observation (positive):**
- **Context cutoff:** requests are **rejected with 400, never truncated** (L4553, L4560), with a `CTX_SLACK = 8` margin and an optional `fit_max_tokens` clamp. This is the correct safety posture.
- **Model switching:** lease-based (`ModelGate.use`), a request holds its model to completion; a switch waits for zero leases and fails all waiters atomically. Correct and deadlock-free.
- **Concurrency:** `slot_q`/`slot_busy`/`slot_held` batch slots with orphan-slot reaping (`_reap_orphan_slots`, L1487) — no slot leak.
- **Streaming:** SSE with keep-alive comments, `wfile.flush()` per chunk, client-disconnect cancellation (`cancel.set()`), and mid-stream error events (EngineDied/StructuredOutputError/ValueError all flush a proper `data:` error then `[DONE]`).
- **Error propagation:** a clean status map — invalid 400, model_busy 409, structured_output 502, memory 503, storage 507, io 500 — with `Retry-After` on the retryable 503s.

**Recommendation:** none required. One low note: the `ANSWER_ROOM_MIN_CAP`/`ANSWER_RESERVE_MIN` thinking-budget cut (L162–L163, L4762–L4778) is a latency safeguard that trades a little reasoning length for guaranteed answer room — fine, but if you ever see "thinking limited" spam under 128K context, raise `ANSWER_ROOM_MIN_CAP` or the budget is being applied more often than desired.

---

## Summary (prioritized)

| # | Tier | Category | What |
|---|---|---|---|
| 1 | High | CPU AVX | Use AVX2-VNNI `_mm256_dpbusd_epi32` for the Q8_K expert dot products (currently AVX2 emulation). |
| 2 | Medium | CPU AVX | Non-temporal `_mm256_stream_ps` on the down-project `out[]` write. |
| 3 | Medium | CPU AVX | Tiered spin/yield/park policy for GPU-wait bridges. |
| 4 | Medium | GPU Kernel | INT8-key decode scoring on tensor cores (m16n8k16) in the qsa fallback. |
| 5 | Low | PCIe Streaming | Fold mapped exchange buffers into the WDDM aperture budget. |
| 6 | Low | GPU Kernel | Optional constant-mem pointer array for the MMQ pick (only if profiled). |
| 7 | — | Server | No change — context cutoff, model gate, streaming, error map are all correct. |

The single highest-value, lowest-risk change is **#1**: the i5-12400's AVX-VNNI is sitting unused on the MoE hot path. Everything else is incremental.
