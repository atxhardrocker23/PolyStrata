"""
audit_benchmark.py — Comprehensive Speed & Metrics Audit for PolyStrata Models
Hardware: Intel i5-12400 (6 P-Cores) + NVIDIA RTX 3060 12GB (Unlocked) + 32GB RAM
Platform: Windows 11 / CUDA 13.5
"""

import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("G:/Strata-3060")
PORT = 8099
API_BASE = f"http://127.0.0.1:{PORT}"

MODELS_ORDER = [
    # NanoPlus 100% VRAM Models
    ("flint-40b-apex-nanoplus", "strata-flint-nanoplus.json"),
    ("tiel-coder-35b-apex-nanoplus", "strata-tiel-coder-nanoplus.json"),
    ("ornith-nanoplus", "strata-ornith-nanoplus.json"),
    ("lfm2.5-24b-nanoplus", "strata-lfm2.5-nanoplus.json"),
    # Standard & Hybrid Models
    ("flint-40b-a2.5b-ud-q3_k_xl", "strata-flint-ud-q3_k_xl.json"),
    ("cyber-tiel-coder-35b-a3b", "strata-cybertiel-q4_k_m.json"),
    ("gemma-4-26b-a4b-ud-q4_k_m", "strata-gemma4-ud-q4_k_m.json"),
    ("gemma-4-26b-a4b-it-abliterated-q6_k", "strata-gemma4-q6_k.json"),
]

PROMPT_SHORT = (
    "You are an expert low-latency C++ systems engineer. "
    "Write a complete, single-file C++20 lock-free Single Producer Single Consumer (SPSC) ring buffer. "
    "Requirements:\n"
    "1. Capacity must be a power of two verified by static_assert with std::has_single_bit.\n"
    "2. Cache-line aligned head and tail indices with alignas(64) to prevent false sharing.\n"
    "3. Proper std::memory_order acquire and release semantics without mutexes.\n"
    "4. Clean enqueue and dequeue template methods returning bool.\n"
    "Provide clear, concise code without unnecessary boilerplate."
)

BASE_CODE_SNIPPET = """
struct TelemetryFrame {
    uint64_t timestamp_ns;
    uint32_t sensor_id;
    float values[8];
};

template <typename T, size_t N>
class RingBuffer {
    static_assert((N & (N - 1)) == 0, "Capacity must be a power of two");
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    std::array<T, N> buffer_;
public:
    bool push(const T& item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t current_head = head_.load(std::memory_order_acquire);
        if ((current_tail - current_head) == N) return false;
        buffer_[current_tail & (N - 1)] = item;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }
    bool pop(T& item) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        size_t current_tail = tail_.load(std::memory_order_acquire);
        if (current_head == current_tail) return false;
        item = buffer_[current_head & (N - 1)];
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }
};
"""
PROMPT_MEDIUM = (
    "Analyze the following high-throughput telemetry engine architecture and suggest AVX2 SIMD "
    "vectorization optimizations for aggregating batches of 64 sensor frames:\n\n"
    + (BASE_CODE_SNIPPET * 18)
    + "\n\nProvide the optimized AVX2 implementation for the batch aggregation loop."
)

SERVE_LOG_RE = re.compile(
    r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d\.]+) ms \(([\d\.]+) tok/s\), "
    r"(\d+) generated in ([\d\.]+) ms \(([\d\.]+) tok/s\)(?:, drafts accepted (\d+) of (\d+))?"
)


def get_gpu_telemetry():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "vram_used_mib": int(parts[0]),
            "vram_total_mib": int(parts[1]),
            "temp_c": int(parts[2]),
            "gpu_util_pct": int(parts[3])
        }
    except Exception:
        return {"vram_used_mib": 0, "vram_total_mib": 12288, "temp_c": 0, "gpu_util_pct": 0}


def build_env():
    env = os.environ.copy()
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    env["PATH"] = f"{cuda_path}\\bin;{cuda_path}\\bin\\x64;" + env.get("PATH", "")
    env["CUDA_MODULE_LOADING"] = "LAZY"
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
    env["STRATA_GDN_MMA"] = "1"
    env["STRATA_FATTN"] = "1"
    env["STRATA_FDEC"] = "1"
    env["STRATA_SLOTS_WARP"] = "1"
    env["STRATA_POLY_STREAM_STAGE"] = "1"
    env["STRATA_POLY_PIN_EXPERTS"] = "0"
    env["STRATA_CACHE_BIAS"] = "0.42"
    env["STRATA_CACHE_BIAS_OUTER"] = "0.15"
    env["STRATA_ROUTE_TAIL_SKIP"] = "7"
    env["STRATA_CPU_SPIN"] = "200000"
    env["OMP_NUM_THREADS"] = "6"
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "cores"
    env["PYTHONPATH"] = f"{ROOT};{ROOT}\\tools;" + env.get("PYTHONPATH", "")
    env["STRATA_ALLOWED_HOSTS"] = "*"
    return env


def parse_latest_log_timing(log_path: Path):
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in reversed(lines[-30:]):
            m = SERVE_LOG_RE.search(line)
            if m:
                return {
                    "prompt_total": int(m.group(1)),
                    "prompt_reused": int(m.group(2)),
                    "prompt_read": int(m.group(3)),
                    "prompt_ms": float(m.group(4)),
                    "prefill_tok_s": float(m.group(5)),
                    "generated": int(m.group(6)),
                    "decode_ms": float(m.group(7)),
                    "decode_tok_s": float(m.group(8)),
                    "drafts_accepted": int(m.group(9)) if m.group(9) else 0,
                    "drafts_total": int(m.group(10)) if m.group(10) else 0,
                }
    except Exception as e:
        print(f"Log parse error: {e}", flush=True)
    return None


def send_chat_completion(model_id: str, prompt: str, max_tokens: int = 128):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        res_body = json.loads(resp.read().decode("utf-8"))
    t1 = time.perf_counter()
    return res_body, (t1 - t0)


def benchmark_model(model_name: str, config_name: str, env: dict):
    config_path = ROOT / config_name
    print(f"\n{'='*75}", flush=True)
    print(f"  AUDITING: {model_name}", flush=True)
    print(f"  Config:   {config_path.name}", flush=True)
    print(f"{'='*75}", flush=True)

    cfg_data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    log_path = Path(cfg_data.get("log") or ROOT / f"{config_path.stem}.log")

    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(ROOT / "serve" / "server.py"),
        "--engine", "strata",
        "--config", str(config_path),
        "--port", str(PORT)
    ]

    t_start_load = time.time()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    server_ready = False
    load_time = 0
    try:
        for sec in range(50):
            time.sleep(1)
            if proc.poll() is not None:
                print(f"  [!] Process exited early with code {proc.returncode}", flush=True)
                break
            try:
                with urllib.request.urlopen(f"{API_BASE}/health", timeout=1) as resp:
                    if resp.status == 200:
                        server_ready = True
                        load_time = time.time() - t_start_load
                        print(f"  [*] Server initialized in {load_time:.1f}s", flush=True)
                        break
            except Exception:
                pass
        
        if not server_ready:
            print("  [!] Failed to initialize server.", flush=True)
            return None

        load_telemetry = get_gpu_telemetry()
        print(f"  [*] VRAM Allocated: {load_telemetry['vram_used_mib']} MiB / {load_telemetry['vram_total_mib']} MiB ({load_telemetry['vram_used_mib']/load_telemetry['vram_total_mib']*100:.1f}%)", flush=True)

        # Warmup pass
        print("  [*] Running warmup pass...", flush=True)
        try:
            send_chat_completion(model_name, "Hello, reply with OK.", max_tokens=16)
        except Exception as e:
            print(f"  [!] Warmup error: {e}", flush=True)

        # Test 1: Short Coding Prompt
        print("  [*] Executing Test 1: Short Prompt (Coding)...", flush=True)
        res1, total1 = send_chat_completion(model_name, PROMPT_SHORT, max_tokens=128)
        timing1 = parse_latest_log_timing(log_path)
        gen_tok1 = res1.get("usage", {}).get("completion_tokens", 128)
        prompt_tok1 = res1.get("usage", {}).get("prompt_tokens", 0)

        prefill1 = timing1["prefill_tok_s"] if timing1 else (prompt_tok1 / (total1 * 0.3) if prompt_tok1 else 0)
        decode1 = timing1["decode_tok_s"] if timing1 else (gen_tok1 / (total1 * 0.7) if gen_tok1 else 0)
        print(f"      Prefill: {prefill1:.1f} tok/s | Decode: {decode1:.1f} tok/s | Gen: {gen_tok1} tokens", flush=True)

        # Test 2: Medium Working Context (~4K tokens)
        print("  [*] Executing Test 2: Medium Working Context (~4,000 tokens)...", flush=True)
        res2, total2 = send_chat_completion(model_name, PROMPT_MEDIUM, max_tokens=128)
        timing2 = parse_latest_log_timing(log_path)
        gen_tok2 = res2.get("usage", {}).get("completion_tokens", 128)
        prompt_tok2 = res2.get("usage", {}).get("prompt_tokens", 0)

        prefill2 = timing2["prefill_tok_s"] if timing2 else (prompt_tok2 / (total2 * 0.5) if prompt_tok2 else 0)
        decode2 = timing2["decode_tok_s"] if timing2 else (gen_tok2 / (total2 * 0.5) if gen_tok2 else 0)
        print(f"      Prefill: {prefill2:.1f} tok/s | Decode: {decode2:.1f} tok/s | Context: {prompt_tok2} tokens", flush=True)

        peak_telemetry = get_gpu_telemetry()
        
        draft_rate = 0.0
        if timing2 and timing2.get("drafts_total", 0) > 0:
            draft_rate = (timing2["drafts_accepted"] / timing2["drafts_total"]) * 100
        elif timing1 and timing1.get("drafts_total", 0) > 0:
            draft_rate = (timing1["drafts_accepted"] / timing1["drafts_total"]) * 100

        result = {
            "model_name": model_name,
            "load_time_s": round(load_time, 1),
            "vram_load_mib": load_telemetry["vram_used_mib"],
            "vram_peak_mib": peak_telemetry["vram_used_mib"],
            "vram_pct": round(peak_telemetry["vram_used_mib"] / peak_telemetry["vram_total_mib"] * 100, 1),
            "short_prefill_tok_s": round(prefill1, 1),
            "short_decode_tok_s": round(decode1, 1),
            "medium_context_tokens": prompt_tok2,
            "medium_prefill_tok_s": round(prefill2, 1),
            "medium_decode_tok_s": round(decode2, 1),
            "speculative_draft_rate_pct": round(draft_rate, 1),
            "gpu_temp_c": peak_telemetry["temp_c"]
        }
        return result

    finally:
        print("  [*] Tearing down engine...", flush=True)
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        except Exception:
            pass
        time.sleep(2)


def main():
    print("===========================================================================", flush=True)
    print("  POLYSTRATA COMPLETE MODEL AUDIT (RTX 3060 12GB + i5-12400 Rig)", flush=True)
    print("===========================================================================", flush=True)
    env = build_env()
    results = []
    bench_dir = ROOT / "bench"
    bench_dir.mkdir(exist_ok=True)
    json_path = bench_dir / "audit_results_20261030.json"
    csv_path = bench_dir / "audit_results_20261030.csv"

    for model_name, config_file in MODELS_ORDER:
        try:
            res = benchmark_model(model_name, config_file, env)
            if res:
                results.append(res)
                # Incremental write
                json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"  [!] Benchmark failed for {model_name}: {e}", flush=True)

    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)

    print("\n" + "=" * 95, flush=True)
    print("  AUDIT COMPLETE — SUMMARY MATRIX", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Model':<34} | {'Short Decode':<13} | {'Short Prefill':<14} | {'Med Prefill':<13} | {'Peak VRAM':<10}", flush=True)
    print("-" * 95, flush=True)
    for r in results:
        print(f"{r['model_name']:<34} | {r['short_decode_tok_s']:>6.1f} tok/s | {r['short_prefill_tok_s']:>7.1f} tok/s | {r['medium_prefill_tok_s']:>6.1f} tok/s | {r['vram_peak_mib']:>5} MiB", flush=True)
    print("=" * 95, flush=True)
    print(f"Results saved to: {json_path}", flush=True)
    print(f"CSV saved to:     {csv_path}\n", flush=True)


if __name__ == "__main__":
    main()
