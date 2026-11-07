#!/usr/bin/env python3
"""
Fast Multi-Threaded Resumable Downloader for Groot-65B-A6B GyroM Weights.
Downloads:
  1. mtp-Groot-65B-A6B-261023-BF16.gguf (5.25 GB)
  2. Groot-65B-A6B-GyroM-Q2_0.gguf (54.52 GB)
Followed by automated pack and MTP runtime preparation.
"""

import os
import sys
import time
import json
import threading
import urllib.request
import concurrent.futures
import subprocess
from pathlib import Path

BASE_DIR = Path(r"G:\Strata-3060")
MODELS_DIR = BASE_DIR / "models" / "groot-gyrom"
NUM_WORKERS = 8
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB read chunks

ITEMS = [
    {
        "name": "MTP Draft Head (mtp-Groot-65B-A6B-261023-BF16.gguf)",
        "url": "https://huggingface.co/GrootTeam/Groot-65B-A6B-261023-GGUF/resolve/main/mtp-Groot-65B-A6B-261023-BF16.gguf",
        "dest": MODELS_DIR / "mtp-Groot-65B-A6B-261023-BF16.gguf",
        "expected_bytes": 5251556576
    },
    {
        "name": "Groot-65B-A6B GyroM Model (Groot-65B-A6B-GyroM-Q2_0.gguf)",
        "url": "https://huggingface.co/BoxyFit/Groot-65B-A6B-GyroM/resolve/main/Groot-65B-A6B-GyroM-Q2_0.gguf",
        "dest": MODELS_DIR / "Groot-65B-A6B-GyroM-Q2_0.gguf",
        "expected_bytes": 54524628160
    }
]

def download_worker(worker_id, url, dest_path, start_byte, end_byte, progress_tracker, stop_event):
    bytes_to_fetch = end_byte - start_byte + 1
    if bytes_to_fetch <= 0:
        return True

    retries = 15
    for attempt in range(retries):
        if stop_event.is_set():
            return False
        try:
            current_start = start_byte + progress_tracker.get_worker_done(worker_id)
            if current_start > end_byte:
                return True

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Range": f"bytes={current_start}-{end_byte}"
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp:
                with open(dest_path, "r+b") as f:
                    f.seek(current_start)
                    while current_start <= end_byte:
                        if stop_event.is_set():
                            return False
                        to_read = min(CHUNK_SIZE, end_byte - current_start + 1)
                        data = resp.read(to_read)
                        if not data:
                            break
                        f.write(data)
                        f.flush()
                        n = len(data)
                        current_start += n
                        progress_tracker.add_bytes(worker_id, n)
            if current_start > end_byte:
                return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 15))
            else:
                print(f"\n[Worker {worker_id}] Error after {retries} attempts: {e}")
                return False
    return False

class ProgressTracker:
    def __init__(self, state_file, num_workers, total_bytes):
        self.state_file = state_file
        self.num_workers = num_workers
        self.total_bytes = total_bytes
        self.lock = threading.Lock()
        self.worker_done = [0] * num_workers
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    if len(data.get("workers", [])) == self.num_workers and data.get("total") == self.total_bytes:
                        self.worker_done = data["workers"]
            except Exception:
                pass

    def save_state(self):
        with self.lock:
            try:
                with open(self.state_file, "w") as f:
                    json.dump({"total": self.total_bytes, "workers": self.worker_done}, f)
            except Exception:
                pass

    def add_bytes(self, worker_id, count):
        with self.lock:
            self.worker_done[worker_id] += count

    def get_worker_done(self, worker_id):
        with self.lock:
            return self.worker_done[worker_id]

    def get_total_done(self):
        with self.lock:
            return sum(self.worker_done)

def download_item(item):
    name = item["name"]
    url = item["url"]
    dest = item["dest"]
    expected = item["expected_bytes"]
    part_file = dest.with_suffix(dest.suffix + ".part")
    state_file = dest.with_suffix(dest.suffix + ".state.json")

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == expected:
        print(f"[{name}] Already downloaded and verified! ({expected / (1024**3):.2f} GB)")
        return True

    print("\n" + "=" * 65)
    print(f"Downloading: {name}")
    print(f"Target:      {dest}")
    print(f"Size:        {expected / (1024**3):.2f} GB ({expected:,} bytes)")
    print(f"Concurrency: {NUM_WORKERS} parallel streams")
    print("=" * 65)

    # Pre-allocate sparse .part file if needed (FSCTL_SET_SPARSE for ReFS Dev Drive)
    if not part_file.exists() or part_file.stat().st_size != expected:
        print(f"Pre-allocating sparse file for {dest.name}...")
        try:
            import msvcrt
            import ctypes
            from ctypes import wintypes
            FSCTL_SET_SPARSE = 0x000900C4
            with open(part_file, "wb") as f:
                handle = msvcrt.get_osfhandle(f.fileno())
                bytes_returned = wintypes.DWORD()
                ctypes.windll.kernel32.DeviceIoControl(
                    handle, FSCTL_SET_SPARSE, None, 0, None, 0, ctypes.byref(bytes_returned), None
                )
                f.seek(expected - 1)
                f.write(b"\0")
        except Exception:
            with open(part_file, "wb") as f:
                f.seek(expected - 1)
                f.write(b"\0")
        if state_file.exists():
            os.remove(state_file)

    tracker = ProgressTracker(state_file, NUM_WORKERS, expected)
    partition_size = expected // NUM_WORKERS

    workers_ranges = []
    for i in range(NUM_WORKERS):
        start = i * partition_size
        end = (i + 1) * partition_size - 1 if i < NUM_WORKERS - 1 else expected - 1
        workers_ranges.append((start, end))

    stop_event = threading.Event()
    start_time = time.time()
    initial_done = tracker.get_total_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(
                download_worker,
                i,
                url,
                part_file,
                workers_ranges[i][0],
                workers_ranges[i][1],
                tracker,
                stop_event
            )
            for i in range(NUM_WORKERS)
        ]

        # Monitor loop
        last_save = time.time()
        while True:
            done_cnt = sum(1 for f in futures if f.done())
            now = time.time()
            elapsed = max(now - start_time, 0.001)
            current_done = tracker.get_total_done()
            downloaded_since_start = current_done - initial_done
            speed = downloaded_since_start / elapsed / (1024 * 1024)
            pct = (current_done / expected) * 100.0

            remaining_bytes = expected - current_done
            if speed > 0:
                eta_sec = remaining_bytes / (speed * 1024 * 1024)
                if eta_sec < 60:
                    eta_str = f"{int(eta_sec)}s"
                elif eta_sec < 3600:
                    eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
                else:
                    eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
            else:
                eta_str = "calculating..."

            print(
                f"\r[{pct:5.1f}%] {current_done / (1024**3):5.2f} / {expected / (1024**3):.2f} GB "
                f"| Speed: {speed:5.1f} MB/s | ETA: {eta_str} | Streams: {NUM_WORKERS - done_cnt}/{NUM_WORKERS}",
                end="",
                flush=True
            )

            if now - last_save >= 5.0:
                tracker.save_state()
                last_save = now

            if done_cnt == NUM_WORKERS:
                break
            time.sleep(1.0)

        results = [f.result() for f in futures]
        tracker.save_state()

    print()
    if not all(results) or tracker.get_total_done() != expected:
        print(f"[FAIL] Download incomplete ({tracker.get_total_done()}/{expected} bytes). Run again to resume.")
        return False

    # Atomic move
    if dest.exists():
        os.remove(dest)
    part_file.rename(dest)
    if state_file.exists():
        os.remove(state_file)

    total_time = time.time() - start_time
    avg_speed = (expected - initial_done) / max(total_time, 0.001) / (1024 * 1024)
    print(f"[SUCCESS] {dest.name} verified! ({expected / (1024**3):.2f} GB in {total_time:.1f}s, avg {avg_speed:.1f} MB/s)")
    return True

def run_pack_setup():
    print("\n" + "=" * 65)
    print(" Running setup_groot_65b_gyrom.py to generate pack & MTP runtime...")
    print("=" * 65)
    script_path = BASE_DIR / "tools" / "setup_groot_65b_gyrom.py"
    res = subprocess.run([sys.executable, str(script_path)])
    return res.returncode == 0

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for item in ITEMS:
        ok = download_item(item)
        if not ok:
            sys.exit(1)

    print("\nAll weights downloaded successfully! Now building pack & MTP runtime...")
    if run_pack_setup():
        print("\n============================================================")
        print(" Groot-65B-A6B GyroM is fully installed, packed, and ready!")
        print(" Launch it from your Desktop: 'Groot 65B GyroM (am2strata)'")
        print("============================================================")
    else:
        print("[ERROR] Pack generation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
