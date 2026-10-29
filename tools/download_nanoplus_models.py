import os
import sys
import time
import urllib.request

MODELS = [
    {
        "name": "Flint 40B-A2.5B APEX-I NanoPlus V2.1",
        "url": "https://huggingface.co/IsValorum/Flint-APEX-I-NanoPlus-V2.1-GGUF/resolve/main/Flint.APEX-I-NanoPlus-V2.1.gguf",
        "dest": r"G:\Strata-3060\models\flint-apex-nanoplus\Flint.APEX-I-NanoPlus-V2.1.gguf",
        "expected_bytes": 13237083360
    },
    {
        "name": "Tiel Coder 35B-A3B APEX-I NanoPlus",
        "url": "https://huggingface.co/IsValorum/Tiel-Coder-35B-A3B-APEX-I-NanoPlus-GGUF/resolve/main/Tiel-Coder-35B-A3B.APEX-I-NanoPlus.gguf",
        "dest": r"G:\Strata-3060\models\tiel-coder-nanoplus\Tiel-Coder-35B-A3B.APEX-I-NanoPlus.gguf",
        "expected_bytes": 12554424672
    }
]

def download_file(item):
    name = item["name"]
    url = item["url"]
    dest = item["dest"]
    expected = item["expected_bytes"]
    part_file = dest + ".part"

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size == expected:
            print(f"[{name}] Already downloaded and verified! Size: {size / 1e9:.2f} GB")
            return True
        else:
            print(f"[{name}] Existing file size mismatch ({size} != {expected}). Re-downloading...")
            os.remove(dest)

    existing_bytes = 0
    if os.path.exists(part_file):
        existing_bytes = os.path.getsize(part_file)
        if existing_bytes > expected:
            print(f"[{name}] Partial file corrupt. Restarting...")
            os.remove(part_file)
            existing_bytes = 0

    print(f"\n========================================================")
    print(f"Downloading: {name}")
    print(f"Target: {dest}")
    print(f"Total Size: {expected / 1e9:.2f} GB ({expected:,} bytes)")
    if existing_bytes > 0:
        print(f"Resuming from: {existing_bytes / 1e9:.2f} GB ({existing_bytes:,} bytes)")
    print(f"========================================================")

    headers = {"User-Agent": "Mozilla/5.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(url, headers=headers)
    
    start_time = time.time()
    last_print = start_time
    bytes_downloaded = existing_bytes
    chunk_size = 8 * 1024 * 1024  # 8 MB chunks

    with urllib.request.urlopen(req, timeout=30) as resp, open(part_file, "ab" if existing_bytes > 0 else "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            bytes_downloaded += len(chunk)
            
            now = time.time()
            if now - last_print >= 5.0 or bytes_downloaded == expected:
                elapsed = max(now - start_time, 0.001)
                speed = (bytes_downloaded - existing_bytes) / elapsed / (1024 * 1024)
                pct = (bytes_downloaded / expected) * 100.0
                eta_sec = (expected - bytes_downloaded) / max((bytes_downloaded - existing_bytes) / elapsed, 1)
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s" if eta_sec < 3600 else f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
                print(f"[{pct:5.1f}%] {bytes_downloaded / 1e9:5.2f} / {expected / 1e9:.2f} GB | Speed: {speed:5.1f} MB/s | ETA: {eta_str}", flush=True)
                last_print = now

    final_size = os.path.getsize(part_file)
    if final_size == expected:
        os.rename(part_file, dest)
        total_time = time.time() - start_time
        avg_speed = (expected - existing_bytes) / max(total_time, 0.001) / (1024 * 1024)
        print(f"[{name}] Download complete! Average speed: {avg_speed:.1f} MB/s in {total_time:.1f}s")
        return True
    else:
        print(f"[{name}] ERROR: File size mismatch after download! ({final_size} != {expected})")
        return False

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    for item in MODELS:
        if target == "all" or target.lower() in item["name"].lower():
            ok = download_file(item)
            if not ok:
                print(f"Failed downloading {item['name']}")
                sys.exit(1)

if __name__ == "__main__":
    main()
