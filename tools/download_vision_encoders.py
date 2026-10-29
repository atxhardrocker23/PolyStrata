import os
import sys
import time
import urllib.request

VISION_MODELS = [
    {
        "name": "Tiel-Coder & Cyber-Tiel Vision Encoder (Ornith 1.5 / Qwen3.5-VL)",
        "url": "https://huggingface.co/IsValorum/Tiel-Coder-35B-A3B-APEX-I-NanoPlus-GGUF/resolve/main/mmproj-Q8_0.gguf",
        "dest": r"G:\Strata-3060\models\mmproj-tiel-qwen35moe-q8_0.gguf",
        "expected_bytes": 614194176
    },
    {
        "name": "Gemma 4 26B Vision Encoder (gemma4)",
        "url": "https://huggingface.co/juan1995-dev/gemma-4-26B-A4B-it.mmproj-f16-GGUF/resolve/main/gemma-4-26B-A4B-it.mmproj-f16.gguf",
        "dest": r"G:\Strata-3060\models\mmproj-gemma-4-26b-f16.gguf",
        "expected_bytes": 1193058528
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
            print(f"[{name}] Already downloaded and verified! Size: {size / 1e6:.1f} MB")
            return True
        else:
            print(f"[{name}] Existing file size mismatch ({size} != {expected}). Re-downloading...")
            os.remove(dest)

    existing_bytes = 0
    if os.path.exists(part_file):
        existing_bytes = os.path.getsize(part_file)
        if existing_bytes > expected:
            os.remove(part_file)
            existing_bytes = 0

    print(f"\nDownloading: {name}")
    print(f"Target: {dest}")
    print(f"Size: {expected / 1e6:.1f} MB ({expected:,} bytes)")

    headers = {"User-Agent": "Mozilla/5.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(url, headers=headers)
    start_time = time.time()
    last_print = start_time
    bytes_downloaded = existing_bytes
    chunk_size = 4 * 1024 * 1024

    with urllib.request.urlopen(req, timeout=30) as resp, open(part_file, "ab" if existing_bytes > 0 else "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            bytes_downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 2.0 or bytes_downloaded == expected:
                elapsed = max(now - start_time, 0.001)
                speed = (bytes_downloaded - existing_bytes) / elapsed / (1024 * 1024)
                pct = (bytes_downloaded / expected) * 100.0
                print(f"[{pct:5.1f}%] {bytes_downloaded / 1e6:6.1f} / {expected / 1e6:.1f} MB | Speed: {speed:5.1f} MB/s", flush=True)
                last_print = now

    final_size = os.path.getsize(part_file)
    if final_size == expected:
        os.rename(part_file, dest)
        total_time = time.time() - start_time
        print(f"[{name}] Download complete in {total_time:.1f}s!")
        return True
    else:
        print(f"[{name}] ERROR: File size mismatch ({final_size} != {expected})")
        return False

def main():
    for item in VISION_MODELS:
        ok = download_file(item)
        if not ok:
            sys.exit(1)

if __name__ == "__main__":
    main()
