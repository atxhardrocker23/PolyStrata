import os
import sys
import time
import urllib.request

URL = "https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-abliterated-i1-GGUF/resolve/main/gemma-4-26B-A4B-it-abliterated.i1-IQ3_S.gguf"
DEST_DIR = r"G:\Strata-3060\models\gemma4-nanoplus"
DEST_FILE = os.path.join(DEST_DIR, "gemma-4-26B-A4B-it-abliterated.i1-IQ3_S.gguf")
PART_FILE = DEST_FILE + ".part"
EXPECTED_BYTES = 12222405504

def download():
    os.makedirs(DEST_DIR, exist_ok=True)
    if os.path.exists(DEST_FILE):
        actual_size = os.path.getsize(DEST_FILE)
        if actual_size == EXPECTED_BYTES:
            print(f"[Done] {DEST_FILE} already exists with full size ({actual_size} bytes).")
            return
        else:
            print(f"[Warning] Incomplete file found ({actual_size} bytes). Re-downloading...")
            os.remove(DEST_FILE)

    print(f"Starting download: Gemma 4 26B-A4B IT Abliterated i1-IQ3_S (~12.2 GB)")
    print(f"URL: {URL}")
    print(f"Destination: {DEST_FILE}")

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    start_time = time.time()
    last_print = start_time

    downloaded = 0
    chunk_size = 1024 * 1024 * 4  # 4 MB chunks

    with urllib.request.urlopen(req) as resp, open(PART_FILE, "wb") as out:
        total_len = int(resp.headers.get("Content-Length", EXPECTED_BYTES))
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 5.0 or downloaded == total_len:
                elapsed = max(now - start_time, 0.001)
                speed_mb = (downloaded / (1024 * 1024)) / elapsed
                pct = (downloaded / total_len) * 100
                dl_gb = downloaded / (1024 ** 3)
                tot_gb = total_len / (1024 ** 3)
                rem_seconds = (total_len - downloaded) / (speed_mb * 1024 * 1024) if speed_mb > 0 else 0
                print(f"[{pct:5.1f}%] {dl_gb:5.2f} / {tot_gb:5.2f} GB | Speed: {speed_mb:5.1f} MB/s | ETA: {rem_seconds:4.0f}s", flush=True)
                last_print = now

    if os.path.exists(DEST_FILE):
        os.remove(DEST_FILE)
    os.rename(PART_FILE, DEST_FILE)
    elapsed_total = time.time() - start_time
    print(f"\n[SUCCESS] Download completed in {elapsed_total:.1f}s. Saved to {DEST_FILE}")

if __name__ == "__main__":
    download()
