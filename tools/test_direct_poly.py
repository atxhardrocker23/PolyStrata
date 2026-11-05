import json
import subprocess
from pathlib import Path

cfg_path = Path(r"G:\Strata-3060\am2strata\configs\strata-tiel-coder-nanoplus.json")
data = json.loads(cfg_path.read_text(encoding="utf-8"))

# Test removing STRATA_POLY_PIN_EXPERTS
if "STRATA_POLY_PIN_EXPERTS" in data.get("env", {}):
    del data["env"]["STRATA_POLY_PIN_EXPERTS"]

temp_cfg = Path(r"G:\Strata-3060\am2strata\configs\test_temp.json")
temp_cfg.write_text(json.dumps(data, indent=2), encoding="utf-8")

cmd = [
    data["exe"],
    "--serve"
] + data["args"]

print("Starting engine test directly with strata-poly...")
env = dict(data.get("env", {}))
env.update({
    "CUDA_MODULE_LOADING": "LAZY"
})

p = subprocess.Popen(
    cmd,
    cwd=data.get("cwd", r"G:\Strata-3060"),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

import time
start = time.time()
while time.time() - start < 40:
    line = p.stdout.readline()
    if not line:
        break
    print("POLY:", line.strip())
    if "cudaMemGetInfo" in line or "out of memory" in line:
        print("[FAIL] Out of memory detected!")
        break
    if "serving on" in line or "ready" in line or "listening" in line or "answers" in line:
        print("[SUCCESS] Engine is ready!")
        break

p.terminate()
try:
    p.wait(timeout=5)
except Exception:
    p.kill()

if temp_cfg.exists():
    temp_cfg.unlink()
