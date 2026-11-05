import glob
import json
import os

configs = glob.glob(r"G:\Strata-3060\am2strata\configs\*.json")
for c in sorted(configs):
    with open(c, "r", encoding="utf-8") as f:
        data = json.load(f)
        args = data.get("args", [])
        ctx = "none"
        res = "none"
        for i in range(len(args) - 1):
            if args[i] == "--max-context":
                ctx = args[i+1]
            if args[i] == "--vram-reserve-mib":
                res = args[i+1]
        env = data.get("env", {})
        pin = env.get("STRATA_POLY_PIN_EXPERTS", "none")
        print(f"{os.path.basename(c):<35} ctx={ctx:<8} pin={pin:<8} vram-res={res:<6}")
