import glob
import json
from pathlib import Path

config_dirs = [
    Path(r"G:\Strata-3060\am2strata\configs"),
    Path(r"G:\Strata-3060")
]

for cdir in config_dirs:
    for cfile in cdir.glob("strata-*.json"):
        try:
            cfg = json.loads(cfile.read_text(encoding="utf-8"))
        except Exception:
            continue

        changed = False

        # 1. Remove STRATA_POLY_PIN_EXPERTS from env
        if "env" in cfg and "STRATA_POLY_PIN_EXPERTS" in cfg["env"]:
            del cfg["env"]["STRATA_POLY_PIN_EXPERTS"]
            changed = True

        # 2. Adjust --vram-reserve-mib to 450 (safe margin on 12GB RTX 3060)
        args = cfg.get("args", [])
        for i in range(len(args) - 1):
            if args[i] == "--vram-reserve-mib":
                if args[i+1] == "300":
                    args[i+1] = "450"
                    changed = True

        if changed:
            cfile.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            print(f"[OK] Fixed VRAM allocation safety in {cfile}")

print("Done adjusting configs.")
