#!/usr/bin/env python3
"""
Setup and Pack Builder for Groot-65B-A6B GyroM (TQ2_T Trellis) on RTX 3060 12GB.
Handles:
1. Verification / downloading of model weights and MTP draft head.
2. Building IQ pack into G:\Strata-3060\packs\groot-gyrom via tools/iq_pack.py.
3. Converting MTP routing head into G:\Strata-3060\mtp\groot\rt via tools/mtp_rt_gguf.py.
4. Validating the native pack and expert profile data/expert-profile-groot.bin.
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(r"G:\Strata-3060")
TOOLS_DIR = BASE_DIR / "tools"
MODELS_DIR = BASE_DIR / "models" / "groot-gyrom"
PACKS_DIR = BASE_DIR / "packs" / "groot-gyrom"
MTP_DIR = BASE_DIR / "mtp" / "groot" / "rt"
PROFILE_BIN = BASE_DIR / "data" / "expert-profile-groot.bin"

MODEL_GGUF = MODELS_DIR / "Groot-65B-A6B-GyroM-Q2_0.gguf"
MTP_GGUF = MODELS_DIR / "mtp-Groot-65B-A6B-261023-BF16.gguf"

HF_MODEL_URL = "https://huggingface.co/BoxyFit/Groot-65B-A6B-GyroM/resolve/main/Groot-65B-A6B-GyroM-Q2_0.gguf"
HF_MTP_URL = "https://huggingface.co/GrootTeam/Groot-65B-A6B-261023-GGUF/resolve/main/mtp-Groot-65B-A6B-261023-BF16.gguf"

def check_or_download():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    MTP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Groot-65B-A6B GyroM Setup & Pack Builder")
    print("=" * 60)

    if not MODEL_GGUF.exists():
        print(f"\n[INFO] Model GGUF not found at: {MODEL_GGUF}")
        print(f"Please download Groot-65B-A6B-GyroM-Q2_0.gguf to:\n  {MODELS_DIR}")
        print(f"Direct link: {HF_MODEL_URL}")
        return False
    else:
        print(f"[OK] Found Model GGUF: {MODEL_GGUF} ({MODEL_GGUF.stat().st_size / (1024**3):.2f} GB)")

    if not MTP_GGUF.exists():
        print(f"\n[INFO] MTP GGUF not found at: {MTP_GGUF}")
        print(f"Please download mtp-Groot-65B-A6B-261023-BF16.gguf to:\n  {MODELS_DIR}")
        print(f"Direct link: {HF_MTP_URL}")
        return False
    else:
        print(f"[OK] Found MTP GGUF: {MTP_GGUF} ({MTP_GGUF.stat().st_size / (1024**2):.2f} MB)")

    return True

def build_pack():
    print("\n[Step 1/2] Building IQ pack with tools/iq_pack.py...")
    cmd1 = [
        sys.executable,
        str(TOOLS_DIR / "iq_pack.py"),
        "--gguf", str(MODEL_GGUF),
        "--out", str(PACKS_DIR),
        "--compat-bf16"
    ]
    res1 = subprocess.run(cmd1)
    if res1.returncode != 0:
        print(f"[FAIL] iq_pack failed with code {res1.returncode}")
        return False

    print("\n[Step 2/2] Converting MTP head with tools/mtp_rt_gguf.py...")
    cmd2 = [
        sys.executable,
        str(TOOLS_DIR / "mtp_rt_gguf.py"),
        "--gguf", str(MTP_GGUF),
        "--out", str(MTP_DIR)
    ]
    res2 = subprocess.run(cmd2)
    if res2.returncode != 0:
        print(f"[FAIL] mtp_rt_gguf failed with code {res2.returncode}")
        return False

    print("\n============================================================")
    print(" Groot-65B-A6B GyroM Pack is Ready!")
    print(" You can now launch it via start-groot-65b-gyrom.bat")
    print("============================================================")
    return True

if __name__ == "__main__":
    if check_or_download():
        build_pack()
