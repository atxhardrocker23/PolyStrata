"""Custom PolyStrata preset: huihui-ai's abliterated Qwen3.6-35B-A3B (MTP GGUF, Q4_K = Q4_K_M).

Registers a `qwen36abl` family next to the stock `qwen36` one (same strata-poly engine, same qwen35moe
architecture, draft block in the file) and then runs PolyStrata's normal setup.py main().  setup.py itself is not
edited, so UPDATE.bat / git pull keep working.

Usage (from this folder):
    .venv\\Scripts\\python.exe setup_abliterated.py --yes --family qwen36abl --model Q4_K_M \\
        --models-dir models --vision no --no-start
"""
import copy
import sys

import setup as S

REPO = "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF"
REV = "acd5abe0c5ebd064f3863ca7d4bba2d924ec6544"           # pinned revision
FILE = "Huihui-Qwen3.6-35B-A3B-abliterated-ggml-model-Q4_K.gguf"
SIZE = 21712409824
SHA256 = "63b75afb4b68fc61059c78115b580037074de88146e0762f4a496081e410cf81"  # from the HF LFS pointer

fam = copy.deepcopy(S.FAMILIES["qwen36"])
fam.update({
    "title": "Qwen3.6-35B-A3B abliterated",
    "by": "Qwen; abliterated and quantized by huihui-ai",
    "about": "huihui-ai's abliterated Qwen3.6-35B-A3B (MTP GGUF, Q4_K_M) on the strata-poly engine",
    "hf": f"{S.hf_endpoint()}/{REPO}/resolve/{REV}/",
    "file": FILE,
    "tag": "qwen36abl-",
    "name": "qwen3.6-35b-a3b-abliterated",
    "vision": False,                                       # text only (no abliterated mmproj pinned here)
    "models": {"Q4_K_M": {"about": "4-bit K-quant experts (llama.cpp Q4_K_M), its draft block in the file",
                          "download_gb": round(SIZE / 1e9, 1), "ram_gb": 26, "arena_gb": 19.2,
                          "vram_beside_gb": 4.7, "draft": "own", "file": FILE}},
    "sha256": {FILE: (SIZE, SHA256)},
})
S.FAMILIES["qwen36abl"] = fam

if __name__ == "__main__":
    sys.exit(S.main())
