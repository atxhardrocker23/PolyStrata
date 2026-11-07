"""tools/mtp_rt_gguf.py - the MTP draft layer's runtime files (the `--mtp` directory) from a llama.cpp qwen4exp MTP GGUF.

Some qwen4exp models ship their MTP head as a standalone llama.cpp GGUF instead of in a BF16 checkpoint that
tools/mtp_fetch.py can read: Groot-65B-A6B-261023 (GrootTeam/Groot-65B-A6B-261023-GGUF,
mtp-Groot-65B-A6B-261023-BF16.gguf, converted by ggml-org/llama.cpp b38960f52's conversion/qwen4exp.py).  This tool
undoes that converter's renames and merges and writes what tools/mtp_pack.py + tools/mtp_rt.py write for the
canonical head, with the GGUF's own expert count (256 for Groot; the engine reads it from experts.bin's size):

    python tools/mtp_rt_gguf.py --gguf mtp-Groot-65B-A6B-261023-BF16.gguf --out <dir>/rt

The converter's transformations, undone here (conversion/qwen4exp.py and qwen.py at b38960f52):
  * fc_embedding + fc_hidden were concatenated along the input axis into nextn.eh_proj (eh([e ; h_s])) -> split.
  * indexer.index_qk_proj was split into indexer.q_proj (the first heads * head_dim rows) and k_proj -> joined.
  * experts.gate_up_proj was split into ffn_gate_exps / ffn_up_exps (gate rows first) -> joined.
  * every RMSNorm weight became 1 + w (Gemma zero-centred gammas).  The engine's dense.txt keeps 1 + w too
    (tools/mtp_rt.py adds it to the checkpoint's raw w), so the GGUF's F32 norms are copied as they are.
  * the router (ffn_gate_inp) and the shared expert gate were stored F32: exact BF16 values, written back as BF16.
The routed experts are quantized to Q2_0 with tools/mtp_pack.py's error-minimizing scales, as setup does for the
canonical head; the large projections become Q8_0 and the hyper-connection weights stay BF16, as in tools/mtp_rt.py.
token_embd / output are not written: the drafter reads the main model's embedding and head.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402

import mtp_pack  # noqa: E402  (q2_0)
import mtp_rt  # noqa: E402  (q8_0, blob_of, Q8, H, FF, BLOB)

H, FF = mtp_rt.H, mtp_rt.FF

# dense.txt name (tools/mtp_rt.py's short names) <- the GGUF tensor after blk.N.
SIMPLE = {
    "hyper_connection_mixer.hc_norm.weight": "nextn.hc_head_norm.weight",
    "hyper_connection_mixer.input_mix_weight_down.weight": "nextn.hc_head_down.weight",
    "hyper_connection_mixer.input_mix_weight_up.weight": "nextn.hc_head_up.weight",
    "attn_hyper_connection.block_inject_weight.weight": "hc_attn_inject.weight",
    "attn_hyper_connection.hc_norm.weight": "hc_attn_norm.weight",
    "attn_hyper_connection.input_mix_weight_down.weight": "hc_attn_down.weight",
    "attn_hyper_connection.input_mix_weight_up.weight": "hc_attn_up.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
    "mlp_hyper_connection.block_inject_weight.weight": "hc_ffn_inject.weight",
    "mlp_hyper_connection.hc_norm.weight": "hc_ffn_norm.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight": "hc_ffn_down.weight",
    "mlp_hyper_connection.input_mix_weight_up.weight": "hc_ffn_up.weight",
    "self_attn.indexer.k_layernorm.weight": "indexer.k_norm.weight",
    "self_attn.indexer.q_layernorm.weight": "indexer.q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "pre_fc_norm_embedding.weight": "nextn.enorm.weight",
    "pre_fc_norm_hidden.weight": "nextn.hnorm.weight",
}
# tools/mtp_rt.py's order (the canonical head's dense.txt), the split / joined ones in their places
ORDER = ["fc_embedding.weight", "fc_hidden.weight",
         "hyper_connection_mixer.hc_norm.weight", "hyper_connection_mixer.input_mix_weight_down.weight",
         "hyper_connection_mixer.input_mix_weight_up.weight",
         "attn_hyper_connection.block_inject_weight.weight", "attn_hyper_connection.hc_norm.weight",
         "attn_hyper_connection.input_mix_weight_down.weight", "attn_hyper_connection.input_mix_weight_up.weight",
         "mlp.gate.weight", "mlp.shared_expert.down_proj.weight", "mlp.shared_expert.gate_proj.weight",
         "mlp.shared_expert.up_proj.weight", "mlp.shared_expert_gate.weight",
         "mlp_hyper_connection.block_inject_weight.weight", "mlp_hyper_connection.hc_norm.weight",
         "mlp_hyper_connection.input_mix_weight_down.weight", "mlp_hyper_connection.input_mix_weight_up.weight",
         "self_attn.indexer.index_qk_proj.weight", "self_attn.indexer.k_layernorm.weight",
         "self_attn.indexer.q_layernorm.weight", "self_attn.k_norm.weight", "self_attn.k_proj.weight",
         "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight",
         "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"]
NORMS = {n for n in ORDER if n.endswith(("norm.weight", "layernorm.weight", "norm_embedding.weight",
                                         "norm_hidden.weight"))}


def values(t) -> np.ndarray:
    """A GGUF tensor's values in its torch shape: gguf-py hands BF16 out as raw bytes (uint8, twice the last axis)."""
    data = np.asarray(t.data)
    if int(t.tensor_type) == int(gguf.GGMLQuantizationType.BF16) and data.dtype == np.uint8:
        data = data.view(np.uint16)
    return data


def as_f32(t) -> np.ndarray:
    """A GGUF tensor (BF16 or F32) as float32 in its torch shape (numpy's view of a GGUF tensor is that shape)."""
    data = values(t)
    tt = int(t.tensor_type)
    if tt == int(gguf.GGMLQuantizationType.F32):
        return data.astype(np.float32)
    if tt == int(gguf.GGMLQuantizationType.BF16):
        u16 = data.view(np.uint16) if data.dtype != np.uint16 else data
        return (u16.astype(np.uint32) << 16).view(np.float32)
    raise ValueError(f"{t.name}: type {t.tensor_type!r}, expected BF16 or F32 (the BF16 MTP companion)")


def to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bits; exact for values that came from BF16 (asserted)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    if np.any(u & 0xFFFF):
        raise ValueError("a tensor stored F32 is not BF16-exact; refusing to round it silently")
    return (u >> 16).astype(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="the llama.cpp qwen4exp MTP GGUF (BF16)")
    ap.add_argument("--out", required=True, help="the runtime directory to write (the engine's --mtp)")
    ap.add_argument("--draft-vocab", default=str(Path(__file__).resolve().parent.parent / "data" / "draft_vocab.bin"),
                    help="the draft head's token subset copied beside it (default data/draft_vocab.bin; '' = none)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = gguf.GGUFReader(a.gguf)
    arch = r.fields["general.architecture"].contents()
    if arch != "qwen4exp":
        print(f"{a.gguf} is a '{arch}' GGUF, not qwen4exp", file=sys.stderr)
        return 1
    n_blk = int(r.fields["qwen4exp.block_count"].contents())
    ne = int(r.fields["qwen4exp.expert_count"].contents())
    nextn = r.fields.get("qwen4exp.nextn_predict_layers")
    if nextn is None or int(nextn.contents()) != 1:
        print("expected one MTP layer (qwen4exp.nextn_predict_layers = 1)", file=sys.stderr)
        return 1
    pre = f"blk.{n_blk - 1}."
    tens = {t.name[len(pre):]: t for t in r.tensors if t.name.startswith(pre)}

    def get(name: str):
        if name not in tens:
            raise KeyError(f"{pre}{name} is not in {a.gguf}")
        return tens[name]

    # ---- the routed experts: gate (ne, FF, H) + up -> gate_up (ne, 2FF, H); down (ne, H, FF); Q2_0 blobs
    g, u, d = (get(f"ffn_{x}_exps.weight") for x in ("gate", "up", "down"))
    for t, shape in ((g, (ne, FF, H)), (u, (ne, FF, H)), (d, (ne, H, FF))):
        if tuple(values(t).shape) != shape:
            raise ValueError(f"{t.name}: shape {values(t).shape}, expected {shape}")
    with open(out / "experts.bin", "wb") as f:
        for e in range(ne):
            gu = np.concatenate([_row_f32(g, e), _row_f32(u, e)], axis=0)
            gu_q = mtp_pack.q2_0(gu).reshape(2 * FF, H // 64 * 18)
            dn_q = mtp_pack.q2_0(_row_f32(d, e)).reshape(H, FF // 64 * 18)
            f.write(mtp_rt.blob_of(gu_q, dn_q))
            if e % 32 == 31:
                print(f"  experts {e + 1}/{ne}", flush=True)

    # ---- the dense tensors
    eh = as_f32(get("nextn.eh_proj.weight"))                       # (H out, 2H in) = [fc_embedding | fc_hidden]
    if eh.shape != (H, 2 * H):
        raise ValueError(f"nextn.eh_proj: shape {eh.shape}, expected {(H, 2 * H)}")
    dense = {"fc_embedding.weight": eh[:, :H], "fc_hidden.weight": eh[:, H:],
             "self_attn.indexer.index_qk_proj.weight":
                 np.concatenate([as_f32(get("indexer.q_proj.weight")), as_f32(get("indexer.k_proj.weight"))], axis=0)}
    for short, gname in SIMPLE.items():
        dense[short] = as_f32(get(gname))
    lines, off = [], 0
    with open(out / "dense.bin", "wb") as f:
        for short in ORDER:
            x = dense[short]
            if short in NORMS:                                    # already 1 + w (see the docstring)
                arr = x.reshape(-1).astype(np.float32)
                raw, kind, rows, cols = arr.tobytes(), "f32", 1, arr.size
            else:
                x2 = x.reshape(x.shape[0], -1)
                rows, cols = x2.shape
                if short in mtp_rt.Q8:
                    raw, kind = mtp_rt.q8_0(x2), "q8_0"
                else:
                    raw, kind = to_bf16(x2).tobytes(), "bf16"
            pad = (-off) % 256
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append(f"{short} {kind} {rows} {cols} {off} {len(raw)}")
            off += len(raw)
    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if a.draft_vocab:
        shutil.copy2(a.draft_vocab, out / "draft_vocab.bin")
    print(f"experts.bin {ne} experts, {ne * mtp_rt.BLOB} B; dense.bin {off} B, {len(lines)} tensors -> {out}")
    for l in lines:
        print("  " + l)
    return 0


def _row_f32(t, e) -> np.ndarray:
    """Expert e of a (ne, rows, cols) GGUF tensor as float32 (rows, cols), without converting the whole tensor."""
    tt = int(t.tensor_type)
    x = values(t)[e]
    if tt == int(gguf.GGMLQuantizationType.BF16):
        u16 = x.view(np.uint16) if x.dtype != np.uint16 else x
        return (u16.astype(np.uint32) << 16).view(np.float32)
    if tt == int(gguf.GGMLQuantizationType.F32):
        return x.astype(np.float32)
    raise ValueError(f"{t.name}: type {t.tensor_type!r}, expected BF16 or F32")


if __name__ == "__main__":
    sys.exit(main())
