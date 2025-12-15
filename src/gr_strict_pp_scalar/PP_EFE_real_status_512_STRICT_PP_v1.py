#!/usr/bin/env python3
import argparse, json, os
from typing import Any, Optional

def load_json(p: str) -> Any:
    with open(p, "r") as f:
        return json.load(f)

def _exists(p: Optional[str]) -> bool:
    return bool(p) and os.path.exists(p)

def _as_bool(v, default: Optional[bool] = None) -> Optional[bool]:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true","t","1","yes","y"): return True
        if s in ("false","f","0","no","n"): return False
    return default

def _find_key_recursive(obj: Any, key: str) -> Optional[Any]:
    if isinstance(obj, dict):
        if key in obj: return obj[key]
        for v in obj.values():
            out = _find_key_recursive(v, key)
            if out is not None: return out
    elif isinstance(obj, list):
        for it in obj:
            out = _find_key_recursive(it, key)
            if out is not None: return out
    return None

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP (512) REAL EFE status: ties together master EFE status + tensor_2p1 gate into a single release-grade PASS key."
    )
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--device", default="gpu", help="Recorded only (gpu/cuda/cpu).")

    ap.add_argument(
        "--master_status_json",
        default="src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.json",
        help="Master scalar+vector+offdiag+deflection+closure+bianchi status JSON."
    )
    ap.add_argument(
        "--tensor_2p1_status_json",
        default="src/gr_strict_pp_scalar/PP_EFE_tensor_2p1_status_512_STRICT_PP_v1.json",
        help="Tensor_2p1 gate status JSON."
    )
    ap.add_argument(
        "--output",
        default="src/gr_strict_pp_scalar/PP_EFE_real_status_512_STRICT_PP_v1.json"
    )
    args = ap.parse_args()

    if not _exists(args.master_status_json):
        raise SystemExit(f"Missing master_status_json: {args.master_status_json}")

    M = load_json(args.master_status_json)

    # Master “already-real” gate (your v3 key)
    k_master = "overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_strict_PP_512_v3"
    master_pass = _as_bool(M.get(k_master), None)
    if master_pass is None:
        master_pass = _as_bool(_find_key_recursive(M, k_master), False)
    master_pass = bool(master_pass)

    # Tensor_2p1 gate (prefer master copy, else read external file)
    k_t2p1 = "overall_PASS_EFE_tensor_2p1_strict_PP_512_v1"
    t2p1_pass = _as_bool(M.get(k_t2p1), None)
    t2p1_flags = M.get("tensor_2p1_flags")

    if t2p1_pass is None and _exists(args.tensor_2p1_status_json):
        T = load_json(args.tensor_2p1_status_json)
        # Very liberal parse
        t2p1_pass = _as_bool(T.get("PASS", {}).get("ALL_PASS_tensor_2p1_status_512_STRICT_PP_v1"), None)
        if t2p1_pass is None:
            t2p1_pass = _as_bool(_find_key_recursive(T, "ALL_PASS_tensor_2p1_status_512_STRICT_PP_v1"), False)
        t2p1_flags = {
            "present": True,
            "tensor_2p1_status_json": args.tensor_2p1_status_json,
            "APPLICABLE": _as_bool(_find_key_recursive(T, "APPLICABLE"), True),
            "PASS": bool(t2p1_pass),
        }
    else:
        t2p1_pass = bool(t2p1_pass is True)

    overall_real = bool(master_pass and t2p1_pass)

    out = {
        "H": int(args.H),
        "W": int(args.W),
        "device": str(args.device),
        "STRICT_PP": True,

        "inputs": {
            "master_status_json": args.master_status_json,
            "tensor_2p1_status_json": args.tensor_2p1_status_json,
        },

        "PASS": {
            "APPLICABLE": True,
            "ALL_PASS_real_EFE_strict_PP_512_v1": overall_real,
            "PASS_master_scalar_vector_offdiag_deflection_closure_bianchi": master_pass,
            "PASS_tensor_2p1": bool(t2p1_pass),
        },

        "refs": {
            "deflection_flags": M.get("deflection_flags"),
            "structural_flags": M.get("structural_flags"),
            "tensor_2p1_flags": t2p1_flags,
            "offdiag_flags": M.get("offdiag_flags"),
        },

        "notes": (
            "REAL EFE status @512 (STRICT PP): declares completion of the 2+1 tensor bundle "
            "using the already-passing master status (scalar+vector+offdiag+deflection+closure+bianchi) "
            "AND the tensor_2p1 gate. No PDE/Poisson/Laplacian smoothing, no GR ansatz, no regression."
        ),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_real_EFE_strict_PP_512_v1 =", out["PASS"]["ALL_PASS_real_EFE_strict_PP_512_v1"])

if __name__ == "__main__":
    main()

