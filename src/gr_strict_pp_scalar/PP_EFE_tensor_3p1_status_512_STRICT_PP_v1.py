#!/usr/bin/env python3
import argparse, json, os
from typing import Any, Optional, Iterable

def load_json(path: str) -> Any:
    with open(path, "r") as f:
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
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return default

def _find_key_recursive(obj: Any, key: str) -> Optional[Any]:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            out = _find_key_recursive(v, key)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for it in obj:
            out = _find_key_recursive(it, key)
            if out is not None:
                return out
    return None

def _find_any(obj: Any, keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        v = _find_key_recursive(obj, k)
        if v is not None:
            return v
    return None

def _find_any_bool(obj: Any, keys: Iterable[str]) -> Optional[bool]:
    return _as_bool(_find_any(obj, keys), None)

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP 512: 3+1 tensor EFE status (binds master v3 + tensor_2p1 + optional invariance)."
    )
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--device", default="gpu", help="Recorded only (gpu/cuda/hip/cpu).")

    ap.add_argument(
        "--master_status_json",
        default="src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.json",
        help="Master status JSON with key ..._v3."
    )
    ap.add_argument(
        "--tensor_2p1_status_json",
        default="src/gr_strict_pp_scalar/PP_EFE_tensor_2p1_status_512_STRICT_PP_v1.json",
        help="Tensor_2p1 status JSON (used if master lacks the key)."
    )
    ap.add_argument(
        "--invariance_status_json",
        default="",
        help="Optional invariance status JSON; if provided and present, must PASS."
    )
    ap.add_argument(
        "--allow_na_invariance",
        action="store_true",
        help="If set, missing/NA invariance does not gate PASS."
    )
    ap.add_argument(
        "--output",
        default="src/gr_strict_pp_scalar/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.json"
    )
    args = ap.parse_args()

    if not _exists(args.master_status_json):
        raise SystemExit(f"Missing master_status_json: {args.master_status_json}")

    M = load_json(args.master_status_json)

    k_master = "overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_strict_PP_512_v3"
    master_pass = _find_any_bool(M, [k_master])
    master_pass = bool(master_pass is True)

    # tensor_2p1: prefer master copy, else read tensor_2p1_status_json
    k_t2p1 = "overall_PASS_EFE_tensor_2p1_strict_PP_512_v1"
    t2p1_pass = _find_any_bool(M, [k_t2p1])
    t2p1_flags = (M or {}).get("tensor_2p1_flags")

    if t2p1_pass is None:
        if not _exists(args.tensor_2p1_status_json):
            raise SystemExit(f"Missing tensor_2p1_status_json: {args.tensor_2p1_status_json}")
        T = load_json(args.tensor_2p1_status_json)
        t2p1_pass = _find_any_bool(T, [
            "ALL_PASS_tensor_2p1_status_512_STRICT_PP_v1",
            "ALL_PASS_tensor_2p1_status_512_STRICT_PP_v1",
            "ALL_PASS",
            "overall_PASS",
            "PASS",
        ])
        t2p1_pass = bool(t2p1_pass is True)
        t2p1_flags = {
            "present": True,
            "tensor_2p1_status_json": args.tensor_2p1_status_json,
            "APPLICABLE": _find_any_bool(T, ["APPLICABLE"]) if _find_any_bool(T, ["APPLICABLE"]) is not None else True,
            "PASS": bool(t2p1_pass),
        }
    else:
        t2p1_pass = bool(t2p1_pass is True)
        if isinstance(t2p1_flags, dict):
            t2p1_flags = dict(t2p1_flags)
            t2p1_flags.setdefault("present", True)

    # invariance optional
    inv_flags = {"present": False}
    if args.invariance_status_json and _exists(args.invariance_status_json):
        I = load_json(args.invariance_status_json)
        inv_pass = _find_any_bool(I, [
            "ALL_PASS_invariance_audit_v1",
            "ALL_PASS_invariance_v1",
            "overall_PASS_invariance_v1",
            "ALL_PASS",
            "overall_PASS",
            "PASS",
        ])
        inv_pass = bool(inv_pass is True)
        inv_flags = {
            "present": True,
            "invariance_status_json": args.invariance_status_json,
            "APPLICABLE": _find_any_bool(I, ["APPLICABLE"]) if _find_any_bool(I, ["APPLICABLE"]) is not None else True,
            "PASS": inv_pass,
        }
        inv_gate = inv_pass
    else:
        inv_flags = {
            "present": False,
            "invariance_status_json": args.invariance_status_json or None,
            "reason_missing": "not provided or not found",
        }
        inv_gate = True if args.allow_na_invariance else (not bool(args.invariance_status_json))

    overall = bool(master_pass and t2p1_pass and inv_gate)

    out = {
        "H": int(args.H),
        "W": int(args.W),
        "device": str(args.device),
        "STRICT_PP": True,
        "inputs": {
            "master_status_json": args.master_status_json,
            "tensor_2p1_status_json": args.tensor_2p1_status_json,
            "invariance_status_json": args.invariance_status_json or None,
        },
        "PASS": {
            "APPLICABLE": True,
            "PASS_master_v3": master_pass,
            "PASS_tensor_2p1": t2p1_pass,
            "PASS_invariance": (inv_flags.get("PASS") if inv_flags.get("present") else None),
            "ALL_PASS_EFE_tensor_3p1_strict_PP_512_v1": overall,
        },
        "refs": {
            "deflection_flags": (M or {}).get("deflection_flags"),
            "structural_flags": (M or {}).get("structural_flags"),
            "offdiag_flags": (M or {}).get("offdiag_flags"),
            "tensor_2p1_flags": t2p1_flags,
            "invariance_flags": inv_flags,
        },
        "notes": (
            "STRICT PP 512: '3+1 tensor EFE' is defined operationally as: "
            "(i) master v3 PASS (scalar+vector+offdiag+deflection+closure+bianchi), "
            "(ii) tensor_2p1 PASS, and (iii) optional invariance PASS if provided. "
            "This is a release-grade binding step; it does not introduce new observables."
        ),
        "policy": {
            "allow_na_invariance": bool(args.allow_na_invariance),
        }
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_EFE_tensor_3p1_strict_PP_512_v1 =", out["PASS"]["ALL_PASS_EFE_tensor_3p1_strict_PP_512_v1"])

if __name__ == "__main__":
    main()

