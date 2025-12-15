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
        description="STRICT PP 512: single top-level REAL EFE status binder (no generators)."
    )
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--device", default="gpu", help="Recorded only.")

    ap.add_argument(
        "--master_json",
        default="src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.json",
        help="Master v3 status JSON (scalar+vector+offdiag+deflection+closure+bianchi).",
    )
    ap.add_argument(
        "--tensor_3p1_json",
        default="src/gr_strict_pp_scalar/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.json",
        help="Single source of truth for tensor 3+1 status (binder output).",
    )
    ap.add_argument(
        "--strongfield_json",
        default="src/gr_strict_pp_scalar/PP_EFE_strongfield_suite_status_512_STRICT_PP_v1.json",
        help="Strongfield suite status JSON.",
    )

    # Optional: if you later create a 512 perihelion status binder, plug it here.
    ap.add_argument("--perihelion_json", default="")
    ap.add_argument("--allow_na_perihelion", action="store_true")

    ap.add_argument(
        "--output",
        default="src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v1.json",
    )
    args = ap.parse_args()

    if not _exists(args.master_json):
        raise SystemExit(f"Missing --master_json: {args.master_json}")
    if not _exists(args.tensor_3p1_json):
        raise SystemExit(f"Missing --tensor_3p1_json: {args.tensor_3p1_json}")
    if not _exists(args.strongfield_json):
        raise SystemExit(f"Missing --strongfield_json: {args.strongfield_json}")

    M = load_json(args.master_json)
    T3 = load_json(args.tensor_3p1_json)
    S = load_json(args.strongfield_json)

    # Master v3 key (single source of truth for scalar+vector+offdiag+deflection+closure+bianchi)
    k_master = "overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_strict_PP_512_v3"
    master_ok = bool(_find_any_bool(M, [k_master]) is True)

    # Tensor 3p1: SINGLE SOURCE OF TRUTH (no fallback inference)
    # Expected schema: T3["PASS"]["ALL_PASS_EFE_tensor_3p1_strict_PP_512_v1"] == True/False
    tensor_ok = bool(
        _as_bool(((T3 or {}).get("PASS") or {}).get("ALL_PASS_EFE_tensor_3p1_strict_PP_512_v1"), False) is True
    )

    # Strongfield suite key (keep robust lookup; schema may evolve)
    k_strong = "ALL_PASS_strongfield_suite_strict_PP_512_v1"
    strong_ok = bool(_find_any_bool(S, [k_strong, "PASS", "ALL_PASS"]) is True) if _find_any_bool(S, [k_strong]) is not None else bool(_find_any_bool(S, [k_strong]) is True)

    # Optional perihelion gate (currently NA in your suite)
    per_ok: Optional[bool] = None
    per_gate = True
    if args.perihelion_json:
        if not _exists(args.perihelion_json):
            per_ok = False
            per_gate = bool(args.allow_na_perihelion)
        else:
            P = load_json(args.perihelion_json)
            per_ok = bool(_find_any_bool(P, ["PASS", "ALL_PASS", "overall_PASS"]) is True)
            per_gate = per_ok if not args.allow_na_perihelion else True
    else:
        per_gate = True  # not gating until we have 512 perihelion artifacts

    overall = bool(master_ok and tensor_ok and strong_ok and per_gate)

    out = {
        "H": int(args.H),
        "W": int(args.W),
        "device": str(args.device),
        "STRICT_PP": True,
        "inputs": {
            "master_json": args.master_json,
            "tensor_3p1_json": args.tensor_3p1_json,
            "strongfield_json": args.strongfield_json,
            "perihelion_json": args.perihelion_json or None,
        },
        "PASS": {
            "PASS_master_v3": master_ok,
            "PASS_tensor_3p1": tensor_ok,
            "PASS_strongfield_suite": strong_ok,
            "PASS_perihelion_optional": per_ok,
            "ALL_PASS_real_EFE_strict_PP_512_v1": overall,
        },
        "policy": {"allow_na_perihelion": bool(args.allow_na_perihelion)},
        "notes": (
            "Top-level STRICT PP 'REAL EFE' binder: requires master v3 + tensor_3p1 (single source of truth JSON) "
            "+ strongfield suite. Perihelion is optional until 512 artifacts exist."
        ),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_real_EFE_strict_PP_512_v1 =", out["PASS"]["ALL_PASS_real_EFE_strict_PP_512_v1"])

if __name__ == "__main__":
    main()

