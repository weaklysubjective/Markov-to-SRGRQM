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
        description="STRICT PP master EFE status @512 v2: scalar+vector + offdiag + deflection + structural(v2)."
    )
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)

    ap.add_argument("--scalar_status_json", default="src/gr_strict_pp_scalar/PP_scalar_EFE00_status_512_strict_PP_v1.json")
    ap.add_argument("--vector_status_json", default="src/gr_strict_pp_vector/PP_vector_EFE_status_512_strict_PP_v1.json")
    ap.add_argument("--offdiag_status_json", default="src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v2.json")

    ap.add_argument("--deflection_status_json", default="src/gr_strict_pp_scalar/PP_deflection_E2_status_512_STRICT_PP_v1.json")
    ap.add_argument("--deflection_json_fallback", default="src/gr_strict_pp_scalar/PP_deflection_markov_front_512_strong_pf010_v6_PPV1.json")

    ap.add_argument("--structural_json", default="src/gr_strict_pp_scalar/PP_EFE_structural_status_512_STRICT_PP_v2.json")

    ap.add_argument("--allow_na_offdiag", action="store_true")
    ap.add_argument("--allow_na_deflection", action="store_true")
    ap.add_argument("--allow_na_structural", action="store_true")

    ap.add_argument("--output", default="src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v2.json")
    args = ap.parse_args()

    # required scalar/vector
    if not _exists(args.scalar_status_json):
        raise SystemExit(f"Missing scalar_status_json: {args.scalar_status_json}")
    if not _exists(args.vector_status_json):
        raise SystemExit(f"Missing vector_status_json: {args.vector_status_json}")

    S = load_json(args.scalar_status_json)
    V = load_json(args.vector_status_json)

    scalar_pass = bool(_find_any_bool(S, ["overall_PASS_scalar_EFE00_strict_PP_v1"]) is True)
    vector_pass = bool(_find_any_bool(V, ["overall_PASS_vector_EFE_strict_PP_v1"]) is True)
    overall_scalar_vector = bool(scalar_pass and vector_pass)

    # offdiag optional
    offdiag_flags: dict[str, Any] = {"present": False}
    offdiag_gate_pass: bool
    if _exists(args.offdiag_status_json):
        O = load_json(args.offdiag_status_json)
        of = (O or {}).get("offdiag_flags") or {}
        offdiag_app  = _as_bool(of.get("APPLICABLE", None), None)
        offdiag_pass = _as_bool(of.get("PASS", None), None)
        if offdiag_pass is None:
            offdiag_pass = _find_any_bool(O, ["ALL_PASS", "overall_PASS", "PASS"])
        offdiag_flags = {
            "present": True,
            "offdiag_status_json": args.offdiag_status_json,
            "APPLICABLE": offdiag_app,
            "PASS": offdiag_pass,
        }
        if offdiag_app is True:
            offdiag_gate_pass = bool(offdiag_pass is True)
        else:
            offdiag_gate_pass = True if args.allow_na_offdiag else False
    else:
        offdiag_flags = {"present": False, "offdiag_status_json": args.offdiag_status_json}
        offdiag_gate_pass = True if args.allow_na_offdiag else False

    # deflection optional
    def _parse_deflection_status(D: Any) -> tuple[Optional[bool], Optional[bool], Optional[str]]:
        AP = _find_any_bool(D, ["APPLICABLE"])
        PAS = _find_any_bool(D, [
            "PASS",
            "ALL_PASS",
            "overall_PASS",
            "ALL_PASS_deflection_E2_strict_PP_512_v1",
            "overall_PASS_deflection_E2_strict_PP_512_v1",
        ])
        reason = _find_any(D, ["reason", "notes"])
        if PAS is not None and AP is None:
            AP = True
        return AP, PAS, (str(reason) if reason is not None else None)

    defl_flags: dict[str, Any] = {"present": False}
    defl_gate_pass: bool

    if _exists(args.deflection_status_json):
        D = load_json(args.deflection_status_json)
        AP, PAS, reason = _parse_deflection_status(D)
        defl_flags = {
            "present": True,
            "deflection_status_json": args.deflection_status_json,
            "APPLICABLE": AP,
            "PASS": PAS,
            "reason": reason,
        }
        if AP is True:
            defl_gate_pass = bool(PAS is True)
        else:
            defl_gate_pass = True if args.allow_na_deflection else False
    elif _exists(args.deflection_json_fallback):
        D = load_json(args.deflection_json_fallback)
        AP, PAS, reason = _parse_deflection_status(D)
        defl_flags = {
            "present": True,
            "deflection_json_fallback": args.deflection_json_fallback,
            "APPLICABLE": AP,
            "PASS": PAS,
            "reason": reason,
            "notes": "fallback deflection artifact; prefer E2 status json if present.",
        }
        if AP is True:
            defl_gate_pass = bool(PAS is True)
        else:
            defl_gate_pass = True if args.allow_na_deflection else False
    else:
        defl_flags = {"present": False}
        defl_gate_pass = True if args.allow_na_deflection else False

    # structural(v2) optional
    structural_flags: dict[str, Any] = {"present": False}
    structural_gate_pass: bool
    if _exists(args.structural_json):
        T = load_json(args.structural_json)
        st_pass = _find_any_bool(T, ["ALL_PASS_structural_status_512_STRICT_PP_v2"])
        st_app  = _find_any_bool(T, ["APPLICABLE"])
        # If file exists but doesn't expose these keys, fall back:
        if st_pass is None:
            st_pass = _find_any_bool(T, ["ALL_PASS", "PASS"])
        if st_app is None and st_pass is not None:
            st_app = True
        structural_flags = {
            "present": True,
            "structural_status_json": args.structural_json,
            "APPLICABLE": st_app,
            "PASS": st_pass,
        }
        if st_app is True:
            structural_gate_pass = bool(st_pass is True)
        else:
            structural_gate_pass = True if args.allow_na_structural else False
    else:
        structural_flags = {
            "present": False,
            "structural_status_json": args.structural_json,
            "reason_missing": f"not found: {args.structural_json}",
        }
        structural_gate_pass = True if args.allow_na_structural else False

    overall_extended_v2 = bool(overall_scalar_vector and offdiag_gate_pass and defl_gate_pass)
    overall_extended_v4 = bool(overall_extended_v2 and structural_gate_pass)

    out = {
        "H": int(args.H),
        "W": int(args.W),
        "STRICT_PP": True,
        "scalar_status_json": args.scalar_status_json,
        "vector_status_json": args.vector_status_json,
        "offdiag_status_json": args.offdiag_status_json,
        "deflection_status_json": args.deflection_status_json,
        "structural_json": args.structural_json,

        "scalar_flags": {"overall_PASS_scalar_EFE00_strict_PP_v1": bool(scalar_pass)},
        "vector_flags": {"overall_PASS_vector_EFE_strict_PP_v1": bool(vector_pass)},
        "offdiag_flags": offdiag_flags,
        "deflection_flags": defl_flags,
        "structural_flags": structural_flags,

        "overall_PASS_EFE_scalar_vector_strict_PP_512_v1": bool(overall_scalar_vector),
        "overall_PASS_EFE_scalar_vector_offdiag_deflection_strict_PP_512_v2": bool(overall_extended_v2),

        "overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_covariance_strict_PP_512_v4": bool(overall_extended_v4),

        "policy": {
            "allow_na_offdiag": bool(args.allow_na_offdiag),
            "allow_na_deflection": bool(args.allow_na_deflection),
            "allow_na_structural": bool(args.allow_na_structural),
        },

        "notes": (
            "STRICT PP master status v2: adds structural(v2)=closure+bianchi+covariance(relabel). "
            "No PDE/Poisson/smoothing/ansatz/regression."
        ),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_covariance_strict_PP_512_v4 =",
          out["overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_covariance_strict_PP_512_v4"])
    print("structural(APPLICABLE,PASS) =",
          out["structural_flags"].get("APPLICABLE"), out["structural_flags"].get("PASS"))

if __name__ == "__main__":
    main()

