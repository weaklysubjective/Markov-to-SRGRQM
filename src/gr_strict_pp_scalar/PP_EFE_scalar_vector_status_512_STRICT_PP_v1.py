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
    """Depth-first search for a key inside nested dict/list JSON."""
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
    v = _find_any(obj, keys)
    return _as_bool(v, None)

def main():
    STRUCTURAL_JSON_DEFAULT = "src/gr_strict_pp_scalar/PP_EFE_structural_status_512_STRICT_PP_v1.json"
    TENSOR_2P1_JSON_DEFAULT = "src/gr_strict_pp_scalar/PP_EFE_tensor_2p1_status_512_STRICT_PP_v1.json"

    ap = argparse.ArgumentParser(
        description="STRICT PP master EFE status @512: scalar EFE00 + vector EFE + optional offdiag 2+1 + optional deflection (E2 preferred) + structural + tensor_2p1 gate."
    )
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)

    # For repo-wide CLI consistency: accept --device gpu|cuda|cpu|hip, but this script aggregates JSON.
    ap.add_argument("--device", default="gpu",
                    help="Accepted for consistency; this script aggregates JSON only. Use gpu|cuda|cpu|hip.")

    ap.add_argument("--scalar_status_json", default="src/gr_strict_pp_scalar/PP_scalar_EFE00_status_512_strict_PP_v1.json")
    ap.add_argument("--vector_status_json", default="src/gr_strict_pp_vector/PP_vector_EFE_status_512_strict_PP_v1.json")
    ap.add_argument("--offdiag_status_json", default="src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v2.json")

    # E2 deflection status (preferred)
    ap.add_argument("--deflection_status_json", default="src/gr_strict_pp_scalar/PP_deflection_E2_status_512_STRICT_PP_v1.json")
    # Old PPV1/v6 fallback (record-only unless E2 missing)
    ap.add_argument("--deflection_json_fallback", default="src/gr_strict_pp_scalar/PP_deflection_markov_front_512_strong_pf010_v6_PPV1.json")

    ap.add_argument("--allow_na_offdiag", action="store_true",
                    help="If set, offdiag missing/NA does not gate the extended PASS.")
    ap.add_argument("--allow_na_deflection", action="store_true",
                    help="If set, deflection missing/NA does not gate the extended PASS.")

    ap.add_argument("--structural_json", default=STRUCTURAL_JSON_DEFAULT,
                    help="Path to PP_EFE_structural_status_512_STRICT_PP_v1.json")

    # NEW: tensor_2p1 aggregated gate
    ap.add_argument("--tensor_2p1_status_json", default=TENSOR_2P1_JSON_DEFAULT,
                    help="Path to PP_EFE_tensor_2p1_status_512_STRICT_PP_v1.json")

    ap.add_argument("--allow_na_tensor_2p1", action="store_true",
                    help="If set, tensor_2p1 missing/NA does not gate the tensor_2p1 key (NOT recommended).")

    ap.add_argument("--output", default="src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.json")
    args = ap.parse_args()

    # ---- required scalar/vector ----
    if not _exists(args.scalar_status_json):
        raise SystemExit(f"Missing scalar_status_json: {args.scalar_status_json}")
    if not _exists(args.vector_status_json):
        raise SystemExit(f"Missing vector_status_json: {args.vector_status_json}")

    S = load_json(args.scalar_status_json)
    V = load_json(args.vector_status_json)

    scalar_pass = _find_any_bool(S, ["overall_PASS_scalar_EFE00_strict_PP_v1"])
    vector_pass = _find_any_bool(V, ["overall_PASS_vector_EFE_strict_PP_v1"])
    scalar_pass = bool(scalar_pass is True)
    vector_pass = bool(vector_pass is True)

    overall_scalar_vector = bool(scalar_pass and vector_pass)

    # ---- offdiag (optional) ----
    offdiag_flags: dict[str, Any] = {"present": False}
    offdiag_gate_pass: bool

    if _exists(args.offdiag_status_json):
        O = load_json(args.offdiag_status_json)

        of = (O or {}).get("offdiag_flags") or {}
        offdiag_app  = _as_bool(of.get("APPLICABLE", None), None)
        offdiag_pass = _as_bool(of.get("PASS", None), None)

        if offdiag_pass is None:
            offdiag_pass = _find_any_bool(O, [
                "ALL_PASS_offdiag_2p1_strict_PP_512_v2",
                "ALL_PASS_offdiag_2p1_strict_PP_512_v1",
                "overall_PASS_offdiag_2p1_strict_PP_512_v2",
                "overall_PASS_offdiag_2p1_strict_PP_512_v1",
                "ALL_PASS",
                "overall_PASS",
                "PASS",
            ])

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
        offdiag_flags = {
            "present": False,
            "offdiag_status_json": args.offdiag_status_json,
            "reason_missing": f"not found: {args.offdiag_status_json}",
            "PASS": None,
        }
        offdiag_gate_pass = True if args.allow_na_offdiag else False

    # ---- deflection (E2 preferred; optional) ----
    defl_flags: dict[str, Any] = {"present": False}
    defl_gate_pass: bool

    def _parse_deflection_status(D: Any) -> tuple[Optional[bool], Optional[bool], Optional[str]]:
        AP = _find_any_bool(D, ["APPLICABLE"])
        PAS = _find_any_bool(D, [
            "PASS",
            "PASS_deflection_markov_front_PP_v7",
            "PASS_deflection_markov_front_PP_v6",
            "PASS_deflection",
            "overall_PASS_deflection_E2_strict_PP_512_v1",
            "ALL_PASS_deflection_E2_strict_PP_512_v1",
            "overall_PASS",
            "ALL_PASS",
        ])
        reason = _find_any(D, ["reason", "notes"])
        if PAS is not None and AP is None:
            AP = True
        return AP, PAS, (str(reason) if reason is not None else None)

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
            "notes": "fallback deflection artifact (PPV1/v6). Prefer E2 deflection_status_json if present.",
        }
        if AP is True:
            defl_gate_pass = bool(PAS is True)
        else:
            defl_gate_pass = True if args.allow_na_deflection else False
    else:
        defl_flags = {
            "present": False,
            "reason_missing": f"not found: {args.deflection_status_json} (and fallback missing)",
        }
        defl_gate_pass = True if args.allow_na_deflection else False

    # ---- structural (closure+bianchi) ----
    structural_flags: dict[str, Any] = {"present": False}
    structural_gate_pass: bool

    def _parse_structural_status(R: Any) -> tuple[Optional[bool], Optional[bool]]:
        AP = _find_any_bool(R, ["APPLICABLE"])
        PAS = _find_any_bool(R, [
            "ALL_PASS_structural_status_512_STRICT_PP_v1",
            "PASS_structural_case_v1",
            "PASS_structural_case",
            "PASS",
            "ALL_PASS",
            "overall_PASS",
        ])
        if PAS is not None and AP is None:
            AP = True
        return AP, PAS

    if _exists(args.structural_json):
        R = load_json(args.structural_json)
        AP, PAS = _parse_structural_status(R)
        structural_flags = {
            "present": True,
            "structural_status_json": args.structural_json,
            "APPLICABLE": AP,
            "PASS": PAS,
        }
        # Structural is STRICT: if applicable => require PASS true; else treat NA as gate-fail
        if AP is True:
            structural_gate_pass = bool(PAS is True)
        else:
            structural_gate_pass = False
    else:
        structural_flags = {
            "present": False,
            "structural_status_json": args.structural_json,
            "APPLICABLE": None,
            "PASS": None,
            "reason_missing": f"not found: {args.structural_json}",
        }
        structural_gate_pass = False

    # ---- tensor_2p1 (NEW) ----
    tensor_2p1_flags: dict[str, Any] = {"present": False}
    tensor_2p1_gate_pass: bool

    def _parse_tensor_2p1_status(T: Any) -> tuple[Optional[bool], Optional[bool]]:
        AP = _find_any_bool(T, ["APPLICABLE"])
        PAS = _find_any_bool(T, [
            "ALL_PASS_tensor_2p1_status_512_STRICT_PP_v1",
            "PASS_tensor_2p1_case_v1",
            "PASS",
            "ALL_PASS",
            "overall_PASS",
        ])
        if PAS is not None and AP is None:
            AP = True
        return AP, PAS

    if _exists(args.tensor_2p1_status_json):
        T = load_json(args.tensor_2p1_status_json)
        AP, PAS = _parse_tensor_2p1_status(T)
        tensor_2p1_flags = {
            "present": True,
            "tensor_2p1_status_json": args.tensor_2p1_status_json,
            "APPLICABLE": AP,
            "PASS": PAS,
        }
        if AP is True:
            tensor_2p1_gate_pass = bool(PAS is True)
        else:
            tensor_2p1_gate_pass = True if args.allow_na_tensor_2p1 else False
    else:
        tensor_2p1_flags = {
            "present": False,
            "tensor_2p1_status_json": args.tensor_2p1_status_json,
            "APPLICABLE": None,
            "PASS": None,
            "reason_missing": f"not found: {args.tensor_2p1_status_json}",
        }
        tensor_2p1_gate_pass = True if args.allow_na_tensor_2p1 else False

    # ---- combined keys ----
    overall_extended_v2 = bool(overall_scalar_vector and offdiag_gate_pass and defl_gate_pass)
    overall_extended_v3 = bool(overall_extended_v2 and structural_gate_pass)

    # New top-level key requested:
    overall_tensor_2p1 = bool(tensor_2p1_gate_pass)

    out = {
        "H": int(args.H),
        "W": int(args.W),
        "STRICT_PP": True,
        "device": str(args.device),

        "notes": (
            "STRICT PP master EFE status @512. "
            "Key overall_PASS_EFE_scalar_vector_strict_PP_512_v1 gates scalar+vector only. "
            "Key overall_PASS_EFE_scalar_vector_offdiag_deflection_strict_PP_512_v2 gates scalar+vector+offdiag+deflection (E2 preferred). "
            "Key overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_strict_PP_512_v3 gates v2 + structural(closure+bianchi). "
            "Key overall_PASS_EFE_tensor_2p1_strict_PP_512_v1 gates tensor_2p1 aggregator (structural + offdiag_2p1), if present. "
            "No PDE/Laplacian/Poisson/GR ansatz/regression."
        ),

        "scalar_status_json": args.scalar_status_json,
        "vector_status_json": args.vector_status_json,
        "offdiag_status_json": args.offdiag_status_json,
        "deflection_status_json": args.deflection_status_json,
        "structural_json": args.structural_json,
        "tensor_2p1_status_json": args.tensor_2p1_status_json,

        "scalar_flags": {"overall_PASS_scalar_EFE00_strict_PP_v1": bool(scalar_pass)},
        "vector_flags": {"overall_PASS_vector_EFE_strict_PP_v1": bool(vector_pass)},

        "overall_PASS_EFE_scalar_vector_strict_PP_512_v1": bool(overall_scalar_vector),

        "offdiag_flags": offdiag_flags,
        "deflection_flags": defl_flags,
        "overall_PASS_EFE_scalar_vector_offdiag_deflection_strict_PP_512_v2": bool(overall_extended_v2),

        "structural_flags": structural_flags,
        "overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_strict_PP_512_v3": bool(overall_extended_v3),

        # NEW:
        "tensor_2p1_flags": tensor_2p1_flags,
        "overall_PASS_EFE_tensor_2p1_strict_PP_512_v1": bool(overall_tensor_2p1),

        "policy": {
            "allow_na_offdiag": bool(args.allow_na_offdiag),
            "allow_na_deflection": bool(args.allow_na_deflection),
            "allow_na_tensor_2p1": bool(args.allow_na_tensor_2p1),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("overall_PASS_EFE_scalar_vector_strict_PP_512_v1 =", out["overall_PASS_EFE_scalar_vector_strict_PP_512_v1"])
    print("overall_PASS_EFE_scalar_vector_offdiag_deflection_strict_PP_512_v2 =", out["overall_PASS_EFE_scalar_vector_offdiag_deflection_strict_PP_512_v2"])
    print("overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_strict_PP_512_v3 =", out["overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_strict_PP_512_v3"])
    print("overall_PASS_EFE_tensor_2p1_strict_PP_512_v1 =", out["overall_PASS_EFE_tensor_2p1_strict_PP_512_v1"])
    print("deflection(APPLICABLE,PASS) =", out["deflection_flags"].get("APPLICABLE"), out["deflection_flags"].get("PASS"))
    print("structural(APPLICABLE,PASS) =", out["structural_flags"].get("APPLICABLE"), out["structural_flags"].get("PASS"))
    print("tensor_2p1(APPLICABLE,PASS) =", out["tensor_2p1_flags"].get("APPLICABLE"), out["tensor_2p1_flags"].get("PASS"))

if __name__ == "__main__":
    main()

