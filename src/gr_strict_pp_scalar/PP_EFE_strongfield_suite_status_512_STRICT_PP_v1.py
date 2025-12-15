#!/usr/bin/env python3
import argparse, json, os
from typing import Any, Optional, Iterable, Tuple

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

def _extract_named_pass_from_PASS_dict(j: Any) -> Optional[bool]:
    """
    Many of your status JSONs follow:
      { "PASS": { "ALL_PASS_<name>": true, ... }, ... }
    This tries to find any boolean True/False in j["PASS"] that looks like an ALL_PASS/overall_PASS/PASS marker.
    """
    if not isinstance(j, dict):
        return None
    P = j.get("PASS")
    if not isinstance(P, dict):
        return None

    # Prefer explicit ALL_PASS / overall_PASS keys if present
    for k in list(P.keys()):
        if k.startswith("ALL_PASS") or k.startswith("overall_PASS") or k == "PASS":
            b = _as_bool(P.get(k), None)
            if b is not None:
                return b

    # Otherwise, if PASS dict has exactly one bool-like value, use it
    bool_vals = []
    for k, v in P.items():
        b = _as_bool(v, None)
        if b is not None:
            bool_vals.append(b)
    if len(bool_vals) == 1:
        return bool_vals[0]
    return None

def _extract_pass_shapiro(j: Any) -> Optional[bool]:
    """
    Shapiro artifacts have varied keying across v1/v2_sparse and case variants.
    This is intentionally liberal and STRICT-PP-safe: it only reads reported PASS flags.
    """
    if j is None:
        return None

    # 1) Common explicit keys (top-level or nested)
    candidates = [
        # Generic
        "ALL_PASS",
        "overall_PASS",
        "PASS",

        # Common Shapiro names (case-insensitive via recursive key search on exact tokens)
        "ALL_PASS_Shapiro",
        "overall_PASS_Shapiro",
        "PASS_Shapiro",

        # Markov-tau Shapiro variants
        "ALL_PASS_Shapiro_markov_tau_v1",
        "ALL_PASS_Shapiro_markov_tau_v2_sparse",
        "PASS_Shapiro_markov_tau_v2_sparse",
        "PASS_Shapiro_markov_tau",

        # 512-named variants (some scripts embed res in named key)
        "ALL_PASS_Shapiro_markov_tau_512_strict_PP_v1",
        "ALL_PASS_Shapiro_markov_tau_512_STRICT_PP_v1",
        "overall_PASS_Shapiro_markov_tau_512_strict_PP_v1",

        # If your JSON stores a nested PASS map, we’ll also catch those below
    ]
    v = _find_any_bool(j, candidates)
    if v is not None:
        return v

    # 2) Look inside {"PASS": {...}} patterns
    v2 = _extract_named_pass_from_PASS_dict(j)
    if v2 is not None:
        return v2

    # 3) Some scripts store under "summary" or "results" with PASS-like key
    v3 = _find_any_bool(j, [
        "ALL_PASS_Shapiro_markov_tau",
        "overall_PASS_Shapiro_markov_tau",
        "PASS_Shapiro_markov_tau",
        "ALL_PASS_Shapiro_markov",
        "PASS_Shapiro_markov",
    ])
    if v3 is not None:
        return v3

    return None

def _extract_pass_deflection(j: Any) -> Optional[bool]:
    if j is None:
        return None
    return _find_any_bool(j, [
        "PASS",
        "ALL_PASS",
        "overall_PASS",
        "PASS_deflection_markov_front_PP_v7",
        "PASS_deflection_markov_front_PP_v6",
        "PASS_deflection",
        "ALL_PASS_deflection_E2_strict_PP_512_v1",
        "overall_PASS_deflection_E2_strict_PP_512_v1",
    ])

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP 512: strong-field suite status binder (no heavy generation)."
    )
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--device", default="gpu", help="Recorded only.")

    # “Real EFE” / structural binder inputs
    ap.add_argument("--tensor_3p1_status_json",
                    default="src/gr_strict_pp_scalar/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.json")
    ap.add_argument("--master_status_json",
                    default="src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.json")

    # Observables (existing artifacts)
    ap.add_argument("--shapiro_mass_json",
                    default="src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_512_mass_ms080_PPV1.json")
    ap.add_argument("--shapiro_strong_json",
                    default="src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_512_strong_pf010_PPV1.json")
    ap.add_argument("--deflection_status_json",
                    default="src/gr_strict_pp_scalar/PP_deflection_E2_status_512_STRICT_PP_v1.json")

    # Optional perihelion artifacts (if you have them at 512; otherwise leave missing)
    ap.add_argument("--perihelion_mass_json", default="")
    ap.add_argument("--perihelion_strong_json", default="")
    ap.add_argument("--allow_na_perihelion", action="store_true")

    ap.add_argument("--output",
                    default="src/gr_strict_pp_scalar/PP_EFE_strongfield_suite_status_512_STRICT_PP_v1.json")
    args = ap.parse_args()

    if not _exists(args.master_status_json):
        raise SystemExit(f"Missing master_status_json: {args.master_status_json}")
    M = load_json(args.master_status_json)

    # Require master v3 “real EFE” gate
    k_master = "overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_strict_PP_512_v3"
    master_pass = bool(_find_any_bool(M, [k_master]) is True)

    # Require tensor_3p1 binder if present
    t3p1_pass = False
    if _exists(args.tensor_3p1_status_json):
        T = load_json(args.tensor_3p1_status_json)
        t3p1_pass = bool(_find_any_bool(T, ["ALL_PASS_EFE_tensor_3p1_strict_PP_512_v1"]) is True)

    # Shapiro gates (strict: file must exist and PASS must be discoverable & true)
    sh_mass_ok = None
    sh_strong_ok = None

    if _exists(args.shapiro_mass_json):
        S = load_json(args.shapiro_mass_json)
        sh_mass_ok = _extract_pass_shapiro(S)
        sh_mass_ok = bool(sh_mass_ok is True) if sh_mass_ok is not None else False
    else:
        sh_mass_ok = False

    if _exists(args.shapiro_strong_json):
        S = load_json(args.shapiro_strong_json)
        sh_strong_ok = _extract_pass_shapiro(S)
        sh_strong_ok = bool(sh_strong_ok is True) if sh_strong_ok is not None else False
    else:
        sh_strong_ok = False

    shapiro_gate = bool(sh_mass_ok and sh_strong_ok)

    # Deflection E2 gate
    defl_ok = False
    if _exists(args.deflection_status_json):
        D = load_json(args.deflection_status_json)
        d = _extract_pass_deflection(D)
        defl_ok = bool(d is True)

    # Optional perihelion
    per_mass_ok = None
    per_strong_ok = None
    per_gate_ok = True

    if args.perihelion_mass_json:
        if _exists(args.perihelion_mass_json):
            P = load_json(args.perihelion_mass_json)
            per_mass_ok = bool(_find_any_bool(P, ["PASS", "ALL_PASS", "overall_PASS"]) is True)
        else:
            per_mass_ok = False

    if args.perihelion_strong_json:
        if _exists(args.perihelion_strong_json):
            P = load_json(args.perihelion_strong_json)
            per_strong_ok = bool(_find_any_bool(P, ["PASS", "ALL_PASS", "overall_PASS"]) is True)
        else:
            per_strong_ok = False

    if (args.perihelion_mass_json or args.perihelion_strong_json):
        if args.allow_na_perihelion:
            per_gate_ok = True
        else:
            ok = True
            if args.perihelion_mass_json:
                ok = ok and bool(per_mass_ok is True)
            if args.perihelion_strong_json:
                ok = ok and bool(per_strong_ok is True)
            per_gate_ok = ok

    overall = bool(master_pass and t3p1_pass and shapiro_gate and defl_ok and per_gate_ok)

    out = {
        "H": int(args.H),
        "W": int(args.W),
        "device": str(args.device),
        "STRICT_PP": True,
        "inputs": {
            "master_status_json": args.master_status_json,
            "tensor_3p1_status_json": args.tensor_3p1_status_json,
            "shapiro_mass_json": args.shapiro_mass_json,
            "shapiro_strong_json": args.shapiro_strong_json,
            "deflection_status_json": args.deflection_status_json,
            "perihelion_mass_json": args.perihelion_mass_json or None,
            "perihelion_strong_json": args.perihelion_strong_json or None,
        },
        "PASS": {
            "PASS_master_v3": master_pass,
            "PASS_tensor_3p1": t3p1_pass,
            "PASS_shapiro_mass": sh_mass_ok,
            "PASS_shapiro_strong": sh_strong_ok,
            "PASS_deflection_E2": defl_ok,
            "PASS_perihelion_mass": per_mass_ok,
            "PASS_perihelion_strong": per_strong_ok,
            "ALL_PASS_strongfield_suite_strict_PP_512_v1": overall,
        },
        "policy": {
            "allow_na_perihelion": bool(args.allow_na_perihelion),
        },
        "notes": (
            "STRICT PP strong-field suite binder: gates on (master v3) + (tensor_3p1) + (Shapiro both cases) + (deflection E2). "
            "Perihelion is optional until 512 artifacts are provided. "
            "Shapiro PASS extraction is liberal over prior keying styles (v1/v2_sparse)."
        ),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_strongfield_suite_strict_PP_512_v1 =", out["PASS"]["ALL_PASS_strongfield_suite_strict_PP_512_v1"])

if __name__ == "__main__":
    main()

