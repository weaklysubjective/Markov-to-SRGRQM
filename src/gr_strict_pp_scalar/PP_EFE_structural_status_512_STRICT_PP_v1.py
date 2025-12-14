#!/usr/bin/env python3
import argparse, json, os
from typing import Any, Dict, Optional, Iterable

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
    v = _find_any(obj, keys)
    return _as_bool(v, None)

def _parse_all_pass(obj: Any, keys: Iterable[str]) -> Optional[bool]:
    return _find_any_bool(obj, keys)

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP structural status @512: aggregates tensor-closure v2 + Bianchi-proxy v2 per case (ms080, strong_pf010)."
    )
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)

    # Defaults: adjust only if you used different filenames
    ap.add_argument("--closure_ms080", default="src/gr_strict_pp_scalar/PP_EFE_tensor_closure_audit_512_ms080_v2.json")
    ap.add_argument("--closure_strong", default="src/gr_strict_pp_scalar/PP_EFE_tensor_closure_audit_512_strong_pf010_v2.json")

    ap.add_argument("--bianchi_ms080", default="src/gr_strict_pp_scalar/PP_EFE_bianchi_audit_512_ms080_v2.json")
    ap.add_argument("--bianchi_strong", default="src/gr_strict_pp_scalar/PP_EFE_bianchi_audit_512_strong_pf010_v2.json")

    ap.add_argument("--allow_na_case", action="store_true",
                    help="If set, a missing case does not fail ALL_PASS_structural (useful while bringing ms080 online).")

    ap.add_argument("--output", default="src/gr_strict_pp_scalar/PP_EFE_structural_status_512_STRICT_PP_v1.json")
    args = ap.parse_args()

    def parse_case(case: str, closure_path: str, bianchi_path: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"case": case}

        # ---- closure ----
        c_present = _exists(closure_path)
        c_pass = None
        c_keys = [
            "ALL_PASS_tensor_closure_audit_v2",
            "ALL_PASS_tensor_closure_audit_v1",
            "ALL_PASS_tensor_closure_audit",
            "ALL_PASS",
            "PASS",  # last-resort
        ]
        if c_present:
            C = load_json(closure_path)
            # Prefer PASS dict keys if present
            c_pass = _parse_all_pass(C, c_keys)
            # If file has PASS.{ALL_PASS_tensor_closure_audit_v2} etc, the recursive search finds it.
        out["closure"] = {
            "present": bool(c_present),
            "path": closure_path,
            "ALL_PASS": c_pass,
        }

        # ---- bianchi ----
        b_present = _exists(bianchi_path)
        b_pass = None
        b_keys = [
            "ALL_PASS_bianchi_proxy_audit_v2",
            "ALL_PASS_bianchi_proxy_audit_v1",
            "ALL_PASS_bianchi_proxy_audit",
            "ALL_PASS",
            "PASS",
        ]
        if b_present:
            B = load_json(bianchi_path)
            b_pass = _parse_all_pass(B, b_keys)
        out["bianchi"] = {
            "present": bool(b_present),
            "path": bianchi_path,
            "ALL_PASS": b_pass,
        }

        # ---- per-case gate ----
        # If present but pass is None => treat as False (structure mismatch)
        if c_present and b_present:
            out["APPLICABLE"] = True
            out["PASS_structural_case_v1"] = bool(c_pass is True and b_pass is True)
        else:
            out["APPLICABLE"] = False
            out["PASS_structural_case_v1"] = None

        return out

    per_case = [
        parse_case("ms080", args.closure_ms080, args.bianchi_ms080),
        parse_case("strong_pf010", args.closure_strong, args.bianchi_strong),
    ]

    # Determine overall applicability and pass
    applicable_cases = [c for c in per_case if c.get("APPLICABLE") is True]
    missing_cases = [c for c in per_case if c.get("APPLICABLE") is False]

    if applicable_cases:
        all_pass_applicable = all(c.get("PASS_structural_case_v1") is True for c in applicable_cases)
    else:
        all_pass_applicable = False

    if missing_cases and not args.allow_na_case:
        overall = False
    else:
        overall = bool(all_pass_applicable)

    out = {
        "H": int(args.H),
        "W": int(args.W),
        "STRICT_PP": True,
        "notes": (
            "STRICT PP structural status @512. Aggregates tensor-closure audit (v2) and Bianchi-proxy audit (v2) per case. "
            "No new observables, no PDE/Poisson, no smoothing, no GR ansatz, no regression."
        ),
        "policy": {
            "allow_na_case": bool(args.allow_na_case),
        },
        "per_case": per_case,
        "PASS": {
            "APPLICABLE": bool(applicable_cases) and (True if args.allow_na_case else len(missing_cases) == 0),
            "ALL_PASS_structural_status_512_STRICT_PP_v1": bool(overall),
        },
        "summary": {
            "n_applicable_cases": int(len(applicable_cases)),
            "n_missing_cases": int(len(missing_cases)),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_structural_status_512_STRICT_PP_v1 =", out["PASS"]["ALL_PASS_structural_status_512_STRICT_PP_v1"])
    print("APPLICABLE =", out["PASS"]["APPLICABLE"])

if __name__ == "__main__":
    main()

