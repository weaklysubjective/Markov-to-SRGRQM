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
        description="STRICT PP structural status @512 v2: closure + bianchi + covariance(relabel) per case."
    )
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)

    # Case list (keep stable naming)
    ap.add_argument("--cases", default="ms080,strong_pf010")

    # Closure (already v2 PASS on both)
    ap.add_argument("--closure_ms080_json", default="src/gr_strict_pp_scalar/PP_EFE_tensor_closure_audit_512_ms080_v2.json")
    ap.add_argument("--closure_strong_json", default="src/gr_strict_pp_scalar/PP_EFE_tensor_closure_audit_512_strong_pf010_v2.json")

    # Bianchi (v2 PASS on both)
    ap.add_argument("--bianchi_ms080_json", default="src/gr_strict_pp_scalar/PP_EFE_bianchi_audit_512_ms080_v2.json")
    ap.add_argument("--bianchi_strong_json", default="src/gr_strict_pp_scalar/PP_EFE_bianchi_audit_512_strong_pf010_v2.json")

    # Covariance / relabel invariance (new)
    ap.add_argument("--cov_ms080_json", default="src/gr_strict_pp_scalar/PP_EFE_covariance_audit_512_ms080_v1.json")
    ap.add_argument("--cov_strong_json", default="src/gr_strict_pp_scalar/PP_EFE_covariance_audit_512_strong_pf010_v1.json")

    ap.add_argument("--output", default="src/gr_strict_pp_scalar/PP_EFE_structural_status_512_STRICT_PP_v2.json")
    args = ap.parse_args()

    case_list = [c.strip() for c in args.cases.split(",") if c.strip()]
    if not case_list:
        raise SystemExit("ERROR: empty --cases")

    def _case_paths(case: str):
        if case == "ms080":
            return dict(
                closure=args.closure_ms080_json,
                bianchi=args.bianchi_ms080_json,
                cov=args.cov_ms080_json,
            )
        if case == "strong_pf010":
            return dict(
                closure=args.closure_strong_json,
                bianchi=args.bianchi_strong_json,
                cov=args.cov_strong_json,
            )
        raise SystemExit(f"ERROR: unknown case: {case}")

    per_case = []
    n_applicable = 0
    n_missing = 0
    all_pass = True

    for case in case_list:
        P = _case_paths(case)
        present = {k: _exists(v) for k, v in P.items()}
        missing = [k for k, ok in present.items() if not ok]

        if missing:
            per_case.append({
                "case": case,
                "APPLICABLE": False,
                "PASS_structural_case_v2": None,
                "missing": missing,
                "paths": P,
            })
            n_missing += 1
            all_pass = False
            continue

        # ---- closure ----
        C = load_json(P["closure"])
        c_pass = _find_any_bool(C, [
            "ALL_PASS_tensor_closure_audit_v2",
            "ALL_PASS_tensor_closure_audit_v1",
            "ALL_PASS",
            "PASS",
        ])
        c_pass = bool(c_pass is True)

        # ---- bianchi ----
        B = load_json(P["bianchi"])
        b_pass = _find_any_bool(B, [
            "ALL_PASS_bianchi_proxy_audit_v2",
            "ALL_PASS_bianchi_proxy_audit_v1",
            "ALL_PASS",
            "PASS",
        ])
        b_pass = bool(b_pass is True)

        # ---- covariance ----
        V = load_json(P["cov"])
        v_pass = _find_any_bool(V, [
            "ALL_PASS_covariance_relabel_audit_v1",
            "ALL_PASS_covariance_audit_v1",
            "ALL_PASS",
            "PASS",
        ])
        v_pass = bool(v_pass is True)

        case_pass = bool(c_pass and b_pass and v_pass)
        per_case.append({
            "case": case,
            "APPLICABLE": True,
            "PASS_structural_case_v2": case_pass,
            "closure": {"path": P["closure"], "ALL_PASS": c_pass},
            "bianchi": {"path": P["bianchi"], "ALL_PASS": b_pass},
            "covariance": {"path": P["cov"], "ALL_PASS": v_pass},
        })
        n_applicable += 1
        if not case_pass:
            all_pass = False

    APPLICABLE = bool(n_applicable == len(case_list) and n_missing == 0)
    out = {
        "H": int(args.H),
        "W": int(args.W),
        "STRICT_PP": True,
        "PASS": {
            "ALL_PASS_structural_status_512_STRICT_PP_v2": bool(all_pass and APPLICABLE),
            "APPLICABLE": bool(APPLICABLE),
        },
        "summary": {
            "n_applicable_cases": int(n_applicable),
            "n_missing_cases": int(n_missing),
        },
        "per_case": per_case,
        "notes": (
            "STRICT PP structural status v2: requires (closure v2) AND (bianchi proxy v2) AND "
            "(covariance/relabel audit v1) per case. No PDE/Poisson/smoothing/ansatz/regression."
        ),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_structural_status_512_STRICT_PP_v2 =", out["PASS"]["ALL_PASS_structural_status_512_STRICT_PP_v2"])
    print("APPLICABLE =", out["PASS"]["APPLICABLE"])

if __name__ == "__main__":
    main()

