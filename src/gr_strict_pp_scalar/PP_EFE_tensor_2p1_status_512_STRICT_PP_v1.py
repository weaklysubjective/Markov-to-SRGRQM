#!/usr/bin/env python3
import argparse, json, os
from typing import Any, Optional, Iterable, Tuple, Dict, List

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

def _case_lookup(root: Any, case: str) -> Optional[dict]:
    """
    Tries to locate per-case record inside common JSON shapes:
      - {"per_case":[{"case": "...", ...}, ...]}
      - {"cases": {"ms080": {...}, ...}}
      - {"cases": [{"case":"ms080",...}, ...]}
    """
    if not isinstance(root, dict):
        return None

    # per_case list
    pc = root.get("per_case")
    if isinstance(pc, list):
        for it in pc:
            if isinstance(it, dict) and str(it.get("case")) == case:
                return it

    # cases dict
    cases = root.get("cases")
    if isinstance(cases, dict) and case in cases and isinstance(cases[case], dict):
        out = dict(cases[case])
        out.setdefault("case", case)
        return out

    # cases list
    if isinstance(cases, list):
        for it in cases:
            if isinstance(it, dict) and str(it.get("case")) == case:
                return it

    return None

def _parse_applicable_pass(obj: Any, prefer_keys: List[str]) -> Tuple[Optional[bool], Optional[bool]]:
    """
    Liberal parse:
      - APPLICABLE: tries "APPLICABLE"
      - PASS: tries prefer_keys first, then generic keys
    """
    AP = _find_any_bool(obj, ["APPLICABLE"])
    PAS = _find_any_bool(obj, prefer_keys + [
        "PASS",
        "ALL_PASS",
        "overall_PASS",
        "PASS_structural_case_v1",
        "PASS_structural_case",
        "PASS_case",
        "ALL_PASS_case",
    ])
    if PAS is not None and AP is None:
        AP = True
    return AP, PAS

def main():
    ap = argparse.ArgumentParser(
        description=(
            "STRICT PP tensor_2p1 status @512: combines (A) structural (closure+bianchi) and "
            "(B) offdiag_2p1 block status into a single per-case and overall gate. "
            "Pure aggregator: no generators, no PDE/Poisson, no smoothing, no regression."
        )
    )

    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)

    # For CLI consistency across the repo: accept --device gpu|cuda|cpu|hip, but do not compute here.
    ap.add_argument("--device", default="gpu",
                    help="Accepted for consistency; aggregator does not compute. Use gpu|cuda|cpu|hip.")

    ap.add_argument("--cases", default="ms080,strong_pf010",
                    help="Comma-separated case list to gate (default: ms080,strong_pf010).")

    ap.add_argument("--structural_json",
                    default="src/gr_strict_pp_scalar/PP_EFE_structural_status_512_STRICT_PP_v1.json",
                    help="Structural status JSON (closure+bianchi).")

    ap.add_argument("--offdiag_status_json",
                    default="src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v2.json",
                    help="Offdiag 2+1 block status JSON (Einstein offdiag / ring flux style).")

    ap.add_argument("--allow_na_structural", action="store_true",
                    help="If set, missing/NA structural does not gate PASS (NOT recommended).")
    ap.add_argument("--allow_na_offdiag", action="store_true",
                    help="If set, missing/NA offdiag_2p1 does not gate PASS (NOT recommended).")

    ap.add_argument("--output",
                    default="src/gr_strict_pp_scalar/PP_EFE_tensor_2p1_status_512_STRICT_PP_v1.json")

    args = ap.parse_args()

    cases = [c.strip() for c in str(args.cases).split(",") if c.strip()]
    if not cases:
        raise SystemExit("No cases provided via --cases")

    structural_root = None
    offdiag_root = None

    if _exists(args.structural_json):
        structural_root = load_json(args.structural_json)
    if _exists(args.offdiag_status_json):
        offdiag_root = load_json(args.offdiag_status_json)

    per_case_out: List[dict] = []
    n_applicable = 0
    n_missing = 0

    # Global summaries (best-effort)
    global_struct_AP, global_struct_PASS = (None, None)
    global_off_AP, global_off_PASS = (None, None)

    if structural_root is not None:
        global_struct_AP, global_struct_PASS = _parse_applicable_pass(
            structural_root,
            prefer_keys=["ALL_PASS_structural_status_512_STRICT_PP_v1"]
        )
    if offdiag_root is not None:
        # Prefer canonical flags if present
        oflags = offdiag_root.get("offdiag_flags") if isinstance(offdiag_root, dict) else None
        if isinstance(oflags, dict):
            global_off_AP = _as_bool(oflags.get("APPLICABLE", None), None)
            global_off_PASS = _as_bool(oflags.get("PASS", None), None)
        if global_off_PASS is None:
            global_off_AP, global_off_PASS = _parse_applicable_pass(
                offdiag_root,
                prefer_keys=[
                    "ALL_PASS_offdiag_2p1_strict_PP_512_v2",
                    "ALL_PASS_offdiag_2p1_strict_PP_512_v1",
                    "overall_PASS_offdiag_2p1_strict_PP_512_v2",
                    "overall_PASS_offdiag_2p1_strict_PP_512_v1",
                ]
            )

    # Per-case gating
    all_cases_pass = True
    all_cases_applicable = True

    for case in cases:
        sc = _case_lookup(structural_root, case) if structural_root is not None else None
        oc = _case_lookup(offdiag_root, case) if offdiag_root is not None else None

        # Structural per-case parse (fall back to global if no per-case record exists)
        if sc is not None:
            sAP, sPAS = _parse_applicable_pass(sc, prefer_keys=["PASS_structural_case_v1"])
        else:
            sAP, sPAS = global_struct_AP, global_struct_PASS

        # Offdiag per-case parse (fall back to global if no per-case record exists)
        if oc is not None:
            oAP, oPAS = _parse_applicable_pass(oc, prefer_keys=[
                "ALL_PASS_offdiag_2p1_strict_PP_512_v2",
                "PASS_offdiag_2p1_case_v2",
                "PASS_offdiag_2p1_case_v1",
            ])
        else:
            oAP, oPAS = global_off_AP, global_off_PASS

        # Presence bookkeeping
        missing_struct = (structural_root is None) or (sc is None and global_struct_PASS is None and global_struct_AP is None)
        missing_offdiag = (offdiag_root is None) or (oc is None and global_off_PASS is None and global_off_AP is None)

        # Applicability for this case:
        # - If either component says APPLICABLE True, we treat that component as applicable.
        # - If APPLICABLE is missing but PASS present, treat as applicable.
        # - If both components unknown/missing, case APPLICABLE is False.
        comp_app_struct = (sAP is True) or (sPAS is not None)
        comp_app_offdiag = (oAP is True) or (oPAS is not None)

        case_applicable = bool(comp_app_struct and comp_app_offdiag)
        # If one side missing/NA, decide whether we allow it
        if not comp_app_struct:
            case_applicable = bool(args.allow_na_structural and comp_app_offdiag)
        if not comp_app_offdiag:
            case_applicable = bool(args.allow_na_offdiag and (comp_app_struct or args.allow_na_structural))

        # PASS logic (STRICT by default):
        # - If component is applicable => require PASS True.
        # - If component not applicable => depends on allow_na_*
        s_gate = True
        if comp_app_struct:
            s_gate = bool(sPAS is True)
        else:
            s_gate = True if args.allow_na_structural else False

        o_gate = True
        if comp_app_offdiag:
            o_gate = bool(oPAS is True)
        else:
            o_gate = True if args.allow_na_offdiag else False

        case_pass = bool(case_applicable and s_gate and o_gate)

        if case_applicable:
            n_applicable += 1
        else:
            all_cases_applicable = False

        if missing_struct or missing_offdiag:
            n_missing += 1

        if not case_pass:
            all_cases_pass = False

        per_case_out.append({
            "case": case,
            "APPLICABLE": case_applicable,
            "PASS_tensor_2p1_case_v1": (case_pass if case_applicable else None),

            "structural": {
                "present_root": structural_root is not None,
                "present_case": sc is not None,
                "APPLICABLE": sAP,
                "PASS": sPAS,
                "source_json": args.structural_json,
            },
            "offdiag_2p1": {
                "present_root": offdiag_root is not None,
                "present_case": oc is not None,
                "APPLICABLE": oAP,
                "PASS": oPAS,
                "source_json": args.offdiag_status_json,
            },
            "debug": {
                "missing_structural": missing_struct,
                "missing_offdiag": missing_offdiag,
            }
        })

    overall_applicable = bool(all_cases_applicable and n_applicable == len(cases))
    overall_pass = bool(overall_applicable and all_cases_pass)

    out = {
        "H": int(args.H),
        "W": int(args.W),
        "STRICT_PP": True,
        "device": str(args.device),
        "inputs": {
            "cases": cases,
            "structural_json": args.structural_json,
            "offdiag_status_json": args.offdiag_status_json,
        },
        "policy": {
            "allow_na_structural": bool(args.allow_na_structural),
            "allow_na_offdiag": bool(args.allow_na_offdiag),
        },
        "PASS": {
            "ALL_PASS_tensor_2p1_status_512_STRICT_PP_v1": bool(overall_pass),
            "APPLICABLE": bool(overall_applicable),
            "PASS_structural_all_cases": None if not overall_applicable else all(
                (pc.get("structural", {}).get("PASS") is True) or (args.allow_na_structural and not pc.get("structural", {}).get("APPLICABLE", True))
                for pc in per_case_out
            ),
            "PASS_offdiag_2p1_all_cases": None if not overall_applicable else all(
                (pc.get("offdiag_2p1", {}).get("PASS") is True) or (args.allow_na_offdiag and not pc.get("offdiag_2p1", {}).get("APPLICABLE", True))
                for pc in per_case_out
            ),
        },
        "summary": {
            "n_cases_total": len(cases),
            "n_applicable_cases": int(n_applicable),
            "n_missing_components_cases": int(n_missing),
        },
        "per_case": per_case_out,
        "notes": (
            "STRICT PP tensor_2p1 status @512: aggregator gate that requires (A) structural closure+bianchi "
            "and (B) offdiag 2+1 block status, per case. "
            "Pure JSON aggregator: no generators, no PDE/Poisson, no smoothing, no GR ansatz, no regression."
        ),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_tensor_2p1_status_512_STRICT_PP_v1 =", out["PASS"]["ALL_PASS_tensor_2p1_status_512_STRICT_PP_v1"])
    print("APPLICABLE =", out["PASS"]["APPLICABLE"])

if __name__ == "__main__":
    main()

