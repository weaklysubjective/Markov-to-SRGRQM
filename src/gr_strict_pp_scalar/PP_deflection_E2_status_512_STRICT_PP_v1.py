#!/usr/bin/env python3
"""
PP_deflection_E2_status_512_STRICT_PP_v1.py

STRICT PP summary aggregator for the 512×512 PPV1_E2 deflection results.

This script:
- Loads the two *already-produced* deflection JSON artifacts (ms080 + strong_pf010),
- Extracts the PASS/APPLICABLE + key scalars,
- Emits a single status JSON with an overall flag.

STRICT PP guarantees:
- No new physics.
- No PDE, no Laplacian/Poisson, no GR ansatz, no regression.
- Pure aggregation over existing STRICT PP artifacts.
"""

import argparse
import json
import os
from typing import Any, Dict, Optional


DEFAULT_MS080 = "src/gr_strict_pp_scalar/PP_deflection_markov_front_512_ms080_v7_PPV1_E2_late.json"
DEFAULT_PF010 = "src/gr_strict_pp_scalar/PP_deflection_markov_front_512_strong_pf010_v7_PPV1_E2_late.json"
DEFAULT_OUT   = "src/gr_strict_pp_scalar/PP_deflection_E2_status_512_STRICT_PP_v1.json"


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _get_deflection_block(P: Dict[str, Any]) -> Dict[str, Any]:
    # v7 files store under deflection_stats
    ds = P.get("deflection_stats")
    if isinstance(ds, dict):
        return ds
    # fallback: accept flat layout if older versions wrote at top-level
    return P


def _extract_case_summary(case: str, path: str) -> Dict[str, Any]:
    P = _load_json(path)
    ds = _get_deflection_block(P)

    applicable = ds.get("APPLICABLE", None)
    # v7 key in deflection_stats:
    pass_key = "PASS_deflection_markov_front_PP_v7" if "PASS_deflection_markov_front_PP_v7" in ds else "PASS"
    passed = ds.get(pass_key, None)

    # Scalars (may be None if NA)
    D_flat = ds.get("D_flat", None)
    D_curved = ds.get("D_curved", None)
    reduction_fraction = ds.get("reduction_fraction", None)
    reason = ds.get("reason", None)

    # Basic sanity: applicable implies we can interpret the scalar values
    if applicable is True:
        assert D_flat is not None, f"{case}: APPLICABLE=True but D_flat is missing"
        assert D_curved is not None, f"{case}: APPLICABLE=True but D_curved is missing"

    return {
        "case": case,
        "path": path,
        "APPLICABLE": applicable,
        "PASS_key_used": pass_key,
        "PASS": passed,
        "D_flat": D_flat,
        "D_curved": D_curved,
        "reduction_fraction": reduction_fraction,
        "reason": reason,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="STRICT PP deflection E2 status aggregator (512×512, v1).")
    ap.add_argument("--ms080_json", default=DEFAULT_MS080, help="ms080 v7 E2 late JSON")
    ap.add_argument("--pf010_json", default=DEFAULT_PF010, help="strong_pf010 v7 E2 late JSON")
    ap.add_argument("--output", default=DEFAULT_OUT)
    args = ap.parse_args()

    ms = _extract_case_summary("ms080", args.ms080_json)
    pf = _extract_case_summary("strong_pf010", args.pf010_json)

    def ok_case(C: Dict[str, Any]) -> bool:
        # Strict reading: must be explicitly APPLICABLE=True and PASS=True
        return bool(C.get("APPLICABLE") is True and C.get("PASS") is True)

    overall = bool(ok_case(ms) and ok_case(pf))

    out = {
        "H": 512,
        "W": 512,
        "notes": (
            "STRICT PP deflection E2 status (512×512). Aggregates v7 late-window deflection-front artifacts "
            "for ms080 and strong_pf010. No new observables, no PDE/Poisson, no GR ansatz, no regression."
        ),
        "inputs": {
            "ms080_json": args.ms080_json,
            "strong_pf010_json": args.pf010_json,
        },
        "cases": [ms, pf],
        "overall_PASS_deflection_E2_strict_PP_v1": overall,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("overall_PASS_deflection_E2_strict_PP_v1 =", overall)


if __name__ == "__main__":
    main()

