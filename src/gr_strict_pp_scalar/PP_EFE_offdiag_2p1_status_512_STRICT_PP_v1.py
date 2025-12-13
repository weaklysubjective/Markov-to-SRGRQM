#!/usr/bin/env python3
import argparse, json, os
from typing import Any, Dict, List, Optional

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def _pick(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP off-diag 2+1 status aggregator (512). "
                    "Aggregates per-case Einstein-block toy off-diag artifacts. "
                    "No new observables; no PDE/Poisson; no GR ansatz; no regression."
    )

    # Backward-compat placeholders (optional, not gating)
    ap.add_argument("--scalar_status", default=None,
                    help="Optional scalar status JSON (informational only).")
    ap.add_argument("--vector_suite", default=None,
                    help="Optional vector suite JSON (informational only).")

    # NEW: explicit blocks list (this is what you tried to run)
    ap.add_argument("--blocks", nargs="+", default=None,
                    help="List of per-case offdiag Einstein-block JSONs (ms080 + strong_pf010).")

    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    blocks = []
    if args.blocks:
        blocks = args.blocks
    else:
        # If no --blocks, we still write a report, but it will be non-applicable.
        blocks = []

    per_case = []
    all_app = True
    all_pass = True

    for p in blocks:
        if not os.path.exists(p):
            all_app = False
            all_pass = False
            per_case.append({
                "path": p,
                "present": False,
                "APPLICABLE": None,
                "PASS": None,
                "reason": "missing_file"
            })
            continue

        J = load_json(p)

        applicable = bool(_pick(J, ["APPLICABLE"], False))
        # accept key variants
        pass_key = _pick(J, [
            "PASS_offdiag_2p1_toy_diag_v3",
            "PASS_offdiag_2p1_toy_diag_v2",
            "PASS_offdiag_2p1_toy_diag",
        ], None)
        passed = bool(pass_key) if pass_key is not None else False

        all_app = all_app and applicable
        all_pass = all_pass and passed

        per_case.append({
            "path": p,
            "present": True,
            "case": _pick(J, ["case"], None),
            "APPLICABLE": applicable,
            "PASS": passed,
            "pass_key_used": "PASS_offdiag_2p1_toy_diag_v3" if "PASS_offdiag_2p1_toy_diag_v3" in J else (
                "PASS_offdiag_2p1_toy_diag_v2" if "PASS_offdiag_2p1_toy_diag_v2" in J else (
                    "PASS_offdiag_2p1_toy_diag" if "PASS_offdiag_2p1_toy_diag" in J else None
                )
            )
        })

    out = {
        "H": 512,
        "W": 512,
        "notes": "STRICT PP off-diag 2+1 status (512×512). Aggregates per-case offdiag Einstein-block artifacts. "
                 "No new observables, no PDE/Poisson, no GR ansatz, no regression.",
        "inputs": {
            "blocks": blocks,
            "scalar_status": args.scalar_status,
            "vector_suite": args.vector_suite,
        },
        "offdiag_flags": {
            "present": True,
            "APPLICABLE": bool(blocks) and all_app,
            "PASS": bool(blocks) and all_pass,
        },
        "per_case": per_case,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("APPLICABLE =", out["offdiag_flags"]["APPLICABLE"])
    print("PASS =", out["offdiag_flags"]["PASS"])

if __name__ == "__main__":
    main()

