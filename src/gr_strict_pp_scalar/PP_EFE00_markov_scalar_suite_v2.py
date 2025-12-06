#!/usr/bin/env python3
"""
PP_EFE00_markov_scalar_suite_v2.py

STRICT PP scalar EFE suite v2.

This script ONLY aggregates existing STRICT PP scalar packs (v3).
It does NOT compute any new geometry or physics.

Input:
  --packs PACK1.json PACK2.json ...

Each pack is assumed to be produced by PP_EFE00_markov_scalar_pack_v3.py and
to contain:
  - overall_PASS_scalar_EFE_strict_PP_v3 (bool or null)

Output:
  A single JSON with:
    - n_packs
    - per-pack PASS flags
    - ALL_PASS_scalar_EFE_strict_PP_v3
    - notes
"""

import argparse
import json
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STRICT PP scalar EFE suite v2 over v3 scalar packs."
    )
    p.add_argument(
        "--packs",
        nargs="+",
        required=True,
        help="List of v3 scalar pack JSON files (from PP_EFE00_markov_scalar_pack_v3.py)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output suite JSON path",
    )
    return p.parse_args()


def main():
    args = parse_args()

    pack_paths: List[str] = args.packs
    results = []
    all_pass = True
    any_defined = False

    for path in pack_paths:
        data = load_json(path)
        flag = data.get("overall_PASS_scalar_EFE_strict_PP_v3", None)
        results.append(
            {
                "pack": path,
                "overall_PASS_scalar_EFE_strict_PP_v3": flag,
            }
        )
        if flag is not None:
            any_defined = True
            if not flag:
                all_pass = False

    if not any_defined:
        all_pass_final = None
    else:
        all_pass_final = all_pass

    suite = {
        "n_packs": len(pack_paths),
        "packs": results,
        "ALL_PASS_scalar_EFE_strict_PP_v3": all_pass_final,
        "notes": (
            "STRICT PP scalar EFE suite v2. Aggregates v3 scalar packs "
            "built purely from Markov/trace-derived geometry and observables "
            "(scalar EFE, perihelion tau, Shapiro tau, deflection basin). "
            "This script introduces no new physics or geometry; it only "
            "reports whether all input packs have overall_PASS_scalar_EFE_strict_PP_v3 = true."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(suite, f, indent=2)

    print(f"Wrote scalar suite v2 to {args.output}")


if __name__ == "__main__":
    main()

