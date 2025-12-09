#!/usr/bin/env python3
"""
PP_EFE00_markov_scalar_pack_v3.py

STRICT PP scalar EFE evidence pack v3.

This script does NOT introduce any new physics or geometry:
it only aggregates existing STRICT PP diagnostics into a single JSON.

Inputs:
  1) A v2 scalar pack JSON (from PP_EFE00_markov_scalar_pack_v2.py),
     which already contains:
       - geom_summary
       - efe00_summary
       - peri_C1_summary
       - peri_C3_summary
       - shapiro_summary
       - overall_PASS_scalar_EFE_strict_PP  (v2: without deflection)
  2) A STRICT PP deflection JSON from PP_deflection_markov_front_PP_v2.py.

Outputs:
  - A v3 scalar pack JSON that adds:
      - deflection_summary
      - overall_PASS_scalar_EFE_strict_PP_v3
    where overall_PASS_scalar_EFE_strict_PP_v3 is TRUE iff
      (v2 overall PASS) AND (deflection PASS).

STRICT PP:
  - This script only reads JSON summaries and combines PASS flags.
  - No PDE, no Laplacian, no Poisson, no GR ansatz, no regression.
  - No distances or times are recomputed here.
"""

import argparse
import json
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STRICT PP scalar EFE evidence pack v3 (adds deflection to v2 pack)."
    )
    p.add_argument(
        "--pack_v2",
        required=True,
        help="Input v2 scalar pack JSON (from PP_EFE00_markov_scalar_pack_v2.py)",
    )
    p.add_argument(
        "--deflection",
        required=True,
        help="Deflection JSON (from PP_deflection_markov_front_PP_v2.py)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output v3 scalar pack JSON path",
    )
    return p.parse_args()


def main():
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        description="STRICT PP scalar EFE evidence pack v3 (aggregate v2 + deflection)"
    )
    parser.add_argument("--pack_v2", required=True, help="Path to v2 scalar pack JSON")
    parser.add_argument("--deflection", required=True, help="Path to deflection JSON (v2 or v3 front)")
    parser.add_argument("--output", required=True, help="Output v3 pack JSON")
    args = parser.parse_args()

    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    pack_v2 = load_json(args.pack_v2)
    defl = load_json(args.deflection)

    # -------------------------------
    # Normalize deflection summary
    # -------------------------------
    deflection_summary = {}

    # If this looks like a raw deflection report, normalize it.
    if isinstance(defl, dict) and "PASS_deflection_markov_front_PP" in defl:
        md = defl.get("markov_distance_to_mass") or {}

        deflection_summary = {
            "H": defl.get("H"),
            "W": defl.get("W"),
            "N": defl.get("N"),
            "edges_flat": defl.get("edges_flat"),
            "edges_curved": defl.get("edges_curved"),
            "mass_mask": defl.get("mass_mask"),
            "src_row_label": defl.get("src_row_label"),
            "src_col_label": defl.get("src_col_label"),
            "steps": defl.get("steps"),

            # v2 endpoint-front fields (may be absent in v3)
            "norm_finite_mass_flat": defl.get("norm_finite_mass_flat"),
            "norm_finite_mass_curved": defl.get("norm_finite_mass_curved"),
            "D_flat": md.get("D_flat"),
            "D_curved": md.get("D_curved"),

            # v3 TIMEQ fields (names may vary; copy if present)
            "t10_flat": defl.get("t10_flat"),
            "t10_curved": defl.get("t10_curved"),
            "t50_flat": defl.get("t50_flat"),
            "t50_curved": defl.get("t50_curved"),
            "t90_flat": defl.get("t90_flat"),
            "t90_curved": defl.get("t90_curved"),

            "PASS_deflection_markov_front_PP": defl.get("PASS_deflection_markov_front_PP"),
            "notes": defl.get("notes"),
        }
    else:
        # Already a summary-like dict
        deflection_summary = dict(defl)

    # -------------------------------
    # Correct applicability detection
    # -------------------------------
    PASS_defl = deflection_summary.get("PASS_deflection_markov_front_PP", None)

    # v2 evidence present?
    has_v2_norms = (
        deflection_summary.get("norm_finite_mass_flat", None) is not None and
        deflection_summary.get("norm_finite_mass_curved", None) is not None
    )

    # v3 TIMEQ evidence present?
    has_timeq = (
        deflection_summary.get("t50_flat", None) is not None or
        deflection_summary.get("t50_curved", None) is not None
    )

    # General applicability:
    # If we have either recognizable evidence fields OR PASS explicitly set.
    DEFLECTION_APPLICABLE = bool(has_v2_norms or has_timeq or (PASS_defl is not None))

    # Gate:
    PASS_deflection_or_NA = True if not DEFLECTION_APPLICABLE else bool(PASS_defl)

    # -------------------------------
    # Core v2 pass
    # -------------------------------
    core = pack_v2.get("overall_PASS_scalar_EFE_strict_PP", None)

    # -------------------------------
    # v3 overall logic (strict PP bookkeeping)
    # -------------------------------
    if core is None:
        overall_v3 = None
    elif core is False:
        overall_v3 = False
    else:
        # core True
        overall_v3 = True if PASS_deflection_or_NA else False

    # -------------------------------
    # Build v3 pack
    # -------------------------------
    pack_v3 = dict(pack_v2)

    deflection_summary = dict(deflection_summary)
    deflection_summary["DEFLECTION_APPLICABLE"] = DEFLECTION_APPLICABLE

    pack_v3["deflection_summary"] = deflection_summary
    pack_v3["PASS_deflection_or_NA"] = PASS_deflection_or_NA
    pack_v3["overall_PASS_scalar_EFE_strict_PP_v3"] = overall_v3

    pack_v3["notes"] = (
        "STRICT PP scalar EFE evidence pack v3. Aggregates v2 scalar evidence with "
        "a strict-PP deflection observable (v2 endpoint-front or v3 time-quantile). "
        "No PDE, no Laplacian/Poisson, no GR ansatz, no regression. "
        "Admissibility gate rules: "
        "deflection is applicable if recognized deflection evidence fields are present "
        "(endpoint norms or time-quantiles) or if PASS_deflection_markov_front_PP is explicit. "
        "If not applicable, PASS_deflection_or_NA=True for bookkeeping; otherwise it equals PASS."
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(pack_v3, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("DEFLECTION_APPLICABLE =", DEFLECTION_APPLICABLE)
    print("PASS_deflection_or_NA =", PASS_deflection_or_NA)
    print("overall_PASS_scalar_EFE_strict_PP_v3 =", overall_v3)



if __name__ == "__main__":
    main()

