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
    args = parse_args()

    pack_v2 = load_json(args.pack_v2)
    defl = load_json(args.deflection)

    # --- Extract deflection summary from STRICT PP deflection JSON --- #
    md = defl.get("markov_distance_to_mass", {})
    D_flat = md.get("D_flat", None)
    D_curved = md.get("D_curved", None)
    pass_defl = defl.get("PASS_deflection_markov_front_PP", None)

    norm_flat = defl.get("norm_finite_mass_flat", None)
    norm_curved = defl.get("norm_finite_mass_curved", None)

    deflection_summary = {
        "norm_finite_mass_flat": norm_flat,
        "norm_finite_mass_curved": norm_curved,
        "D_flat": D_flat,
        "D_curved": D_curved,
        "PASS_deflection_markov_front_PP": pass_defl,
    }

    # --- Combine PASS flags --- #
    # v2 overall PASS:
    pass_v2 = pack_v2.get("overall_PASS_scalar_EFE_strict_PP", None)

    if pass_v2 is None or pass_defl is None:
        overall_v3 = None
    else:
        overall_v3 = bool(pass_v2 and pass_defl)

    # --- Build v3 pack (do NOT mutate original dict silently) --- #
    pack_v3 = dict(pack_v2)  # shallow copy is fine for this structure
    pack_v3["deflection_summary"] = deflection_summary
    pack_v3["overall_PASS_scalar_EFE_strict_PP_v3"] = overall_v3
    pack_v3["notes"] = (
        "STRICT PP scalar EFE evidence pack v3. Based purely on Markov/trace-derived "
        "geometry and observables: scalar EFE (kappa over mass core), perihelion-like "
        "Markov tau (C1/C3), Shapiro-like Markov tau, and deflection via Markov basin "
        "capture. This script only aggregates existing STRICT PP diagnostics from "
        "PP_EFE00_markov_scalar_pack_v2.py and PP_deflection_markov_front_PP_v2.py; "
        "it introduces no PDE, no Laplacian/Poisson, no GR ansatz, and no regression."
    )

    with open(args.output, "w") as f:
        json.dump(pack_v3, f, indent=2)

    print(f"Wrote v3 scalar pack to {args.output}")


if __name__ == "__main__":
    main()

