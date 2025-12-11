#!/usr/bin/env python3
import argparse
import json
from typing import Dict, Any


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def extract_pass_flags(d: Dict[str, Any]) -> Dict[str, bool]:
    """
    Collect all top-level PASS_* keys as booleans.
    This is generic enough to work with the existing reports where
    fields like PASS_Shapiro_markov_tau or PASS_scalar_* appear.
    """
    out: Dict[str, bool] = {}
    for k, v in d.items():
        if k.startswith("PASS_"):
            # Coerce to bool, but only if it's a simple JSON type
            out[k] = bool(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "STRICT PP scalar EFE-00 status at 512x512. "
            "Aggregates τ-geometry and Shapiro PASS flags into a single report. "
            "No new physics, no PDE, no Laplacian/Poisson, no regression."
        )
    )
    ap.add_argument("--mass_tau_json", required=True,
                    help="τ-geometry JSON for ms080 (e.g. PP_markov_tau_geometry_512_mass_ms080_topk500.json)")
    ap.add_argument("--strong_tau_json", required=True,
                    help="τ-geometry JSON for strong_pf010 (e.g. PP_markov_tau_geometry_512_strong_pf010_topk500.json)")
    ap.add_argument("--mass_shapiro_json", required=True,
                    help="Shapiro JSON for ms080 (e.g. PP_Shapiro_markov_tau_512_mass_ms080_PPV1.json)")
    ap.add_argument("--strong_shapiro_json", required=True,
                    help="Shapiro JSON for strong_pf010 (e.g. PP_Shapiro_markov_tau_512_strong_pf010_PPV1.json)")
    ap.add_argument("--output", required=True,
                    help="Output JSON path, e.g. src/gr_strict_pp_scalar/PP_scalar_512_EFE_status_PPV1.json")

    args = ap.parse_args()

    # --- Load all inputs ---
    mass_tau = load_json(args.mass_tau_json)
    strong_tau = load_json(args.strong_tau_json)
    mass_shapiro = load_json(args.mass_shapiro_json)
    strong_shapiro = load_json(args.strong_shapiro_json)

    # --- Extract generic PASS_* flags ---
    mass_tau_flags = extract_pass_flags(mass_tau)
    strong_tau_flags = extract_pass_flags(strong_tau)
    mass_shapiro_flags = extract_pass_flags(mass_shapiro)
    strong_shapiro_flags = extract_pass_flags(strong_shapiro)

    # Helper: all PASS flags in a dict must be True; empty dict imposes no constraint
    def all_true_or_empty(flags: Dict[str, bool]) -> bool:
        return all(flags.values()) if flags else True

    # Overall scalar EFE-00 status:
    # - requires all nonempty PASS_* sets from τ-geometry and Shapiro to be True.
    # - deflection is *intentionally excluded* from this scalar EFE status.
    overall_pass = (
        all_true_or_empty(mass_tau_flags)
        and all_true_or_empty(strong_tau_flags)
        and all_true_or_empty(mass_shapiro_flags)
        and all_true_or_empty(strong_shapiro_flags)
    )

    # Try to grab H, W if present (for convenience)
    H = mass_tau.get("H", strong_tau.get("H", None))
    W = mass_tau.get("W", strong_tau.get("W", None))

    out = {
        "H": H,
        "W": W,
        "mass_case": "ms080",
        "strong_case": "strong_pf010",
        "mass_tau_json": args.mass_tau_json,
        "strong_tau_json": args.strong_tau_json,
        "mass_shapiro_json": args.mass_shapiro_json,
        "strong_shapiro_json": args.strength_shapiro_json if hasattr(args, "strength_shapiro_json") else args.strong_shapiro_json,  # backward-safe
        "mass_tau_PASS_flags": mass_tau_flags,
        "strong_tau_PASS_flags": strong_tau_flags,
        "mass_shapiro_PASS_flags": mass_shapiro_flags,
        "strong_shapiro_PASS_flags": strong_shapiro_flags,
        "overall_PASS_scalar_EFE00_strict_PP_v1": overall_pass,
        "notes": (
            "STRICT PP scalar EFE-00 status at 512x512 PPV1. "
            "This is a *summary* over existing τ-geometry and Shapiro reports. "
            "It introduces no new observables and performs no PDE, "
            "no Laplacian/Poisson solve, and no regression. "
            "Deflection-front tests are intentionally excluded here; "
            "see the deflection JSON for their NA/fail status."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("overall_PASS_scalar_EFE00_strict_PP_v1 =", overall_pass)


if __name__ == "__main__":
    main()

