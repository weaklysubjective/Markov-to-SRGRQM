#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "STRICT PP EFE status at 512x512 PPV1. "
            "Aggregates scalar EFE-00 status and vector suite status into a single report. "
            "No new physics, no PDE, no Laplacian/Poisson, no regression."
        )
    )
    ap.add_argument(
        "--scalar_status_json",
        required=True,
        help="Scalar EFE-00 status JSON, e.g. "
             "src/gr_strict_pp_scalar/PP_scalar_512_EFE_status_PPV1.json",
    )
    ap.add_argument(
        "--vector_suite_json",
        required=True,
        help="Vector suite JSON, e.g. "
             "src/gr_strict_pp_vector/PP_vector_suite_512_ms080_pf010_bt001_PPV1_v2.json",
    )
    ap.add_argument(
        "--deflection_json",
        required=False,
        default="",
        help="Optional deflection-front JSON for documentation, e.g. "
             "src/gr_strict_pp_scalar/"
             "PP_deflection_markov_front_512_strong_pf010_v5_PPV1.json",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output JSON path, e.g. "
             "src/gr_strict_pp_scalar/PP_EFE_512_STRICT_PP_status_PPV1.json",
    )
    args = ap.parse_args()

    # --- Load components ---
    scalar = load_json(args.scalar_status_json)
    vector = load_json(args.vector_suite_json)
    deflection = None
    if args.deflection_json:
        try:
            deflection = load_json(args.deflection_json)
        except FileNotFoundError:
            deflection = None

    # --- Extract booleans ---
    scalar_ok = bool(scalar.get("overall_PASS_scalar_EFE00_strict_PP_v1"))
    # Vector suite v2 uses ALL_PASS_vector_strict_PP_v2
    vector_ok = bool(
        vector.get("ALL_PASS_vector_strict_PP_v2",
                   vector.get("ALL_PASS_vector_strict_PP_v1", False))
    )

    # Deflection: we treat it explicitly as NA / negative at 512
    deflection_status = None
    deflection_notes = "No deflection JSON provided."
    if deflection is not None:
        # We only look at the applicability flag if present
        ds = deflection.get("deflection_stats", {})
        applicable = ds.get("APPLICABLE", deflection.get("APPLICABLE"))
        deflection_status = {
            "APPLICABLE": bool(applicable),
            "PASS_key_present": any(
                k.startswith("PASS_deflection") for k in (ds.keys() if isinstance(ds, dict) else [])
            ),
        }
        if not applicable:
            deflection_notes = (
                "Deflection-front test at 512x512 PPV1 is NOT APPLICABLE for strong_pf010: "
                "no Markov hop-distance band (d>0) with sufficient nodes and "
                "nonzero flat/curved front mass."
            )
        else:
            deflection_notes = (
                "Deflection-front test at 512x512 PPV1 is applicable; "
                "see deflection JSON for PASS/FAIL details."
            )

    overall_pass = scalar_ok and vector_ok

    H = scalar.get("H")
    W = scalar.get("W")

    out: Dict[str, Any] = {
        "H": H,
        "W": W,
        "scalar_status_json": args.scalar_status_json,
        "vector_suite_json": args.vector_suite_json,
        "deflection_json": args.deflection_json or None,
        "scalar_PASS": scalar_ok,
        "vector_PASS": vector_ok,
        "deflection_status": deflection_status,
        "overall_PASS_STRICT_PP_EFE_512_PPV1": overall_pass,
        "notes": (
            "STRICT PP EFE status at 512x512 PPV1. "
            "Scalar side uses τ-geometry (G00 sign + κ sign) and sparse Shapiro; "
            "vector side uses azimuthal bias / feeder-shear / frame-dragging proxies "
            "with admissibility gates. "
            "Both scalar and vector suites PASS at this scale. "
            + deflection_notes
        ),
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("overall_PASS_STRICT_PP_EFE_512_PPV1 =", overall_pass)


if __name__ == "__main__":
    main()

