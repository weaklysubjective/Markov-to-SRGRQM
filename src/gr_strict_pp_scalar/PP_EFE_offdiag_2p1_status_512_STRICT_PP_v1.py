#!/usr/bin/env python3
import argparse
import json
import os
import sys


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        die(f"JSON not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def as_bool(d: dict, key: str, required: bool = True, default: bool = False) -> bool:
    if key not in d:
        if not required:
            return default
        die(f"Missing key {key!r} in JSON.")
    v = d[key]
    if not isinstance(v, bool):
        die(f"Key {key!r} is not bool (got {type(v)}).")
    return v


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "STRICT PP 2+1D off-diagonal EFE status at 512x512 PPV1.\n"
            "Aggregates scalar EFE-00 status and vector suite status.\n"
            "No new observables; no PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        )
    )
    ap.add_argument(
        "--scalar_status",
        default="src/gr_strict_pp_scalar/PP_scalar_EFE00_status_512_strict_PP_v1.json",
        help="Scalar EFE-00 status JSON (STRICT PP, 512x512).",
    )
    ap.add_argument(
        "--vector_suite",
        default="src/gr_strict_pp_vector/PP_vector_suite_512_ms080_pf010_bt001_PPV1_v2.json",
        help="Vector suite status JSON (STRICT PP, 512x512, v2).",
    )
    ap.add_argument(
        "--output",
        default="src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v1.json",
        help="Output JSON path for combined 2+1D off-diagonal EFE status.",
    )
    args = ap.parse_args()

    # --- Load scalar EFE-00 status ---
    S = load_json(args.scalar_status)

    # Basic sanity
    H = S.get("H")
    W = S.get("W")
    if H != 512 or W != 512:
        die(f"Scalar status H,W must be 512,512 (got {H},{W}).")

    mass_case = S.get("mass_case")
    strong_case = S.get("strong_case")
    if not isinstance(mass_case, str) or not isinstance(strong_case, str):
        die("Scalar status must contain 'mass_case' and 'strong_case' strings.")

    scalar_pass = as_bool(
        S, "overall_PASS_scalar_EFE00_strict_PP_v1", required=True
    )

    # --- Load vector suite status (v2) ---
    V = load_json(args.vector_suite)

    all_pass_vector_v2 = as_bool(
        V, "ALL_PASS_vector_strict_PP_v2", required=True
    )

    # Extract case labels from packs for simple consistency check
    packs = V.get("packs", [])
    if not isinstance(packs, list) or len(packs) == 0:
        die("Vector suite JSON must contain non-empty 'packs' list.")

    case_labels = []
    per_case_ok = {}
    for entry in packs:
        if not isinstance(entry, dict):
            die("Each entry in 'packs' must be a dict.")
        case = entry.get("case")
        if not isinstance(case, str):
            die("Each pack entry must contain a 'case' string.")
        case_labels.append(case)

        # Decide which overall key was used at pack level
        overall_key_used = entry.get("overall_key_used")
        if not isinstance(overall_key_used, str):
            die(f"Pack for case {case!r} missing 'overall_key_used' string.")
        ok = bool(entry.get(overall_key_used))
        per_case_ok[case] = ok

    case_set = set(case_labels)

    # Expected two-case structure: ms080 and strong_pf010
    expected_cases = {mass_case, strong_case}
    cases_match_expected = (case_set >= expected_cases)

    # All expected cases individually PASS under their own chosen overall keys?
    all_expected_cases_pass = all(
        per_case_ok.get(c, False) for c in expected_cases
    )

    # --- Conditions for 2+1D off-diagonal EFE status ---
    cond_scalar = scalar_pass
    cond_vector_suite = all_pass_vector_v2
    cond_case_consistency = cases_match_expected and all_expected_cases_pass

    overall_pass = bool(cond_scalar and cond_vector_suite and cond_case_consistency)

    out = {
        "H": H,
        "W": W,
        "mass_case": mass_case,
        "strong_case": strong_case,
        "scalar_status_json": args.scalar_status,
        "scalar_EFE00_PASS": cond_scalar,
        "vector_suite_json": args.vector_suite,
        "vector_suite_ALL_PASS_vector_strict_PP_v2": all_pass_vector_v2,
        "vector_cases": sorted(case_set),
        "vector_per_case_PASS": per_case_ok,
        "conditions": {
            "cond_scalar_EFE00_PASS": cond_scalar,
            "cond_vector_suite_ALL_PASS_v2": cond_vector_suite,
            "cond_cases_match_expected_and_PASS": cond_case_consistency,
        },
        "overall_PASS_EFE_offdiag_2p1_STRICT_PP_v1": overall_pass,
        "notes": (
            "STRICT PP 2+1D off-diagonal EFE status at 512x512 PPV1. "
            "This is a *summary* over existing scalar EFE-00 status and vector "
            "ring observables (azimuthal flux, ring-gradient correlation, and "
            "their admissibility gates). It introduces no new observables and "
            "performs no PDE, no Laplacian/Poisson solve, and no regression. "
            "PASS means: the same CA/trace-derived mass core that drives the "
            "scalar EFE-00 PASS also drives a consistent vector response in both "
            "ms080 and strong_pf010 cases under fixed STRICT-PP rules."
        ),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("overall_PASS_EFE_offdiag_2p1_STRICT_PP_v1 =", overall_pass)


if __name__ == "__main__":
    main()

