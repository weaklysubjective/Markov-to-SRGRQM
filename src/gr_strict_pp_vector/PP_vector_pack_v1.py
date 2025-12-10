#!/usr/bin/env python3
import argparse
import json
import os
import sys


def load_json(path: str):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def infer_hw_from_subreports(az, fd, fs):
    # Prefer any explicit H/W from whichever subreport is present.
    for R in (az, fd, fs):
        if isinstance(R, dict) and R.get("H") and R.get("W"):
            return int(R["H"]), int(R["W"])
    return 0, 0


def infer_n_from_hw(H, W):
    if H > 0 and W > 0:
        return int(H * W)
    return 0


def pass_or_na(applicable_val, pass_val):
    # Applicable False => treat as NA => PASS-or-NA True
    if applicable_val is False:
        return True
    # If applicable is True or missing, use pass_val truthiness
    return bool(pass_val)


def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP vector evidence pack v1 with admissibility gates."
    )
    ap.add_argument("--case", required=True)
    ap.add_argument("--azimuthal_json", default="")
    ap.add_argument("--frame_dragging_json", default="")
    ap.add_argument("--feeder_shear_json", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    az = load_json(args.azimuthal_json) if args.azimuthal_json else None
    fd = load_json(args.frame_dragging_json) if args.frame_dragging_json else None
    fs = load_json(args.feeder_shear_json) if args.feeder_shear_json else None

    H, W = infer_hw_from_subreports(az, fd, fs)
    N = infer_n_from_hw(H, W)

    P = {
        "case": args.case,
        "H": H,
        "W": W,
        "N": N,
        "azimuthal_flux": az,
        "frame_dragging": fd,
        "feeder_shear": fs,
        "notes": (
            "STRICT PP vector pack v1. Aggregates vector observables with "
            "admissibility gates. No PDE, no Laplacian/Poisson, "
            "no GR ansatz, no regression."
        ),
    }

    # --- Compute PASS-or-NA flags ---
    # Azimuthal
    az_app = None
    az_pass = None
    if isinstance(az, dict):
        az_app = az.get("VECTOR_AZIMUTHAL_FLUX_APPLICABLE")
        az_pass = az.get("PASS_vector_azimuthal_flux_PP")

    P["PASS_azimuthal_or_NA"] = pass_or_na(az_app, az_pass)

    # Frame dragging
    fd_app = None
    fd_pass = None
    if isinstance(fd, dict):
        fd_app = fd.get("FRAME_DRAGGING_APPLICABLE")
        fd_pass = fd.get("PASS_frame_dragging_PP")

    P["PASS_frame_dragging_or_NA"] = pass_or_na(fd_app, fd_pass)

    # Feeder shear
    fs_app = None
    fs_pass = None
    if isinstance(fs, dict):
        fs_app = fs.get("FEEDER_SHEAR_APPLICABLE")
        fs_pass = fs.get("PASS_vector_feeder_shear_PP")

    P["PASS_feeder_shear_or_NA"] = pass_or_na(fs_app, fs_pass)

    # --- Overall v1 (legacy strict gate) ---
    P["overall_PASS_vector_strict_PP_v1"] = bool(
        P["PASS_azimuthal_or_NA"]
        and P["PASS_frame_dragging_or_NA"]
        and P["PASS_feeder_shear_or_NA"]
    )

    # --- Overall v2 (512-aware) ---
    # At 512+, feeder shear is treated as informative (non-gating).
    if H >= 512:
        P["overall_PASS_vector_strict_PP_v2"] = bool(
            P["PASS_azimuthal_or_NA"]
            and P["PASS_frame_dragging_or_NA"]
        )
    else:
        P["overall_PASS_vector_strict_PP_v2"] = P[
            "overall_PASS_vector_strict_PP_v1"
        ]

    # --- Write ---
    with open(args.output, "w") as f:
        json.dump(P, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("overall_PASS_vector_strict_PP_v1 =", P["overall_PASS_vector_strict_PP_v1"])
    print("overall_PASS_vector_strict_PP_v2 =", P["overall_PASS_vector_strict_PP_v2"])


if __name__ == "__main__":
    main()

