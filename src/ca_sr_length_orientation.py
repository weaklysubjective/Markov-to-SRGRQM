#!/usr/bin/env python3
# ca_sr_length_orientation_wrapper.py
# Orientation sweep for length contraction by delegating to the trusted ca_sr_length_contraction.py

import argparse, json, subprocess, sys, math

def run_one(T, v, L0):
    # Call the proven script and parse its JSON
    cmd = [sys.executable, "ca_sr_length_contraction.py",
           "--T", str(T), "--v", str(v), "--L0", str(L0)]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)

def main():
    ap = argparse.ArgumentParser(description="Orientation sweep (sign flips) via ca_sr_length_contraction.py")
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--v", type=float, default=0.6)
    ap.add_argument("--L0", type=int, default=200)
    ap.add_argument("--angles", type=int, default=8)   # in 1D: sign flips suffice
    ap.add_argument("--tol_cells", type=float, default=14.0)
    args = ap.parse_args()

    # In 1D, 'angles' just alternates the sign of v
    results = []
    errs = []
    target = args.L0 * math.sqrt(max(0.0, 1.0 - args.v*args.v))
    for k in range(args.angles):
        v_k = args.v if (k % 2 == 0) else -args.v
        js = run_one(args.T, v_k, args.L0)
        Lp = float(js.get("L_prime_measured_cells", js.get("L_prime", 0.0)))
        results.append(Lp)
        errs.append(abs(Lp - target))

    max_err = max(errs) if errs else float("inf")
    summary = {
        "T": args.T, "v": args.v, "L0": args.L0, "angles": args.angles,
        "L_prime_per_angle": [float(x) for x in results],
        "L_prime_target": target,
        "max_abs_err_cells": max_err,
        "tol_cells": args.tol_cells,
        "PASS_length_orientation": (max_err <= args.tol_cells),
        "notes": "Delegates to ca_sr_length_contraction.py (the passing implementation). Orientation in 1D = sign flip of v."
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

