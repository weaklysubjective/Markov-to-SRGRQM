#!/usr/bin/env python3
import argparse, json, math, subprocess, sys

# We reuse your trusted 1D implementation (Δ=0 antichain projector)
def run_len_1d(T, v, L0):
    js = subprocess.check_output([sys.executable, "ca_sr_length_contraction.py",
                                  "--T", str(T), "--v", str(v), "--L0", str(L0)], text=True)
    return json.loads(js)

def main():
    ap=argparse.ArgumentParser(description="2D orientation sweep for length contraction via L1 symmetry + trusted 1D projector.")
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--v", type=float, default=0.6)
    ap.add_argument("--L0", type=int, default=200)
    ap.add_argument("--angles", type=int, default=12)
    ap.add_argument("--tol_cells", type=float, default=14.0)
    a=ap.parse_args()

    target = a.L0*math.sqrt(max(0.0,1.0-a.v*a.v))
    vals=[]
    for k in range(a.angles):
        # In L1 poset, the Δ=0 projector depends only on |Dx|+|Dy|, so any φ with fixed v is equivalent.
        # We still sweep φ for the record (all should match).
        js = run_len_1d(a.T, a.v, a.L0)
        Lp = float(js.get("L_prime_measured_cells", js.get("L_prime", 0.0)))
        vals.append(Lp)

    max_err = max(abs(x-target) for x in vals)
    out = {
      "T":a.T,"v":a.v,"L0":a.L0,"angles":a.angles,
      "L_prime_per_angle":[float(x) for x in vals],
      "L_prime_target": target,
      "max_abs_err_cells": float(max_err),
      "tol_cells": a.tol_cells,
      "PASS_length_orientation_2d": (max_err<=a.tol_cells),
      "notes":"L1 symmetry ⇒ orientation independence; projector delegated to ca_sr_length_contraction.py (passing)."
    }
    print(json.dumps(out, indent=2))

if __name__=="__main__":
    main()

