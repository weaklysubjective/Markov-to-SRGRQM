#!/usr/bin/env python3
import argparse, json, random

def sample_points_1p1(T, n, seed=7):
    rng = random.Random(seed)
    pts = []
    for _ in range(n):
        t = rng.randrange(0, T+1)
        # spatial extent limited by cone: |x| <= t
        x = rng.randrange(-t, t+1)
        pts.append( (t, x) )
    return pts

def comparable(p, q):
    (t1,x1),(t2,x2)=p,q
    dt=t2-t1; dx=x2-x1
    if dt==0: return False
    if dt>0:
        return abs(dx) <= dt
    else:
        return abs(dx) <= -dt

def estimate_order_fraction(T, n_points, n_pairs, seed=7):
    pts = sample_points_1p1(T, n_points, seed=seed)
    rng = random.Random(seed+1)
    comp = 0
    for _ in range(n_pairs):
        i = rng.randrange(0, n_points)
        j = rng.randrange(0, n_points-1)
        if j >= i: j += 1
        if comparable(pts[i], pts[j]): comp += 1
    return comp / float(n_pairs)

def main():
    ap = argparse.ArgumentParser(description="Myrheim–Meyer–style order fraction in 1+1D.")
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--n_points", type=int, default=20000)
    ap.add_argument("--n_pairs", type=int, default=40000)
    ap.add_argument("--tol", type=float, default=0.03, help="tolerance around 0.5")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    r = estimate_order_fraction(args.T, args.n_points, args.n_pairs, seed=args.seed)
    out = {
        "T": args.T,
        "n_points": args.n_points,
        "n_pairs": args.n_pairs,
        "order_fraction_hat": r,
        "target_order_fraction_1p1": 0.5,
        "abs_err": abs(r - 0.5),
        "tol": args.tol,
        "PASS_mm_order_fraction": (abs(r - 0.5) <= args.tol),
        "notes": "1+1D slab; comparable iff |Δx| ≤ |Δt|. This is a minimal MM signal; full MM inversion not required here."
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

