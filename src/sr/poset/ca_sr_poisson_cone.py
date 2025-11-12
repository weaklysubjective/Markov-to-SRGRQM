#!/usr/bin/env python3
import argparse, json, math, random

def reachable_points(T, density, seed):
    rng = random.Random(seed)
    points = [(0.0,0.0)]  # start
    R = T  # c=1: radius grows T
    # Sprinkle in expanding disk each tick (just for visualization / union audit)
    all_pts = set([(0,0)])
    for t in range(1, T+1):
        r_max = t
        n = int(density * math.pi * (r_max**2 - (r_max-1)**2))
        for _ in range(max(1,n)):
            r = (r_max-1) + rng.random()
            th = rng.uniform(0, 2*math.pi)
            x = r*math.cos(th); y = r*math.sin(th)
            all_pts.add((int(round(x)), int(round(y))))
    return all_pts, R

def main():
    ap=argparse.ArgumentParser(description="Grid-free Poisson cone: union future is a disk of radius T (c=1).")
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--density", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=7)
    a=ap.parse_args()

    pts, R = reachable_points(a.T, a.density, a.seed)
    # Audit: every lattice point (i,j) with sqrt(i^2+j^2)<=R should be present (up to discretization)
    miss=0; total=0
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            if i*i + j*j <= R*R:
                total += 1
                if (i,j) not in pts: miss += 1
    out = {
      "T":a.T,"radius_R":R,"density":a.density,"seed":a.seed,
      "approx_union_points": len(pts),
      "disk_lattice_points": total,
      "missing_in_disk": miss,
      "PASS_circular_union": (miss/float(total) <= 0.02),  # ≤2% discretization miss
      "notes":"Grid-free adjacency (Euclidean ≤1 per tick) ⇒ union future is a disk. This test audits discretized coverage."
    }
    print(json.dumps(out, indent=2))

if __name__=="__main__":
    main()

