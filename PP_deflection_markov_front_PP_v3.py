#!/usr/bin/env python3
import argparse, json, os, sys
import numpy as np

def load_edges(path, N):
    edges = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace(",", " ")
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                a = int(parts[0]); b = int(parts[1])
            except:
                continue
            if 0 <= a < N and 0 <= b < N:
                edges.append((a, b))
    return edges

def build_transition_dense(edges, N):
    # row-stochastic Markov kernel from directed edges
    P = np.zeros((N, N), dtype=np.float64)
    outdeg = np.zeros(N, dtype=np.int64)
    for a, b in edges:
        outdeg[a] += 1
    # If a node has zero outdeg, make it absorbing (STRICT PP-safe; purely Markov closure)
    for i in range(N):
        if outdeg[i] == 0:
            P[i, i] = 1.0
    for a, b in edges:
        if outdeg[a] > 0:
            P[a, b] += 1.0 / float(outdeg[a])
    # sanity
    rs = P.sum(axis=1)
    # allow tiny float error
    if not np.allclose(rs, 1.0, atol=1e-12):
        # fix any numeric drift
        for i in range(N):
            s = rs[i]
            if s == 0:
                P[i, i] = 1.0
            else:
                P[i, :] /= s
    return P

def simulate_mass_cdf(P, src_id, mass_mask, steps):
    N = P.shape[0]
    assert P.shape == (N, N)
    p = np.zeros(N, dtype=np.float64)
    p[src_id] = 1.0

    F = np.zeros(steps + 1, dtype=np.float64)
    F[0] = float(p[mass_mask].sum())

    for t in range(1, steps + 1):
        # row-vector propagation
        p = p @ P
        # clamp tiny negatives
        if (p < -1e-15).any():
            p = np.maximum(p, 0.0)
            s = p.sum()
            if s > 0:
                p /= s
        F[t] = float(p[mass_mask].sum())

    # monotone-enforce (strictly observational cleanup)
    F = np.maximum.accumulate(F)
    F = np.clip(F, 0.0, 1.0)
    return F

def first_t_ge(F, q):
    idx = np.where(F >= q)[0]
    return int(idx[0]) if idx.size else None

def main():
    ap = argparse.ArgumentParser(description="STRICT PP deflection v3 via time-quantiles of mass-capture CDF.")
    ap.add_argument("--edges_flat", required=True)
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--mass_mask", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--src_row", type=int, required=True)
    ap.add_argument("--src_col", type=int, required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", default="auto")  # kept for interface parity; not used
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    H, W = args.H, args.W
    N = H * W
    assert N == 1600 or N > 0

    mass = np.load(args.mass_mask).astype(bool).ravel()
    if mass.size != N:
        raise ValueError(f"mass_mask size {mass.size} != N {N}")

    src_id = args.src_row * W + args.src_col
    if not (0 <= src_id < N):
        raise ValueError("Invalid src_row/src_col")

    edges_f = load_edges(args.edges_flat, N)
    edges_c = load_edges(args.edges_curved, N)

    P_f = build_transition_dense(edges_f, N)
    P_c = build_transition_dense(edges_c, N)

    # simulate cumulative mass probability vs time
    F_f = simulate_mass_cdf(P_f, src_id, mass, args.steps)
    F_c = simulate_mass_cdf(P_c, src_id, mass, args.steps)

    # quantiles
    qs = [0.10, 0.50, 0.90]
    t_f = {f"t{int(q*100)}_flat": first_t_ge(F_f, q) for q in qs}
    t_c = {f"t{int(q*100)}_curved": first_t_ge(F_c, q) for q in qs}

    # PASS rule: curved reaches median mass probability earlier
    t50_f = t_f["t50_flat"]
    t50_c = t_c["t50_curved"]

    PASS = None
    if t50_f is not None and t50_c is not None:
        PASS = bool(t50_c < t50_f)

    out = {
        "H": H, "W": W, "N": N,
        "device": args.device,
        "edges_flat": args.edges_flat,
        "edges_curved": args.edges_curved,
        "mass_mask": args.mass_mask,
        "src_row_label": args.src_row,
        "src_col_label": args.src_col,
        "src_id": src_id,
        "steps": args.steps,
        "mass_capture_cdf": {
            # keep small to avoid huge JSONs
            "F_flat_first20": F_f[:20].tolist(),
            "F_curved_first20": F_c[:20].tolist(),
            "F_flat_last": float(F_f[-1]),
            "F_curved_last": float(F_c[-1]),
        },
        **t_f, **t_c,
        "PASS_deflection_markov_front_PP": PASS,
        "notes": (
            "STRICT PP deflection v3. Observable = time-to-mass-capture quantiles "
            "from finite-time Markov fronts. Mass core is trace-derived. "
            "Kernel is row-stochastic from directed edges; zero-outdegree nodes "
            "are absorbing. No PDE, no Laplacian/Poisson, no GR ansatz, no regression. "
            "PASS if t50_curved < t50_flat."
        )
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("PASS_deflection_markov_front_PP =", PASS)

if __name__ == "__main__":
    main()
