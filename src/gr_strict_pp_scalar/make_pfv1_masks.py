#!/usr/bin/env python3
"""
STRICT PP helper: build TOPK mass mask + TO-mass orbit band mask
from canonical PPV1 edges and trace weights.

- No PDE
- No Laplacian/Poisson
- No GR ansatz
- No regression
- Distances are directed Markov/BFS hop counts on the adjacency only.
"""

import argparse
import json
import os
import sys
import numpy as np
import collections


# --------------------------
# Robust trace loader
# --------------------------
def load_trace_weights(path: str, N: int) -> np.ndarray:
    raw = np.loadtxt(path)
    raw = np.asarray(raw)

    # Case A: 1D length N
    if raw.ndim == 1 and raw.size == N:
        w = raw.astype(float)
        assert w.shape == (N,)
        return w

    # Case B: 2D with >=2 columns -> assume (id, weight) in first two
    if raw.ndim == 2 and raw.shape[1] >= 2:
        ids = raw[:, 0].astype(int, copy=False)
        wcol = raw[:, 1].astype(float, copy=False)

        # If ids look valid, map them
        if ids.size > 0 and ids.min() >= 0 and ids.max() < N:
            w = np.zeros(N, dtype=float)
            # allow repeated ids; last wins
            for i, val in zip(ids.tolist(), wcol.tolist()):
                w[i] = val
            assert w.shape == (N,)
            return w

        # fallback: if second column length matches N, accept direct
        if wcol.size == N:
            w = wcol.copy()
            assert w.shape == (N,)
            return w

    # Case C: 1D size 2N -> interpret as pairs
    if raw.ndim == 1 and raw.size == 2 * N:
        pairs = raw.reshape(N, 2)
        ids = pairs[:, 0].astype(int, copy=False)
        wcol = pairs[:, 1].astype(float, copy=False)

        if ids.size > 0 and ids.min() >= 0 and ids.max() < N:
            w = np.zeros(N, dtype=float)
            for i, val in zip(ids.tolist(), wcol.tolist()):
                w[i] = val
            assert w.shape == (N,)
            return w

        w = wcol.copy()
        assert w.shape == (N,)
        return w

    raise ValueError(
        f"Unrecognized trace_weights format: shape={raw.shape}, size={raw.size}. "
        "Expected 1-col weights, 2-col (id weight), or flat 2N pairs."
    )


# --------------------------
# Edges parsing
# --------------------------
def load_edges_adj(edges_path: str, N: int):
    adj = [[] for _ in range(N)]
    n_edges = 0

    with open(edges_path, "r") as f:
        for line in f:
            s = line.strip().replace(",", " ")
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) < 2:
                continue
            try:
                a = int(p[0]); b = int(p[1])
            except:
                continue
            if 0 <= a < N and 0 <= b < N:
                adj[a].append(b)
                n_edges += 1

    return adj, n_edges


def build_reverse_adj(adj, N: int):
    rev = [[] for _ in range(N)]
    for a in range(N):
        for b in adj[a]:
            rev[b].append(a)
    return rev


# --------------------------
# Directed TO-mass distances
# --------------------------
def directed_to_mass_dist(rev, mass_bool: np.ndarray):
    N = mass_bool.size
    dq = collections.deque()
    dist = np.full(N, -1, dtype=int)

    mass_ids = np.where(mass_bool)[0]
    assert mass_ids.size > 0, "empty mass set"

    for m in mass_ids:
        dist[m] = 0
        dq.append(m)

    while dq:
        u = dq.popleft()
        for v in rev[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                dq.append(v)

    return dist


def print_histogram(dist: np.ndarray, max_d: int = 30):
    vals = dist[dist >= 0]
    hist = collections.Counter(vals.tolist())
    print("reachable_to_mass:", int(vals.size))
    for d in sorted(hist):
        if d > max_d:
            break
        print(d, hist[d])
    return hist


# --------------------------
# Band picking
# --------------------------
def pick_auto_band(dist: np.ndarray, min_ring_nodes: int):
    # Choose the smallest contiguous band [lo, hi] with enough nodes.
    # Start from lo=1 upward.
    dvals = dist[dist >= 0]
    if dvals.size == 0:
        return None

    dmax = int(dvals.max())
    for lo in range(1, dmax + 1):
        # try small widths first
        for width in range(0, 4):
            hi = lo + width
            ring = (dist >= lo) & (dist <= hi)
            if int(ring.sum()) >= min_ring_nodes:
                return lo, hi
    return None


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP helper: build TOPK mass mask + TO-mass orbit band from PPV1 edges + trace."
    )
    ap.add_argument("--edges", required=True)
    ap.add_argument("--trace_weights", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)

    ap.add_argument("--mass_topk", type=int, default=80)
    ap.add_argument("--band_lo", type=int, default=None)
    ap.add_argument("--band_hi", type=int, default=None)
    ap.add_argument("--auto_band", action="store_true",
                    help="If set and band_lo/hi not provided, pick first nonempty band.")
    ap.add_argument("--min_ring_nodes", type=int, default=12)

    ap.add_argument("--mass_out", default=None)
    ap.add_argument("--orbit_out", default=None)
    ap.add_argument("--report_out", default=None)

    ap.add_argument("--hist_max_d", type=int, default=30)

    args = ap.parse_args()

    H, W = args.H, args.W
    N = H * W
    assert N > 0

    if not os.path.exists(args.edges):
        raise FileNotFoundError(f"Edges file not found: {args.edges}")
    if not os.path.exists(args.trace_weights):
        raise FileNotFoundError(f"Trace weights file not found: {args.trace_weights}")

    # Defaults
    case_hint = "case"
    if "ms080" in args.edges or "ms080" in args.trace_weights:
        case_hint = "ms080"
    elif "pf010" in args.edges or "pf010" in args.trace_weights:
        case_hint = "strong_pf010"

    mass_out = args.mass_out or f"PP_mass_mask_{H}x{W}_{case_hint}_PPV1.npy"
    orbit_out = args.orbit_out or f"PP_orbit_ring_mask_{H}x{W}_{case_hint}_PPV1.npy"

    # Load trace weights with robust format support
    w = load_trace_weights(args.trace_weights, N)
    assert w.shape == (N,), f"weights shape mismatch {w.shape} vs {(N,)}"
    assert np.all(np.isfinite(w)), "non-finite weights"

    # TOPK mass
    k = int(args.mass_topk)
    assert 1 <= k <= N
    mass_ids = np.argsort(w)[-k:]
    mass_bool = np.zeros(N, dtype=bool)
    mass_bool[mass_ids] = True
    assert int(mass_bool.sum()) == k

    # Save mass mask
    np.save(mass_out, mass_bool.reshape(H, W).astype(np.uint8))

    # Load edges
    adj, n_edges = load_edges_adj(args.edges, N)
    outdeg = np.array([len(x) for x in adj], dtype=int)
    sink_frac = float(np.mean(outdeg == 0))

    # Reverse adjacency
    rev = build_reverse_adj(adj, N)

    # Directed TO-mass distances
    dist = directed_to_mass_dist(rev, mass_bool)
    hist = print_histogram(dist, max_d=args.hist_max_d)

    # Determine band
    band_lo = args.band_lo
    band_hi = args.band_hi

    if band_lo is None or band_hi is None:
        if args.auto_band:
            picked = pick_auto_band(dist, args.min_ring_nodes)
            if picked is None:
                band_lo = band_hi = None
            else:
                band_lo, band_hi = picked
        else:
            band_lo = band_hi = None

    ring_bool = None
    ring_size = 0
    if band_lo is not None and band_hi is not None:
        assert band_lo >= 0 and band_hi >= band_lo
        ring_bool = (dist >= band_lo) & (dist <= band_hi)
        ring_size = int(ring_bool.sum())
        np.save(orbit_out, ring_bool.reshape(H, W).astype(np.uint8))
    else:
        # still write an empty orbit mask for explicitness
        ring_bool = np.zeros(N, dtype=bool)
        np.save(orbit_out, ring_bool.reshape(H, W).astype(np.uint8))

    # Optional report
    report = {
        "H": H, "W": W, "N": N,
        "edges": args.edges,
        "trace_weights": args.trace_weights,
        "mass_topk": k,
        "mass_out": mass_out,
        "orbit_out": orbit_out,
        "n_edges": int(n_edges),
        "sink_frac_outdeg0": sink_frac,
        "band_lo": band_lo,
        "band_hi": band_hi,
        "ring_size": int(ring_size),
        "reachable_to_mass": int(np.sum(dist >= 0)),
        "hist_preview": {str(d): int(hist[d]) for d in sorted(hist)[:min(len(hist), 20)]},
        "notes": (
            "STRICT PP masks-from-PPV1 v1. Mass core = TOPK(trace weights). "
            "Orbit band = directed TO-mass hop shell(s) on adjacency. "
            "No PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        )
    }

    if args.report_out:
        with open(args.report_out, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    # Console summary
    print("WROTE mass_mask:", mass_out, "sum", int(mass_bool.sum()))
    if band_lo is not None and band_hi is not None:
        print("WROTE orbit_mask:", orbit_out, "ring_size", ring_size, "band", band_lo, band_hi)
    else:
        print("WROTE orbit_mask (empty):", orbit_out, "no band specified; use --band_lo/--band_hi or --auto_band")

    if args.report_out:
        print("WROTE report:", args.report_out)


if __name__ == "__main__":
    main()

