#!/usr/bin/env python3
"""
PP_build_edges_from_trace_v1.py

STRICT PP canonical edge builder from trace weights.

Goal
----
Replace ad-hoc "PY blob" edge artifacts with a reproducible, CLI-based generator
so we can:

1) Reproduce 40x40 edges from the same rule,
2) Diff vs legacy artifacts,
3) Scale to 512+ with provenance.

STRICT PP guarantees
--------------------
- Uses ONLY trace-derived weights + local neighborhood connectivity.
- No PDE, no Laplacian/Poisson, no GR ansatz, no regression.
- Grid angles/geometry are NOT used as metrics (only implicit adjacency by
  neighborhood indices).
- Deterministic output.

Default rule (v1)
-----------------
"gradient_topk":
For each node u, look at its local neighbors. Add directed edges
u -> v for the top-k neighbors by weight[v]. This naturally produces
sink-heavy cores when weights are sharply peaked (consistent with pf010 behavior).

This is an operational, trace-to-kernel rule that matches the spirit of the
one-off blobs we likely used.

You can change the rule explicitly via --rule, but DO NOT silently change defaults
without freezing a new version tag.
"""

import argparse
import json
import math
import os
from collections import deque, defaultdict

import numpy as np


# ----------------------------
# Parsing trace weights
# ----------------------------
def build_edges_uphill_topk(w: np.ndarray, H: int, W: int, k_out: int, neighbors_mode: str):
    """
    STRICT PP likely legacy rule:
    For each node u, consider local neighbors with w[v] > w[u].
    Add edges u -> top-k of those by w[v].
    If no uphill neighbor exists, u has no outgoing edge.
    This produces sink-heavy sparse graphs.
    """
    N = H * W
    neigh = get_neighbors_fn(neighbors_mode)

    edges = []
    for u in range(N):
        wu = w[u]
        cand = []
        for v in neigh(u, H, W):
            if w[v] > wu:
                cand.append((w[v], v))
        if not cand:
            continue
        cand.sort(reverse=True, key=lambda x: x[0])
        for i in range(min(k_out, len(cand))):
            edges.append((u, cand[i][1]))
    return edges


def read_trace_weights(path: str, N: int) -> np.ndarray:
    """
    Attempts to parse trace weights from common ad-hoc formats:
    1) One float per line (length N).
    2) Two columns: idx weight.
    3) Three columns: row col weight.
    4) Four columns: row col idx weight (we ignore redundant idx).

    Returns float64 weights of shape (N,).

    STRICT ASSERTS:
    - Must produce exactly N weights after parsing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"trace_weights not found: {path}")

    # Load raw lines
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace(",", " ")
            parts = s.split()
            rows.append(parts)

    if not rows:
        raise ValueError("trace_weights file appears empty after comment/blank filtering")

    # Case 1: single-column floats
    if all(len(r) == 1 for r in rows):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[0]))
            except:
                raise ValueError("Failed to parse single-column trace weights as floats")
        w = np.asarray(vals, dtype=np.float64)
        assert w.size == N, f"weights length mismatch: got {w.size}, expected {N}"
        return w

    # Case 2/3/4: structured formats
    # We'll attempt to infer by column count and content.
    # Build an array filled with NaN, then fill by index.
    w = np.full(N, np.nan, dtype=np.float64)

    # Heuristic: if first col looks like integer index in [0, N)
    # and there are exactly 2 cols => (idx, weight)
    # else if >=3 cols => (row, col, weight) or variants
    for r in rows:
        if len(r) < 2:
            continue

        # Try (idx, weight)
        if len(r) == 2:
            try:
                idx = int(r[0])
                val = float(r[1])
            except:
                continue
            if 0 <= idx < N:
                w[idx] = val
            continue

        # Try (row, col, weight)
        if len(r) >= 3:
            # Prefer last token as weight
            try:
                val = float(r[-1])
            except:
                continue

            # Try to interpret first two as row/col
            try:
                a = int(r[0]); b = int(r[1])
            except:
                continue

            # If a,b look like row/col within plausible ranges,
            # we can't know H/W here, so we accept them later only
            # if they map inside N when flattened by inferred W.
            # Because we don't have W here, we also try direct idx in r[2] if present.
            # Safer approach: if r has an explicit idx token in middle, use it.
            # Common pattern in quick blobs: row col idx weight.
            if len(r) >= 4:
                # Try third token as idx
                try:
                    idx = int(r[2])
                except:
                    idx = None
                if idx is not None and 0 <= idx < N:
                    w[idx] = val
                    continue

            # Fallback: treat a as idx if it fits.
            if 0 <= a < N and (len(r) == 3):
                # Ambiguous (idx, ?, weight) -> ignore
                # We won't guess here.
                continue

            # Otherwise we leave row/col mapping to a later pass.
            # We'll store these triples to remap when H/W is known.
            # For simplicity, we stash them in a global collector.
            pass

    # If we already filled most entries, accept if complete.
    if np.isfinite(w).all():
        return w

    # Second pass: attempt (row, col, weight) remap using H/W supplied later.
    # We can't do that here without H/W, so we will handle this in main()
    # with a separate parser function.
    return w  # partially filled; main() will attempt row/col remap if needed


def fill_weights_from_rowcol(path: str, H: int, W: int, w: np.ndarray) -> np.ndarray:
    """
    If w has NaNs, attempt to fill them using (row, col, weight) lines.
    """
    N = H * W
    if w.size != N:
        raise ValueError("weight array size mismatch during row/col fill")

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace(",", " ")
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                val = float(parts[-1])
            except:
                continue
            try:
                r = int(parts[0]); c = int(parts[1])
            except:
                continue
            if 0 <= r < H and 0 <= c < W:
                idx = r * W + c
                if not np.isfinite(w[idx]):
                    w[idx] = val

    return w


# ----------------------------
# Neighborhood utilities
# ----------------------------

def neighbors_4(idx: int, H: int, W: int):
    r = idx // W
    c = idx % W
    if r > 0:
        yield idx - W
    if r < H - 1:
        yield idx + W
    if c > 0:
        yield idx - 1
    if c < W - 1:
        yield idx + 1


def neighbors_8(idx: int, H: int, W: int):
    r = idx // W
    c = idx % W
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr * W + cc


def get_neighbors_fn(mode: str):
    if mode == "4":
        return neighbors_4
    if mode == "8":
        return neighbors_8
    raise ValueError("neighbors must be '4' or '8'")


# ----------------------------
# Edge building rules
# ----------------------------

def build_edges_gradient_topk(w: np.ndarray, H: int, W: int, k_out: int, neighbors_mode: str):
    """
    For each node u, add edges to the top-k neighbors by weight.
    """
    N = H * W
    neigh = get_neighbors_fn(neighbors_mode)

    edges = []
    for u in range(N):
        cand = []
        for v in neigh(u, H, W):
            cand.append((w[v], v))
        if not cand:
            continue
        cand.sort(reverse=True, key=lambda x: x[0])
        for i in range(min(k_out, len(cand))):
            edges.append((u, cand[i][1]))
    return edges


def build_edges_threshold(w: np.ndarray, H: int, W: int, eps: float, neighbors_mode: str):
    """
    Add edge u->v if w[v] >= w[u] + eps.
    """
    N = H * W
    neigh = get_neighbors_fn(neighbors_mode)

    edges = []
    for u in range(N):
        wu = w[u]
        for v in neigh(u, H, W):
            if w[v] >= wu + eps:
                edges.append((u, v))
    return edges


def build_edges_sym_topk(w: np.ndarray, H: int, W: int, k_out: int, neighbors_mode: str):
    """
    Symmetric variant: add top-k by max(w[u], w[v]) ranking from u's neighborhood.
    Less sink-heavy, more connectivity.
    """
    N = H * W
    neigh = get_neighbors_fn(neighbors_mode)

    edges = []
    for u in range(N):
        cand = []
        for v in neigh(u, H, W):
            score = max(w[u], w[v])
            cand.append((score, v))
        if not cand:
            continue
        cand.sort(reverse=True, key=lambda x: x[0])
        for i in range(min(k_out, len(cand))):
            edges.append((u, cand[i][1]))
    return edges


# ----------------------------
# Graph stats / SCC (Tarjan)
# ----------------------------

def build_adj(edges, N):
    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
    return adj


def tarjan_scc_sizes(adj):
    N = len(adj)
    idx = [-1] * N
    low = [0] * N
    onstack = [False] * N
    stack = []
    index = 0
    sizes = []

    def strongconnect(v):
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack[v] = True

        for w in adj[v]:
            if idx[w] == -1:
                strongconnect(w)
                low[v] = low[v] if low[v] < low[w] else low[w]
            elif onstack[w]:
                low[v] = low[v] if low[v] < idx[w] else idx[w]

        if low[v] == idx[v]:
            sz = 0
            while True:
                w = stack.pop()
                onstack[w] = False
                sz += 1
                if w == v:
                    break
            sizes.append(sz)

    for v in range(N):
        if idx[v] == -1:
            strongconnect(v)

    sizes.sort(reverse=True)
    return sizes


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP build edges from trace weights (canonical v1)."
    )
    ap.add_argument("--trace_weights", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--case", required=True)

    #ap.add_argument("--rule", choices=["gradient_topk", "threshold", "sym_topk"],
    #                default="gradient_topk")
    ap.add_argument("--rule", choices=["gradient_topk", "uphill_topk", "threshold", "sym_topk"],
                default="gradient_topk")

    ap.add_argument("--neighbors", choices=["4", "8"], default="4")
    ap.add_argument("--k_out", type=int, default=2,
                    help="Used by *_topk rules")
    ap.add_argument("--eps", type=float, default=0.0,
                    help="Used by threshold rule")

    ap.add_argument("--edges_out", required=True)
    ap.add_argument("--report_out", required=True)

    ap.add_argument("--normalize", action="store_true",
                    help="Normalize weights to sum=1 for report only (edges use raw ordering).")

    ap.add_argument("--scc_max_N", type=int, default=300000,
                    help="Skip full SCC analysis if N exceeds this.")
    args = ap.parse_args()

    H, W = args.H, args.W
    N = H * W

    # ---- Load weights ----
    w = read_trace_weights(args.trace_weights, N)

    # Attempt row/col fill if needed
    if not np.isfinite(w).all():
        w = fill_weights_from_rowcol(args.trace_weights, H, W, w)

    # Final validation
    if not np.isfinite(w).all():
        missing = int(np.sum(~np.isfinite(w)))
        raise ValueError(f"trace_weights parsing incomplete: {missing} entries still NaN")

    assert w.shape == (N,), "weights shape mismatch"

    # ---- Build edges ----
    if args.rule == "gradient_topk":
        edges = build_edges_gradient_topk(w, H, W, args.k_out, args.neighbors)
    elif args.rule == "uphill_topk":
        edges = build_edges_uphill_topk(w, H, W, args.k_out, args.neighbors)
    elif args.rule == "sym_topk":
        edges = build_edges_sym_topk(w, H, W, args.k_out, args.neighbors)
    elif args.rule == "threshold":
        edges = build_edges_threshold(w, H, W, args.eps, args.neighbors)
    else:
        raise ValueError("unknown rule")

    # Determinism
    edges = sorted(set(edges))

    # ---- Stats ----
    outdeg = np.zeros(N, dtype=np.int32)
    indeg = np.zeros(N, dtype=np.int32)
    for a, b in edges:
        outdeg[a] += 1
        indeg[b] += 1

    w_rep = w.copy()
    if args.normalize:
        s = float(w_rep.sum())
        if s > 0:
            w_rep = w_rep / s

    stats = {
        "case": args.case,
        "H": H, "W": W, "N": N,
        "trace_weights": args.trace_weights,
        "rule": args.rule,
        "neighbors": args.neighbors,
        "k_out": args.k_out if "topk" in args.rule else None,
        "eps": args.eps if args.rule == "threshold" else None,
        "n_edges": len(edges),

        "weight_min": float(w_rep.min()),
        "weight_max": float(w_rep.max()),
        "weight_mean": float(w_rep.mean()),
        "weight_std": float(w_rep.std()),

        "outdeg_min": int(outdeg.min()),
        "outdeg_max": int(outdeg.max()),
        "outdeg_mean": float(outdeg.mean()),
        "outdeg_std": float(outdeg.std()),
        "indeg_min": int(indeg.min()),
        "indeg_max": int(indeg.max()),
        "indeg_mean": float(indeg.mean()),
        "indeg_std": float(indeg.std()),

        "n_sinks_outdeg0": int(np.sum(outdeg == 0)),
        "sink_frac_outdeg0": float(np.sum(outdeg == 0)) / float(N),
        "n_sources_indeg0": int(np.sum(indeg == 0)),
        "source_frac_indeg0": float(np.sum(indeg == 0)) / float(N),

        "edges_out": args.edges_out,
        "report_out": args.report_out,
        "notes": (
            "STRICT PP edges-from-trace v1. Builds a directed local adjacency from "
            "trace-derived weights using an explicit rule. "
            "No PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        )
    }

    # SCC (optional)
    if N <= args.scc_max_N:
        adj = build_adj(edges, N)
        scc_sizes = tarjan_scc_sizes(adj)
        stats["scc"] = {
            "n_scc": int(len(scc_sizes)),
            "largest_scc_size": int(scc_sizes[0]) if scc_sizes else 0,
            "top5_scc_sizes": [int(x) for x in scc_sizes[:5]],
        }
    else:
        stats["scc"] = {
            "skipped": True,
            "reason": f"N={N} > scc_max_N={args.scc_max_N}"
        }

    # ---- Write edges ----
    os.makedirs(os.path.dirname(args.edges_out) or ".", exist_ok=True)
    with open(args.edges_out, "w") as f:
        f.write(f"# STRICT PP edges from trace v1\n")
        f.write(f"# case={args.case} H={H} W={W} rule={args.rule} neighbors={args.neighbors}\n")
        for a, b in edges:
            f.write(f"{a} {b}\n")

    # ---- Write report ----
    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print("WROTE", args.edges_out)
    print("WROTE", args.report_out)
    print("n_edges =", stats["n_edges"])
    print("sink_frac_outdeg0 =", stats["sink_frac_outdeg0"])
    scc = stats.get("scc", {})
    if scc and not scc.get("skipped", False):
        print("largest_scc_size =", scc.get("largest_scc_size"))


if __name__ == "__main__":
    main()

