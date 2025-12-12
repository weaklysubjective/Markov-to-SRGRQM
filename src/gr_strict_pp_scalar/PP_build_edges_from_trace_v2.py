#!/usr/bin/env python3
"""
PP_build_edges_from_trace_v2.py

STRICT PP canonical edge builder from trace weights (v2).

Motivation (operational)
------------------------
v1 'gradient_topk' / 'uphill_topk' can be extremely sink-heavy around sharply-peaked
trace weights. That is fine for attractive cores (τ/Shapiro), but may prevent
Markov fronts from producing a grazing flux band outside the mass.

v2 introduces a single, FIXED, globally reusable rule to preserve an attractive core
while increasing outer-shell connectivity—still STRICT PP and deterministic.

STRICT PP guarantees
--------------------
- Uses ONLY trace-derived weights + local neighborhood adjacency (4/8-neighborhood).
- No PDE, no Laplacian/Poisson, no GR ansatz, no regression.
- Grid geometry is NOT used as metric; only adjacency indices and optional LABELS.
- Deterministic output (stable tie-breaking by index).

Default rule (v2)
-----------------
"core_uphill_else_sym":
- Define a "core" as the top core_topk nodes by weight (or empty if weights ~ uniform).
- For u in core: build edges using 'uphill_topk' (sink-heavy, attractive).
- For u outside core: build edges using 'sym_topk' (more connectivity).
- Ensures at least min_out outgoing edges for non-core nodes via deterministic fallback.

This is a single fixed policy; parameters are explicit CLI args.
"""

import argparse
import json
import os
import numpy as np


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
# Trace weight parsing (same intent as v1)
# ----------------------------

def read_trace_weights(path: str, N: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"trace_weights not found: {path}")

    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace(",", " ")
            rows.append(s.split())

    if not rows:
        raise ValueError("trace_weights file empty after filtering")

    # Case 1: single-column floats
    if all(len(r) == 1 for r in rows):
        vals = [float(r[0]) for r in rows]
        w = np.asarray(vals, dtype=np.float64)
        if w.size != N:
            raise ValueError(f"weights length mismatch: got {w.size}, expected {N}")
        return w

    # Otherwise fill by idx if possible; row/col fill later
    w = np.full(N, np.nan, dtype=np.float64)

    for r in rows:
        if len(r) == 2:
            try:
                idx = int(r[0]); val = float(r[1])
            except:
                continue
            if 0 <= idx < N:
                w[idx] = val
            continue

        if len(r) >= 3:
            try:
                val = float(r[-1])
            except:
                continue

            # common: row col idx weight
            if len(r) >= 4:
                try:
                    idx = int(r[2])
                except:
                    idx = None
                if idx is not None and 0 <= idx < N:
                    w[idx] = val
                    continue

    return w


def fill_weights_from_rowcol(path: str, H: int, W: int, w: np.ndarray) -> np.ndarray:
    N = H * W
    if w.size != N:
        raise ValueError("weight array size mismatch in row/col fill")

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
                r = int(parts[0])
                c = int(parts[1])
            except:
                continue
            if 0 <= r < H and 0 <= c < W:
                idx = r * W + c
                if not np.isfinite(w[idx]):
                    w[idx] = val

    return w


# ----------------------------
# Edge building primitives
# ----------------------------

def build_edges_uphill_topk(w: np.ndarray, H: int, W: int, k_out: int, neighbors_mode: str):
    """
    For each u, consider neighbors v with w[v] > w[u]. Add u->topk(v) by w[v].
    Stable tie-break: by (-w[v], v).
    """
    N = H * W
    neigh = get_neighbors_fn(neighbors_mode)
    edges = []

    for u in range(N):
        wu = w[u]
        cand = []
        for v in neigh(u, H, W):
            if w[v] > wu:
                cand.append(v)
        if not cand:
            continue
        # stable sort: higher weight first, then smaller index
        cand.sort(key=lambda v: (-w[v], v))
        for v in cand[:min(k_out, len(cand))]:
            edges.append((u, v))
    return edges


def build_edges_sym_topk(w: np.ndarray, H: int, W: int, k_out: int, neighbors_mode: str):
    """
    Symmetric-ish: rank neighbors by score=max(w[u], w[v]); add top-k.
    Stable tie-break by (-score, v).
    """
    N = H * W
    neigh = get_neighbors_fn(neighbors_mode)
    edges = []

    for u in range(N):
        cand = []
        wu = w[u]
        for v in neigh(u, H, W):
            score = wu if wu >= w[v] else w[v]
            cand.append((score, v))
        if not cand:
            continue
        cand.sort(key=lambda x: (-x[0], x[1]))
        for i in range(min(k_out, len(cand))):
            edges.append((u, cand[i][1]))
    return edges


def ensure_min_out(edges, H, W, min_out, neighbors_mode, w):
    """
    For nodes with outdeg < min_out, add deterministic fallback edges to best neighbors.
    (Strict PP: still local adjacency + trace weights only.)
    """
    N = H * W
    neigh = get_neighbors_fn(neighbors_mode)
    outdeg = np.zeros(N, dtype=np.int32)
    adjset = set(edges)
    for a, b in edges:
        outdeg[a] += 1

    for u in range(N):
        need = int(min_out - outdeg[u])
        if need <= 0:
            continue
        cand = list(neigh(u, H, W))
        # choose highest w[v], tie by v
        cand.sort(key=lambda v: (-w[v], v))
        for v in cand:
            if need <= 0:
                break
            e = (u, v)
            if e in adjset:
                continue
            adjset.add(e)
            edges.append(e)
            outdeg[u] += 1
            need -= 1

    return edges


# ----------------------------
# v2 rule: core_uphill_else_sym
# ----------------------------

def core_mask_topk(w: np.ndarray, core_topk: int, uniform_std_eps: float):
    """
    Deterministic core selection:
    - If weights are ~uniform (std < eps), return empty core (avoid arbitrary picks).
    - Else choose top core_topk indices by (w desc, idx asc).
    """
    if float(np.std(w)) < float(uniform_std_eps):
        return np.zeros(w.shape[0], dtype=bool)

    N = w.shape[0]
    k = int(max(0, min(core_topk, N)))
    if k == 0:
        return np.zeros(N, dtype=bool)

    # stable ranking: primary -w, secondary idx
    idx = np.arange(N, dtype=np.int64)
    order = np.lexsort((idx, -w))  # sorts by (-w, idx)
    top = order[:k]
    m = np.zeros(N, dtype=bool)
    m[top] = True
    return m


def build_edges_core_uphill_else_sym(
    w: np.ndarray,
    H: int,
    W: int,
    core_topk: int,
    core_k_out: int,
    shell_k_out: int,
    core_neighbors: str,
    shell_neighbors: str,
    shell_min_out: int,
    uniform_std_eps: float
):
    N = H * W
    core = core_mask_topk(w, core_topk=core_topk, uniform_std_eps=uniform_std_eps)

    edges = []
    # build core edges (sink-heavy)
    if core.any():
        edges_core = build_edges_uphill_topk(w, H, W, core_k_out, core_neighbors)
        # keep only edges from core nodes
        edges.extend([(u, v) for (u, v) in edges_core if core[u]])

    # build shell edges (connectivity)
    edges_shell = build_edges_sym_topk(w, H, W, shell_k_out, shell_neighbors)
    edges.extend([(u, v) for (u, v) in edges_shell if not core[u]])

    # ensure minimal outdegree on shell (do NOT force core)
    edges = sorted(set(edges))
    edges = ensure_min_out(edges, H, W, min_out=shell_min_out, neighbors_mode=shell_neighbors, w=w)
    edges = sorted(set(edges))

    # stats
    outdeg = np.zeros(N, dtype=np.int32)
    indeg = np.zeros(N, dtype=np.int32)
    for a, b in edges:
        outdeg[a] += 1
        indeg[b] += 1

    stats = {
        "core_topk": int(core_topk),
        "core_frac": float(core.mean()),
        "core_k_out": int(core_k_out),
        "shell_k_out": int(shell_k_out),
        "core_neighbors": core_neighbors,
        "shell_neighbors": shell_neighbors,
        "shell_min_out": int(shell_min_out),
        "uniform_std_eps": float(uniform_std_eps),

        "n_edges": int(len(edges)),
        "n_sinks_outdeg0": int(np.sum(outdeg == 0)),
        "sink_frac_outdeg0": float(np.sum(outdeg == 0)) / float(N),
        "outdeg_min": int(outdeg.min()),
        "outdeg_max": int(outdeg.max()),
        "outdeg_mean": float(outdeg.mean()),
        "indeg_min": int(indeg.min()),
        "indeg_max": int(indeg.max()),
        "indeg_mean": float(indeg.mean()),
    }

    return edges, stats


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="STRICT PP build edges from trace weights (canonical v2).")
    ap.add_argument("--trace_weights", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--case", required=True)

    ap.add_argument(
        "--rule",
        choices=["core_uphill_else_sym"],
        default="core_uphill_else_sym"
    )

    ap.add_argument("--edges_out", required=True)
    ap.add_argument("--report_out", required=True)

    # v2 params (explicit, fixed per run)
    ap.add_argument("--core_topk", type=int, default=500,
                    help="Top-K weight nodes treated as 'core' (sink-heavy).")
    ap.add_argument("--core_k_out", type=int, default=2)
    ap.add_argument("--shell_k_out", type=int, default=4)
    ap.add_argument("--core_neighbors", choices=["4", "8"], default="4")
    ap.add_argument("--shell_neighbors", choices=["4", "8"], default="8")
    ap.add_argument("--shell_min_out", type=int, default=1,
                    help="Ensure at least this outdegree for non-core nodes (local fallback).")
    ap.add_argument("--uniform_std_eps", type=float, default=1e-12,
                    help="If std(weights)<eps treat as uniform (empty core).")

    ap.add_argument("--normalize", action="store_true",
                    help="Normalize weights to sum=1 for report only (edges use raw ordering).")
    args = ap.parse_args()

    H, W = args.H, args.W
    N = H * W

    # ---- Load weights ----
    w = read_trace_weights(args.trace_weights, N)
    if not np.isfinite(w).all():
        w = fill_weights_from_rowcol(args.trace_weights, H, W, w)
    if not np.isfinite(w).all():
        missing = int(np.sum(~np.isfinite(w)))
        raise ValueError(f"trace_weights parsing incomplete: {missing} NaNs remain")
    if w.shape != (N,):
        raise ValueError("weights shape mismatch")

    # ---- Build edges ----
    if args.rule == "core_uphill_else_sym":
        edges, rule_stats = build_edges_core_uphill_else_sym(
            w=w,
            H=H,
            W=W,
            core_topk=args.core_topk,
            core_k_out=args.core_k_out,
            shell_k_out=args.shell_k_out,
            core_neighbors=args.core_neighbors,
            shell_neighbors=args.shell_neighbors,
            shell_min_out=args.shell_min_out,
            uniform_std_eps=args.uniform_std_eps,
        )
    else:
        raise ValueError("unknown rule")

    # ---- Report ----
    w_rep = w.copy()
    if args.normalize:
        s = float(w_rep.sum())
        if s > 0:
            w_rep = w_rep / s

    report = {
        "case": args.case,
        "H": H, "W": W, "N": N,
        "trace_weights": args.trace_weights,
        "rule": args.rule,
        "edges_out": args.edges_out,
        "report_out": args.report_out,

        "weight_min": float(w_rep.min()),
        "weight_max": float(w_rep.max()),
        "weight_mean": float(w_rep.mean()),
        "weight_std": float(w_rep.std()),

        "rule_stats": rule_stats,
        "notes": (
            "STRICT PP edges-from-trace v2. Core uses uphill_topk (attractive/sink), "
            "shell uses sym_topk + min-out fallback (connectivity). "
            "No PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        ),
    }

    os.makedirs(os.path.dirname(args.edges_out) or ".", exist_ok=True)
    with open(args.edges_out, "w") as f:
        f.write("# STRICT PP edges from trace v2\n")
        f.write(f"# case={args.case} H={H} W={W} rule={args.rule}\n")
        f.write(f"# core_topk={args.core_topk} core_k_out={args.core_k_out} "
                f"shell_k_out={args.shell_k_out} core_neighbors={args.core_neighbors} "
                f"shell_neighbors={args.shell_neighbors} shell_min_out={args.shell_min_out}\n")
        for a, b in edges:
            f.write(f"{a} {b}\n")

    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("WROTE", args.edges_out)
    print("WROTE", args.report_out)
    print("n_edges =", report["rule_stats"]["n_edges"])
    print("sink_frac_outdeg0 =", report["rule_stats"]["sink_frac_outdeg0"])
    print("core_frac =", report["rule_stats"]["core_frac"])


if __name__ == "__main__":
    main()

