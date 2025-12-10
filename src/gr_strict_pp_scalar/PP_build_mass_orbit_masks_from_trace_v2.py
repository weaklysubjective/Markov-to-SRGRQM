#!/usr/bin/env python3
"""
PP_build_mass_orbit_masks_from_trace_v2.py

STRICT PP mask builder for Scalar 512 PPV1 inputs.

Generates:
  1) Mass core mask from trace weights (top-k).
  2) Orbit ring mask from flat PPV1 graph hop-shells around that mass core.

STRICT PP compliance:
  - Mass derives ONLY from trace weights.
  - Ring derives ONLY from graph connectivity (flat PPV1 edges) and
    the trace-derived mass mask.
  - No Euclidean distance, no PDE, no Poisson/Laplacian injection.

Why hop-shells?
  - This provides a reproducible, sparse, coordinate-free ring selection
    that is stable at 512 scale.
  - You can tune ring_min_hops/ring_max_hops to match prior ring sizes
    if you want continuity.

Defaults are geared toward your 512 runner usage patterns.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


# -----------------------------
# Helpers
# -----------------------------

def die(msg: str):
    print("[ERROR]", msg, file=sys.stderr)
    raise SystemExit(1)


def read_edges(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        die(f"edges file not found: {path}")

    rows = []
    cols = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            s = int(parts[0])
            t = int(parts[1])
            rows.append(s)
            cols.append(t)

    if not rows:
        die(f"no edges parsed from: {path}")

    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def build_undirected_adj_list(
    rows: np.ndarray,
    cols: np.ndarray,
    N: int,
) -> List[List[int]]:
    """
    Build an undirected adjacency list from directed edge pairs.
    This is PP-friendly: we're using the flat Markov support only,
    not any coordinate metric.

    Memory:
      - Adjacency list is far cheaper than NxN matrices at 512.
    """
    adj = [[] for _ in range(N)]
    for s, t in zip(rows, cols):
        if 0 <= s < N and 0 <= t < N:
            adj[s].append(t)
            adj[t].append(s)
    return adj


def load_trace_weights(trace_path: str, N: int) -> np.ndarray:
    if not os.path.exists(trace_path):
        die(f"trace_weights file not found: {trace_path}")

    # Expect either:
    #  - one weight per line
    #  - or "index weight" pairs
    weights = []

    with open(trace_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                weights.append(float(parts[0]))
            elif len(parts) >= 2:
                # ignore explicit index if present
                weights.append(float(parts[1]))
            else:
                continue

    w = np.asarray(weights, dtype=np.float64)

    if w.size != N:
        die(f"trace_weights length {w.size} != N={N}. "
            f"Provide a properly sized trace_weights file.")

    if not np.isfinite(w).all():
        die("trace_weights contains non-finite values.")

    return w


def build_mass_mask_from_topk(trace_w: np.ndarray, topk: int) -> np.ndarray:
    N = trace_w.size
    if topk <= 0 or topk > N:
        die(f"mass_topk must be in [1, N]. Got {topk} with N={N}")

    # Strict PP: mass = highest trace intensity
    idx = np.argpartition(-trace_w, topk - 1)[:topk]
    mask = np.zeros((N,), dtype=np.uint8)
    mask[idx] = 1
    return mask


def bfs_multi_source_hops(
    adj: List[List[int]],
    sources: np.ndarray,
) -> np.ndarray:
    """
    Compute hop distance to the nearest source in an undirected graph,
    using BFS.

    Returns:
      dist[N] with -1 for unreachable.
    """
    N = len(adj)
    dist = np.full((N,), -1, dtype=np.int32)

    queue = []
    for s in sources.tolist():
        if 0 <= s < N:
            dist[s] = 0
            queue.append(s)

    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        du = dist[u]
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                queue.append(v)

    return dist


def build_ring_mask_from_hops(
    dist: np.ndarray,
    ring_min_hops: int,
    ring_max_hops: int,
) -> np.ndarray:
    if ring_min_hops < 1:
        die("ring_min_hops must be >= 1")
    if ring_max_hops < ring_min_hops:
        die("ring_max_hops must be >= ring_min_hops")

    mask = np.zeros((dist.size,), dtype=np.uint8)
    ok = (dist >= ring_min_hops) & (dist <= ring_max_hops)
    mask[ok] = 1
    return mask


# -----------------------------
# CLI
# -----------------------------

def default_paths(case: str, H: int, W: int) -> Dict[str, str]:
    hw = f"{H}x{W}"
    return {
        "edges_flat": f"edges_ca_v3_flat_{hw}_PPV1.txt",
        "trace_weights": f"trace_weights_ca_v3_{case}_{hw}.txt",
        "mass_mask": f"PP_mass_mask_{hw}_{case}_PPV1.npy",
        "ring_mask": f"PP_orbit_ring_mask_{hw}_{case}_PPV1.npy",
    }


def parse_args():
    ap = argparse.ArgumentParser(
        description="STRICT PP mass + orbit ring mask builder from trace + flat PPV1 edges."
    )
    ap.add_argument("--case", required=True, help="e.g., mass_ms080 or strong_pf010")
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)

    ap.add_argument("--edges_flat", type=str, default="")
    ap.add_argument("--trace_weights", type=str, default="")

    ap.add_argument("--mass_topk", type=int, default=500)

    # PP-friendly ring definition: hop shells
    ap.add_argument("--ring_min_hops", type=int, default=2)
    ap.add_argument("--ring_max_hops", type=int, default=6)

    ap.add_argument("--out_mass_mask", type=str, default="")
    ap.add_argument("--out_ring_mask", type=str, default="")

    return ap.parse_args()


def main():
    args = parse_args()
    H, W = int(args.H), int(args.W)
    if H <= 0 or W <= 0:
        die("H and W must be positive.")
    N = H * W

    paths = default_paths(args.case, H, W)

    edges_flat = args.edges_flat or paths["edges_flat"]
    trace_path = args.trace_weights or paths["trace_weights"]

    out_mass = args.out_mass_mask or paths["mass_mask"]
    out_ring = args.out_ring_mask or paths["ring_mask"]

    # Load trace weights
    trace_w = load_trace_weights(trace_path, N)

    # Build mass mask
    mass_mask = build_mass_mask_from_topk(trace_w, args.mass_topk)

    # Build hop-based ring on flat graph
    rows, cols = read_edges(edges_flat)

    # Sanity-check edge indices range
    if rows.min() < 0 or cols.min() < 0 or rows.max() >= N or cols.max() >= N:
        die("edges_flat contains out-of-range indices for given H,W.")

    adj = build_undirected_adj_list(rows, cols, N)

    mass_idx = np.nonzero(mass_mask > 0)[0].astype(np.int64)
    if mass_idx.size == 0:
        die("mass_mask ended up empty unexpectedly.")

    dist = bfs_multi_source_hops(adj, mass_idx)

    ring_mask = build_ring_mask_from_hops(dist, args.ring_min_hops, args.ring_max_hops)

    # Save outputs
    np.save(out_mass, mass_mask.reshape(H, W))
    np.save(out_ring, ring_mask.reshape(H, W))

    ring_size = int((ring_mask > 0).sum())
    mass_size = int((mass_mask > 0).sum())
    unreachable = int((dist < 0).sum())

    print("[OK] Wrote:")
    print("  mass_mask:", out_mass, f"(mass_count={mass_size})")
    print("  ring_mask:", out_ring, f"(ring_count={ring_size})")
    print("[META]")
    print("  case:", args.case)
    print("  H,W,N:", H, W, N)
    print("  edges_flat:", edges_flat)
    print("  trace_weights:", trace_path)
    print("  mass_topk:", args.mass_topk)
    print("  ring_hops:", f"{args.ring_min_hops}..{args.ring_max_hops}")
    print("  unreachable_nodes_in_flat_graph:", unreachable)


if __name__ == "__main__":
    main()

