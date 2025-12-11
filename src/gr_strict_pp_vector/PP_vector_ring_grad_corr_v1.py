#!/usr/bin/env python3
import argparse
import json
from typing import Dict, Any, Tuple, List

import numpy as np


def load_edges(path: str, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load directed edges from a text file.
    Each non-comment, non-blank line is expected to contain at least two
    integers: src dst [optional ...].
    Returns src, dst as int64 arrays of length M.
    """
    src: List[int] = []
    dst: List[int] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Bad edge line in {path!r}: {line!r}")
            try:
                i = int(parts[0])
                j = int(parts[1])
            except ValueError:
                raise ValueError(f"Non-integer edge endpoints in {path!r}: {line!r}")
            if i < 0 or i >= N or j < 0 or j >= N:
                raise ValueError(
                    f"Edge ({i},{j}) out of range for N={N} in {path!r}"
                )
            src.append(i)
            dst.append(j)
    if not src:
        raise ValueError(f"No edges parsed from {path!r}")
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    return src_arr, dst_arr


def load_mask_bool(path: str, N: int, name: str) -> np.ndarray:
    """
    Load a boolean mask from .npy and sanity-check shape and dtype.
    """
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.size != N:
        raise ValueError(
            f"{name} mask size {arr.size} != N={N}. "
            f"Got shape {arr.shape}, expected total {N} elements."
        )
    mask = arr.reshape(-1).astype(bool)
    return mask


def build_reverse_adj(src: np.ndarray, dst: np.ndarray, N: int) -> List[List[int]]:
    """
    Build reverse adjacency: for each node v, list of u with edges u->v.
    """
    rev: List[List[int]] = [[] for _ in range(N)]
    for u, v in zip(src, dst):
        rev[v].append(int(u))
    return rev


def bfs_dist_to_mass(rev_adj: List[List[int]], mass_mask: np.ndarray) -> np.ndarray:
    """
    Multi-source BFS on the reverse graph to compute hop distance TO mass.

    dist[i] = minimum number of (forward) hops from i to any mass node,
              or +inf if no path exists.

    We use reverse adjacency so that a single BFS from all mass nodes
    covers the entire graph.
    """
    N = len(rev_adj)
    dist = np.full(N, np.inf, dtype=np.float64)

    mass_idx = np.nonzero(mass_mask)[0]
    if mass_idx.size == 0:
        raise ValueError("mass_mask has no True entries.")
    # Initialize queue with all mass nodes at distance 0
    from collections import deque
    q = deque()
    for m in mass_idx:
        dist[m] = 0.0
        q.append(m)

    while q:
        v = q.popleft()
        dv = dist[v]
        for u in rev_adj[v]:
            if not np.isfinite(dist[u]):
                dist[u] = dv + 1.0
                q.append(u)

    return dist


def ring_order_by_angle(
    ring_mask: np.ndarray, mass_mask: np.ndarray, H: int, W: int
) -> np.ndarray:
    """
    Order ring nodes by angular label around the mass centroid.

    This uses grid coordinates ONLY as labels. No Euclidean physics is
    introduced; angles are a way to index ring nodes around mass.
    """
    N = H * W
    assert ring_mask.size == N
    assert mass_mask.size == N

    ring_idx = np.nonzero(ring_mask)[0]
    if ring_idx.size == 0:
        raise ValueError("orbit/ring mask has no True entries.")

    # Mass centroid (labels only)
    rows = np.arange(N) // W
    cols = np.arange(N) % W
    mass_idx = np.nonzero(mass_mask)[0]
    if mass_idx.size == 0:
        raise ValueError("mass_mask has no True entries.")
    mass_r = rows[mass_idx].mean()
    mass_c = cols[mass_idx].mean()

    r_ring = rows[ring_idx]
    c_ring = cols[ring_idx]
    # Angle label (no metric claim)
    angles = np.arctan2(r_ring - mass_r, c_ring - mass_c)

    order = np.argsort(angles)
    ring_ordered = ring_idx[order]

    return ring_ordered


def build_ring_index(ring_ordered: np.ndarray, N: int) -> np.ndarray:
    """
    Build a lookup: ring_index[i] = position of node i in the ring ordering,
    or -1 if not in the ring.
    """
    ring_index = np.full(N, -1, dtype=np.int64)
    for pos, node in enumerate(ring_ordered):
        ring_index[int(node)] = pos
    return ring_index


def signed_ring_direction(idx_u: int, idx_v: int, n: int) -> int:
    """
    Given positions idx_u and idx_v on a ring of length n, return +1 for cw,
    -1 for ccw, 0 for 'no azimuthal preference' when they coincide.

    We define cw/ccw purely in terms of the ordering indices.
    """
    if n <= 0:
        raise ValueError("Ring length must be positive.")
    diff = (idx_v - idx_u) % n
    if diff == 0:
        return 0
    # Map diff to (-n/2, n/2]
    if diff > n // 2:
        diff = diff - n
    # Now diff>0 means cw, diff<0 means ccw
    return 1 if diff > 0 else -1


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "STRICT PP vector ring gradient correlation v1. "
            "Measures whether Markov edges that move inward (toward mass, "
            "in hop-distance) on the orbit ring prefer one azimuthal direction "
            "(cw vs ccw) around mass."
        )
    )
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--mass_mask", required=True)
    ap.add_argument("--orbit_mask", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--min_edges_used", type=int, default=50)
    ap.add_argument("--bias_threshold", type=float, default=0.01)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    H = args.H
    W = args.W
    N = H * W
    if N <= 0:
        raise ValueError(f"Non-positive grid size H={H}, W={W}")

    # --- Load inputs ---
    src, dst = load_edges(args.edges_curved, N)
    mass_mask = load_mask_bool(args.mass_mask, N, "mass_mask")
    orbit_mask = load_mask_bool(args.orbit_mask, N, "orbit_mask")

    if not mass_mask.any():
        raise ValueError("mass_mask has no True entries.")
    if not orbit_mask.any():
        raise ValueError("orbit_mask has no True entries.")

    # We consider ring nodes as orbit minus mass
    ring_mask = orbit_mask & (~mass_mask)
    if not ring_mask.any():
        raise ValueError("ring_mask (orbit & ~mass) has no True entries.")

    # --- Build Markov distance TO mass ---
    rev_adj = build_reverse_adj(src, dst, N)
    dist_to_mass = bfs_dist_to_mass(rev_adj, mass_mask)

    # Exclude unreachable nodes from ring
    finite_mask = np.isfinite(dist_to_mass)
    ring_mask = ring_mask & finite_mask
    if not ring_mask.any():
        raise ValueError(
            "After excluding unreachable nodes, ring_mask has no True entries."
        )

    # --- Order ring nodes by angle around mass ---
    ring_ordered = ring_order_by_angle(ring_mask, mass_mask, H, W)
    n_ring = ring_ordered.size
    if n_ring < 2:
        raise ValueError(f"Ring has too few nodes: {n_ring}")

    ring_index = build_ring_index(ring_ordered, N)

    # --- Scan edges that lie within the ring and move inward ---
    cw_decrease = 0
    ccw_decrease = 0
    other_count = 0
    edges_in_ring_total = 0

    dist = dist_to_mass  # alias

    for u, v in zip(src, dst):
        iu = ring_index[u]
        iv = ring_index[v]
        if iu < 0 or iv < 0:
            continue  # at least one endpoint not on ring
        edges_in_ring_total += 1

        du = dist[u]
        dv = dist[v]
        if not np.isfinite(du) or not np.isfinite(dv):
            continue

        # inward if dv < du (toward mass); outward if dv > du
        delta = dv - du
        if delta < 0.0:
            # inward edge; we care about azimuthal direction
            s = signed_ring_direction(iu, iv, n_ring)
            if s > 0:
                cw_decrease += 1
            elif s < 0:
                ccw_decrease += 1
            else:
                other_count += 1
        else:
            # we treat outward / neutral edges as 'other'
            other_count += 1

    edges_used = cw_decrease + ccw_decrease
    if edges_in_ring_total == 0:
        raise ValueError("No edges with both endpoints on the ring.")

    # Applicability: we require at least min_edges_used inward edges
    applicable = edges_used >= args.min_edges_used

    if applicable and edges_used > 0:
        bias_grad = (cw_decrease - ccw_decrease) / float(edges_used)
    else:
        bias_grad = 0.0

    PASS = bool(applicable and abs(bias_grad) >= args.bias_threshold)

    out: Dict[str, Any] = {
        "H": H,
        "W": W,
        "N": N,
        "edges_curved": args.edges_curved,
        "mass_mask": args.mass_mask,
        "orbit_mask": args.orbit_mask,
        "n_ring_nodes": int(n_ring),
        "edges_in_ring_total": int(edges_in_ring_total),
        "edges_used_inward": int(edges_used),
        "cw_decrease": int(cw_decrease),
        "ccw_decrease": int(ccw_decrease),
        "other_count": int(other_count),
        "bias_grad": float(bias_grad),
        "bias_threshold": float(args.bias_threshold),
        "min_edges_used": int(args.min_edges_used),
        "VECTOR_RING_GRAD_CORR_APPLICABLE": bool(applicable),
        "PASS_vector_ring_grad_corr_PP": bool(PASS),
        "notes": (
            "STRICT PP vector ring gradient correlation v1. "
            "Ring is defined by orbit_mask & ~mass_mask. "
            "Scalar structure is Markov hop distance TO mass from a reverse-BFS. "
            "We count only edges on the ring whose endpoint distance decreases "
            "(toward mass) and ask whether they preferentially move in one "
            "azimuthal direction (cw vs ccw) in the ring ordering. "
            "Grid coordinates are used as labels only to define an ordering; "
            "no PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("VECTOR_RING_GRAD_CORR_APPLICABLE =", applicable)
    print("PASS_vector_ring_grad_corr_PP =", PASS)


if __name__ == "__main__":
    main()

