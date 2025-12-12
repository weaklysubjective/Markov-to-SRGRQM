#!/usr/bin/env python3
"""
PP_deflection_markov_front_PP_v5.py

STRICT PP Markov deflection front at scale H×W, with an
ADAPTIVE MARKOV RING (Option A).

Key changes vs v4:
    - Ring is NOT taken from a pre-baked orbit_mask.
    - Instead, we define a Markov-hop "ring" by distance-to-mass
      on the CURVED graph.

Fixed STRICT-PP rule for the ring:
    Let dist_to_mass[i] be the directed Markov hop distance
    (via reverse BFS on the curved edges) from node i to ANY mass node.

    For d = 1, 2, 3, ... in ascending order:
        Define ring_d = { i : dist_to_mass[i] == d }.
        Require:
            * |ring_d| >= min_ring_nodes, and
            * front_flat_mass(ring_d)  >= min_front_mass, and
            * front_curved_mass(ring_d) >= min_front_mass.
        Take the FIRST d that satisfies this.
        If no such d exists, the observable is APPLICABLE = False.

Distances, fronts, and masks are all STRICT PP:
    - Distance/time from Markov hops only (reverse BFS, no PDE).
    - Fronts from finite-time Markov evolution p_{t+1} = p_t @ P
      on the given edge sets.
    - No Laplacian/Poisson, no GR ansatz, no regression.
    - Grid indices (row/col) are LABELS ONLY (used for masks, reporting).

Edge format:
    Accepts either:
      - 3 columns: src dst weight   (interpreted as weights, row-normalized)
      - 2 columns: src dst          (weight=1, row-normalized)
"""

import argparse
import json
from collections import deque
from typing import Tuple, Dict, Tuple as Tup

import numpy as np
import scipy.sparse as sp


# ---------------------------
# IO and basic helpers
# ---------------------------

def load_edges(path: str, N: int) -> Tup[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load edges as (src, dst, weight).

    Formats per non-comment, non-blank line:
        src dst
        src dst weight

    src, dst: integers in [0, N).
    weight: float (default 1.0 if omitted).
    """
    srcs, dsts, ws = [], [], []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) == 2:
                i = int(parts[0])
                j = int(parts[1])
                w = 1.0
            elif len(parts) >= 3:
                i = int(parts[0])
                j = int(parts[1])
                w = float(parts[2])
            else:
                raise ValueError(f"Bad edge line in {path!r}: {line!r}")
            if not (0 <= i < N and 0 <= j < N):
                raise ValueError(
                    f"Edge indices out of range in {path}: ({i}, {j}), N={N}"
                )
            srcs.append(i)
            dsts.append(j)
            ws.append(w)
    if not srcs:
        raise ValueError(f"No edges loaded from {path}")
    srcs = np.asarray(srcs, dtype=np.int64)
    dsts = np.asarray(dsts, dtype=np.int64)
    ws = np.asarray(ws, dtype=float)
    return srcs, dsts, ws


def build_markov_matrix(
    srcs: np.ndarray, dsts: np.ndarray, ws: np.ndarray, N: int
) -> sp.csr_matrix:
    """
    Build a row-stochastic Markov matrix P (N×N) in CSR form.

    P[src, dst] ∝ weight; rows with positive outgoing sum are normalized
    to sum to 1. Rows with zero outgoing weight remain all zeros (sinks).
    """
    P = sp.csr_matrix((ws, (srcs, dsts)), shape=(N, N))

    row_sums = np.asarray(P.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(row_sums, dtype=float)
    mask = row_sums > 0.0
    inv[mask] = 1.0 / row_sums[mask]

    if np.any(mask):
        D = sp.diags(inv)
        P = D @ P

    new_row_sums = np.asarray(P.sum(axis=1)).reshape(-1)
    tol = 1e-6
    for r, s in enumerate(new_row_sums):
        if s == 0.0:
            continue
        if not (abs(s - 1.0) <= tol):
            raise ValueError(
                f"Markov row {r} sum={s}, expected ~1 or 0. "
                f"Edge construction may be inconsistent."
            )

    return P


def load_mask(path: str, N: int, H: int, W: int, name: str) -> np.ndarray:
    """
    Load a boolean mask from .npy (1D or 2D) and flatten to (N,).

    Accepts:
        (H, W) -> flattened
        (N,)   -> used directly
    """
    arr = np.load(path)
    if arr.ndim == 2:
        if arr.shape != (H, W):
            raise ValueError(
                f"{name} mask {path!r} has shape {arr.shape}, expected ({H},{W})"
            )
        arr = arr.astype(bool).reshape(-1)
    elif arr.ndim == 1:
        if arr.size != N:
            raise ValueError(
                f"{name} mask {path!r} has size {arr.size}, expected N={N}"
            )
        arr = arr.astype(bool)
    else:
        raise ValueError(
            f"{name} mask {path!r} must be 1D or 2D, got shape {arr.shape}"
        )
    if not np.any(arr):
        raise ValueError(f"{name} mask {path!r} has no True entries")
    return arr


# ---------------------------
# STRICT-PP geometry helpers
# ---------------------------

def bfs_markov_distance_to_mass(
    srcs: np.ndarray, dsts: np.ndarray, N: int, mass_mask: np.ndarray
) -> np.ndarray:
    """
    Directed Markov hop distance TO mass via reverse BFS on the curved graph.

    If edges are i -> j, with srcs[i], dsts[i], we want, for each node k,
    the minimum number of forward Markov hops to reach any mass node.

    We build reverse adjacency:
        rev_adj[j].append(i)
    and BFS from all mass nodes in the reversed graph.

    Unreachable nodes get distance = np.inf.
    """
    assert mass_mask.shape == (N,)
    mass_indices = np.nonzero(mass_mask)[0]
    if mass_indices.size == 0:
        raise ValueError("mass_mask has no True entries in BFS helper")

    rev_adj = [[] for _ in range(N)]
    for s, d in zip(srcs, dsts):
        rev_adj[d].append(s)

    dist = np.full(N, np.inf, dtype=float)
    dq = deque()
    for m in mass_indices:
        dist[m] = 0.0
        dq.append(m)

    while dq:
        u = dq.popleft()
        du = dist[u]
        for v in rev_adj[u]:
            if not np.isfinite(dist[v]):
                dist[v] = du + 1.0
                dq.append(v)

    return dist


def build_auto_source_mask(
    H: int, W: int, mass_mask: np.ndarray, band_width: int
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Build an automatic OUTER-BAND source mask using LABELS ONLY.

    Rule:
        - Compute label-based mass center (row, col).
        - If center_col < W/2: launch from RIGHT boundary band.
          Else: launch from LEFT boundary band.
        - Band has width 'band_width' in columns (clipped to [1, W]).

    Returns:
        src_mask: bool (N,)
        meta: dict with side and counts for reporting.
    """
    N = H * W
    idx = np.nonzero(mass_mask)[0]
    if idx.size == 0:
        raise ValueError("mass_mask has no True entries for auto source builder")

    rows = idx // W
    cols = idx % W
    center_row = float(rows.mean())
    center_col = float(cols.mean())

    bw = max(1, min(int(band_width), W))

    if center_col < W / 2.0:
        side = "right"
        col_start = max(0, W - bw)
        col_end = W
    else:
        side = "left"
        col_start = 0
        col_end = bw

    src_mask_2d = np.zeros((H, W), dtype=bool)
    src_mask_2d[:, col_start:col_end] = True
    src_mask = src_mask_2d.reshape(-1)

    meta = {
        "auto_source": True,
        "source_side": side,
        "source_cols_start_label": int(col_start),
        "source_cols_end_label_exclusive": int(col_end),
        "source_band_width": int(bw),
        "mass_center_row_label": center_row,
        "mass_center_col_label": center_col,
        "N": N,
    }
    return src_mask, meta


def build_source_distribution(
    H: int,
    W: int,
    mass_mask: np.ndarray,
    src_mask_path: str,
    source_band_width: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Build initial Markov distribution p0 over sources.

    - If src_mask_path is provided:
        * Load bool mask, use True entries as sources.
    - Else:
        * Use outer-band auto source from build_auto_source_mask.

    Returns:
        p0: shape (N,), sum(p0) == 1
        meta: dict with source meta data.
    """
    N = H * W
    if src_mask_path:
        arr = np.load(src_mask_path)
        if arr.ndim == 2:
            if arr.shape != (H, W):
                raise ValueError(
                    f"src_mask {src_mask_path!r} has shape {arr.shape}, "
                    f"expected ({H},{W})"
                )
            arr = arr.astype(bool).reshape(-1)
        elif arr.ndim == 1:
            if arr.size != N:
                raise ValueError(
                    f"src_mask {src_mask_path!r} size {arr.size}, expected N={N}"
                )
            arr = arr.astype(bool)
        else:
            raise ValueError(
                f"src_mask {src_mask_path!r} must be 1D or 2D, got shape {arr.shape}"
            )
        if not np.any(arr):
            raise ValueError(f"src_mask {src_mask_path!r} has no True entries")
        src_mask = arr
        meta = {
            "auto_source": False,
            "source_mask_path": src_mask_path,
        }
    else:
        src_mask, meta = build_auto_source_mask(
            H, W, mass_mask, band_width=source_band_width
        )

    src_idx = np.nonzero(src_mask)[0]
    if src_idx.size == 0:
        raise ValueError("Source mask yielded no nodes")

    p0 = np.zeros(N, dtype=float)
    p0[src_idx] = 1.0 / src_idx.size
    meta["n_source_nodes"] = int(src_idx.size)
    meta["n_mass_nodes"] = int(np.count_nonzero(mass_mask))
    return p0, meta


# ---------------------------
# Markov front evolution (global)
# ---------------------------

def evolve_fronts_global(
    P_flat: sp.csr_matrix,
    P_curved: sp.csr_matrix,
    p0: np.ndarray,
    steps: int,
    burn_in: int,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolve Markov fronts on flat/curved graphs and accumulate node-wise
    occupancy over the last `window` steps (after burn_in).

    p_{t+1} = p_t @ P

    We store:
        front_flat_sum[i]  ~ sum_t p_flat_t[i]
        front_curved_sum[i] ~ sum_t p_curved_t[i]
    for t in [start_collect, steps-1], where
        start_collect = max(burn_in, steps - window).
    """
    N = p0.size
    p_flat = p0[None, :]
    p_curv = p0[None, :]

    front_flat = np.zeros(N, dtype=float)
    front_curved = np.zeros(N, dtype=float)

    burn_in = max(0, int(burn_in))
    steps = max(1, int(steps))
    window = max(1, min(int(window), steps))

    start_collect = max(burn_in, steps - window)

    for t in range(steps):
        p_flat = p_flat @ P_flat
        p_curv = p_curv @ P_curved

        if t >= start_collect:
            pf = np.asarray(p_flat).reshape(-1)
            pc = np.asarray(p_curv).reshape(-1)
            front_flat += pf
            front_curved += pc

    return front_flat, front_curved


# ---------------------------
# Ring selection and deflection
# ---------------------------

def pick_markov_ring(
    dist_to_mass: np.ndarray,
    front_flat: np.ndarray,
    front_curved: np.ndarray,
    min_ring_nodes: int,
    min_front_mass: float,
) -> Dict[str, object]:
    """
    Pick a Markov-defined ring via a FIXED STRICT-PP rule:

        - Convert dist_to_mass to integer hops (ignore inf).
        - For d in ascending order (d >= 1):
             ring_d = { i : dist_int[i] == d }.
             If |ring_d| >= min_ring_nodes and
                flat_mass(ring_d) >= min_front_mass and
                curved_mass(ring_d) >= min_front_mass:
                 choose this d and ring_d.
        - If no such d exists, APPLICABLE = False (no deflection band).

    Returns a dict with:
        {
          "APPLICABLE": bool,
          "reason": ...,
          "ring_idx": np.ndarray or None,
          "ring_distance_hops": int or None,
          "front_mass_flat_ring": float,
          "front_mass_curved_ring": float,
        }
    """
    N = dist_to_mass.size
    assert front_flat.shape == (N,)
    assert front_curved.shape == (N,)

    finite = np.isfinite(dist_to_mass)
    if not np.any(finite):
        return {
            "APPLICABLE": False,
            "reason": "No nodes with finite Markov distance to mass.",
            "ring_idx": None,
            "ring_distance_hops": None,
            "front_mass_flat_ring": 0.0,
            "front_mass_curved_ring": 0.0,
        }

    dist_int = dist_to_mass.astype(np.int64)
    # Candidate hop distances: positive integers only
    d_vals = np.unique(dist_int[finite & (dist_int > 0)])
    if d_vals.size == 0:
        return {
            "APPLICABLE": False,
            "reason": "All finite distances are 0 (mass band only).",
            "ring_idx": None,
            "ring_distance_hops": None,
            "front_mass_flat_ring": 0.0,
            "front_mass_curved_ring": 0.0,
        }

    min_ring_nodes = max(1, int(min_ring_nodes))
    min_front_mass = float(min_front_mass)

    for d in d_vals:
        ring_mask = finite & (dist_int == d)
        ring_idx = np.nonzero(ring_mask)[0]
        if ring_idx.size < min_ring_nodes:
            continue

        flat_mass = float(front_flat[ring_idx].sum())
        curv_mass = float(front_curved[ring_idx].sum())

        if flat_mass >= min_front_mass and curv_mass >= min_front_mass:
            return {
                "APPLICABLE": True,
                "reason": None,
                "ring_idx": ring_idx,
                "ring_distance_hops": int(d),
                "front_mass_flat_ring": flat_mass,
                "front_mass_curved_ring": curv_mass,
            }

    return {
        "APPLICABLE": False,
        "reason": (
            "No Markov hop-distance band (d>0) with sufficient nodes and "
            "nonzero flat/curved front mass."
        ),
        "ring_idx": None,
        "ring_distance_hops": None,
        "front_mass_flat_ring": float(front_flat.sum()),
        "front_mass_curved_ring": float(front_curved.sum()),
    }


def compute_deflection_on_ring(
    ring_idx: np.ndarray,
    dist_to_mass: np.ndarray,
    front_flat: np.ndarray,
    front_curved: np.ndarray,
    delta_threshold: float,
) -> Dict[str, object]:
    """
    Given a chosen ring (indices), compute D_flat, D_curved, DeltaD.

    D_flat   = E_flat[dist_to_mass | ring]
    D_curved = E_curved[dist_to_mass | ring]
    DeltaD   = D_flat - D_curved

    PASS_deflection_markov_front_PP_v5 if DeltaD > delta_threshold.
    """
    d_used = dist_to_mass[ring_idx]
    finite = np.isfinite(d_used)
    if not np.any(finite):
        return {
            "PASS_deflection_markov_front_PP_v5": False,
            "D_flat": None,
            "D_curved": None,
            "DeltaD": None,
        }

    idx = ring_idx[finite]
    flat = front_flat[idx]
    curv = front_curved[idx]
    dvals = d_used[finite]

    flat_mass = float(flat.sum())
    curv_mass = float(curv.sum())
    if flat_mass <= 0.0 or curv_mass <= 0.0:
        return {
            "PASS_deflection_markov_front_PP_v5": False,
            "D_flat": None,
            "D_curved": None,
            "DeltaD": None,
        }

    pf = flat / flat_mass
    pc = curv / curv_mass

    D_flat = float((pf * dvals).sum())
    D_curved = float((pc * dvals).sum())
    DeltaD = D_flat - D_curved
    passed = bool(DeltaD > float(delta_threshold))

    return {
        "PASS_deflection_markov_front_PP_v5": passed,
        "D_flat": D_flat,
        "D_curved": D_curved,
        "DeltaD": DeltaD,
        "DeltaD_threshold": float(delta_threshold),
        "front_mass_flat_ring": flat_mass,
        "front_mass_curved_ring": curv_mass,
        "n_ring_nodes_used": int(idx.size),
    }


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "STRICT PP Markov deflection v5 with adaptive Markov ring. "
            "Finite-time Markov fronts from an outer band, distances via "
            "reverse BFS to mass, ring picked by a fixed hop-distance rule."
        )
    )
    ap.add_argument("--edges_flat", required=True)
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--mass_mask", required=True, help=".npy, bool (H×W or N)")

    # orbit_mask is kept for compatibility + reporting, but NOT used
    # to define the ring in v5.
    ap.add_argument("--orbit_mask", required=True, help=".npy, compatibility only")

    ap.add_argument("--src_mask", default="", help=".npy, optional explicit sources")

    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--burn_in", type=int, default=64)
    ap.add_argument("--window", type=int, default=128)

    ap.add_argument(
        "--min_front_mass",
        type=float,
        default=1e-4,
        help="Minimum ring front mass (flat and curved) to accept a band.",
    )
    ap.add_argument(
        "--min_ring_nodes",
        type=int,
        default=100,
        help="Minimum number of nodes in a ring band."
    )
    ap.add_argument(
        "--delta_threshold",
        type=float,
        default=0.05,
        help="PASS if DeltaD > delta_threshold."
    )

    ap.add_argument(
        "--source_band_width",
        type=int,
        default=1,
        help="Width (columns) of auto source band at boundary (labels only).",
    )

    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    H = int(args.H)
    W = int(args.W)
    if H <= 0 or W <= 0:
        raise ValueError(f"H and W must be positive, got H={H}, W={W}")
    N = H * W

    # Masks
    mass_mask = load_mask(args.mass_mask, N, H, W, "mass")
    orbit_mask = load_mask(args.orbit_mask, N, H, W, "orbit")  # for reporting only

    # Edges
    src_f, dst_f, w_f = load_edges(args.edges_flat, N)
    src_c, dst_c, w_c = load_edges(args.edges_curved, N)

    # Markov matrices
    P_flat = build_markov_matrix(src_f, dst_f, w_f, N)
    P_curved = build_markov_matrix(src_c, dst_c, w_c, N)

    # Distances to mass on CURVED graph
    dist_to_mass = bfs_markov_distance_to_mass(src_c, dst_c, N, mass_mask)

    # Source distribution
    p0, src_meta = build_source_distribution(
        H, W, mass_mask, args.src_mask, args.source_band_width
    )

    # Global fronts
    front_flat, front_curved = evolve_fronts_global(
        P_flat=P_flat,
        P_curved=P_curved,
        p0=p0,
        steps=args.steps,
        burn_in=args.burn_in,
        window=args.window,
    )

    # Pick Markov ring
    ring_info = pick_markov_ring(
        dist_to_mass=dist_to_mass,
        front_flat=front_flat,
        front_curved=front_curved,
        min_ring_nodes=args.min_ring_nodes,
        min_front_mass=args.min_front_mass,
    )

    if not ring_info["APPLICABLE"]:
        out = {
            "H": H,
            "W": W,
            "N": N,
            "edges_flat": args.edges_flat,
            "edges_curved": args.edges_curved,
            "mass_mask": args.mass_mask,
            "orbit_mask": args.orbit_mask,
            "src_mask": args.src_mask or None,
            "steps": int(args.steps),
            "burn_in": int(args.burn_in),
            "window": int(args.window),
            "min_front_mass": float(args.min_front_mass),
            "min_ring_nodes": int(args.min_ring_nodes),
            "delta_threshold": float(args.delta_threshold),
            "source_band_width": int(args.source_band_width),
            "source_meta": src_meta,
            "ring_selection": ring_info,
            "deflection_stats": {
                "APPLICABLE": False,
                "reason": ring_info["reason"],
                "PASS_deflection_markov_front_PP_v5": False,
                "D_flat": None,
                "D_curved": None,
                "DeltaD": None,
            },
            "notes": (
                "STRICT PP deflection v5 with adaptive Markov ring. "
                "No PDE, no Laplacian/Poisson, no GR ansatz, no regression. "
                "Ring is defined solely by Markov hop distance to mass and "
                "front masses, via a fixed rule. orbit_mask is kept only for "
                "compatibility and reporting."
            ),
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print("WROTE", args.output)
        print("APPLICABLE = False")
        print("reason:", ring_info["reason"])
        return

    ring_idx = ring_info["ring_idx"]

    # Deflection on the chosen ring
    deflect = compute_deflection_on_ring(
        ring_idx=ring_idx,
        dist_to_mass=dist_to_mass,
        front_flat=front_flat,
        front_curved=front_curved,
        delta_threshold=args.delta_threshold,
    )

    deflect["APPLICABLE"] = True
    deflect["ring_distance_hops"] = ring_info["ring_distance_hops"]

    out = {
        "H": H,
        "W": W,
        "N": N,
        "edges_flat": args.edges_flat,
        "edges_curved": args.edges_curved,
        "mass_mask": args.mass_mask,
        "orbit_mask": args.orbit_mask,
        "src_mask": args.src_mask or None,
        "steps": int(args.steps),
        "burn_in": int(args.burn_in),
        "window": int(args.window),
        "min_front_mass": float(args.min_front_mass),
        "min_ring_nodes": int(args.min_ring_nodes),
        "delta_threshold": float(args.delta_threshold),
        "source_band_width": int(args.source_band_width),
        "source_meta": src_meta,
        "ring_selection": ring_info,
        "deflection_stats": deflect,
        "notes": (
            "STRICT PP deflection v5. Distances/time from Markov hops and "
            "reverse BFS only; no PDE, no Laplacian/Poisson, no GR ansatz, "
            "no regression. Ring is defined by a fixed Markov-hop rule "
            "independent of the observed DeltaD. orbit_mask is retained "
            "only for compatibility and metadata."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("APPLICABLE = True")
    print("ring_distance_hops =", ring_info["ring_distance_hops"])
    print("front_mass_flat_ring =", ring_info["front_mass_flat_ring"])
    print("front_mass_curved_ring =", ring_info["front_mass_curved_ring"])
    print("D_flat   =", deflect["D_flat"])
    print("D_curved =", deflect["D_curved"])
    print("DeltaD   =", deflect["DeltaD"])
    print("PASS_deflection_markov_front_PP_v5 =", deflect["PASS_deflection_markov_front_PP_v5"])


if __name__ == "__main__":
    main()

