#!/usr/bin/env python3
"""
PP_deflection_markov_front_PP_v4.py

STRICT PP Markov deflection front at scale H×W.

Motivation:
    Earlier deflection versions tended to give
        D_flat = 0.0, D_curved = 0.0, closer_curved_than_flat = null
    because they effectively measured from a front that was already at the
    mass band, so distance-to-mass was trivially zero for both cases.

v4 fix (STRICT PP):
    - Launch a finite-time Markov front from an OUTER source band
      (auto-chosen boundary column opposite the mass in LABEL-space,
       or provided via src_mask).
    - Observe arrivals on an ORBIT ring (orbit_mask).
    - Weight ring arrivals by Markov distance-to-mass, computed via
      reverse BFS on the CURVED graph only.

v4 (band extension):
    - Auto source can be a boundary band of configurable width
      (--source_band_width), still using labels only.

All primitives obey STRICT PP:
    - Distance & time from Markov hops / commute structure only.
    - No PDE, no Laplacian/Poisson, no GR ansatz, no regression.
    - Grid indices (row/col) are LABELS ONLY (used for masks, source placement).

Edge format:
    Accepts either:
      - 3 columns: src dst weight   (row-stochastic or raw weights)
      - 2 columns: src dst          (interpreted as weight=1 and
                                     row-normalized to build Markov P)
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

    Expected formats (per line, ignoring comments/blank):
        src dst weight
        src dst

    src, dst: 0-based integer indices in [0, N).
    weight: float, typically a Markov probability or raw weight.
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
                    f"Edge indices out of range in {path}: ({i}, {j}) with N={N}"
                )
            srcs.append(i)
            dsts.append(j)
            ws.append(w)
    srcs = np.asarray(srcs, dtype=np.int64)
    dsts = np.asarray(dsts, dtype=np.int64)
    ws = np.asarray(ws, dtype=float)
    if srcs.size == 0:
        raise ValueError(f"No edges loaded from {path}")
    return srcs, dsts, ws


def build_markov_matrix(
    srcs: np.ndarray, dsts: np.ndarray, ws: np.ndarray, N: int
) -> sp.csr_matrix:
    """
    Build a row-stochastic Markov matrix P (N×N) in CSR.

    Interpreting edges as:
        P[src, dst] ∝ weight

    We ALWAYS enforce row-stochasticity via row normalization on
    rows with positive outgoing weight (sums to 1). Rows with zero
    outgoing weight remain all zeros (sinks).

    This is still STRICT PP: the only input is the edge structure and
    weights from the trace/experience pipeline; no external geometry.
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
                f"After normalization, row {r} of Markov matrix has sum {s}, "
                "expected ~1 or 0 (sink). Edge construction may be inconsistent."
            )

    return P


def load_mask(path: str, N: int, H: int, W: int, name: str) -> np.ndarray:
    """
    Load a boolean mask from .npy and flatten.

    Accepts shapes:
        (H, W) -> flattened to (N,)
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

    If we have edges i -> j (from srcs to dsts), and we want distance in
    hops from node k to ANY mass node along forward Markov edges, we do:

        * Build reverse adjacency: rev_adj[j].append(i)
        * BFS outward from all mass nodes in the reversed graph.

    The resulting dist[v] is the MIN number of Markov hops along forward edges
    needed to reach a mass node.

    Unreachable nodes are assigned np.inf.
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
    H: int, W: int, mass_mask: np.ndarray, band_width: int = 1
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Build an automatic outer-band source mask using LABELS ONLY.

    Strategy:
        * Compute label-based mass "center" row/col from mass_mask.
        * Choose the boundary side opposite the mass in LABEL-space:
            - If center_col < W/2 -> use right boundary
            - Else -> use left boundary
        * Use a vertical band of width 'band_width' at that boundary.

    band_width is clipped to [1, W].
    """
    N = H * W
    idx = np.nonzero(mass_mask)[0]
    if idx.size == 0:
        raise ValueError("mass_mask has no True entries in auto source builder")

    rows = idx // W
    cols = idx % W
    center_row = float(rows.mean())
    center_col = float(cols.mean())

    bw = max(1, min(band_width, W))

    if center_col < W / 2.0:
        # Mass is left-ish; launch from right boundary band
        side = "right"
        col_start = max(0, W - bw)
        col_end = W  # exclusive
    else:
        # Mass is right-ish; launch from left boundary band
        side = "left"
        col_start = 0
        col_end = bw  # exclusive

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
    orbit_mask: np.ndarray,
    src_mask_path: str,
    source_band_width: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Build the initial Markov distribution over sources.

    Cases:
        * If src_mask_path is provided:
            - Load src_mask as .npy, flatten to (N,), use True entries as sources.
        * Else:
            - Use the auto source band built from mass_mask and source_band_width.

    Returns:
        p0: np.ndarray, shape (N,), sum(p0) == 1.
        meta: dict with 'auto_source' flag and labels for documentation.
    """
    N = H * W
    if src_mask_path:
        arr = np.load(src_mask_path)
        if arr.ndim == 2:
            if arr.shape != (H, W):
                raise ValueError(
                    f"src_mask {src_mask_path!r} has shape {arr.shape}, expected ({H},{W})"
                )
            arr = arr.astype(bool).reshape(-1)
        elif arr.ndim == 1:
            if arr.size != N:
                raise ValueError(
                    f"src_mask {src_mask_path!r} has size {arr.size}, expected N={N}"
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

    src_indices = np.nonzero(src_mask)[0]
    if src_indices.size == 0:
        raise ValueError("Source mask yielded no nodes")

    p0 = np.zeros(N, dtype=float)
    p0[src_indices] = 1.0 / src_indices.size
    meta["n_source_nodes"] = int(src_indices.size)
    meta["n_orbit_nodes"] = int(np.count_nonzero(orbit_mask))
    meta["n_mass_nodes"] = int(np.count_nonzero(mass_mask))
    return p0, meta


# ---------------------------
# Markov front evolution
# ---------------------------

def evolve_fronts_on_ring(
    P_flat: sp.csr_matrix,
    P_curved: sp.csr_matrix,
    p0: np.ndarray,
    orbit_mask: np.ndarray,
    steps: int,
    burn_in: int,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolve Markov fronts on flat/curved graphs, collecting mass on the orbit ring.

    p_{t+1} = p_t @ P   (row-vector times CSR).

    We accumulate orbit-ring mass over the last `window` steps:

        for t in 0..steps-1:
            p_flat = p_flat @ P_flat
            p_curv = p_curv @ P_curved
            if t >= burn_in and t >= steps - window:
                ring_flat_sum += p_flat[orbit_mask]
                ring_curv_sum += p_curv[orbit_mask]
    """
    N = P_flat.shape[0]
    assert p0.shape == (N,)
    assert orbit_mask.shape == (N,)

    p_flat = p0[None, :]
    p_curv = p0[None, :]

    ring_flat_sum = np.zeros(N, dtype=float)
    ring_curv_sum = np.zeros(N, dtype=float)

    burn_in = max(0, burn_in)
    steps = max(1, steps)
    window = max(1, min(window, steps))

    start_collect = max(burn_in, steps - window)

    for t in range(steps):
        p_flat = p_flat @ P_flat
        p_curv = p_curv @ P_curved

        if t >= start_collect:
            pf = np.asarray(p_flat).reshape(-1)
            pc = np.asarray(p_curv).reshape(-1)
            ring_flat_sum += pf * orbit_mask
            ring_curv_sum += pc * orbit_mask

    return ring_flat_sum, ring_curv_sum


# ---------------------------
# Deflection statistic
# ---------------------------

def compute_deflection_stats(
    ring_flat_sum: np.ndarray,
    ring_curv_sum: np.ndarray,
    orbit_mask: np.ndarray,
    dist_to_mass: np.ndarray,
    min_front_mass: float,
    delta_threshold: float,
) -> Dict[str, object]:
    """
    Compute D_flat, D_curved and DeltaD on the orbit ring.
    """
    N = ring_flat_sum.size
    assert orbit_mask.shape == (N,)
    assert dist_to_mass.shape == (N,)

    ring_idx = np.nonzero(orbit_mask)[0]
    if ring_idx.size == 0:
        raise ValueError("orbit_mask has no True entries in deflection stats")

    finite_mask = np.isfinite(dist_to_mass[ring_idx])
    use_idx = ring_idx[finite_mask]
    if use_idx.size == 0:
        return {
            "APPLICABLE": False,
            "reason": "No orbit-ring nodes with finite Markov distance to mass",
            "D_flat": None,
            "D_curved": None,
            "DeltaD": None,
            "front_mass_flat": float(ring_flat_sum[ring_idx].sum()),
            "front_mass_curved": float(ring_curv_sum[ring_idx].sum()),
        }

    flat_used = ring_flat_sum[use_idx]
    curv_used = ring_curv_sum[use_idx]
    d_used = dist_to_mass[use_idx]

    front_flat = float(flat_used.sum())
    front_curv = float(curv_used.sum())

    applicable = (
        front_flat >= min_front_mass and front_curv >= min_front_mass
    )

    if not applicable:
        return {
            "APPLICABLE": False,
            "reason": (
                f"Insufficient front mass on orbit ring: "
                f"flat={front_flat}, curved={front_curv}, "
                f"min_front_mass={min_front_mass}"
            ),
            "D_flat": None,
            "D_curved": None,
            "DeltaD": None,
            "front_mass_flat": front_flat,
            "front_mass_curved": front_curv,
        }

    pf = flat_used / front_flat
    pc = curv_used / front_curv

    D_flat = float((pf * d_used).sum())
    D_curved = float((pc * d_used).sum())
    DeltaD = D_flat - D_curved

    passed = bool(DeltaD > delta_threshold)

    return {
        "APPLICABLE": True,
        "reason": None,
        "D_flat": D_flat,
        "D_curved": D_curved,
        "DeltaD": DeltaD,
        "DeltaD_threshold": float(delta_threshold),
        "PASS_deflection_markov_front_PP_v4": passed,
        "front_mass_flat": front_flat,
        "front_mass_curved": front_curv,
        "n_ring_nodes_total": int(ring_idx.size),
        "n_ring_nodes_used": int(use_idx.size),
    }


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "STRICT PP Markov deflection front v4. "
            "Finite-time Markov fronts from an outer band, observed on an orbit ring, "
            "weighted by Markov distance-to-mass."
        )
    )
    ap.add_argument("--edges_flat", required=True)
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--mass_mask", required=True, help=".npy, bool (H×W or N)")
    ap.add_argument("--orbit_mask", required=True, help=".npy, bool (H×W or N)")
    ap.add_argument("--src_mask", default="", help=".npy, optional explicit sources")

    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--burn_in", type=int, default=64)
    ap.add_argument("--window", type=int, default=128)

    ap.add_argument("--min_front_mass", type=float, default=1e-6)
    ap.add_argument("--delta_threshold", type=float, default=0.05)

    ap.add_argument(
        "--source_band_width",
        type=int,
        default=1,
        help="Width (in columns) of auto source band at boundary (labels only).",
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
    orbit_mask = load_mask(args.orbit_mask, N, H, W, "orbit")

    # Edges
    src_f, dst_f, w_f = load_edges(args.edges_flat, N)
    src_c, dst_c, w_c = load_edges(args.edges_curved, N)

    # Markov matrices (CSR, row-normalized)
    P_flat = build_markov_matrix(src_f, dst_f, w_f, N)
    P_curved = build_markov_matrix(src_c, dst_c, w_c, N)

    # Markov distance-to-mass (curved graph)
    dist_to_mass = bfs_markov_distance_to_mass(src_c, dst_c, N, mass_mask)

    # Source distribution
    p0, src_meta = build_source_distribution(
        H,
        W,
        mass_mask,
        orbit_mask,
        args.src_mask,
        source_band_width=args.source_band_width,
    )

    # Evolve fronts and accumulate orbit-ring mass
    ring_flat_sum, ring_curv_sum = evolve_fronts_on_ring(
        P_flat=P_flat,
        P_curved=P_curved,
        p0=p0,
        orbit_mask=orbit_mask,
        steps=args.steps,
        burn_in=args.burn_in,
        window=args.window,
    )

    # Deflection stats
    stats = compute_deflection_stats(
        ring_flat_sum=ring_flat_sum,
        ring_curv_sum=ring_curv_sum,
        orbit_mask=orbit_mask,
        dist_to_mass=dist_to_mass,
        min_front_mass=args.min_front_mass,
        delta_threshold=args.delta_threshold,
    )

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
        "delta_threshold": float(args.delta_threshold),
        "source_band_width": int(args.source_band_width),
        "source_meta": src_meta,
        "deflection_stats": stats,
        "notes": (
            "STRICT PP deflection v4. "
            "Distances/time from Markov hops and BFS only; "
            "no PDE, no Laplacian/Poisson, no GR ansatz, no regression. "
            "Orbit and mass masks define all 'geometry'; "
            "grid indices are used purely as labels. "
            "This version avoids the D_flat=D_curved=0 degeneracy by "
            "measuring on an orbit ring rather than at the mass band, "
            "supports both 2-column and 3-column edge formats, "
            "and uses an outer boundary band (configurable width) as source."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    if stats.get("APPLICABLE", False):
        print("APPLICABLE = True")
        print("D_flat   =", stats["D_flat"])
        print("D_curved =", stats["D_curved"])
        print("DeltaD   =", stats["DeltaD"])
        print("PASS_deflection_markov_front_PP_v4 =", stats["PASS_deflection_markov_front_PP_v4"])
    else:
        print("APPLICABLE = False")
        print("reason:", stats.get("reason"))


if __name__ == "__main__":
    main()

