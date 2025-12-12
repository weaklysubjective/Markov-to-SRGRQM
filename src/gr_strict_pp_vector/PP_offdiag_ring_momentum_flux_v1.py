#!/usr/bin/env python3
import argparse
import json
from collections import deque

import numpy as np


def load_edges_allow_optional_weight(path: str):
    """
    Accepts edge lines:
      "u v"         (weight defaults to 1.0)
      "u v w"       (float weight)
    Ignores empty lines and lines starting with '#'.
    """
    src = []
    dst = []
    w = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                u = int(parts[0]); v = int(parts[1]); ww = 1.0
            elif len(parts) == 3:
                u = int(parts[0]); v = int(parts[1]); ww = float(parts[2])
            else:
                raise ValueError(f"Bad edge line in {path!r}: {line!r}")
            src.append(u); dst.append(v); w.append(ww)
    return (np.asarray(src, dtype=np.int64),
            np.asarray(dst, dtype=np.int64),
            np.asarray(w, dtype=np.float64))


def mask_to_bool_flat(mask, N: int, name: str):
    """
    Accepts mask in any shape (e.g., 512x512) and returns flat bool mask (N,).
    """
    mask = np.asarray(mask)
    mask = mask.reshape(-1)
    if mask.size != N:
        raise ValueError(f"{name} has size {mask.size}, expected N={N}")
    if mask.dtype == np.bool_:
        return mask
    return (mask != 0)


def mass_centroid_rc(mass_mask_bool_flat: np.ndarray, H: int, W: int):
    idx = np.flatnonzero(mass_mask_bool_flat)
    if idx.size == 0:
        raise ValueError("mass_mask has zero true entries")
    r = (idx // W).astype(np.float64)
    c = (idx % W).astype(np.float64)
    return float(np.mean(r)), float(np.mean(c)), int(idx.size)


def build_reverse_csr(src: np.ndarray, dst: np.ndarray, N: int):
    """
    Reverse adjacency for BFS-to-mass: for each node v, rev_neighbors(v) are all u with u->v.
    Stored as CSR:
      ptr[v]..ptr[v+1] indexes into rev_src_sorted[]
    """
    order = np.argsort(dst, kind="mergesort")
    dst_sorted = dst[order]
    rev_src_sorted = src[order]

    counts = np.bincount(dst_sorted, minlength=N).astype(np.int64)
    ptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ptr, rev_src_sorted


def directed_hops_to_mass(src: np.ndarray, dst: np.ndarray, mass_mask_bool_flat: np.ndarray, N: int):
    """
    Computes shortest directed hop distance TO the mass set:
      dist[u] = min hops along directed edges u->...->m for any mass node m.
    Uses reverse BFS from mass nodes over reverse adjacency.
    """
    ptr, rev_src_sorted = build_reverse_csr(src, dst, N)
    dist = np.full(N, -1, dtype=np.int32)

    q = deque()
    mass_nodes = np.flatnonzero(mass_mask_bool_flat).astype(np.int64)
    for m in mass_nodes:
        dist[m] = 0
        q.append(int(m))

    while q:
        v = q.popleft()
        dv = int(dist[v])
        a = int(ptr[v]); b = int(ptr[v + 1])
        neigh = rev_src_sorted[a:b]  # u such that u->v
        for u in neigh:
            uu = int(u)
            if dist[uu] < 0:
                dist[uu] = dv + 1
                q.append(uu)

    return dist


def ring_order_from_angles(orbit_nodes: np.ndarray, center_r: float, center_c: float, H: int, W: int):
    """
    Orders ring nodes by angle around (center_r, center_c).
    Uses row/col as LABELS ONLY.
    """
    r = (orbit_nodes // W).astype(np.float64)
    c = (orbit_nodes % W).astype(np.float64)
    ang = np.arctan2(r - center_r, c - center_c)  # [-pi, pi]
    order = np.argsort(ang, kind="mergesort")
    return orbit_nodes[order]


def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP off-diagonal proxy v1: ring-local tangential flux conditioned on radial in/out steps."
    )
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--orbit_mask", required=True)
    ap.add_argument("--mass_mask", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--step_window", type=int, default=3)
    ap.add_argument("--min_ring_nodes", type=int, default=200)
    ap.add_argument("--min_edges_used", type=int, default=1000)
    ap.add_argument("--min_edges_per_class", type=int, default=200)
    ap.add_argument("--deltaJ_threshold", type=float, default=0.01)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    H = int(args.H); W = int(args.W)
    N = H * W

    # Load masks (FLATTENED)
    orbit_mask = np.load(args.orbit_mask)
    mass_mask = np.load(args.mass_mask)
    orbit_bool = mask_to_bool_flat(orbit_mask, N, "orbit_mask")
    mass_bool = mask_to_bool_flat(mass_mask, N, "mass_mask")

    orbit_nodes = np.flatnonzero(orbit_bool).astype(np.int64)
    n_orbit = int(orbit_nodes.size)

    center_r, center_c, n_mass = mass_centroid_rc(mass_bool, H, W)

    # Load edges
    src, dst, w = load_edges_allow_optional_weight(args.edges_curved)
    if src.size == 0:
        raise ValueError("edges_curved is empty")

    # Hard bounds check on node IDs
    smin = int(src.min()); smax = int(src.max())
    dmin = int(dst.min()); dmax = int(dst.max())
    if smin < 0 or dmin < 0 or smax >= N or dmax >= N:
        raise ValueError(
            f"Edge node IDs out of range for N={N}. "
            f"src[min,max]=[{smin},{smax}], dst[min,max]=[{dmin},{dmax}]. "
            f"Check H/W or edge builder."
        )

    # Directed hop distance TO mass (Markov-only)
    dist = directed_hops_to_mass(src, dst, mass_bool, N)

    # Ring ordering (angles are LABELS ONLY)
    ring_nodes = ring_order_from_angles(orbit_nodes, center_r, center_c, H, W)
    K = int(ring_nodes.size)

    idx_map = np.full(N, -1, dtype=np.int32)
    idx_map[ring_nodes] = np.arange(K, dtype=np.int32)

    # Filter edges to ring-to-ring
    ring_u = orbit_bool[src]
    ring_v = orbit_bool[dst]
    m_rr = ring_u & ring_v
    src_rr = src[m_rr].astype(np.int64)
    dst_rr = dst[m_rr].astype(np.int64)

    if src_rr.size == 0:
        out = {
            "H": H, "W": W, "N": N,
            "edges_curved": args.edges_curved,
            "orbit_mask": args.orbit_mask,
            "mass_mask": args.mass_mask,
            "source_meta": {
                "mass_center_row_label": center_r,
                "mass_center_col_label": center_c,
                "n_mass_nodes": n_mass,
                "n_orbit_nodes": n_orbit,
                "n_ring_ordered": K,
            },
            "params": {
                "step_window": args.step_window,
                "min_ring_nodes": args.min_ring_nodes,
                "min_edges_used": args.min_edges_used,
                "min_edges_per_class": args.min_edges_per_class,
                "deltaJ_threshold": args.deltaJ_threshold,
            },
            "result": {
                "APPLICABLE": False,
                "reason": "No edges with both endpoints on orbit ring.",
                "PASS_offdiag_ring_momentum_flux_PP": False,
            }
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print("WROTE", args.output)
        print("APPLICABLE = False")
        print("reason:", out["result"]["reason"])
        return

    iu = idx_map[src_rr]
    iv = idx_map[dst_rr]

    # Robust (should be unnecessary, but keep strict)
    valid = (iu >= 0) & (iv >= 0)
    src_rr = src_rr[valid]
    dst_rr = dst_rr[valid]
    iu = iu[valid]
    iv = iv[valid]

    K = int(ring_nodes.size)
    step_window = int(args.step_window)

    delta = (iv - iu).astype(np.int64)
    delta_mod = (delta + K) % K
    is_cw = (delta_mod > 0) & (delta_mod <= step_window)
    is_ccw = (delta_mod >= (K - step_window)) & (delta_mod < K)

    # Radial in/out via Markov hop distance TO mass
    du = dist[src_rr]
    dv = dist[dst_rr]
    finite = (du >= 0) & (dv >= 0)
    du = du[finite]; dv = dv[finite]
    is_cw_f = is_cw[finite]
    is_ccw_f = is_ccw[finite]

    delta_d = (dv.astype(np.int32) - du.astype(np.int32))
    is_in = (delta_d == -1)
    is_out = (delta_d == +1)

    cw_total = int(np.count_nonzero(is_cw_f))
    ccw_total = int(np.count_nonzero(is_ccw_f))
    tang_total = cw_total + ccw_total

    cw_in = int(np.count_nonzero(is_cw_f & is_in))
    ccw_in = int(np.count_nonzero(is_ccw_f & is_in))
    tang_in = cw_in + ccw_in

    cw_out = int(np.count_nonzero(is_cw_f & is_out))
    ccw_out = int(np.count_nonzero(is_ccw_f & is_out))
    tang_out = cw_out + ccw_out

    eps = 1e-12
    J_total = float((cw_total - ccw_total) / (tang_total + eps)) if tang_total > 0 else None
    J_in = float((cw_in - ccw_in) / (tang_in + eps)) if tang_in > 0 else None
    J_out = float((cw_out - ccw_out) / (tang_out + eps)) if tang_out > 0 else None
    deltaJ = float(J_in - J_out) if (J_in is not None and J_out is not None) else None

    applicable = True
    reason = None
    if K < int(args.min_ring_nodes):
        applicable = False
        reason = f"Too few ring nodes: {K} < min_ring_nodes={args.min_ring_nodes}"
    elif tang_total < int(args.min_edges_used):
        applicable = False
        reason = f"Too few tangential edges used: {tang_total} < min_edges_used={args.min_edges_used}"
    elif tang_in < int(args.min_edges_per_class) or tang_out < int(args.min_edges_per_class):
        applicable = False
        reason = (
            f"Insufficient tangential edges in classes: "
            f"in={tang_in}, out={tang_out}, min_edges_per_class={args.min_edges_per_class}"
        )

    passed = False
    if applicable:
        passed = bool(abs(deltaJ) >= float(args.deltaJ_threshold))

    out = {
        "H": H, "W": W, "N": N,
        "edges_curved": args.edges_curved,
        "orbit_mask": args.orbit_mask,
        "mass_mask": args.mass_mask,
        "notes": (
            "STRICT PP off-diagonal ring momentum-flux proxy v1. "
            "Ring ordering uses grid angles as LABELS ONLY. "
            "Radial in/out is defined solely by directed Markov hop distance TO mass "
            "(multi-source reverse BFS on the given edge set). "
            "No PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        ),
        "source_meta": {
            "mass_center_row_label": center_r,
            "mass_center_col_label": center_c,
            "n_mass_nodes": n_mass,
            "n_orbit_nodes": n_orbit,
            "n_ring_ordered": K,
            "n_edges_total": int(src.size),
            "n_edges_ring_to_ring": int(src_rr.size),
        },
        "params": {
            "step_window": step_window,
            "min_ring_nodes": int(args.min_ring_nodes),
            "min_edges_used": int(args.min_edges_used),
            "min_edges_per_class": int(args.min_edges_per_class),
            "deltaJ_threshold": float(args.deltaJ_threshold),
        },
        "counts": {
            "cw_total": cw_total,
            "ccw_total": ccw_total,
            "tang_total": tang_total,
            "cw_in": cw_in,
            "ccw_in": ccw_in,
            "tang_in": tang_in,
            "cw_out": cw_out,
            "ccw_out": ccw_out,
            "tang_out": tang_out,
        },
        "flux": {
            "J_total": J_total,
            "J_in": J_in,
            "J_out": J_out,
            "DeltaJ_in_minus_out": deltaJ,
        },
        "result": {
            "APPLICABLE": bool(applicable),
            "reason": reason,
            "PASS_offdiag_ring_momentum_flux_PP": bool(passed) if applicable else False,
        }
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("APPLICABLE =", out["result"]["APPLICABLE"])
    if not out["result"]["APPLICABLE"]:
        print("reason:", out["result"]["reason"])
    else:
        print("PASS_offdiag_ring_momentum_flux_PP =", out["result"]["PASS_offdiag_ring_momentum_flux_PP"])
        print("DeltaJ =", out["flux"]["DeltaJ_in_minus_out"])
        print("J_in =", out["flux"]["J_in"], "J_out =", out["flux"]["J_out"])


if __name__ == "__main__":
    main()

