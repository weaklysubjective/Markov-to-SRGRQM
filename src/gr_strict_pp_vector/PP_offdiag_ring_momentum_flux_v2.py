#!/usr/bin/env python3
import argparse, json, math, sys
import numpy as np
from collections import deque

def die(msg: str):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(2)

def load_mask_any(path: str, H: int, W: int, name: str) -> np.ndarray:
    arr = np.load(path)
    if arr.shape == (H, W):
        arr = arr.reshape(-1)
    elif arr.shape == (H * W,):
        pass
    else:
        die(f"{name} shape {arr.shape} not compatible with H,W=({H},{W})")
    arr = arr.astype(bool, copy=False)
    if arr.size != H * W:
        die(f"{name} size {arr.size} != N={H*W}")
    return arr

def load_edges_any(path: str, N: int):
    src = []
    dst = []
    w = []
    bad = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                u, v = parts
                ww = 1.0
            elif len(parts) == 3:
                u, v, ww = parts
                ww = float(ww)
            else:
                bad += 1
                continue
            u = int(u); v = int(v)
            if u < 0 or u >= N or v < 0 or v >= N:
                bad += 1
                continue
            src.append(u); dst.append(v); w.append(float(ww))
    if bad > 0:
        print(f"[WARN] skipped {bad} bad edge lines while reading {path}", file=sys.stderr)
    if len(src) == 0:
        die(f"no valid edges loaded from {path}")
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), np.asarray(w, dtype=np.float64)

def idx_to_rc(idx: np.ndarray, W: int):
    r = idx // W
    c = idx - r * W
    return r.astype(np.float64), c.astype(np.float64)

def mass_center_from_mask(mass_bool: np.ndarray, H: int, W: int):
    mass_idx = np.flatnonzero(mass_bool)
    if mass_idx.size == 0:
        die("mass_mask has zero true entries")
    r, c = idx_to_rc(mass_idx, W)
    return float(np.mean(r)), float(np.mean(c)), int(mass_idx.size)

def ring_order_from_angles(orbit_bool: np.ndarray, H: int, W: int, center_r: float, center_c: float):
    ring_idx = np.flatnonzero(orbit_bool)
    if ring_idx.size == 0:
        die("orbit_mask has zero true entries")
    rr, cc = idx_to_rc(ring_idx, W)
    ang = np.arctan2(rr - center_r, cc - center_c)  # LABELS ONLY
    order = np.argsort(ang, kind="mergesort")
    ring_sorted = ring_idx[order]
    pos = np.full(H * W, -1, dtype=np.int64)
    pos[ring_sorted] = np.arange(ring_sorted.size, dtype=np.int64)
    return ring_sorted, pos

def build_reverse_adj(src: np.ndarray, dst: np.ndarray, N: int):
    # reversed adjacency: v <- u (so reverse edges are dst->src)
    rev = [[] for _ in range(N)]
    for u, v in zip(src.tolist(), dst.tolist()):
        rev[v].append(u)
    return rev

def directed_hops_to_mass(src: np.ndarray, dst: np.ndarray, mass_bool: np.ndarray, N: int):
    # dist_to_mass[x] = shortest directed hops from x to ANY mass node
    # computed by BFS on reversed graph starting from mass nodes.
    rev = build_reverse_adj(src, dst, N)
    dist = np.full(N, np.inf, dtype=np.float64)

    q = deque()
    mass_idx = np.flatnonzero(mass_bool)
    for m in mass_idx.tolist():
        dist[m] = 0.0
        q.append(m)

    while q:
        v = q.popleft()
        dv = dist[v]
        for u in rev[v]:
            if dist[u] == np.inf:
                dist[u] = dv + 1.0
                q.append(u)
    return dist

def main():
    ap = argparse.ArgumentParser(description="STRICT PP off-diagonal ring momentum-flux v2 (signed smallest-arc tangential classifier).")
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--mass_mask", required=True)
    ap.add_argument("--orbit_mask", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)

    # kept for CLI compatibility; v2 does NOT use window gating by default
    ap.add_argument("--step_window", type=int, default=0,
                    help="Compatibility only. v2 uses signed smallest-arc tangential jumps over the full ring; this does not gate edges.")
    ap.add_argument("--min_edges_used", type=int, default=1000)
    ap.add_argument("--min_edges_per_class", type=int, default=200)
    ap.add_argument("--deltaJ_threshold", type=float, default=0.01)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    H, W = args.H, args.W
    N = H * W

    mass_bool = load_mask_any(args.mass_mask, H, W, "mass_mask")
    orbit_bool = load_mask_any(args.orbit_mask, H, W, "orbit_mask")

    src, dst, w = load_edges_any(args.edges_curved, N)

    # centers + ring ordering (LABELS ONLY)
    center_r, center_c, n_mass = mass_center_from_mask(mass_bool, H, W)
    ring_sorted, ring_pos = ring_order_from_angles(orbit_bool, H, W, center_r, center_c)
    K = int(ring_sorted.size)

    # Markov hop distance to mass (STRICT PP distance proxy)
    dist_to_mass = directed_hops_to_mass(src, dst, mass_bool, N)

    # Tangential classification: signed smallest-arc jump along ring ordering
    cw_count = 0
    ccw_count = 0
    cw_w = 0.0
    ccw_w = 0.0
    tangential_used = 0

    # Optional: radial bookkeeping (not gating here, but useful for later mapping)
    rad_in_count = 0
    rad_out_count = 0
    rad_zero_count = 0

    for u, v, ww in zip(src, dst, w):
        pu = ring_pos[u]
        if pu < 0:
            continue
        pv = ring_pos[v]
        if pv < 0:
            continue

        # tangential (ring-to-ring) edge
        du = int(pv - pu)
        du %= K
        if du > K // 2:
            du -= K  # now in (-K/2, K/2]
        if du == 0:
            continue

        tangential_used += 1
        if du > 0:
            cw_count += 1
            cw_w += float(ww)
        else:
            ccw_count += 1
            ccw_w += float(ww)

        # radial sign wrt hop-distance-to-mass (if both finite)
        du_mass = dist_to_mass[v] - dist_to_mass[u]
        if not np.isfinite(du_mass):
            continue
        if du_mass < 0:
            rad_in_count += 1
        elif du_mass > 0:
            rad_out_count += 1
        else:
            rad_zero_count += 1

    out = {
        "H": H, "W": W, "N": N,
        "edges_curved": args.edges_curved,
        "mass_mask": args.mass_mask,
        "orbit_mask": args.orbit_mask,
        "notes": (
            "STRICT PP off-diagonal ring momentum-flux v2. "
            "Tangential edges are all ring-to-ring transitions, classified by signed smallest-arc jump "
            "in the ring ordering (angles are LABELS ONLY). "
            "No PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        ),
        "source_meta": {
            "mass_center_row_label": center_r,
            "mass_center_col_label": center_c,
            "n_mass_nodes": n_mass,
            "n_orbit_nodes": K,
            "step_window_arg_ignored": int(args.step_window),
        },
        "counts": {
            "tangential_edges_used": int(tangential_used),
            "cw_count": int(cw_count),
            "ccw_count": int(ccw_count),
            "rad_in_count": int(rad_in_count),
            "rad_out_count": int(rad_out_count),
            "rad_zero_count": int(rad_zero_count),
        },
        "weights": {
            "cw_w": float(cw_w),
            "ccw_w": float(ccw_w),
        },
        "thresholds": {
            "min_edges_used": int(args.min_edges_used),
            "min_edges_per_class": int(args.min_edges_per_class),
            "deltaJ_threshold": float(args.deltaJ_threshold),
        }
    }

    # Applicability gates
    if tangential_used < args.min_edges_used:
        out["APPLICABLE"] = False
        out["reason"] = f"Too few ring-to-ring tangential edges used: {tangential_used} < min_edges_used={args.min_edges_used}"
    elif cw_count < args.min_edges_per_class or ccw_count < args.min_edges_per_class:
        out["APPLICABLE"] = False
        out["reason"] = (
            f"Too few edges per class: cw={cw_count}, ccw={ccw_count} "
            f"< min_edges_per_class={args.min_edges_per_class}"
        )
    else:
        # Tangential momentum-flux proxy (dimensionless)
        denom = (cw_w + ccw_w)
        if denom <= 0:
            out["APPLICABLE"] = False
            out["reason"] = "Nonpositive total tangential weight; cannot form deltaJ"
        else:
            deltaJ = (cw_w - ccw_w) / denom
            out["APPLICABLE"] = True
            out["deltaJ"] = float(deltaJ)
            out["PASS_offdiag_ring_momentum_flux_PP_v2"] = bool(abs(deltaJ) >= args.deltaJ_threshold)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("APPLICABLE =", out.get("APPLICABLE"))
    if out.get("APPLICABLE"):
        print("deltaJ =", out.get("deltaJ"), "PASS =", out.get("PASS_offdiag_ring_momentum_flux_PP_v2"))
        print("tangential_edges_used =", out["counts"]["tangential_edges_used"],
              "cw =", out["counts"]["cw_count"], "ccw =", out["counts"]["ccw_count"])
    else:
        print("reason:", out.get("reason"))

if __name__ == "__main__":
    main()

