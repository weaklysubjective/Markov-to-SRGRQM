#!/usr/bin/env python3
"""
PP_offdiag_ring_cov_toy_v1.py

STRICT PP 2D+1 off-diagonal toy observable on an orbit ring.

Goal:
  On a given curved geometry (edges + mass_mask + orbit_mask),
  measure the covariance between radial and tangential displacement
  of Markov edges that start on the orbit ring.

Inputs:
  - Curved edges:       edges_ca_v3_*_512x512_PPV1.txt   (PPV1 style)
  - Mass mask:          PP_mass_mask_512x512_*_PPV1.npy
  - Orbit ring mask:    PP_orbit_ring_mask_512x512_*_PPV1.npy
  - H, W:               grid dimensions

STRICT PP:
  - Uses only:
      * PPV1 edges (Markov transitions),
      * mass mask,
      * orbit mask,
      * grid indices as labels.
  - No PDE, no Laplacian/Poisson, no GR ansatz, no regression.
  - "Radius" / "angle" are label-based and only used to define local
    radial/tangential directions and bin edges.

Observable (toy):
  - For each ring node i, take all outgoing edges i -> j.
  - If j is also on the ring, compute displacement in label space:
      d = (dr, dc) = (row_j - row_i, col_j - col_i).
  - Define a local radial unit vector r_hat from mass center to node i,
    and a tangential unit vector t_hat (perpendicular).
  - Decompose d into (d_r, d_t) via dot products in label space.
  - Aggregate (d_r, d_t) over all ring edges and compute:
      Var_r, Var_t, Cov_rt, corr_rt.

This is a diagnostic 2x2 covariance "metric-like" object on the ring.
It does NOT define a full EFE mapping; it just exposes off-diagonal
structure between radial and tangential flow.

CLI example (strong_pf010, 512×512 PPV1):

  python src/gr_strict_pp_vector/PP_offdiag_ring_cov_toy_v1.py \\
    --edges_curved edges_ca_v3_strong_pf010_512x512_PPV1.txt \\
    --mass_mask   PP_mass_mask_512x512_strong_pf010_PPV1.npy \\
    --orbit_mask  PP_orbit_ring_mask_512x512_strong_pf010_PPV1.npy \\
    --H 512 --W 512 \\
    --min_edges_used 1000 \\
    --corr_small_threshold 0.1 \\
    --output src/gr_strict_pp_vector/PP_offdiag_ring_cov_512_strong_pf010_toy_v1.json

"""

import argparse
import json
import numpy as np
from typing import Tuple


def load_edges_ppv1(path: str, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PPV1-style edges from text file.

    Accepted formats per line:
      - "src dst"
      - "src dst w"

    Returns:
      src, dst as int64 arrays of length M.

    STRICT PP sanity:
      - 0 <= src, dst < N
    """
    src = []
    dst = []
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
                raise ValueError(f"Non-integer indices in {path!r}: {line!r}")
            src.append(i)
            dst.append(j)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)

    # Sanity checks
    assert src.shape == dst.shape
    assert src.ndim == 1
    if src.size == 0:
        raise ValueError(f"No edges loaded from {path!r}")
    if src.min() < 0 or dst.min() < 0 or src.max() >= N or dst.max() >= N:
        raise ValueError(
            f"Edge indices out of range for N={N}: "
            f"src range [{src.min()}, {src.max()}], "
            f"dst range [{dst.min()}, {dst.max()}]"
        )
    return src, dst


def idx_to_rc(idx: np.ndarray, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert flat indices to (row, col) using row-major convention:
      idx = row * W + col

    Returns:
      rows, cols as int64 arrays.
    """
    idx = np.asarray(idx, dtype=np.int64)
    rows = idx // W
    cols = idx % W
    return rows, cols


def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP 2D+1 off-diagonal toy on an orbit ring "
                    "(radial/tangential covariance)."
    )
    ap.add_argument("--edges_curved", required=True,
                    help="Curved PPV1 edges file (txt).")
    ap.add_argument("--mass_mask", required=True,
                    help="Mass mask .npy (bool or 0/1).")
    ap.add_argument("--orbit_mask", required=True,
                    help="Orbit ring mask .npy (bool or 0/1).")
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--min_edges_used", type=int, default=1000,
                    help="Minimum number of ring edges required for applicability.")
    ap.add_argument("--corr_small_threshold", type=float, default=0.1,
                    help="Threshold for |corr_rt| considered 'small off-diagonal' "
                         "in the toy PASS flag.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    H = int(args.H)
    W = int(args.W)
    N = H * W

    # Load masks
    mass_mask = np.load(args.mass_mask)
    orbit_mask = np.load(args.orbit_mask)

    assert mass_mask.shape == (H, W), (
        f"mass_mask shape {mass_mask.shape} != ({H}, {W})"
    )
    assert orbit_mask.shape == (H, W), (
        f"orbit_mask shape {orbit_mask.shape} != ({H}, {W})"
    )

    mass_mask = mass_mask.astype(bool)
    orbit_mask = orbit_mask.astype(bool)

    # Sanity: non-empty mass and orbit sets
    mass_indices = np.flatnonzero(mass_mask)
    orbit_indices = np.flatnonzero(orbit_mask)
    if mass_indices.size == 0:
        raise ValueError("mass_mask has no True entries.")
    if orbit_indices.size == 0:
        raise ValueError("orbit_mask has no True entries.")

    # Mass center in label space (row/col means)
    mass_rows, mass_cols = idx_to_rc(mass_indices, W)
    mass_center_row = float(mass_rows.mean())
    mass_center_col = float(mass_cols.mean())

    # Load curved edges
    src, dst = load_edges_ppv1(args.edges_curved, N)

    # Restrict to edges starting on the orbit ring and ending also on ring
    ring_mask_flat = orbit_mask.reshape(-1)
    on_ring_src = ring_mask_flat[src]
    on_ring_dst = ring_mask_flat[dst]
    keep = on_ring_src & on_ring_dst

    if not np.any(keep):
        out = {
            "H": H,
            "W": W,
            "N": N,
            "edges_curved": args.edges_curved,
            "mass_mask": args.mass_mask,
            "orbit_mask": args.orbit_mask,
            "min_edges_used": int(args.min_edges_used),
            "corr_small_threshold": float(args.corr_small_threshold),
            "APPLICABLE": False,
            "reason": "No edges with both endpoints on orbit ring.",
            "n_edges_total": int(src.size),
            "n_edges_ring_ring": 0,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print("WROTE", args.output)
        print("APPLICABLE = False (no ring-ring edges)")
        return

    src_ring = src[keep]
    dst_ring = dst[keep]
    n_ring_edges = int(src_ring.size)

    if n_ring_edges < args.min_edges_used:
        out = {
            "H": H,
            "W": W,
            "N": N,
            "edges_curved": args.edges_curved,
            "mass_mask": args.mass_mask,
            "orbit_mask": args.orbit_mask,
            "min_edges_used": int(args.min_edges_used),
            "corr_small_threshold": float(args.corr_small_threshold),
            "APPLICABLE": False,
            "reason": (
                f"Insufficient ring-ring edges: {n_ring_edges} < min_edges_used."
            ),
            "n_edges_total": int(src.size),
            "n_edges_ring_ring": n_ring_edges,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print("WROTE", args.output)
        print("APPLICABLE = False (insufficient ring-ring edges)")
        return

    # Compute row/col for ring edges
    src_rows, src_cols = idx_to_rc(src_ring, W)
    dst_rows, dst_cols = idx_to_rc(dst_ring, W)

    # Displacement in label space
    d_rows = dst_rows - src_rows
    d_cols = dst_cols - src_cols

    # Local radial/tangential basis at each source node
    # Row is "y", col is "x".
    dy_center = src_rows.astype(float) - mass_center_row
    dx_center = src_cols.astype(float) - mass_center_col
    radial_norm = np.sqrt(dx_center**2 + dy_center**2)

    # Guard against any zero-length radial vectors (unlikely on orbit ring)
    valid = radial_norm > 0
    if not np.any(valid):
        out = {
            "H": H,
            "W": W,
            "N": N,
            "edges_curved": args.edges_curved,
            "mass_mask": args.mass_mask,
            "orbit_mask": args.orbit_mask,
            "min_edges_used": int(args.min_edges_used),
            "corr_small_threshold": float(args.corr_small_threshold),
            "APPLICABLE": False,
            "reason": "All orbit-ring nodes have zero radial norm (degenerate).",
            "n_edges_total": int(src.size),
            "n_edges_ring_ring": n_ring_edges,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print("WROTE", args.output)
        print("APPLICABLE = False (degenerate radial basis)")
        return

    # Filter to valid only
    src_rows = src_rows[valid]
    src_cols = src_cols[valid]
    d_rows = d_rows[valid]
    d_cols = d_cols[valid]
    dy_center = dy_center[valid]
    dx_center = dx_center[valid]
    radial_norm = radial_norm[valid]
    n_used = int(radial_norm.size)

    if n_used < args.min_edges_used:
        out = {
            "H": H,
            "W": W,
            "N": N,
            "edges_curved": args.edges_curved,
            "mass_mask": args.mass_mask,
            "orbit_mask": args.orbit_mask,
            "min_edges_used": int(args.min_edges_used),
            "corr_small_threshold": float(args.corr_small_threshold),
            "APPLICABLE": False,
            "reason": (
                f"Insufficient ring-ring edges after radial filtering: "
                f"{n_used} < min_edges_used."
            ),
            "n_edges_total": int(src.size),
            "n_edges_ring_ring": n_ring_edges,
            "n_edges_used": n_used,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print("WROTE", args.output)
        print("APPLICABLE = False (insufficient after filtering)")
        return

    # Radial unit vector r_hat = (dx, dy) / ||(dx, dy)||
    r_hat_x = dx_center / radial_norm
    r_hat_y = dy_center / radial_norm

    # Tangential unit vector t_hat: rotate r_hat by +90 degrees
    # In (x, y), a +90° rotation is (-y, x).
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x

    # Displacement in (x, y) coordinates (label space)
    # x ≡ col, y ≡ row
    disp_x = d_cols.astype(float)
    disp_y = d_rows.astype(float)

    # Project displacement onto radial and tangential directions
    d_r = disp_x * r_hat_x + disp_y * r_hat_y
    d_t = disp_x * t_hat_x + disp_y * t_hat_y

    assert d_r.shape == d_t.shape
    assert d_r.ndim == 1

    if d_r.size < args.min_edges_used:
        # This should be redundant, but keep as safety.
        out = {
            "H": H,
            "W": W,
            "N": N,
            "edges_curved": args.edges_curved,
            "mass_mask": args.mass_mask,
            "orbit_mask": args.orbit_mask,
            "min_edges_used": int(args.min_edges_used),
            "corr_small_threshold": float(args.corr_small_threshold),
            "APPLICABLE": False,
            "reason": (
                f"Insufficient (d_r, d_t) samples: {d_r.size} < min_edges_used."
            ),
            "n_edges_total": int(src.size),
            "n_edges_ring_ring": n_ring_edges,
            "n_edges_used": int(d_r.size),
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print("WROTE", args.output)
        print("APPLICABLE = False (insufficient samples)")
        return

    # Compute covariance / correlation
    d_r_mean = float(d_r.mean())
    d_t_mean = float(d_t.mean())

    d_r_centered = d_r - d_r_mean
    d_t_centered = d_t - d_t_mean

    var_r = float((d_r_centered ** 2).mean())
    var_t = float((d_t_centered ** 2).mean())
    cov_rt = float((d_r_centered * d_t_centered).mean())

    # Correlation coefficient, guard against zero variance
    if var_r > 0.0 and var_t > 0.0:
        corr_rt = float(
            cov_rt / (np.sqrt(var_r) * np.sqrt(var_t))
        )
    else:
        corr_rt = None

    # Toy PASS flag: "off-diagonal small" (approx diagonal covariance)
    if corr_rt is None:
        PASS_offdiag_toy_diagonal = None
    else:
        PASS_offdiag_toy_diagonal = bool(
            abs(corr_rt) < args.corr_small_threshold
        )

    out = {
        "H": H,
        "W": W,
        "N": N,
        "edges_curved": args.edges_curved,
        "mass_mask": args.mass_mask,
        "orbit_mask": args.orbit_mask,
        "min_edges_used": int(args.min_edges_used),
        "corr_small_threshold": float(args.corr_small_threshold),
        "APPLICABLE": True,
        "reason": None,
        "n_edges_total": int(src.size),
        "n_edges_ring_ring": n_ring_edges,
        "n_edges_used": int(d_r.size),
        "mass_center_row_label": mass_center_row,
        "mass_center_col_label": mass_center_col,
        "stats": {
            "d_r_mean": d_r_mean,
            "d_t_mean": d_t_mean,
            "var_r": var_r,
            "var_t": var_t,
            "cov_rt": cov_rt,
            "corr_rt": corr_rt,
            "PASS_offdiag_toy_diagonal": PASS_offdiag_toy_diagonal,
        },
        "notes": (
            "STRICT PP 2D+1 off-diagonal toy on orbit ring. "
            "Local basis (radial/tangential) is defined from mass center "
            "and grid labels only; displacement uses Markov edges restricted "
            "to ring->ring transitions. No PDE, no Laplacian/Poisson, "
            "no GR ansatz, no regression."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("APPLICABLE = True")
    print("n_edges_used =", d_r.size)
    print("var_r =", var_r, "var_t =", var_t, "cov_rt =", cov_rt, "corr_rt =", corr_rt)
    print("PASS_offdiag_toy_diagonal =", PASS_offdiag_toy_diagonal)


if __name__ == "__main__":
    main()

