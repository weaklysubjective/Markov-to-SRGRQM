#!/usr/bin/env python3
"""
PP_Shapiro_markov_tau_v1.py

STRICT PP Shapiro-like test from Markov + Doyle commute time.

Inputs (you choose the pairs = what "through" vs "around" means):
  --edges_flat    : flat CA/Markov graph (e.g. edges_ca_v3_40x40.txt)
  --edges_curved  : curved mass graph (e.g. edges_ca_v3_mass_ms080_40x40.txt)
  --H, --W        : grid dims (e.g. 40, 40; used only for sanity, N = H*W)
  --src_through   : node index for source of "through-mass" pair
  --dst_through   : node index for destination of "through-mass" pair
  --src_around    : node index for source of "around-mass" pair
  --dst_around    : node index for destination of "around-mass" pair
  --output        : JSON output

What it does (per graph: flat, curved):
  * Builds a symmetric weighted graph from the directed edges
    (standard Doyle/Steiner convention: conductances = symmetric weights).
  * Builds Laplacian L and its Moore–Penrose pseudoinverse L^+.
  * Volume vol = sum of all (symmetric) edge weights.
  * Commute time τ(u,v) = vol * (e_u - e_v)^T L^+ (e_u - e_v).

Then computes:
  - τ_flat_through,  τ_curved_through
  - τ_flat_around,   τ_curved_around

And Shapiro-like deltas:
  - dτ_through = τ_curved_through - τ_flat_through
  - dτ_around  = τ_curved_around  - τ_flat_around

PASS criterion (very simple, no GR fit):
  - PASS_Shapiro_markov_tau = (dτ_through > 0) and (|dτ_through| >= |dτ_around|)

No PDE, no GR formula, no regression — pure graph/Doyle math on your Markov edges.
"""

import argparse
import json
import sys

import numpy as np


def load_edges(path):
    """
    Load edges from text file.
    Expected columns:
      s d w   (src, dst, weight)
    or:
      s d     (weight assumed 1.0)
    Node indices are assumed 0-based.
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] == 2:
        src = data[:, 0].astype(int)
        dst = data[:, 1].astype(int)
        w = np.ones_like(src, dtype=float)
    elif data.shape[1] == 3:
        src = data[:, 0].astype(int)
        dst = data[:, 1].astype(int)
        w = data[:, 2].astype(float)
    else:
        raise ValueError(f"Unexpected edge format in {path}, shape={data.shape}")
    return src, dst, w


def build_symmetric_laplacian(N, src, dst, w):
    """
    Build symmetric Laplacian for Doyle/Steiner commute time.

    We interpret weights as conductances on an undirected graph:
      A[i,j] = sum of weights on i->j and j->i.
    Then Laplacian L:
      L[i,i] = sum_j A[i,j]
      L[i,j] = -A[i,j] for i != j
    """
    A = np.zeros((N, N), dtype=float)

    # Accumulate directed weights into symmetric adjacency
    for s, d, wt in zip(src, dst, w):
        if s < 0 or s >= N or d < 0 or d >= N:
            raise ValueError(f"Edge index out of range: {s}->{d} (N={N})")
        if wt <= 0:
            # We allow zero or negative weights? For PP we assume non-negative conductances.
            # If negative, you should clean edges upstream.
            raise ValueError(f"Non-positive weight {wt} for edge {s}->{d}")
        A[s, d] += wt
        A[d, s] += wt

    # Laplacian
    L = np.zeros((N, N), dtype=float)
    row_sums = A.sum(axis=1)
    for i in range(N):
        L[i, i] = row_sums[i]
    # Off-diagonals
    L -= A

    # Graph volume (sum of all symmetric weights)
    vol = A.sum()

    return L, vol


def commute_time_from_laplacian(L_pinv, vol, u, v):
    """
    Commute time τ(u,v) = vol * (e_u - e_v)^T L^+ (e_u - e_v).
    L_pinv is the Moore–Penrose pseudoinverse of the Laplacian.
    """
    N = L_pinv.shape[0]
    if not (0 <= u < N and 0 <= v < N):
        raise ValueError(f"u={u}, v={v} out of range for N={N}")
    if u == v:
        return 0.0
    e = np.zeros(N, dtype=float)
    e[u] = 1.0
    e[v] = -1.0
    # L^+ e
    Le = L_pinv @ e
    # e^T L^+ e
    eff_res = float(e @ Le)
    tau = vol * eff_res
    return tau


def compute_commute_pairset(edges_path, N, src_through, dst_through, src_around, dst_around):
    """
    For one graph (flat or curved):
      - load edges
      - build symmetric Laplacian
      - compute L^+
      - compute τ_through and τ_around
    """
    src, dst, w = load_edges(edges_path)
    if src.size == 0:
        raise ValueError(f"No edges in {edges_path}")

    L, vol = build_symmetric_laplacian(N, src, dst, w)

    # Moore–Penrose pseudoinverse (Doyle commute time)
    # N is small (e.g. 1600), so this is feasible.
    L_pinv = np.linalg.pinv(L)

    tau_through = commute_time_from_laplacian(L_pinv, vol, src_through, dst_through)
    tau_around = commute_time_from_laplacian(L_pinv, vol, src_around, dst_around)

    return {
        "edges_file": edges_path,
        "N": N,
        "vol": float(vol),
        "tau_through": float(tau_through),
        "tau_around": float(tau_around),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Strict-PP Shapiro-like test from Markov + Doyle commute time."
    )
    parser.add_argument(
        "--edges_flat",
        required=True,
        help="Flat Markov edges file (e.g. edges_ca_v3_40x40.txt).",
    )
    parser.add_argument(
        "--edges_curved",
        required=True,
        help="Curved mass Markov edges file (e.g. edges_ca_v3_mass_ms080_40x40.txt).",
    )
    parser.add_argument("--H", type=int, required=True, help="Grid height (e.g. 40).")
    parser.add_argument("--W", type=int, required=True, help="Grid width (e.g. 40).")

    parser.add_argument(
        "--src_through",
        type=int,
        required=True,
        help="Source node index for 'through-mass' pair.",
    )
    parser.add_argument(
        "--dst_through",
        type=int,
        required=True,
        help="Destination node index for 'through-mass' pair.",
    )
    parser.add_argument(
        "--src_around",
        type=int,
        required=True,
        help="Source node index for 'around-mass' pair.",
    )
    parser.add_argument(
        "--dst_around",
        type=int,
        required=True,
        help="Destination node index for 'around-mass' pair.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for Shapiro Markov τ evidence.",
    )

    args = parser.parse_args()

    H = args.H
    W = args.W
    N = H * W

    # Basic checks
    for name, val in [
        ("src_through", args.src_through),
        ("dst_through", args.dst_through),
        ("src_around", args.src_around),
        ("dst_around", args.dst_around),
    ]:
        if not (0 <= val < N):
            raise ValueError(f"{name}={val} out of range for N={N}")

    # Flat case
    flat_res = compute_commute_pairset(
        args.edges_flat,
        N,
        args.src_through,
        args.dst_through,
        args.src_around,
        args.dst_around,
    )

    # Curved case
    curved_res = compute_commute_pairset(
        args.edges_curved,
        N,
        args.src_through,
        args.dst_through,
        args.src_around,
        args.dst_around,
    )

    tau_flat_through = flat_res["tau_through"]
    tau_flat_around = flat_res["tau_around"]
    tau_curved_through = curved_res["tau_through"]
    tau_curved_around = curved_res["tau_around"]

    d_tau_through = tau_curved_through - tau_flat_through
    d_tau_around = tau_curved_around - tau_flat_around

    # Simple Shapiro-like PASS: curved has extra delay on the "through" pair,
    # and that excess is at least as large as any change on the "around" pair.
    PASS_Shapiro = (d_tau_through > 0.0) and (abs(d_tau_through) >= abs(d_tau_around))

    report = {
        "H": H,
        "W": W,
        "N": N,
        "edges_flat": args.edges_flat,
        "edges_curved": args.edges_curved,
        "src_through": int(args.src_through),
        "dst_through": int(args.dst_through),
        "src_around": int(args.src_around),
        "dst_around": int(args.dst_around),
        "flat": flat_res,
        "curved": curved_res,
        "d_tau_through": float(d_tau_through),
        "d_tau_around": float(d_tau_around),
        "PASS_Shapiro_markov_tau": bool(PASS_Shapiro),
        "notes": (
            "Shapiro-like delay from Markov + Doyle commute time. "
            "Commute times computed via τ = vol * (e_u - e_v)^T L^+ (e_u - e_v). "
            "No PDE, no GR ansatz, no regression; 'through' vs 'around' semantics "
            "are entirely defined by the chosen node pairs."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote Shapiro Markov τ report to {args.output}")
    if PASS_Shapiro:
        print("PASS_Shapiro_markov_tau = TRUE")
    else:
        print("PASS_Shapiro_markov_tau = FALSE")


if __name__ == "__main__":
    main()

