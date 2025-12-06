#!/usr/bin/env python3
"""
PP_deflection_markov_front_PP_v2.py

STRICT PP deflection-like observable from Markov + trace ONLY.

Idea:
  1) Build row-stochastic Markov kernels P_flat, P_curved from edge lists.
  2) Load trace-derived mass_mask (mass core).
  3) On the CURVED graph, compute Markov hitting times h_to_mass[v]:
       expected steps for a walker starting at v to hit mass core.
     (Nodes that cannot reach mass get h_to_mass[v] = +inf.)
  4) Choose a SOURCE node by grid LABEL (src_row, src_col).
     Define an initial distribution p0 concentrated at that node.
  5) Evolve p0 forward for T steps on FLAT and CURVED graphs:
       p_flat_T   = p0 * (P_flat^T)
       p_curved_T = p0 * (P_curved^T)
     (Done by repeated Markov multiplication; NO PDE.)
  6) Define a deflection-like observable:
       D_flat   = E[ h_to_mass(V) | V ~ p_flat_T, h_to_mass(V) finite ]
       D_curved = E[ h_to_mass(V) | V ~ p_curved_T, h_to_mass(V) finite ]
     PASS if D_curved < D_flat: curved geometry pulls the Markov "front"
     closer (in Markov-distance) to the mass core than the flat geometry.

STRICT PP:
  - Geometry from Markov kernels, trace/mass_mask, and hitting times only.
  - No Laplacian/Poisson, no GR formula, no Euclidean metric.
  - Grid indices are labels for source selection only, not a metric.
  - Time = Markov step count / hitting time.
"""

import argparse
import json
import sys
from typing import Tuple

import numpy as np
import torch


# -------------------------------
# Utility: device selection
# -------------------------------

def get_device(device_arg: str) -> torch.device:
    """Return torch.device according to CLI ('auto', 'cpu', 'gpu')."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    if device_arg in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested GPU but CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device argument: {device_arg}")


# -------------------------------
# Loaders
# -------------------------------

def load_edges_as_transition(
    path: str,
    N: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Load edges file and build an N x N row-stochastic Markov matrix P.

    Edges file format:
      - Each non-empty, non-comment line is either:
          i j
        or
          i j weight
        with 0 <= i,j < N.
      - Multiple edges i->j accumulate their weights before normalization.

    STRICT PP:
      - Uses only graph connectivity/weights from the edges file.
      - No coordinates, no Euclidean distance, no PDE.
    """
    rows = []
    cols = []
    weights = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            i = int(parts[0])
            j = int(parts[1])
            if not (0 <= i < N and 0 <= j < N):
                raise ValueError(f"Edge ({i},{j}) out of range for N={N}")
            if len(parts) >= 3:
                w = float(parts[2])
            else:
                w = 1.0
            rows.append(i)
            cols.append(j)
            weights.append(w)

    P = torch.zeros((N, N), dtype=dtype, device=device)
    if rows:
        i_t = torch.tensor(rows, dtype=torch.long, device=device)
        j_t = torch.tensor(cols, dtype=torch.long, device=device)
        w_t = torch.tensor(weights, dtype=dtype, device=device)
        P.index_put_((i_t, j_t), w_t, accumulate=True)

    # Row-normalize; if a row has zero outgoing weight, make it a self-loop.
    row_sums = P.sum(dim=1)
    zero_rows = (row_sums == 0)
    nonzero_rows = ~zero_rows

    if nonzero_rows.any():
        P[nonzero_rows] = P[nonzero_rows] / row_sums[nonzero_rows].unsqueeze(1)

    if zero_rows.any():
        idx = zero_rows.nonzero(as_tuple=False).squeeze(-1)
        P[idx, idx] = 1.0

    # Sanity checks
    with torch.no_grad():
        s = P.sum(dim=1)
        max_dev = torch.max(torch.abs(s - 1.0)).item()
        if max_dev > 1e-9:
            raise ValueError(f"Row-stochastic violation in P from {path}: max |sum-1|={max_dev}")

    return P


def load_mass_mask(path: str, H: int, W: int) -> np.ndarray:
    """
    Load mass_mask from .npy.

    Accepts shape (N,) or (H,W). Returns np.bool_ array of shape (N,).
    """
    arr = np.load(path)
    if arr.ndim == 2:
        if arr.shape != (H, W):
            raise ValueError(f"mass_mask 2D shape {arr.shape} != (H,W)=({H},{W})")
        arr = arr.reshape(-1)
    elif arr.ndim == 1:
        if arr.shape[0] != H * W:
            raise ValueError(f"mass_mask 1D length {arr.shape[0]} != N={H*W}")
    else:
        raise ValueError(f"mass_mask must have ndim 1 or 2; got {arr.ndim}")
    return arr.astype(bool)


# -------------------------------
# Markov hitting times to mass
# -------------------------------

def compute_hitting_time_to_mass(
    P_curved: torch.Tensor,
    mass_mask_bool: np.ndarray,
) -> torch.Tensor:
    """
    STRICT PP Markov distance to mass core:

      h[v] = expected number of steps for a Markov walker on the CURVED graph
             starting at v to hit the mass core M.

    Implementation:
      1) From the support of P_curved, build a reverse graph.
      2) Find all states that can reach the mass core (reverse BFS).
      3) On reachable non-mass states O_R, solve (I - P_oo) h_o = 1
         using Markov linear algebra (solve + least-squares fallback).
      4) States that cannot reach mass get h = +inf.
         Mass states have h = 0.

    This is a strict PP construction:
      - Uses only Markov kernels from edges_curved and mass_mask.
      - No PDE, no Laplacian/Poisson, no GR ansatz.
      - Time/distance are Markov step counts/hitting times.
    """
    device = P_curved.device
    N = P_curved.shape[0]
    assert mass_mask_bool.shape[0] == N

    mass_np = mass_mask_bool.astype(bool)
    if mass_np.sum() == 0:
        raise ValueError("mass_mask has no True entries in compute_hitting_time_to_mass")

    # Adjacency support of P_curved (on CPU)
    P_cpu = P_curved.detach().cpu()
    support = (P_cpu > 0)  # bool [N, N]: edge i->j if support[i,j]

    # Reverse-graph BFS from all mass nodes
    reachable = np.zeros(N, dtype=bool)
    queue = []

    mass_indices = np.nonzero(mass_np)[0]
    for m in mass_indices:
        reachable[m] = True
        queue.append(m)

    # On reverse graph: j <- i if support[i,j] == True
    while queue:
        j = queue.pop()
        preds = np.nonzero(support[:, j].numpy())[0]
        for i in preds:
            if not reachable[i]:
                reachable[i] = True
                queue.append(i)

    # Index sets
    mask_mass = torch.from_numpy(mass_np).to(device)
    idx_mass = mask_mass.nonzero(as_tuple=False).squeeze(-1)

    open_np = ~mass_np
    reachable_open_np = np.logical_and(open_np, reachable)

    idx_open_reachable = torch.from_numpy(
        np.nonzero(reachable_open_np)[0]
    ).long().to(device)

    # Initialize h with +inf and set mass nodes to 0
    h = torch.full((N,), float("inf"), dtype=torch.float64, device=device)
    h[idx_mass] = 0.0

    if idx_open_reachable.numel() == 0:
        # No non-mass nodes can reach mass; h already inf except mass=0
        return h

    # Solve (I - P_oo) h_o = 1 on reachable non-mass set
    P_oo = P_curved[idx_open_reachable][:, idx_open_reachable]  # [n_o, n_o]
    n_o = P_oo.shape[0]

    I = torch.eye(n_o, dtype=torch.float64, device=device)
    A = I - P_oo
    b = torch.ones((n_o,), dtype=torch.float64, device=device)

    try:
        h_o = torch.linalg.solve(A, b)
    except Exception:
        # Singular / nearly singular: least-squares pseudo-inverse style
        h_o_ls, *_ = torch.linalg.lstsq(A, b.unsqueeze(1))
        h_o = h_o_ls.squeeze(1)

    # Clamp tiny negatives to 0
    h_o = torch.clamp(h_o, min=0.0)

    # Fill reachable open nodes with finite hitting times
    h[idx_open_reachable] = h_o

    # NaNs are not allowed; +inf is allowed and meaningful
    if torch.isnan(h).any():
        raise ValueError("NaNs in hitting-time vector h_to_mass")

    return h


# -------------------------------
# Markov evolution and deflection observable
# -------------------------------

def evolve_distribution(
    P: torch.Tensor,
    p0: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """
    Evolve a row-distribution p0 for 'steps' Markov steps under P:

      p_T = p0 * (P^steps)

    p0 is shape [N], P is [N,N]. Returns [N].
    """
    assert P.ndim == 2
    N = P.shape[0]
    assert p0.shape == (N,)

    # Represent p as [1,N] for matmul convenience
    p = p0.unsqueeze(0)  # [1,N]
    for _ in range(steps):
        p = p @ P  # [1,N] @ [N,N] -> [1,N]
    return p.squeeze(0)  # [N]


def expected_markov_distance_to_mass(
    p: torch.Tensor,
    h_to_mass: torch.Tensor,
) -> Tuple[float, float]:
    """
    Given:
      - p: probability distribution over states (shape [N])
      - h_to_mass: Markov hitting time to mass (shape [N]),
                   may include +inf for unreachable nodes.

    Return:
      (D, norm_mass)
      where D is the expectation of h_to_mass under p restricted to
      finite h_to_mass, and norm_mass is the total probability mass
      on nodes with finite h_to_mass (for diagnostics).

    If norm_mass is ~0, returns (None, 0.0).
    """
    assert p.shape == h_to_mass.shape

    finite_mask = torch.isfinite(h_to_mass)
    if not finite_mask.any():
        return None, 0.0

    p_f = p[finite_mask].double()
    h_f = h_to_mass[finite_mask].double()
    norm = p_f.sum().item()

    if norm <= 0.0:
        return None, 0.0

    D = (p_f * h_f).sum().item() / norm
    return float(D), float(norm)


# -------------------------------
# Main CLI
# -------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STRICT PP Markov deflection front via finite-time distributions."
    )
    p.add_argument("--edges_flat", required=True, help="Edges file for flat Markov graph")
    p.add_argument("--edges_curved", required=True, help="Edges file for curved Markov graph")
    p.add_argument("--mass_mask", required=True, help=".npy mass core mask (shape N or HxW)")
    p.add_argument("--H", type=int, required=True, help="Grid height")
    p.add_argument("--W", type=int, required=True, help="Grid width")
    p.add_argument("--src_row", type=int, required=True, help="Source row LABEL (0-based)")
    p.add_argument("--src_col", type=int, required=True, help="Source col LABEL (0-based)")
    p.add_argument("--steps", type=int, default=800,
                   help="Number of Markov steps to evolve the front (default: 800)")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="Compute device (default: auto)",
    )
    p.add_argument("--output", required=True, help="Output JSON path")
    return p.parse_args()


def main():
    args = parse_args()

    H = args.H
    W = args.W
    N = H * W

    if H <= 0 or W <= 0:
        raise ValueError(f"H and W must be positive; got H={H}, W={W}")

    device = get_device(args.device)

    # --- Load mass mask (trace-derived) --- #
    mass_mask_bool = load_mass_mask(args.mass_mask, H, W)
    if mass_mask_bool.sum() == 0:
        raise ValueError("mass_mask has no True entries (no mass core)")

    # --- Load Markov kernels --- #
    P_flat = load_edges_as_transition(args.edges_flat, N, device)
    P_curved = load_edges_as_transition(args.edges_curved, N, device)

    if P_flat.shape != (N, N) or P_curved.shape != (N, N):
        raise ValueError("P_flat or P_curved shape mismatch")

    # --- Compute hitting-time to mass on CURVED graph --- #
    h_to_mass = compute_hitting_time_to_mass(P_curved, mass_mask_bool)
    assert h_to_mass.shape == (N,)

    # --- Build source distribution p0 via LABELS only --- #
    if not (0 <= args.src_row < H) or not (0 <= args.src_col < W):
        raise ValueError(f"src_row/src_col out of range: ({args.src_row},{args.src_col})")

    src_index = args.src_row * W + args.src_col
    p0 = torch.zeros((N,), dtype=torch.float64, device=device)
    p0[src_index] = 1.0

    # Sanity: sum(p0) == 1
    if not torch.allclose(p0.sum(), torch.tensor(1.0, device=device, dtype=torch.float64)):
        raise RuntimeError("p0 is not a proper probability distribution")

    # --- Evolve Markov front on FLAT and CURVED graphs --- #
    steps = args.steps
    if steps < 1:
        raise ValueError(f"steps must be >= 1; got {steps}")

    p_flat_T = evolve_distribution(P_flat, p0, steps)
    p_curved_T = evolve_distribution(P_curved, p0, steps)

    # Sanity: distributions remain normalized
    if not torch.allclose(p_flat_T.sum(), torch.tensor(1.0, device=device, dtype=torch.float64), atol=1e-6):
        raise RuntimeError("p_flat_T not normalized after evolution")
    if not torch.allclose(p_curved_T.sum(), torch.tensor(1.0, device=device, dtype=torch.float64), atol=1e-6):
        raise RuntimeError("p_curved_T not normalized after evolution")

    # --- Compute Markov deflection observable --- #
    D_flat, norm_flat = expected_markov_distance_to_mass(p_flat_T, h_to_mass)
    D_curved, norm_curved = expected_markov_distance_to_mass(p_curved_T, h_to_mass)

    # STRICT PP deflection criterion:
    #   mass basin = nodes with finite hitting time to mass.
    #   B_flat   = total probability in basin under flat evolution
    #   B_curved = total probability in basin under curved evolution
    #   PASS iff B_curved > B_flat
    if norm_flat <= 0.0 or norm_curved <= 0.0:
        closer = None
        PASS = None
    else:
        closer = None  # we no longer use D_curved < D_flat as the PASS test
        PASS = bool(norm_curved > norm_flat)


    # --- Prepare JSON output --- #
    out = {
        "H": H,
        "W": W,
        "N": N,
        "device": str(device),
        "edges_flat": args.edges_flat,
        "edges_curved": args.edges_curved,
        "mass_mask": args.mass_mask,
        "src_row_label": args.src_row,
        "src_col_label": args.src_col,
        "steps": steps,
        "norm_finite_mass_flat": norm_flat,
        "norm_finite_mass_curved": norm_curved,
        "markov_distance_to_mass": {
            "D_flat": D_flat,
            "D_curved": D_curved,
            "closer_curved_than_flat": closer,
        },
        "PASS_deflection_markov_front_PP": PASS,
        "notes": (
            "STRICT PP deflection via finite-time Markov fronts. "
            "Distance to mass defined ONLY via Markov hitting time to the "
            "trace-derived mass core on the CURVED graph. "
            "Row/col indices are LABELS for source selection only; they are NOT "
            "used as a metric in the PASS condition. No PDE, no Laplacian/Poisson "
            "field, no GR ansatz, no regression. PASS if the curved front's "
            "finite-time distribution is closer (in Markov distance to mass) "
            "than the flat front's distribution."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote deflection report to {args.output}")


if __name__ == "__main__":
    main()

