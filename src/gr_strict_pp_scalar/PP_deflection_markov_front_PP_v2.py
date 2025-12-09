#!/usr/bin/env python3
"""
PP_deflection_markov_front_PP_v2.py  (sparse-safe, 512-ready)

STRICT PP deflection-like observable from Markov + trace ONLY.

Pipeline:
  1) Build row-stochastic Markov kernels P_flat, P_curved from edge lists (SPARSE).
  2) Load trace-derived mass_mask (mass core).
  3) On the CURVED graph, compute Markov hitting times h_to_mass[v]:
       expected steps for a walker starting at v to hit mass core.
     - Nodes that cannot reach mass get h_to_mass[v] = +inf.
     - Mass nodes have h_to_mass[v] = 0.
     - Implementation is sparse-safe:
         * build reverse adjacency from sparse indices
         * find reachable set
         * build small dense P_oo over reachable non-mass nodes
         * solve (I - P_oo) h = 1
  4) Choose a SOURCE node by grid LABEL (src_row, src_col).
     Define p0 concentrated at that node.
  5) Evolve p0 forward for T steps on FLAT and CURVED graphs:
       p_T = p0 * P^T
     Using sparse multiplication via P^T acting on a column vector.
  6) Define deflection-like observable:
       norm_finite_mass_* = total probability that lands in the mass-basin
                            (nodes with finite h_to_mass).
     PASS if norm_finite_mass_curved > norm_finite_mass_flat.

STRICT PP:
  - No Laplacian/Poisson, no PDE, no Euclidean metric.
  - Grid indices are labels only for choosing a source node.
  - Time/distance = Markov step counts / hitting times.
"""

import argparse
import json
from typing import Tuple, Optional

import numpy as np
import torch


# -------------------------------
# Utility: device selection
# -------------------------------

def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested GPU but CUDA/HIP is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device argument: {device_arg}")


# -------------------------------
# Loaders
# -------------------------------

def load_mass_mask(path: str, H: int, W: int) -> np.ndarray:
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


def load_edges_as_transition_sparse(
    edges_path: str,
    N: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build sparse row-stochastic transition matrix P from edgelist.
    Directed PP edges; no symmetrization.
    Adds self-loops for zero-outdegree rows to preserve probability mass.
    """
    rows = []
    cols = []
    vals = []

    with open(edges_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            s = int(parts[0]); t = int(parts[1])
            w = 1.0
            if len(parts) >= 3:
                try:
                    w = float(parts[2])
                except Exception:
                    w = 1.0
            rows.append(s); cols.append(t); vals.append(w)

    if not rows:
        raise ValueError(f"No edges parsed from {edges_path}")

    rows_t = torch.tensor(rows, device=device, dtype=torch.int64)
    cols_t = torch.tensor(cols, device=device, dtype=torch.int64)
    vals_t = torch.tensor(vals, device=device, dtype=dtype)

    # Row sums
    deg = torch.zeros((N,), device=device, dtype=dtype)
    deg.scatter_add_(0, rows_t, vals_t)

    # Normalize rows with outgoing edges
    deg_safe = torch.where(deg > 0, deg, torch.ones_like(deg))
    vals_norm = vals_t / deg_safe[rows_t]

    # Add self-loops for zero-outdegree rows
    zero_rows = (deg == 0).nonzero(as_tuple=False).squeeze(1)
    if zero_rows.numel() > 0:
        rows_extra = zero_rows
        cols_extra = zero_rows
        vals_extra = torch.ones((zero_rows.numel(),), device=device, dtype=dtype)

        rows_all = torch.cat([rows_t, rows_extra], dim=0)
        cols_all = torch.cat([cols_t, cols_extra], dim=0)
        vals_all = torch.cat([vals_norm, vals_extra], dim=0)
    else:
        rows_all, cols_all, vals_all = rows_t, cols_t, vals_norm

    idx = torch.stack([rows_all, cols_all], dim=0)
    P = torch.sparse_coo_tensor(idx, vals_all, (N, N), device=device, dtype=dtype).coalesce()
    return P

# -------------------------------
# Markov evolution (sparse-safe)
# -------------------------------

def evolve_distribution(P: torch.Tensor, p0: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Evolve row distribution: p_{t+1} = p_t P.
    Use column form: v_{t+1} = P^T v_t.
    """
    assert P.is_sparse
    N = P.shape[0]
    assert p0.shape == (N,)
    if steps < 1:
        raise ValueError("steps must be >= 1")

    v = p0
    PT = P.transpose(0, 1).coalesce()

    for _ in range(steps):
        v = torch.sparse.mm(PT, v.unsqueeze(1)).squeeze(1)

    return v


# -------------------------------
# Hitting times to mass (reachable-subgraph solve)
# -------------------------------

def compute_hitting_time_to_mass(
    P_curved: torch.Tensor,
    mass_mask_bool: np.ndarray,
) -> torch.Tensor:
    """
    Compute h[v] = expected steps to hit mass core on CURVED graph.

    Sparse-safe approach:
      1) Extract (rows, cols, vals) from sparse COO.
      2) Build reverse adjacency from edges.
      3) Reverse-BFS from mass to get reachable set.
      4) Build a SMALL dense P_oo over reachable non-mass nodes only,
         using the sparse edges restricted to this set.
      5) Solve (I - P_oo) h = 1.
    """
    if not P_curved.is_sparse:
        raise ValueError("P_curved must be sparse in STRICT PP deflection v2.")

    device = P_curved.device
    N = P_curved.shape[0]
    assert mass_mask_bool.shape[0] == N

    mass_np = mass_mask_bool.astype(bool)
    if mass_np.sum() == 0:
        raise ValueError("mass_mask has no True entries")

    # Pull sparse structure to CPU for graph traversal
    P_cpu = P_curved.detach().to("cpu").coalesce()
    idx = P_cpu.indices().numpy()
    vals = P_cpu.values().numpy()
    rows = idx[0]
    cols = idx[1]

    # Reverse adjacency list
    rev = [[] for _ in range(N)]
    for i, j in zip(rows, cols):
        rev[j].append(i)

    # Reverse BFS from mass nodes
    reachable = np.zeros(N, dtype=bool)
    mass_indices = np.nonzero(mass_np)[0].astype(int).tolist()
    stack = mass_indices[:]
    for m in mass_indices:
        reachable[m] = True

    while stack:
        j = stack.pop()
        for i in rev[j]:
            if not reachable[i]:
                reachable[i] = True
                stack.append(i)

    # Open (non-mass) reachable set
    open_np = ~mass_np
    reachable_open_np = np.logical_and(open_np, reachable)
    open_ids = np.nonzero(reachable_open_np)[0].astype(int)

    # Initialize full h
    h = torch.full((N,), float("inf"), dtype=torch.float64, device=device)
    if mass_indices:
        h[torch.tensor(mass_indices, device=device)] = 0.0

    if open_ids.size == 0:
        return h

    # Build dense P_oo on CPU for the reachable open subset
    # Map global node id -> local index
    pos = {nid: k for k, nid in enumerate(open_ids)}
    n_o = open_ids.size

    P_oo = np.zeros((n_o, n_o), dtype=np.float64)

    # Fill P_oo using sparse edges restricted to open_ids
    for i, j, w in zip(rows, cols, vals):
        pi = pos.get(int(i), None)
        pj = pos.get(int(j), None)
        if pi is not None and pj is not None:
            P_oo[pi, pj] += float(w)

    # Solve (I - P_oo) h = 1
    A = np.eye(n_o, dtype=np.float64) - P_oo
    b = np.ones((n_o,), dtype=np.float64)

    try:
        h_o = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Least-squares fallback
        h_o, *_ = np.linalg.lstsq(A, b, rcond=None)

    h_o = np.clip(h_o, 0.0, None)

    # Write back to torch on device
    open_ids_t = torch.tensor(open_ids, device=device, dtype=torch.long)
    h[open_ids_t] = torch.tensor(h_o, device=device, dtype=torch.float64)

    if torch.isnan(h).any():
        raise ValueError("NaNs in hitting time vector")

    return h


# -------------------------------
# Expectation diagnostics
# -------------------------------

def expected_markov_distance_to_mass(
    p: torch.Tensor,
    h_to_mass: torch.Tensor,
) -> Tuple[Optional[float], float]:
    """
    Returns:
      (D, norm)
    where:
      norm = total probability on finite-h nodes
      D    = E[h | finite], or None if norm==0
    """
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
        description="STRICT PP Markov deflection front (sparse, scale-safe)."
    )
    p.add_argument("--edges_flat", required=True)
    p.add_argument("--edges_curved", required=True)
    p.add_argument("--mass_mask", required=True)
    p.add_argument("--H", type=int, required=True)
    p.add_argument("--W", type=int, required=True)
    p.add_argument("--src_row", type=int, required=True)
    p.add_argument("--src_col", type=int, required=True)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda"],
    )
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    H, W = args.H, args.W
    N = H * W

    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")

    if not (0 <= args.src_row < H) or not (0 <= args.src_col < W):
        raise ValueError("src_row/src_col out of range")

    device = get_device(args.device)

    # Mass mask
    mass_mask_bool = load_mass_mask(args.mass_mask, H, W)
    if mass_mask_bool.sum() == 0:
        raise ValueError("mass_mask has no True entries")

    # Sparse Markov kernels
    P_flat = load_edges_as_transition_sparse(args.edges_flat, N, device, dtype=torch.float64)
    P_curved = load_edges_as_transition_sparse(args.edges_curved, N, device, dtype=torch.float64)

    # Hitting times on CURVED graph
    h_to_mass = compute_hitting_time_to_mass(P_curved, mass_mask_bool)

    # Source distribution
    src_index = args.src_row * W + args.src_col
    p0 = torch.zeros((N,), dtype=torch.float64, device=device)
    p0[src_index] = 1.0

    # Evolve fronts
    steps = args.steps
    p_flat_T = evolve_distribution(P_flat, p0, steps)
    p_curved_T = evolve_distribution(P_curved, p0, steps)

    # Normalize checks (row-stochastic => should stay normalized)
    if not torch.allclose(p_flat_T.sum(), torch.tensor(1.0, device=device, dtype=torch.float64), atol=1e-6):
        raise RuntimeError("p_flat_T not normalized")
    if not torch.allclose(p_curved_T.sum(), torch.tensor(1.0, device=device, dtype=torch.float64), atol=1e-6):
        raise RuntimeError("p_curved_T not normalized")

    # Diagnostics
    D_flat, norm_flat = expected_markov_distance_to_mass(p_flat_T, h_to_mass)
    D_curved, norm_curved = expected_markov_distance_to_mass(p_curved_T, h_to_mass)

    # STRICT PP PASS criterion (basin mass)
    if norm_flat <= 0.0 or norm_curved <= 0.0:
        PASS = None
    else:
        PASS = bool(norm_curved > norm_flat)

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
            "closer_curved_than_flat": None
        },
        "PASS_deflection_markov_front_PP": PASS,
        "notes": (
            "STRICT PP deflection via finite-time Markov fronts (SPARSE, scale-safe). "
            "Mass basin defined ONLY by finite Markov hitting time to the trace-derived "
            "mass core on the CURVED graph. "
            "PASS condition uses basin-mass shift: PASS iff norm_finite_mass_curved > "
            "norm_finite_mass_flat. "
            "No Laplacian/Poisson, no PDE, no Euclidean metric assumptions, no GR ansatz, "
            "no regression."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote deflection report to {args.output}")
    print(f"PASS_deflection_markov_front_PP = {PASS}")


if __name__ == "__main__":
    main()

