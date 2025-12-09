#!/usr/bin/env python3
"""
PP_deflection_markov_front_PP_v3_sparse_only.py

STRICT PP deflection-like observable from Markov + trace ONLY (sparse-only, 512+ safe).

What this version fixes / guarantees:
  - Removes all dense NxN Markov construction paths.
  - Uses sparse COO row-stochastic P built directly from edge lists.
  - Evolves ROW distributions via consistent update: p_{t+1} = p_t @ P
      implemented as column update with P^T: v_{t+1} = P^T @ v_t.
  - Computes Markov hitting times to mass with a matrix-free fixed-point iteration
      (no P_oo slicing, no dense solves).
  - Adds --src_auto ring_mid with --ring_mask for deterministic, case-native source.
  - Emits tighter PASS/INDETERMINATE policy using BOTH:
        (A) basin probability gain  (norm_curved > norm_flat + eps_norm)
        (B) expected hitting time drop (D_curved < D_flat - eps_D)
    with minimum basin mass gate min_norm.

STRICT PP constraints respected:
  - No Laplacian/Poisson, no PDE, no Euclidean metric injection.
  - Mass core is trace-derived mask only.
  - Distance/time are Markov step counts / hitting times only.

Typical 512 runs:
  ms080:
    python src/gr_strict_pp_scalar/PP_deflection_markov_front_PP_v3_sparse_only.py \
      --edges_flat   edges_ca_v3_flat_512x512_PPV1.txt \
      --edges_curved edges_ca_v3_mass_ms080_512x512_PPV1.txt \
      --mass_mask    PP_mass_mask_512x512_mass_ms080_PPV1.npy \
      --ring_mask    PP_orbit_ring_mask_512x512_mass_ms080_PPV1.npy \
      --H 512 --W 512 \
      --src_auto ring_mid \
      --steps 800 \
      --ht_maxiter 20000 \
      --ht_tol 1e-5 \
      --output src/gr_strict_pp_scalar/PP_deflection_front_512_mass_ms080_PPV1.json

  strong_pf010:
    python src/gr_strict_pp_scalar/PP_deflection_markov_front_PP_v3_sparse_only.py \
      --edges_flat   edges_ca_v3_flat_512x512_PPV1.txt \
      --edges_curved edges_ca_v3_strong_pf010_512x512_PPV1.txt \
      --mass_mask    PP_mass_mask_512x512_strong_pf010_PPV1.npy \
      --ring_mask    PP_orbit_ring_mask_512x512_strong_pf010_PPV1.npy \
      --H 512 --W 512 \
      --src_auto ring_mid \
      --steps 800 \
      --ht_maxiter 20000 \
      --ht_tol 1e-5 \
      --output src/gr_strict_pp_scalar/PP_deflection_front_512_strong_pf010_PPV1.json
"""

import argparse
import json
import time
from typing import Optional, Tuple, Dict

import numpy as np
import torch


# -------------------------------
# Device
# -------------------------------

def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device_arg in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested GPU but CUDA/HIP is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device argument: {device_arg}")


# -------------------------------
# Masks
# -------------------------------

def load_mask_1d(path: str, H: int, W: int, name: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        if arr.shape != (H, W):
            raise ValueError(f"{name} 2D shape {arr.shape} != (H,W)=({H},{W})")
        arr = arr.reshape(-1)
    elif arr.ndim == 1:
        if arr.shape[0] != H * W:
            raise ValueError(f"{name} 1D length {arr.shape[0]} != N={H*W}")
    else:
        raise ValueError(f"{name} must have ndim 1 or 2; got {arr.ndim}")
    return arr.astype(bool)


# -------------------------------
# Sparse Markov construction
# -------------------------------

def load_edges_as_transition_sparse(
    edges_path: str,
    N: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build sparse row-stochastic P from edgelist.

    Edges format:
      i j [weight]

    STRICT PP:
      - Uses only given directed edges.
      - No dense NxN allocation.
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
            if len(parts) < 2:
                continue
            s = int(parts[0]); t = int(parts[1])
            w = 1.0
            if len(parts) >= 3:
                try:
                    w = float(parts[2])
                except Exception:
                    w = 1.0
            if not (0 <= s < N and 0 <= t < N):
                raise ValueError(f"Edge ({s},{t}) out of range for N={N}")
            rows.append(s); cols.append(t); vals.append(w)

    if len(rows) == 0:
        raise ValueError(f"No edges parsed from {edges_path}")

    rows_t = torch.tensor(rows, device=device, dtype=torch.int64)
    cols_t = torch.tensor(cols, device=device, dtype=torch.int64)
    vals_t = torch.tensor(vals, device=device, dtype=dtype)

    # Row sums

    deg = torch.zeros((N,), device=device, dtype=dtype)
    deg.scatter_add_(0, rows_t, vals_t)

    # Normalize existing edges
    deg_safe = torch.where(deg > 0, deg, torch.ones_like(deg))
    vals_norm = vals_t / deg_safe[rows_t]

    # --- Add self-loops for sink rows to preserve probability mass ---
    zero_rows = (deg == 0)
    if zero_rows.any():
        zr_idx = zero_rows.nonzero(as_tuple=False).squeeze(-1)
        # add i->i with weight 1.0
        rows_t = torch.cat([rows_t, zr_idx])
        cols_t = torch.cat([cols_t, zr_idx])
        vals_norm = torch.cat([vals_norm, torch.ones_like(zr_idx, dtype=dtype, device=device)])

    idx = torch.stack([rows_t, cols_t], dim=0)
    P = torch.sparse_coo_tensor(idx, vals_norm, (N, N), device=device, dtype=dtype).coalesce()

    return P


def apply_P_sparse(P: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # P @ v
    return torch.sparse.mm(P, v.unsqueeze(1)).squeeze(1)


# -------------------------------
# Reachability (reverse BFS) from sparse support
# -------------------------------

def reverse_reachable_from_mass(
    P: torch.Tensor,
    mass_np: np.ndarray,
) -> np.ndarray:
    """
    Compute boolean reachable[i] = True if i can reach *any* mass node
    along directed edges of P (support only).
    """
    N = mass_np.shape[0]

    if not P.is_sparse:
        raise ValueError("reverse_reachable_from_mass expects sparse P")

    P_cpu = P.detach().to("cpu").coalesce()
    idx = P_cpu.indices()
    rows = idx[0].numpy()
    cols = idx[1].numpy()

    rev = [[] for _ in range(N)]
    for i, j in zip(rows, cols):
        rev[j].append(i)

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

    return reachable


# -------------------------------
# Matrix-free hitting times to mass
# -------------------------------

def compute_hitting_time_to_mass_fixedpoint(
    P_curved: torch.Tensor,
    mass_np: np.ndarray,
    *,
    ht_tol: float = 1e-5,
    ht_maxiter: int = 20000,
    ht_check_every: int = 50,
    damping: float = 1.0,
) -> torch.Tensor:
    """
    Solve mean hitting time equation (strict PP, matrix-free):

      h[i] = 0                         for i in M
      h[i] = 1 + sum_j P[i,j] h[j]     for i not in M

    We:
      1) compute reachable set via reverse BFS on support(P)
      2) iterate fixed-point only on reachable non-mass nodes

    Unreachable non-mass nodes get +inf.

    damping in (0,1] allowed for stability:
      h <- (1-d)*h + d*h_new
    """
    if not P_curved.is_sparse:
        raise ValueError("compute_hitting_time_to_mass_fixedpoint expects sparse P")

    device = P_curved.device
    N = P_curved.shape[0]
    if mass_np.shape[0] != N:
        raise ValueError("mass mask length mismatch")

    if mass_np.sum() == 0:
        raise ValueError("mass_mask has no True entries")

    reachable = reverse_reachable_from_mass(P_curved, mass_np)
    open_np = ~mass_np
    reachable_open_np = np.logical_and(open_np, reachable)

    # Torch masks
    mask_mass = torch.from_numpy(mass_np).to(device)
    mask_reach_open = torch.from_numpy(reachable_open_np).to(device)

    # Initialize h
    h = torch.zeros((N,), dtype=torch.float64, device=device)

    # Iteration
    t0 = time.time()
    last_report = 0.0
    max_dev = float("inf")

    for it in range(1, ht_maxiter + 1):
        # h_prop = 1 + P h
        Ph = apply_P_sparse(P_curved, h)
        h_prop = Ph + 1.0

        # boundary
        h_prop = torch.where(mask_mass, torch.zeros_like(h_prop), h_prop)

        # Only update reachable open nodes
        h_new = h.clone()
        h_new[mask_reach_open] = h_prop[mask_reach_open]

        if damping < 1.0:
            h_new = (1.0 - damping) * h + damping * h_new

        # Check convergence
        if it % ht_check_every == 0:
            diff = torch.abs(h_new[mask_reach_open] - h[mask_reach_open])
            max_dev = diff.max().item() if diff.numel() else 0.0
            now = time.time()
            last_report = now - t0
            if max_dev <= ht_tol:
                h = h_new
                break

        h = h_new

    # Set unreachable open nodes to +inf
    h_out = torch.full((N,), float("inf"), dtype=torch.float64, device=device)
    h_out[mask_mass] = 0.0
    h_out[mask_reach_open] = torch.clamp(h[mask_reach_open], min=0.0)

    if torch.isnan(h_out).any():
        raise ValueError("NaNs in h_to_mass")

    meta = {
        "ht_tol": ht_tol,
        "ht_maxiter": ht_maxiter,
        "ht_check_every": ht_check_every,
        "ht_damping": damping,
        "ht_last_max_dev": max_dev,
        "ht_runtime_sec": float(last_report) if last_report else float(time.time() - t0),
        "reachable_open_count": int(reachable_open_np.sum()),
        "mass_count": int(mass_np.sum()),
    }
    return h_out, meta


# -------------------------------
# Row-distribution evolution
# -------------------------------

def evolve_row_distribution_sparse(
    P: torch.Tensor,
    p0: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """
    Evolve ROW distribution:
      p_{t+1} = p_t @ P

    Implement as column update:
      v_{t+1} = P^T @ v_t
    """
    if not P.is_sparse:
        raise ValueError("evolve_row_distribution_sparse expects sparse P")

    N = P.shape[0]
    if p0.shape != (N,):
        raise ValueError("p0 shape mismatch")

    v = p0.contiguous()
    PT = P.transpose(0, 1).coalesce()

    for _ in range(steps):
        v = torch.sparse.mm(PT, v.unsqueeze(1)).squeeze(1)

    return v


# -------------------------------
# Observable
# -------------------------------

def expected_markov_distance_to_mass(
    p: torch.Tensor,
    h_to_mass: torch.Tensor,
) -> Tuple[Optional[float], float]:
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
# Source selection
# -------------------------------

def pick_src_from_ring_mid(
    ring_mask_path: str,
    H: int,
    W: int,
) -> Tuple[int, int, int, int]:
    ring_np = load_mask_1d(ring_mask_path, H, W, "ring_mask")
    idxs = np.where(ring_np > 0)[0].astype(int)
    if idxs.size == 0:
        raise ValueError("ring_mask has no True entries")

    idxs.sort()
    src_index = int(idxs[len(idxs) // 2])
    src_row, src_col = divmod(src_index, W)
    return src_index, src_row, src_col, int(idxs.size)


# -------------------------------
# CLI
# -------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STRICT PP deflection via sparse Markov fronts (v3)."
    )

    p.add_argument("--edges_flat", required=True)
    p.add_argument("--edges_curved", required=True)

    p.add_argument("--mass_mask", required=True)
    p.add_argument("--ring_mask", default=None,
                   help="Orbit ring mask (.npy) for --src_auto ring_mid")

    p.add_argument("--H", type=int, required=True)
    p.add_argument("--W", type=int, required=True)

    # Manual source
    p.add_argument("--src_row", type=int, default=None)
    p.add_argument("--src_col", type=int, default=None)

    # Auto source
    p.add_argument("--src_auto", type=str, default="none",
                   choices=["none", "ring_mid"],
                   help="Auto source selector (default: none)")

    p.add_argument("--steps", type=int, default=800)

    # Hitting time iteration controls
    p.add_argument("--ht_tol", type=float, default=1e-5)
    p.add_argument("--ht_maxiter", type=int, default=20000)
    p.add_argument("--ht_check_every", type=int, default=50)
    p.add_argument("--ht_damping", type=float, default=1.0,
                   help="Fixed-point damping in (0,1], default 1.0")

    # PASS policy controls
    p.add_argument("--min_norm", type=float, default=1e-6,
                   help="Minimum finite-mass basin probability required to decide PASS/FAIL")
    p.add_argument("--eps_norm", type=float, default=1e-8,
                   help="Required margin for norm_curved > norm_flat + eps_norm")
    p.add_argument("--eps_D", type=float, default=0.0,
                   help="Required margin for D_curved < D_flat - eps_D")

    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "gpu", "cuda"])

    p.add_argument("--output", required=True)

    return p.parse_args()


def main():
    args = parse_args()
    H, W = args.H, args.W
    N = H * W

    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")

    device = get_device(args.device)

    # --- Load masks ---
    mass_np = load_mask_1d(args.mass_mask, H, W, "mass_mask")
    if mass_np.sum() == 0:
        raise ValueError("mass_mask has no True entries")

    # --- Source selection ---
    src_index = None
    src_row = None
    src_col = None
    ring_size = None

    if args.src_auto == "ring_mid":
        if not args.ring_mask:
            raise ValueError("--src_auto ring_mid requires --ring_mask")
        src_index, src_row, src_col, ring_size = pick_src_from_ring_mid(args.ring_mask, H, W)
    else:
        if args.src_row is None or args.src_col is None:
            raise ValueError("Provide --src_row/--src_col or use --src_auto ring_mid")
        if not (0 <= args.src_row < H and 0 <= args.src_col < W):
            raise ValueError("src_row/src_col out of range")
        src_row, src_col = int(args.src_row), int(args.src_col)
        src_index = src_row * W + src_col

    # --- Build sparse Markov kernels ---
    t0 = time.time()
    P_flat = load_edges_as_transition_sparse(args.edges_flat, N, device, dtype=torch.float64)
    P_curved = load_edges_as_transition_sparse(args.edges_curved, N, device, dtype=torch.float64)

    # --- Build p0 ---
    p0 = torch.zeros((N,), dtype=torch.float64, device=device)
    p0[src_index] = 1.0

    # --- Hitting times to mass (curved) ---
    h_to_mass, ht_meta = compute_hitting_time_to_mass_fixedpoint(
        P_curved,
        mass_np,
        ht_tol=args.ht_tol,
        ht_maxiter=args.ht_maxiter,
        ht_check_every=args.ht_check_every,
        damping=args.ht_damping,
    )

    # --- Evolve fronts ---
    if args.steps < 1:
        raise ValueError("steps must be >= 1")

    p_flat_T = evolve_row_distribution_sparse(P_flat, p0, args.steps)
    p_curved_T = evolve_row_distribution_sparse(P_curved, p0, args.steps)

    # --- Normalization diagnostics (do not hard-fail for tiny drift) ---
    sum_flat = float(p_flat_T.sum().item())
    sum_curv = float(p_curved_T.sum().item())
    norm_warn = bool((abs(sum_flat - 1.0) > 1e-4) or (abs(sum_curv - 1.0) > 1e-4))

    # --- Observables ---
    D_flat, norm_flat = expected_markov_distance_to_mass(p_flat_T, h_to_mass)
    D_curved, norm_curved = expected_markov_distance_to_mass(p_curved_T, h_to_mass)

    # --- Tighter PASS/INDETERMINATE policy ---
    min_norm = float(args.min_norm)
    eps_norm = float(args.eps_norm)
    eps_D = float(args.eps_D)

    if (norm_flat < min_norm) or (norm_curved < min_norm) or (D_flat is None) or (D_curved is None):
        PASS = None
        pass_reason = "insufficient_basin_probability_or_undefined_D"
        closer_by_norm = None
        closer_by_D = None
    else:
        closer_by_norm = bool(norm_curved > norm_flat + eps_norm)
        closer_by_D = bool(D_curved < D_flat - eps_D)
        PASS = bool(closer_by_norm and closer_by_D)
        pass_reason = "both_norm_and_D_criteria"

    runtime_sec = float(time.time() - t0)

    out: Dict = {
        "H": H,
        "W": W,
        "N": N,
        "device_used": str(device),

        "edges_flat": args.edges_flat,
        "edges_curved": args.edges_curved,
        "mass_mask": args.mass_mask,
        "ring_mask": args.ring_mask if args.ring_mask else None,

        "src_auto": args.src_auto,
        "ring_size": ring_size,
        "src_index": int(src_index),
        "src_row_label": int(src_row),
        "src_col_label": int(src_col),

        "steps": int(args.steps),

        "front_sum_flat": sum_flat,
        "front_sum_curved": sum_curv,
        "warn_front_normalization": norm_warn,

        "norm_finite_mass_flat": float(norm_flat),
        "norm_finite_mass_curved": float(norm_curved),

        "markov_distance_to_mass": {
            "D_flat": D_flat,
            "D_curved": D_curved,
            "closer_by_norm": closer_by_norm,
            "closer_by_D": closer_by_D,
            "eps_norm": eps_norm,
            "eps_D": eps_D,
            "min_norm": min_norm,
        },

        "PASS_deflection_markov_front_PP": PASS,
        "PASS_reason": pass_reason,

        "hitting_time_meta": ht_meta,

        "runtime_sec": runtime_sec,

        "notes": (
            "STRICT PP deflection via sparse finite-time Markov fronts. "
            "Row distribution evolved by p_{t+1}=p_t@P using P^T@v implementation. "
            "Mass core is trace-derived mask only. "
            "Markov distance to mass defined via matrix-free mean hitting times "
            "on the CURVED graph with boundary h[M]=0. "
            "No PDE, no Poisson/Laplacian field injection, no Euclidean metric, "
            "no GR ansatz, no regression. "
            "PASS requires BOTH: (1) norm_curved > norm_flat + eps_norm and "
            "(2) D_curved < D_flat - eps_D with minimum basin mass min_norm; "
            "otherwise INDETERMINATE (PASS=None)."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote deflection report to {args.output}")
    print(f"PASS_deflection_markov_front_PP = {PASS}")


if __name__ == "__main__":
    main()

