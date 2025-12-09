#!/usr/bin/env python3
"""
PP_markov_tau_geometry_v3_multiShellIntersect.py

STRICT PP Markov τ-geometry (v3)

What it does (design):
- Loads flat + curved CA graph edges.
- Loads trace weights (experiences only).
- Defines T00(i) = normalized trace weights.
- Defines a MASS CORE strictly from traces via:
    1) --mass_mask (explicit .npy boolean mask) OR
    2) --mass_topk K OR
    3) --mass_threshold t OR
    4) --mass_quantile q
  (in that precedence order)
- Computes multi-source hop distances from the mass core on BOTH graphs.
- Builds shell bands from the intersection of flat/curved hop-bands.
- For each valid band S_k, computes mean hitting time h(i -> S_k) on:
    - flat graph Markov P_flat
    - curved graph Markov P_curved
  using the absorbing-hitting-time linear system:
      h_S = 0
      (I - P_oo) h_o = 1
- Defines:
      G00_k(i) = tau_curved_k(i) / tau_flat_k(i) - 1
  and aggregates
      G00_med(i) = median_k G00_k(i) (nanmedian across valid shells)
      kappa_med(i) = G00_med(i) / T00(i)   (where T00(i) > 0)

STRICT PP constraints honored:
- No Laplacian/Poisson.
- No PDE.
- No Euclidean distance used as a metric.
- No GR ansatz.
- No regression.
- Grid (row/col) used only for label convenience in reporting.

Outputs:
- JSON report with shell counts & summary stats.
- Optional NPZ with T00, G00_med, kappa_med, distances, and per-shell taus.

This v3 adds:
- --mass_mask support
- --mass_topk support
- --mass_threshold support
- safer degeneracy guards for quantized/flat traces
- robust linear-solve fallback (solve -> lstsq -> pinv)
"""

import os
import sys
import json
import math
import time
import argparse
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import torch
except Exception as e:
    print("ERROR: PyTorch is required for this script.", file=sys.stderr)
    raise


# ----------------------------
# Utilities
# ----------------------------

def get_device(dev: str) -> torch.device:
    if dev is None:
        dev = "auto"
    dev = dev.lower()
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(dev)


def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_trace_weights(path: str, expected_N: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trace weights file not found: {path}")
    vals = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # allow "idx weight" or just "weight"
            parts = s.split()
            if len(parts) == 1:
                vals.append(float(parts[0]))
            else:
                vals.append(float(parts[-1]))
    w = np.asarray(vals, dtype=np.float64)
    if w.size != expected_N:
        raise ValueError(f"Trace weights length {w.size} != expected N={expected_N}")
    return w


def load_mask(path: str, expected_N: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask file not found: {path}")
    m = np.load(path, allow_pickle=False).astype(bool)
    if m.ndim == 2:
        m = m.ravel()
    if m.shape != (expected_N,):
        raise ValueError(f"Mask shape {m.shape} != ({expected_N},)")
    return m


def load_edges_as_adjacency(path: str, N: int) -> List[List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Edges file not found: {path}")

    adj = [[] for _ in range(N)]
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            i = int(parts[0]); j = int(parts[1])
            if i < 0 or i >= N or j < 0 or j >= N:
                continue
            if j not in adj[i]:
                adj[i].append(j)
            if i not in adj[j]:
                adj[j].append(i)
    return adj


def build_P_from_adj(adj, device="cpu"):
    """
    Build row-stochastic Markov operator P from adjacency.
    STRICT PP: uses only graph edges; no geometry; no dense NxN.
    adj: scipy.sparse CSR/COO with nonnegative weights (typically 1s).
    Returns: torch.sparse_coo_tensor on device (float64).
    """
    import numpy as np
    import torch
    import scipy.sparse as sp

    adj = adj.tocsr()
    N = adj.shape[0]

    # Row sums
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg = np.where(deg == 0, 1.0, deg)

    coo = adj.tocoo()
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    weights = coo.data.astype(np.float64)

    # Row-stochastic values
    vals = weights / deg[rows]

    r = torch.from_numpy(rows).to(device=device)
    c = torch.from_numpy(cols).to(device=device)
    v = torch.from_numpy(vals).to(device=device, dtype=torch.float64)

    idx = torch.stack([r, c], dim=0)
    P = torch.sparse_coo_tensor(
        idx, v, size=(N, N), dtype=torch.float64, device=device
    ).coalesce()
    return P

def apply_P(P, x):
    import torch
    if getattr(P, "is_sparse", False):
        if x.ndim == 1:
            return torch.sparse.mm(P, x.unsqueeze(1)).squeeze(1)
        return torch.sparse.mm(P, x)
    return P @ x

def multisource_bfs(adj: List[List[int]], sources: np.ndarray) -> np.ndarray:
    """
    Unweighted multi-source BFS distances.
    Returns dist array length N with np.inf for unreachable nodes.
    """
    N = len(adj)
    dist = np.full((N,), np.inf, dtype=np.float64)
    q = deque()
    for s in sources:
        dist[s] = 0.0
        q.append(int(s))
    while q:
        u = q.popleft()
        du = dist[u]
        for v in adj[u]:
            if dist[v] == np.inf:
                dist[v] = du + 1.0
                q.append(v)
    return dist


def parse_shell_bands(bands: List[str]) -> List[Dict[str, int]]:
    out = []
    for s in bands:
        if ":" not in s:
            raise ValueError(f"Invalid shell band '{s}', expected 'rmin:rmax'.")
        a, b = s.split(":")
        rmin = int(a); rmax = int(b)
        if rmin < 0 or rmax <= rmin:
            raise ValueError(f"Invalid shell band '{s}'.")
        out.append({"rmin": rmin, "rmax": rmax})
    return out


def safe_solve_linear(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Robust solver:
      1) solve
      2) lstsq
      3) pinv
    """
    try:
        return torch.linalg.solve(A, b)
    except Exception:
        pass

    try:
        # torch.linalg.lstsq available in newer torch
        sol = torch.linalg.lstsq(A, b).solution
        return sol
    except Exception:
        pass

    # final fallback
    Ap = torch.linalg.pinv(A)
    return Ap @ b


def mean_hitting_time_to_set(P: torch.Tensor, target_mask: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Compute mean hitting time h(i -> S) for all i, for a Markov chain with transition P.
    target nodes have h=0.
    Solves (I - P_oo) h_o = 1.
    """
    N = P.shape[0]
    S = target_mask.astype(bool)
    if S.sum() == 0:
        return np.full((N,), np.nan, dtype=np.float64)

    O_idx = np.where(~S)[0]
    S_idx = np.where(S)[0]

    # If all nodes are targets
    if O_idx.size == 0:
        return np.zeros((N,), dtype=np.float64)

    # Extract P_oo
    O_t = torch.tensor(O_idx, dtype=torch.int64, device=device)
    P_oo = P.index_select(0, O_t).index_select(1, O_t)

    I = torch.eye(P_oo.shape[0], dtype=torch.float64, device=device)
    A = I - P_oo
    b = torch.ones((P_oo.shape[0],), dtype=torch.float64, device=device)

    h_o = safe_solve_linear(A, b)

    # Guard non-finite
    if not torch.isfinite(h_o).all():
        # try pinv directly as last rescue
        h_o = (torch.linalg.pinv(A) @ b)
    if not torch.isfinite(h_o).all():
        # return NaNs rather than crash downstream
        h = np.full((N,), np.nan, dtype=np.float64)
        h[S] = 0.0
        return h

    h = np.zeros((N,), dtype=np.float64)
    h[O_idx] = h_o.detach().cpu().numpy()
    h[S_idx] = 0.0
    return h


# ----------------------------
# Mass core selection
# ----------------------------

def select_mass_core(
    weights: np.ndarray,
    N: int,
    mass_mask_path: Optional[str],
    mass_topk: Optional[int],
    mass_threshold: Optional[float],
    mass_quantile: float,
) -> Tuple[np.ndarray, float, str]:
    """
    Returns (mask_bool, threshold_value_or_nan, mode_str)
    """

    # 1) explicit mask
    if mass_mask_path:
        mask = load_mask(mass_mask_path, expected_N=N)
        if mask.sum() == 0:
            raise ValueError("Provided mass_mask is empty.")
        return mask, float("nan"), "mask"

    # 2) topk
    if mass_topk is not None:
        k = int(mass_topk)
        if k <= 0 or k > N:
            raise ValueError(f"Invalid --mass_topk {k}")
        idx = np.argsort(weights)[::-1][:k]
        mask = np.zeros((N,), dtype=bool)
        mask[idx] = True
        thr = float(weights[idx[-1]]) if k > 0 else float("nan")
        return mask, thr, f"topk_{k}"

    # 3) threshold
    if mass_threshold is not None:
        t = float(mass_threshold)
        mask = (weights >= t)
        if mask.sum() == 0:
            raise ValueError("Empty mass_core from --mass_threshold.")
        return mask, t, f"threshold_{t:g}"

    # 4) quantile
    q = float(mass_quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("--mass_quantile must be in (0,1)")
    thr = float(np.quantile(weights, q))
    mask = (weights >= thr)
    if mask.sum() == 0:
        raise ValueError("Empty mass_core; quantile too high or trace degenerate.")

    # Guard against quantized/flat weights selecting (almost) everything
    n_mass = int(mask.sum())
    if n_mass >= int(0.9 * N):
        raise ValueError(
            f"Mass core too large under quantile_{q} (n_mass={n_mass}/{N}). "
            f"For quantized traces, use --mass_topk or --mass_mask."
        )

    return mask, thr, f"quantile_{q}"


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PP Markov τ-geometry v3 (multi-shell intersect). STRICT PP."
    )

    parser.add_argument("--edges_flat", type=str, required=True)
    parser.add_argument("--edges_curved", type=str, required=True)
    parser.add_argument("--trace_weights", type=str, required=True)

    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--W", type=int, required=True)

    # Mass selection controls
    parser.add_argument("--mass_quantile", type=float, default=0.995)
    parser.add_argument("--mass_topk", type=int, default=None)
    parser.add_argument("--mass_threshold", type=float, default=None)
    parser.add_argument("--mass_mask", type=str, default=None)

    # Shell bands
    parser.add_argument(
        "--shell_bands",
        type=str,
        nargs="+",
        default=["4:6", "6:8", "8:10", "10:12", "12:14", "14:16"],
        help="List of 'rmin:rmax' hop-distance bands.",
    )

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--report", type=str, required=True)
    parser.add_argument("--npz_out", type=str, default=None)

    args = parser.parse_args()

    H, W = int(args.H), int(args.W)
    N = H * W

    device = get_device(args.device)

    t0 = time.time()

    # Load weights and normalize to T00
    raw_weights = load_trace_weights(args.trace_weights, expected_N=N)
    wsum = float(raw_weights.sum())
    if wsum <= 0:
        raise ValueError("Trace weights sum <= 0.")
    T00 = (raw_weights / wsum).astype(np.float64)

    # Mass core selection
    mass_core_mask, mass_thr, mass_mode = select_mass_core(
        raw_weights, N,
        mass_mask_path=args.mass_mask,
        mass_topk=args.mass_topk,
        mass_threshold=args.mass_threshold,
        mass_quantile=args.mass_quantile,
    )
    mass_indices = np.where(mass_core_mask)[0]
    n_mass = int(mass_core_mask.sum())

    # Load graphs
    adj_flat = load_edges_as_adjacency(args.edges_flat, N=N)
    adj_curved = load_edges_as_adjacency(args.edges_curved, N=N)

    # Distances from mass core on both graphs
    d_flat = multisource_bfs(adj_flat, mass_indices)
    d_curved = multisource_bfs(adj_curved, mass_indices)

    # Build Markov matrices (dense)
    P_flat = build_P_from_adj(adj_flat, device=device)
    P_curved = build_P_from_adj(adj_curved, device=device)

    # Parse shell bands
    shell_bands = parse_shell_bands(args.shell_bands)

    shells_report = []
    tau_flat_shells = []
    tau_curved_shells = []
    G00_shells = []

    # Create shells
    for k, band in enumerate(shell_bands):
        rmin = int(band["rmin"]); rmax = int(band["rmax"])

        shell_flat_mask = np.isfinite(d_flat) & (d_flat >= rmin) & (d_flat < rmax)
        shell_curved_mask = np.isfinite(d_curved) & (d_curved >= rmin) & (d_curved < rmax)
        shell_intersect_mask = shell_flat_mask & shell_curved_mask

        n_shell_flat = int(shell_flat_mask.sum())
        n_shell_curved = int(shell_curved_mask.sum())
        n_shell_intersect = int(shell_intersect_mask.sum())

        valid = (n_shell_intersect > 0)

        shell_entry = {
            "band_index": k,
            "rmin": rmin,
            "rmax": rmax,
            "n_shell_flat": n_shell_flat,
            "n_shell_curved": n_shell_curved,
            "n_shell_intersect": n_shell_intersect,
            "valid": bool(valid),
            "G00_stats_mass_core_shell": None,
        }

        if not valid:
            shells_report.append(shell_entry)
            continue

        # Mean hitting times to the INTERSECT shell
        h_flat = mean_hitting_time_to_set(P_flat, shell_intersect_mask, device=device)
        h_curved = mean_hitting_time_to_set(P_curved, shell_intersect_mask, device=device)

        # Compute G00_k
        with np.errstate(divide="ignore", invalid="ignore"):
            G00_k = (h_curved / h_flat) - 1.0

        tau_flat_shells.append(h_flat)
        tau_curved_shells.append(h_curved)
        G00_shells.append(G00_k)

        # Stats over MASS CORE for this shell
        G_mass = G00_k[mass_core_mask]
        G_mass = G_mass[np.isfinite(G_mass)]
        if G_mass.size > 0:
            shell_entry["G00_stats_mass_core_shell"] = {
                "count": int(G_mass.size),
                "mean": float(np.mean(G_mass)),
                "std": float(np.std(G_mass)),
                "min": float(np.min(G_mass)),
                "max": float(np.max(G_mass)),
                "q25": float(np.quantile(G_mass, 0.25)),
                "median": float(np.quantile(G_mass, 0.50)),
                "q75": float(np.quantile(G_mass, 0.75)),
            }

        shells_report.append(shell_entry)

    # Stack shells if any valid
    if len(G00_shells) == 0:
        G00_med = np.full((N,), np.nan, dtype=np.float64)
    else:
        G00_shell_stack = np.stack(G00_shells, axis=0)  # [K, N]
        G00_med = np.nanmedian(G00_shell_stack, axis=0)

    # kappa_med
    kappa_med = np.full((N,), np.nan, dtype=np.float64)
    pos_T = (T00 > 0)
    kappa_med[pos_T] = G00_med[pos_T] / T00[pos_T]

    # Summaries
    def stats(vec: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
        if mask is not None:
            v = vec[mask]
        else:
            v = vec
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {
                "count": 0, "mean": None, "std": None, "min": None, "max": None,
                "q25": None, "median": None, "q75": None
            }
        return {
            "count": int(v.size),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "q25": float(np.quantile(v, 0.25)),
            "median": float(np.quantile(v, 0.50)),
            "q75": float(np.quantile(v, 0.75)),
        }

    # Correlation G00_med vs rho (T00) over all finite points
    corr_G00_rho = None
    finite_mask = np.isfinite(G00_med) & np.isfinite(T00)
    if finite_mask.sum() >= 3:
        g = G00_med[finite_mask]
        r = T00[finite_mask]
        if np.std(g) > 0 and np.std(r) > 0:
            corr_G00_rho = float(np.corrcoef(g, r)[0, 1])

    # PASS signals (strictly scalar, sign-based)
    # "Attractive" means median G00 in mass core negative OR kappa median negative.
    G_mass_med = None
    kappa_mass_med = None
    if np.isfinite(G00_med[mass_core_mask]).any():
        G_mass_med = float(np.nanmedian(G00_med[mass_core_mask]))
    if np.isfinite(kappa_med[mass_core_mask]).any():
        kappa_mass_med = float(np.nanmedian(kappa_med[mass_core_mask]))

    PASS_G00_sign_attractive = None
    PASS_kappa_median_sign = None
    if G_mass_med is not None:
        PASS_G00_sign_attractive = bool(G_mass_med < 0.0)
    if kappa_mass_med is not None:
        PASS_kappa_median_sign = bool(kappa_mass_med < 0.0)

    report = {
        "N": N, "H": H, "W": W,
        "edges_flat": args.edges_flat,
        "edges_curved": args.edges_curved,
        "trace_weights": args.trace_weights,
        "mass_mode": mass_mode,
        "mass_quantile": float(args.mass_quantile),
        "mass_topk": None if args.mass_topk is None else int(args.mass_topk),
        "mass_threshold": None if args.mass_threshold is None else float(args.mass_threshold),
        "mass_mask": args.mass_mask,
        "mass_threshold_used": None if (math.isnan(mass_thr) if isinstance(mass_thr, float) else True) else float(mass_thr),
        "n_mass_nodes": int(n_mass),
        "device": str(device),
        "shell_bands": shell_bands,
        "shells": shells_report,

        "T00_stats_all": stats(T00),
        "G00_med_stats_all": stats(G00_med),
        "T00_stats_mass_core": stats(T00, mass_core_mask),
        "G00_med_stats_mass_core": stats(G00_med, mass_core_mask),
        "kappa_med_stats_mass_core": stats(kappa_med, mass_core_mask),

        "corr_G00_rho": corr_G00_rho,

        "PASS_G00_sign_attractive": PASS_G00_sign_attractive,
        "PASS_kappa_median_sign": PASS_kappa_median_sign,

        "runtime_sec": float(time.time() - t0),

        "notes": (
            "PP_markov_tau_geometry_v3_multiShellIntersect: STRICT PP Markov τ-geometry. "
            "T00(i) from normalized trace weights (experiences only). "
            "Mass core selected from traces via --mass_mask/--mass_topk/--mass_threshold/--mass_quantile. "
            "Shells S_k defined by flat∩curved hop-distance bands from mass_core. "
            "Mean hitting times to S_k computed via absorbing Markov linear system. "
            "G00_k(i)=tau_curved_k/tau_flat_k - 1; G00_med is median over valid shells; "
            "kappa_med=G00_med/T00. "
            "No Laplacian, no Poisson, no PDE, no GR ansatz, no regression."
        )
    }

    save_json(args.report, report)
    print(f"Wrote JSON report to {args.report}")

    if args.npz_out:
        out = {
            "T00": T00.astype(np.float64),
            "G00_med": G00_med.astype(np.float64),
            "kappa_med": kappa_med.astype(np.float64),
            "mass_core_mask": mass_core_mask.astype(bool),
            "d_flat": d_flat.astype(np.float64),
            "d_curved": d_curved.astype(np.float64),
        }

        if len(tau_flat_shells) > 0:
            out["tau_flat_shells"] = np.stack(tau_flat_shells, axis=0).astype(np.float64)
            out["tau_curved_shells"] = np.stack(tau_curved_shells, axis=0).astype(np.float64)
            out["G00_shells"] = np.stack(G00_shells, axis=0).astype(np.float64)

        np.savez_compressed(args.npz_out, **out)
        print(f"Saved NPZ to {args.npz_out}")


if __name__ == "__main__":
    main()

