#!/usr/bin/env python3
"""
PP_deflection_markov_front_PP_v7.py

STRICT PP deflection via Markov fronts — v7 (robust applicability).

Key change vs v6:
- v6 required *instantaneous* front mass on a ring to be nonzero in BOTH geometries,
  causing frequent APPLICABLE=False when curved collapses inward.
- v7 uses a STRICT PP, robust observable:
    cumulative ring visitation (time-integrated probability mass on the ring)
  so curved can be ~0 and that becomes a *meaningful focusing signal* rather than NA.

STRICT PP guarantees:
- No PDE, no Laplacian/Poisson, no GR ansatz, no regression.
- Ring and sources defined purely from Markov hop distance to mass on FLAT (fixed rule),
  plus optional orbit_mask as a label-universe.
- Uses torch on GPU if available (GPU-first), CPU fallback.

Observable (v7):
- Choose sources from max-distance band on FLAT: S.
- Choose ring as a fixed quantile band of flat dist_to_mass within orbit universe: R.
- Evolve Markov distributions for T steps:
    p_{t+1} = p_t P
  for FLAT and CURVED with the same source initialization.
- Compute:
    F_ring = sum_{t=0..T-1} sum_{i in R} p_t[i]
    D = F_ring / T
  Then deflection proxy is reduction fraction:
    red = (D_flat - D_curved) / max(D_flat, eps)
  PASS if red >= delta_threshold and D_flat > min_ring_visit_flat.

This avoids the "curved ring mass must be nonzero" brittleness.
"""

import argparse, json, os
from collections import deque
from typing import Tuple

import numpy as np

try:
    import torch
except Exception:
    torch = None


def _ensure_bool_mask(a: np.ndarray, N: int, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2:
        a = a.reshape(-1)
    assert a.ndim == 1, f"{name} must be 1D or 2D; got shape={a.shape}"
    assert a.size == N, f"{name} size mismatch: got {a.size}, expected {N}"
    return (a.astype(np.int64) != 0)


def load_edges(path: str, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accepts:
      - "u v" (weight defaults to 1.0)
      - "u v w"
    Skips blank lines and comments starting with '#'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    src = []
    dst = []
    w = []

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            if len(parts) == 2:
                a, b = int(parts[0]), int(parts[1])
                ww = 1.0
            else:
                a, b = int(parts[0]), int(parts[1])
                ww = float(parts[2])
            if not (0 <= a < N and 0 <= b < N):
                raise ValueError(f"Edge index out of range in {path}: {a} {b} (N={N})")
            src.append(a); dst.append(b); w.append(ww)

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    assert src.shape == dst.shape == w.shape
    return src, dst, w


def build_csr_from_edges(src: np.ndarray, dst: np.ndarray, N: int):
    """
    CSR for adjacency: row u -> list of v.
    Returns indptr, indices, (optionally) weights aligned with indices.
    """
    assert src.shape == dst.shape
    E = src.size
    order = np.argsort(src, kind="mergesort")
    src_s = src[order]
    dst_s = dst[order]

    indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(indptr, src_s + 1, 1)
    np.cumsum(indptr, out=indptr)

    indices = np.empty(E, dtype=np.int64)
    indices[:] = dst_s
    return indptr, indices


def build_reverse_csr(src: np.ndarray, dst: np.ndarray, N: int):
    return build_csr_from_edges(dst, src, N)


def reverse_bfs_dist_to_mass(rev_indptr: np.ndarray, rev_indices: np.ndarray, mass_nodes: np.ndarray) -> np.ndarray:
    N = rev_indptr.size - 1
    dist = np.full(N, -1, dtype=np.int32)

    q = deque()
    for m in mass_nodes:
        dist[m] = 0
        q.append(int(m))

    while q:
        v = q.popleft()
        dv = dist[v]
        start = rev_indptr[v]
        end = rev_indptr[v + 1]
        for u in rev_indices[start:end]:
            if dist[u] == -1:
                dist[u] = dv + 1
                q.append(int(u))
    return dist


def select_sources_maxdist(dist: np.ndarray, n_sources: int, band_width: int, exclude_mask: np.ndarray) -> np.ndarray:
    finite = (dist >= 0) & (~exclude_mask)
    if not finite.any():
        return np.zeros((0,), dtype=np.int64)

    dmax = int(dist[finite].max())
    dmin = max(0, dmax - int(band_width) + 1)

    band = np.where(finite & (dist >= dmin) & (dist <= dmax))[0]
    if band.size == 0:
        return np.zeros((0,), dtype=np.int64)

    order = np.lexsort((band, -dist[band].astype(np.int64)))
    band_sorted = band[order]

    if band_sorted.size > n_sources:
        band_sorted = band_sorted[:n_sources]

    return band_sorted.astype(np.int64)


def quantile_ring_from_dist(dist: np.ndarray, orbit_universe: np.ndarray, q_lo: float, q_hi: float, exclude_mass: np.ndarray):
    finite = (dist >= 0)
    U = orbit_universe & finite & (~exclude_mass)
    vals = dist[U].astype(np.float64)
    if vals.size == 0:
        return {
            "ok": False,
            "reason": "No finite dist values in orbit universe for quantile ring."
        }, np.zeros_like(orbit_universe, dtype=bool)

    q_lo = float(q_lo); q_hi = float(q_hi)
    assert 0.0 <= q_lo < q_hi <= 1.0

    d_lo = int(np.floor(np.quantile(vals, q_lo)))
    d_hi = int(np.ceil(np.quantile(vals, q_hi)))

    ring = U & (dist >= d_lo) & (dist <= d_hi)

    meta = {
        "ok": True,
        "q_lo": q_lo,
        "q_hi": q_hi,
        "d_lo": int(d_lo),
        "d_hi": int(d_hi),
        "ring_nodes": int(ring.sum()),
        "orbit_universe_nodes": int(orbit_universe.sum()),
        "finite_in_universe": int((orbit_universe & finite).sum()),
    }
    return meta, ring


def build_row_stochastic(src: np.ndarray, dst: np.ndarray, w: np.ndarray, N: int):
    """
    Build per-edge probability p_e = w_e / sum_{e:src=u} w_e
    If a node has outdeg=0, it contributes no mass (sink) under this kernel,
    matching prior sink-heavy STRICT PP behavior.
    """
    assert src.shape == dst.shape == w.shape
    out_sum = np.zeros(N, dtype=np.float64)
    np.add.at(out_sum, src, w)
    denom = out_sum[src]
    prob = np.zeros_like(w, dtype=np.float64)
    good = denom > 0
    prob[good] = w[good] / denom[good]
    return prob, out_sum


def evolve_markov_torch(src: np.ndarray, dst: np.ndarray, prob: np.ndarray, p0: np.ndarray, ring_mask: np.ndarray,
                        steps: int, burn_in: int, window: int, device: str):
    """
    Evolve p_{t+1} = p_t P (row-stochastic on edges).
    Report cumulative ring visitation over [burn_in, burn_in+window) windows repeated until steps.
    v7 uses a single contiguous window; we keep CLI burn_in/window for compatibility.

    Returns:
      - F_ring (sum_t sum_{i in ring} p_t[i]) over t in [burn_in, burn_in+window)
      - D = F_ring / window
      - peak_ring_mass over the window
      - final p
    """
    assert torch is not None, "torch not available"
    N = p0.size
    assert src.ndim == dst.ndim == prob.ndim == 1
    assert ring_mask.size == N

    dev = torch.device(device)
    src_t = torch.from_numpy(src.astype(np.int64)).to(dev)
    dst_t = torch.from_numpy(dst.astype(np.int64)).to(dev)
    prob_t = torch.from_numpy(prob.astype(np.float32)).to(dev)

    p = torch.from_numpy(p0.astype(np.float32)).to(dev)
    ring_t = torch.from_numpy(ring_mask.astype(np.bool_)).to(dev)

    # burn-in
    for _ in range(int(burn_in)):
        msg = p.index_select(0, src_t) * prob_t
        p_next = torch.zeros_like(p)
        p_next.index_add_(0, dst_t, msg)
        p = p_next

    F_ring = 0.0
    peak = 0.0
    T = int(window)
    for _ in range(T):
        ring_mass = float(p[ring_t].sum().item())
        F_ring += ring_mass
        if ring_mass > peak:
            peak = ring_mass

        msg = p.index_select(0, src_t) * prob_t
        p_next = torch.zeros_like(p)
        p_next.index_add_(0, dst_t, msg)
        p = p_next

    D = F_ring / max(T, 1)
    return float(F_ring), float(D), float(peak), p.detach().cpu().numpy()


def evolve_markov_numpy(src: np.ndarray, dst: np.ndarray, prob: np.ndarray, p0: np.ndarray, ring_mask: np.ndarray,
                        burn_in: int, window: int):
    """
    CPU fallback: scatter-add using numpy.
    """
    N = p0.size
    p = p0.astype(np.float64, copy=True)

    for _ in range(int(burn_in)):
        msg = p[src] * prob
        p_next = np.zeros(N, dtype=np.float64)
        np.add.at(p_next, dst, msg)
        p = p_next

    F_ring = 0.0
    peak = 0.0
    T = int(window)
    for _ in range(T):
        ring_mass = float(p[ring_mask].sum())
        F_ring += ring_mass
        peak = max(peak, ring_mass)

        msg = p[src] * prob
        p_next = np.zeros(N, dtype=np.float64)
        np.add.at(p_next, dst, msg)
        p = p_next

    D = F_ring / max(T, 1)
    return float(F_ring), float(D), float(peak), p.astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description="STRICT PP deflection via Markov fronts (v7).")
    ap.add_argument("--edges_flat", required=True)
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--mass_mask", required=True)
    ap.add_argument("--orbit_mask", default=None)

    ap.add_argument("--n_sources", type=int, default=2048)
    ap.add_argument("--source_band_width", type=int, default=4)

    # ring selection from FLAT dist within orbit universe
    ap.add_argument("--ring_q_lo", type=float, default=0.60)
    ap.add_argument("--ring_q_hi", type=float, default=0.80)
    ap.add_argument("--min_ring_nodes", type=int, default=100)

    # markov evolution windowing
    ap.add_argument("--steps", type=int, default=512, help="Kept for compatibility; v7 uses burn_in+window only.")
    ap.add_argument("--burn_in", type=int, default=64)
    ap.add_argument("--window", type=int, default=128)

    # v7 applicability & PASS gate
    ap.add_argument("--min_ring_visit_flat", type=float, default=1e-6,
                    help="Minimum D_flat (avg ring mass) to consider measurement meaningful.")
    ap.add_argument("--delta_threshold", type=float, default=0.05,
                    help="PASS if (D_flat - D_curved)/D_flat >= delta_threshold.")

    # device
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="GPU-first if torch+cuda available. (ROCm typically reports as cuda).")

    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    H, W = int(args.H), int(args.W)
    N = H * W

    mass = _ensure_bool_mask(np.load(args.mass_mask), N, "mass_mask")
    orbit = None
    if args.orbit_mask is not None:
        orbit = _ensure_bool_mask(np.load(args.orbit_mask), N, "orbit_mask")
    else:
        orbit = np.ones(N, dtype=bool)

    mass_nodes = np.where(mass)[0].astype(np.int64)
    assert mass_nodes.size > 0, "mass_mask has zero nodes."

    # load edges
    sf, df, wf = load_edges(args.edges_flat, N)
    sc, dc, wc = load_edges(args.edges_curved, N)

    # reverse CSR for dist_to_mass
    revptr_f, revidx_f = build_reverse_csr(sf, df, N)
    dist_f = reverse_bfs_dist_to_mass(revptr_f, revidx_f, mass_nodes)
    finite_f = (dist_f >= 0)
    if not finite_f.any():
        raise RuntimeError("No finite dist_to_mass nodes in FLAT; cannot define sources/ring.")

    # sources from flat max-dist band (exclude mass)
    sources = select_sources_maxdist(dist_f, n_sources=int(args.n_sources),
                                     band_width=int(args.source_band_width), exclude_mask=mass)
    if sources.size == 0:
        raise RuntimeError("No sources selected (flat max-dist band empty).")

    # ring selection from flat dist within orbit universe
    ring_meta, ring = quantile_ring_from_dist(
        dist_f, orbit_universe=orbit, q_lo=args.ring_q_lo, q_hi=args.ring_q_hi, exclude_mass=mass
    )
    ring_ok = bool(ring_meta.get("ok", False)) and int(ring.sum()) >= int(args.min_ring_nodes)

    # build row-stochastic edge probs (flat and curved)
    prob_f, outsum_f = build_row_stochastic(sf, df, wf, N)
    prob_c, outsum_c = build_row_stochastic(sc, dc, wc, N)

    # init p0 over sources (same for flat and curved)
    p0 = np.zeros(N, dtype=np.float64)
    p0[sources] = 1.0 / float(sources.size)

    # choose device
    use_torch = (torch is not None)
    if args.device == "cpu":
        dev = "cpu"
        use_torch = False
    elif args.device == "cuda":
        if (torch is None) or (not torch.cuda.is_available()):
            raise RuntimeError("Requested --device cuda but torch/cuda not available.")
        dev = "cuda"
    else:
        # auto
        if (torch is not None) and torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
            use_torch = False

    burn_in = int(args.burn_in)
    window = int(args.window)
    assert burn_in >= 0 and window >= 1

    # evolve and measure cumulative ring visitation over [burn_in, burn_in+window)
    if use_torch:
        Ff, Df, peakf, _ = evolve_markov_torch(sf, df, prob_f, p0, ring, steps=int(args.steps),
                                               burn_in=burn_in, window=window, device=dev)
        Fc, Dc, peakc, _ = evolve_markov_torch(sc, dc, prob_c, p0, ring, steps=int(args.steps),
                                               burn_in=burn_in, window=window, device=dev)
    else:
        Ff, Df, peakf, _ = evolve_markov_numpy(sf, df, prob_f, p0, ring, burn_in=burn_in, window=window)
        Fc, Dc, peakc, _ = evolve_markov_numpy(sc, dc, prob_c, p0, ring, burn_in=burn_in, window=window)

    # applicability: ring must exist AND flat must actually visit it above min
    applicable = bool(ring_ok) and (Df > float(args.min_ring_visit_flat))
    reason = None
    if not ring_ok:
        reason = "Ring selection failed or too few ring nodes."
    elif not (Df > float(args.min_ring_visit_flat)):
        reason = f"Flat avg ring visitation too small: D_flat={Df} <= min_ring_visit_flat={args.min_ring_visit_flat}"

    # PASS: reduction fraction above threshold
    eps = 1e-12
    red = (Df - Dc) / max(Df, eps) if Df > 0 else None
    passed = False
    if applicable:
        passed = bool(red is not None and red >= float(args.delta_threshold))

    out = {
        "H": H, "W": W, "N": N,
        "edges_flat": args.edges_flat,
        "edges_curved": args.edges_curved,
        "mass_mask": args.mass_mask,
        "orbit_mask": args.orbit_mask,

        "device_used": dev,
        "torch_used": bool(use_torch),

        "sources": {
            "policy": "maxdist_flat_band",
            "n_sources": int(sources.size),
            "source_band_width": int(args.source_band_width),
        },

        "ring_selection": {
            "mode": "quantile_band_from_flat_dist_within_orbit",
            **ring_meta,
            "min_ring_nodes": int(args.min_ring_nodes),
            "PASS_ring_nodes_ge_min": bool(ring_ok),
        },

        "evolution": {
            "burn_in": burn_in,
            "window": window,
            "steps_arg_ignored_for_v7": int(args.steps),
        },

        "deflection_stats": {
            "APPLICABLE": bool(applicable),
            "reason": reason,
            "F_ring_flat": float(Ff),
            "F_ring_curved": float(Fc),
            "D_flat": float(Df),
            "D_curved": float(Dc),
            "peak_ring_mass_flat": float(peakf),
            "peak_ring_mass_curved": float(peakc),
            "reduction_fraction": float(red) if red is not None else None,
            "delta_threshold": float(args.delta_threshold),
            "min_ring_visit_flat": float(args.min_ring_visit_flat),
            "PASS_deflection_markov_front_PP_v7": bool(passed),
        },

        "notes": (
            "STRICT PP deflection v7. Uses cumulative ring visitation on a fixed quantile-band ring "
            "defined from flat directed hop distance to mass (within orbit universe). "
            "This makes deflection applicable even when curved rapidly collapses inward (ring mass ~ 0), "
            "which is interpreted as strong focusing rather than NA. "
            "No PDE/Poisson/GR ansatz/regression."
        )
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("APPLICABLE =", out["deflection_stats"]["APPLICABLE"])
    if not out["deflection_stats"]["APPLICABLE"]:
        print("reason:", out["deflection_stats"]["reason"])
    else:
        print("D_flat =", out["deflection_stats"]["D_flat"], "D_curved =", out["deflection_stats"]["D_curved"])
        print("reduction_fraction =", out["deflection_stats"]["reduction_fraction"])
        print("PASS =", out["deflection_stats"]["PASS_deflection_markov_front_PP_v7"])


if __name__ == "__main__":
    main()

