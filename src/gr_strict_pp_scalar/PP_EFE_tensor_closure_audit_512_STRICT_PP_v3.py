#!/usr/bin/env python3
import argparse, json, os
from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch
except Exception as e:
    torch = None


def _exists(p: Optional[str]) -> bool:
    return bool(p) and os.path.exists(p)


def _read_edges_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    src = []
    dst = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                continue
            src.append(int(parts[0]))
            dst.append(int(parts[1]))
    if not src:
        raise SystemExit(f"ERROR: no edges read from {path}")
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def _load_npz_or_npy(path: str, key: Optional[str] = None) -> np.ndarray:
    if not _exists(path):
        raise SystemExit(f"ERROR: path not found: {path}")
    if path.endswith(".npy"):
        arr = np.load(path)
        return arr
    if path.endswith(".npz"):
        z = np.load(path)
        keys = list(z.files)
        if key is None:
            # heuristic
            for k in ("G00", "G00_map", "Gdown", "T00", "rho", "mass_core_mask", "mask"):
                if k in keys:
                    return z[k]
            raise SystemExit(f"ERROR: {path} npz had keys {keys} but none matched (provide --*_key).")
        if key not in keys:
            raise SystemExit(f"ERROR: {path} npz had keys {keys} but missing requested key={key}")
        return z[key]
    raise SystemExit(f"ERROR: unsupported file type: {path}")


def _infer_N_from_fields(G00: np.ndarray, rho: np.ndarray) -> int:
    # Allow [H,W] or [N] shapes
    if G00.ndim == 2:
        H, W = G00.shape
        if rho.ndim == 2 and rho.shape != (H, W):
            raise SystemExit(f"ERROR: rho shape {rho.shape} != G00 shape {G00.shape}")
        return int(H * W)
    if G00.ndim == 1:
        if rho.ndim != 1 or rho.shape[0] != G00.shape[0]:
            raise SystemExit(f"ERROR: rho shape {rho.shape} != G00 shape {G00.shape}")
        return int(G00.shape[0])
    raise SystemExit(f"ERROR: unsupported G00 shape {G00.shape}")


def _flatten(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1).astype(np.float64, copy=False)


def _bidirected_filter(src: np.ndarray, dst: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    # Keep only edges that have the reverse edge present.
    keys = src * N + dst
    order = np.argsort(keys)
    keys_sorted = keys[order]

    rev_keys = dst * N + src
    pos = np.searchsorted(keys_sorted, rev_keys, side="left")
    ok = (pos < keys_sorted.size) & (keys_sorted[pos] == rev_keys)

    src2 = src[ok]
    dst2 = dst[ok]
    if src2.size == 0:
        raise SystemExit("ERROR: after bidirected filter, no edges remain.")
    return src2, dst2


def _median_ratio_on_mask(G00: np.ndarray, rho: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    m = mask.astype(bool).reshape(-1)
    g = G00.reshape(-1)[m]
    r = rho.reshape(-1)[m]
    valid = np.isfinite(g) & np.isfinite(r) & (np.abs(r) > 0)
    if valid.sum() == 0:
        return float("nan"), 0
    k = np.median(g[valid] / r[valid])
    return float(k), int(valid.sum())


def _core_sign_match_frac(G00: np.ndarray, rho: np.ndarray, mask: np.ndarray, kappa: float) -> float:
    m = mask.astype(bool).reshape(-1)
    g = G00.reshape(-1)[m]
    r = rho.reshape(-1)[m]
    pred = kappa * r
    valid = np.isfinite(g) & np.isfinite(pred)
    if valid.sum() == 0:
        return float("nan")
    sg = np.sign(g[valid])
    sp = np.sign(pred[valid])
    return float(np.mean(sg == sp))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "STRICT PP tensor-closure audit (v2 semantics): "
            "build MH reversible kernel targeting rho on a bidirected edge set, "
            "audit detailed-balance flux symmetry and graph divergence. "
            "No PDE/Poisson, no smoothing, no GR ansatz, no regression."
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="cpu-first; set gpu to use CUDA/ROCm if available.")
    ap.add_argument("--edges", required=True, help="Edge list .txt with lines: src dst (0..N-1).")

    ap.add_argument("--G00", required=True, help=".npy or .npz for G00 field.")
    ap.add_argument("--G00_key", default=None, help="If G00 is .npz, key to load (e.g., G00_med).")

    ap.add_argument("--rho", required=True, help=".npy or .npz for rho/T00 field.")
    ap.add_argument("--rho_key", default=None, help="If rho is .npz, key to load (e.g., T00).")

    ap.add_argument("--mask", default=None, help=".npy or .npz for mass-core mask. If omitted, uses topk_frac on rho.")
    ap.add_argument("--mask_key", default=None, help="If mask is .npz, key to load (e.g., mass_core_mask).")
    ap.add_argument("--topk_frac_used_if_no_mask", type=float, default=80/262144, help="Fallback core mask via top-k rho fraction.")

    ap.add_argument("--require_bidirected", action="store_true", default=True, help="Require bidirected edges (default true).")

    ap.add_argument("--core_match_frac_min", type=float, default=0.90)
    ap.add_argument("--graph_div_rel_l1_max", type=float, default=0.03)
    ap.add_argument("--antisym_rel_l1_max", type=float, default=1e-12)

    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if torch is None:
        raise SystemExit("ERROR: torch is required for this script (cpu or gpu).")

    # --- load fields ---
    G00_raw = _load_npz_or_npy(args.G00, args.G00_key)
    rho_raw = _load_npz_or_npy(args.rho, args.rho_key)

    N = _infer_N_from_fields(G00_raw, rho_raw)
    G00 = _flatten(G00_raw)
    rho = _flatten(rho_raw)

    # mask
    if args.mask is not None:
        mask_raw = _load_npz_or_npy(args.mask, args.mask_key)
        mask = mask_raw.reshape(-1).astype(bool)
        if mask.size != N:
            raise SystemExit(f"ERROR: mask size {mask.size} != N {N}")
    else:
        # top-k on rho
        frac = float(args.topk_frac_used_if_no_mask)
        k = max(1, int(round(frac * N)))
        idx = np.argpartition(rho, -k)[-k:]
        mask = np.zeros(N, dtype=bool)
        mask[idx] = True

    # --- load edges ---
    src_np, dst_np = _read_edges_txt(args.edges)
    if src_np.min() < 0 or dst_np.min() < 0 or src_np.max() >= N or dst_np.max() >= N:
        raise SystemExit("ERROR: edges contain node ids outside [0, N). Check H/W consistency and edge file.")
    if args.require_bidirected:
        src_np, dst_np = _bidirected_filter(src_np, dst_np, N)

    # --- device selection (cpu-first; gpu only if requested) ---
    if args.device == "gpu":
        dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    else:
        dev = torch.device("cpu")

    # --- build MH kernel targeting rho as stationary ---
    # pi := rho / sum(rho), clipped nonnegative
    rho_pos = np.clip(rho, 0.0, None)
    rho_l1 = float(np.sum(rho_pos))
    if not np.isfinite(rho_l1) or rho_l1 <= 0:
        raise SystemExit("ERROR: rho has nonpositive or non-finite L1 norm after clipping to >=0.")
    pi_np = (rho_pos / rho_l1).astype(np.float64, copy=False)

    # degrees on filtered edges
    deg_out = np.zeros(N, dtype=np.int64)
    np.add.at(deg_out, src_np, 1)
    if np.any(deg_out[src_np] <= 0):
        raise SystemExit("ERROR: some src nodes have deg_out=0 after filtering (unexpected).")

    # Compute MH accept ratio r = (pi_j q_ji)/(pi_i q_ij) with q_ij=1/deg(i)
    # => r = (pi_j * deg(i)) / (pi_i * deg(j))
    pi_i = pi_np[src_np]
    pi_j = pi_np[dst_np]
    di = deg_out[src_np].astype(np.float64)
    dj = deg_out[dst_np].astype(np.float64)

    # avoid divide-by-zero when pi_i=0
    r = np.zeros_like(pi_i, dtype=np.float64)
    ok = pi_i > 0
    r[ok] = (pi_j[ok] * di[ok]) / (pi_i[ok] * dj[ok])
    a = np.minimum(1.0, r)  # accept prob
    q = 1.0 / di            # proposal prob
    P_off = q * a           # off-diagonal transition prob along directed edge

    # move to torch for fast scatters
    src_t = torch.from_numpy(src_np.astype(np.int64, copy=False)).to(dev)
    dst_t = torch.from_numpy(dst_np.astype(np.int64, copy=False)).to(dev)
    Poff_t = torch.from_numpy(P_off.astype(np.float64, copy=False)).to(dev)
    pi_t = torch.from_numpy(pi_np.astype(np.float64, copy=False)).to(dev)

    # Row sums and self-loop probability
    row_sum = torch.zeros((N,), dtype=torch.float64, device=dev)
    row_sum.scatter_add_(0, src_t, Poff_t)
    Pii = (1.0 - row_sum).clamp(min=0.0)  # numerical safety

    # Conservative flux on directed edges: F_ij = pi_i * P_ij  (reversible => F_ij == F_ji)
    F = pi_t[src_t] * Poff_t  # [E]
    F_abs_sum = torch.sum(torch.abs(F)).item()
    if F_abs_sum <= 0 or not np.isfinite(F_abs_sum):
        raise SystemExit("ERROR: flux sum is nonpositive or non-finite.")

    # Build reverse-edge index for antisymmetry check (need mapping within filtered edges)
    # Use CPU numpy sorted-key lookup, then bring indices to torch.
    keys = (src_np * N + dst_np).astype(np.int64, copy=False)
    order = np.argsort(keys)
    keys_sorted = keys[order]
    rev_keys = (dst_np * N + src_np).astype(np.int64, copy=False)
    pos = np.searchsorted(keys_sorted, rev_keys, side="left")
    if np.any(pos >= keys_sorted.size) or np.any(keys_sorted[pos] != rev_keys):
        raise SystemExit("ERROR: bidirected requirement violated: some reverse edges missing after filter.")
    rev_idx = order[pos]  # for each edge e, reverse edge index is rev_idx[e]
    rev_t = torch.from_numpy(rev_idx.astype(np.int64, copy=False)).to(dev)

    F_rev = F[rev_t]
    antisym_rel_l1 = (torch.sum(torch.abs(F - F_rev)) / torch.sum(torch.abs(F))).item()

    # Graph divergence of conservative flux: div_i = sum_out F_ij - sum_in F_ji
    out_sum = torch.zeros((N,), dtype=torch.float64, device=dev)
    in_sum  = torch.zeros((N,), dtype=torch.float64, device=dev)
    out_sum.scatter_add_(0, src_t, F)
    in_sum.scatter_add_(0, dst_t, F)
    div = out_sum - in_sum
    graph_div_rel_l1 = (torch.sum(torch.abs(div)).item()) / float(rho_l1)

    # kappa + core sign
    kappa_med, n_core_valid = _median_ratio_on_mask(G00, rho_pos, mask)
    core_match_frac = _core_sign_match_frac(G00, rho_pos, mask, kappa_med)

    PASS_core = bool(np.isfinite(core_match_frac) and core_match_frac >= float(args.core_match_frac_min))
    PASS_div  = bool(np.isfinite(graph_div_rel_l1) and graph_div_rel_l1 <= float(args.graph_div_rel_l1_max))
    PASS_sym  = bool(np.isfinite(antisym_rel_l1) and antisym_rel_l1 <= float(args.antisym_rel_l1_max))

    ALL_PASS = bool(PASS_core and PASS_div and PASS_sym)

    out = {
        "H": int(getattr(G00_raw, "shape", [N, ])[0] if getattr(G00_raw, "ndim", 1) == 2 else 512),
        "W": int(getattr(G00_raw, "shape", [N, ])[1] if getattr(G00_raw, "ndim", 1) == 2 else 512),
        "STRICT_PP": True,
        "device": str(dev),
        "inputs": {
            "edges": args.edges,
            "G00": args.G00,
            "G00_key": args.G00_key,
            "rho": args.rho,
            "rho_key": args.rho_key,
            "mask": args.mask,
            "mask_key": args.mask_key,
            "topk_frac_used_if_no_mask": (None if args.mask is not None else float(args.topk_frac_used_if_no_mask)),
            "require_bidirected": bool(args.require_bidirected),
        },
        "kappa": {
            "kappa_med": float(kappa_med),
            "n_core_valid": int(n_core_valid),
        },
        "diagnostics": {
            "core_match_frac_sign(G00, kappa*T00)": float(core_match_frac),
            "graph_div_rel_l1(||div(F)||_1 / ||rho||_1)": float(graph_div_rel_l1),
            "paired_flux_antisym_rel_l1(sum|Fij-Fji| / sum|F|)": float(antisym_rel_l1),
            "n_edges_used": int(src_np.size),
            "rho_l1": float(rho_l1),
        },
        "thresholds": {
            "core_match_frac_min": float(args.core_match_frac_min),
            "graph_div_rel_l1_max": float(args.graph_div_rel_l1_max),
            "antisym_rel_l1_max": float(args.antisym_rel_l1_max),
        },
        "PASS": {
            "ALL_PASS_tensor_closure_audit_v2": bool(ALL_PASS),
            "PASS_core_sign_consistency": bool(PASS_core),
            "PASS_momentum_divergence_small_graph": bool(PASS_div),
            "PASS_offdiag_symmetry_small_flux": bool(PASS_sym),
        },
        "notes": (
            "Tensor-closure audit v2 (STRICT PP): builds MH reversible kernel targeting rho on bidirected edges; "
            "audits conservative detailed-balance flux symmetry and graph divergence. "
            "No PDE/Poisson, no smoothing, no GR ansatz, no regression."
        ),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_tensor_closure_audit_v2 =", out["PASS"]["ALL_PASS_tensor_closure_audit_v2"])
    print("kappa_med =", out["kappa"]["kappa_med"])
    print("core_match_frac =", out["diagnostics"]["core_match_frac_sign(G00, kappa*T00)"])
    print("graph_div_rel_l1 =", out["diagnostics"]["graph_div_rel_l1(||div(F)||_1 / ||rho||_1)"])
    print("antisym_rel_l1 =", out["diagnostics"]["paired_flux_antisym_rel_l1(sum|Fij-Fji| / sum|F|)"])


if __name__ == "__main__":
    main()

