#!/usr/bin/env python3
import argparse, json, os
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
try:
    import torch
except Exception:
    torch = None


def _torch_required():
    if torch is None:
        raise SystemExit("ERROR: torch is required.")

def _get_device(device: str):
    _torch_required()
    if device == "cpu":
        return torch.device("cpu")
    if device == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit("ERROR: --device gpu requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    raise SystemExit(f"ERROR: unknown --device {device}")

def _exists(p: Optional[str]) -> bool:
    return bool(p) and os.path.exists(p)

def _as_torch(a, device, dtype):
    return torch.as_tensor(a, device=device, dtype=dtype)


# -------- edges --------
def _read_edges_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    src, dst = [], []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "," in s:
                a, b = s.split(",", 1)
            else:
                parts = s.split()
                if len(parts) < 2:
                    continue
                a, b = parts[0], parts[1]
            src.append(int(a)); dst.append(int(b))
    if not src:
        raise SystemExit(f"ERROR: no edges parsed from {path}")
    return np.asarray(src, np.int64), np.asarray(dst, np.int64)


# -------- tau fields --------
def _reduce_to_2d(a: np.ndarray, H: int, W: int) -> np.ndarray:
    if a.shape == (H, W):
        return a
    if a.ndim == 1 and a.size == H * W:
        return a.reshape(H, W)
    raise SystemExit(f"ERROR: cannot reduce {a.shape} to {(H,W)}")

def _load_tau_fields(tau_npz: str, H: int, W: int, dflat_key: str, mask_key: str) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(tau_npz, allow_pickle=False)
    if dflat_key not in z or mask_key not in z:
        raise SystemExit(f"ERROR: tau_npz missing {dflat_key} or {mask_key}. Keys={list(z.keys())}")
    d_flat = _reduce_to_2d(z[dflat_key], H, W).astype(np.float64)
    core   = _reduce_to_2d(z[mask_key], H, W).astype(bool)
    return d_flat, core

def _quantile_band(d_flat: np.ndarray, core: np.ndarray, qlo: float, qhi: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    vals = d_flat[~core]
    vals = vals[np.isfinite(vals)]
    if vals.size < 1000:
        raise SystemExit(f"ERROR: too few finite d_flat values outside core: {vals.size}")
    lo = float(np.quantile(vals, qlo))
    hi = float(np.quantile(vals, qhi))
    ring = (~core) & np.isfinite(d_flat) & (d_flat >= lo) & (d_flat <= hi)
    return ring, {"qlo": float(qlo), "qhi": float(qhi), "d_flat_lo": lo, "d_flat_hi": hi, "ring_frac": float(ring.mean())}

def _mass_center_from_mask(core: np.ndarray) -> Tuple[float, float]:
    rr, cc = np.where(core)
    if rr.size < 10:
        raise SystemExit("ERROR: mass_core_mask too small.")
    return float(rr.mean()), float(cc.mean())

def _parse_bands(bands_str: str) -> List[Tuple[float, float]]:
    out = []
    for part in bands_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise SystemExit(f"ERROR: band '{part}' must be like qlo-qhi")
        a, b = part.split("-", 1)
        qlo, qhi = float(a), float(b)
        if not (0.0 < qlo < qhi < 1.0):
            raise SystemExit(f"ERROR: invalid band {part}")
        out.append((qlo, qhi))
    if not out:
        raise SystemExit("ERROR: no bands parsed")
    return out


# -------- utilities --------
@torch.no_grad()
def _outdeg(src: torch.Tensor, N: int) -> torch.Tensor:
    out = torch.zeros((N,), device=src.device, dtype=torch.int32)
    ones = torch.ones_like(src, dtype=torch.int32)
    out.index_add_(0, src, ones)
    return out

@torch.no_grad()
def _ring_keep(src: torch.Tensor, dst: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    return ring[src] & ring[dst]

@torch.no_grad()
def _paired_keys(src: torch.Tensor, dst: torch.Tensor, keep: torch.Tensor, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    s = src[keep]
    d = dst[keep]
    a = torch.minimum(s, d)
    b = torch.maximum(s, d)
    key = a * N + b
    uniq, inv, counts = torch.unique(key, return_inverse=True, return_counts=True)
    paired_on_kept = (counts[inv] == 2)
    return key, paired_on_kept

def _intersect_keys_cpu(k1: torch.Tensor, k2: torch.Tensor) -> np.ndarray:
    a = k1.detach().cpu().numpy()
    b = k2.detach().cpu().numpy()
    if a.size == 0 or b.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.intersect1d(np.unique(a), np.unique(b), assume_unique=False)

@torch.no_grad()
def _select_common_paired_edges(src: torch.Tensor, dst: torch.Tensor,
                               ring: torch.Tensor, N: int,
                               common_keys: torch.Tensor) -> torch.Tensor:
    keep0 = _ring_keep(src, dst, ring)
    key0, paired0 = _paired_keys(src, dst, keep0, N)

    m = torch.zeros_like(keep0)
    m[keep0] = paired0
    if common_keys.numel() == 0:
        return m & False

    s = src[m]
    d = dst[m]
    a = torch.minimum(s, d)
    b = torch.maximum(s, d)
    k = a * N + b
    in_common = torch.isin(k, common_keys)

    out = torch.zeros_like(m)
    idx = torch.nonzero(m, as_tuple=False).squeeze(1)
    out[idx[in_common]] = True
    return out


# -------- local subgraph stationary pi --------
@torch.no_grad()
def _compress_subgraph(src: torch.Tensor, dst: torch.Tensor, mask_edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (src2, dst2, nodes_orig) where nodes_orig maps compact index -> original node id.
    """
    s = src[mask_edges]
    d = dst[mask_edges]
    nodes = torch.unique(torch.cat([s, d], dim=0), sorted=True)
    # map original node ids to 0..M-1
    # searchsorted works because nodes sorted unique
    src2 = torch.searchsorted(nodes, s)
    dst2 = torch.searchsorted(nodes, d)
    return src2.to(torch.int64), dst2.to(torch.int64), nodes.to(torch.int64)

@torch.no_grad()
def _power_iter_stationary_compact(src2: torch.Tensor, dst2: torch.Tensor, M: int,
                                  iters: int, tol: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
    od = _outdeg(src2, M).to(torch.float32)
    if (od == 0).any():
        # sinks inside band; stationary still defined but power iter with uniform outgoing breaks
        raise SystemExit("ERROR: band-subgraph has sinks (outdeg=0).")
    pi = od / (od.sum() + 1e-12)
    hist = {"iters": 0, "tol": float(tol), "l1_last": None, "reached_tol": False}
    for t in range(iters):
        pi_new = torch.zeros_like(pi)
        w = pi[src2] / od[src2]
        pi_new.index_add_(0, dst2, w)
        pi_new = pi_new / (pi_new.sum() + 1e-12)
        l1 = torch.sum(torch.abs(pi_new - pi)).item()
        pi = pi_new
        hist["iters"] = t + 1
        hist["l1_last"] = float(l1)
        if l1 < tol:
            hist["reached_tol"] = True
            break
    return pi, hist


# -------- circulation on masked edges --------
@torch.no_grad()
def _tangential_frac_masked(src: torch.Tensor, dst: torch.Tensor,
                            od_full: torch.Tensor,
                            pi_full: Optional[torch.Tensor],
                            mask_edges: torch.Tensor,
                            H: int, W: int,
                            cy: float, cx: float,
                            weight_mode: str) -> Dict[str, Any]:
    n = int(mask_edges.sum().item())
    if n < 1:
        return {"APPLICABLE": False, "n_edges": 0, "weight_mode": weight_mode}

    s = src[mask_edges]
    d = dst[mask_edges]

    sr = torch.div(s, W, rounding_mode="floor").to(torch.float32)
    sc = (s - (sr.to(torch.int64) * W)).to(torch.float32)
    dr = torch.div(d, W, rounding_mode="floor").to(torch.float32)
    dc = (d - (dr.to(torch.int64) * W)).to(torch.float32)

    mr = 0.5 * (sr + dr)
    mc = 0.5 * (sc + dc)

    dx = (dc - sc)
    dy = (dr - sr)
    step = torch.sqrt(dx * dx + dy * dy + 1e-12)

    x = (mc - float(cx))
    y = (mr - float(cy))
    rad = torch.sqrt(x * x + y * y + 1e-12)
    tx = (-y) / rad
    ty = (x) / rad
    comp = dx * tx + dy * ty

    if weight_mode == "uniform":
        w = 1.0 / od_full[s].to(torch.float32)
    elif weight_mode == "pi":
        if pi_full is None:
            raise SystemExit("ERROR: weight_mode=pi requires pi_full")
        w = pi_full[s] / od_full[s].to(torch.float32)
    else:
        raise SystemExit("ERROR: unknown weight_mode")

    net = torch.sum(w * comp).item()
    denom = torch.sum(w * step).item() + 1e-18
    frac = net / denom

    return {
        "APPLICABLE": True,
        "n_edges": n,
        "net": float(net),
        "denom_step": float(denom),
        "frac_net_over_step": float(frac),
        "weight_mode": weight_mode,
    }


def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP ring circulation (v7): common bidirected pairs AND local per-band stationary pi (induced subgraph) to avoid global-pi contamination."
    )
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--case", default="strong_pf010")

    ap.add_argument("--tau_npz", required=True)
    ap.add_argument("--dflat_key", default="d_flat")
    ap.add_argument("--mask_key", default="mass_core_mask")

    ap.add_argument("--edges_flat", required=True)
    ap.add_argument("--edges_curved", required=True)

    ap.add_argument("--bands", default="0.55-0.60,0.60-0.65,0.65-0.70,0.70-0.75,0.75-0.80")
    ap.add_argument("--min_common_edges", type=int, default=200,
                    help="Band applicable iff n_edges_flat==n_edges_curved >= this after common-paired filtering.")
    ap.add_argument("--abs_max", type=float, default=0.03)
    ap.add_argument("--delta_max", type=float, default=0.03)

    ap.add_argument("--pi_iters", type=int, default=5000)
    ap.add_argument("--pi_tol", type=float, default=1e-6)
    ap.add_argument("--pi_l1_last_max", type=float, default=5e-4)

    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--output", required=True)

    args = ap.parse_args()
    H, W = int(args.H), int(args.W)
    N = H * W

    for p in (args.tau_npz, args.edges_flat, args.edges_curved):
        if not _exists(p):
            raise SystemExit(f"ERROR: not found: {p}")

    device = _get_device(args.device)

    d_flat, core = _load_tau_fields(args.tau_npz, H, W, args.dflat_key, args.mask_key)
    cy, cx = _mass_center_from_mask(core)
    bands = _parse_bands(args.bands)

    def load_edges(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        src_np, dst_np = _read_edges_txt(path)
        if src_np.max() >= N or dst_np.max() >= N:
            raise SystemExit(f"ERROR: edges {path} contain node id >= {N}")
        return _as_torch(src_np, device=device, dtype=torch.int64), _as_torch(dst_np, device=device, dtype=torch.int64)

    srcF, dstF = load_edges(args.edges_flat)
    srcC, dstC = load_edges(args.edges_curved)

    # full-graph outdeg used for uniform weights
    odF_full = _outdeg(srcF, N).to(torch.float32)
    odC_full = _outdeg(srcC, N).to(torch.float32)

    per_band = []
    deltas_u, deltas_p = [], []
    abs_fail_u, abs_fail_p = 0, 0
    pi_hist_per_band = []

    for (qlo, qhi) in bands:
        ring_np, ring_meta = _quantile_band(d_flat, core, qlo, qhi)
        ring = _as_torch(ring_np.reshape(-1), device=device, dtype=torch.bool)

        keepF0 = _ring_keep(srcF, dstF, ring)
        keepC0 = _ring_keep(srcC, dstC, ring)

        keyF, pairedF = _paired_keys(srcF, dstF, keepF0, N)
        keyC, pairedC = _paired_keys(srcC, dstC, keepC0, N)

        keysF = keyF[pairedF]
        keysC = keyC[pairedC]
        common_cpu = _intersect_keys_cpu(keysF, keysC)
        common_keys = _as_torch(common_cpu, device=device, dtype=torch.int64)

        mF = _select_common_paired_edges(srcF, dstF, ring, N, common_keys)
        mC = _select_common_paired_edges(srcC, dstC, ring, N, common_keys)

        nF = int(mF.sum().item())
        nC = int(mC.sum().item())
        n_common = int(common_keys.numel())
        band_app = bool((nF == nC) and (nF >= int(args.min_common_edges)) and (n_common > 0))

        # uniform (uses full-graph outdeg but on common-paired edges only)
        flat_u = _tangential_frac_masked(srcF, dstF, odF_full, None, mF, H, W, cy, cx, "uniform")
        curv_u = _tangential_frac_masked(srcC, dstC, odC_full, None, mC, H, W, cy, cx, "uniform")
        dU = None
        if band_app and flat_u["APPLICABLE"] and curv_u["APPLICABLE"]:
            dU = float(curv_u["frac_net_over_step"] - flat_u["frac_net_over_step"])
            deltas_u.append(dU)
            if abs(curv_u["frac_net_over_step"]) > float(args.abs_max) or abs(flat_u["frac_net_over_step"]) > float(args.abs_max):
                abs_fail_u += 1

        # local pi per band per graph (induced subgraph on common-paired edges)
        pi_local_flat = None
        pi_local_curv = None
        hist_flat = {"APPLICABLE": False}
        hist_curv = {"APPLICABLE": False}
        flat_p = {"APPLICABLE": False, "n_edges": nF, "weight_mode": "pi_local"}
        curv_p = {"APPLICABLE": False, "n_edges": nC, "weight_mode": "pi_local"}
        dP = None

        if band_app:
            # compress each band-subgraph
            srcF2, dstF2, nodesF = _compress_subgraph(srcF, dstF, mF)
            srcC2, dstC2, nodesC = _compress_subgraph(srcC, dstC, mC)
            M_F = int(nodesF.numel())
            M_C = int(nodesC.numel())

            # compute local outdeg (compact)
            try:
                piF_loc, histF = _power_iter_stationary_compact(srcF2, dstF2, M_F, int(args.pi_iters), float(args.pi_tol))
                piC_loc, histC = _power_iter_stationary_compact(srcC2, dstC2, M_C, int(args.pi_iters), float(args.pi_tol))
                hist_flat = {"APPLICABLE": True, **histF, "M_nodes": M_F}
                hist_curv = {"APPLICABLE": True, **histC, "M_nodes": M_C}

                # expand to full N only on participating nodes (sparse map)
                pi_local_flat = torch.zeros((N,), device=device, dtype=torch.float32)
                pi_local_curv = torch.zeros((N,), device=device, dtype=torch.float32)
                pi_local_flat[nodesF] = piF_loc.to(torch.float32)
                pi_local_curv[nodesC] = piC_loc.to(torch.float32)

                # for weights on masked edges, need subgraph outdeg on original node ids
                odF_sub = torch.zeros((N,), device=device, dtype=torch.int32)
                odC_sub = torch.zeros((N,), device=device, dtype=torch.int32)
                odF_sub.index_add_(0, srcF[mF], torch.ones_like(srcF[mF], dtype=torch.int32))
                odC_sub.index_add_(0, srcC[mC], torch.ones_like(srcC[mC], dtype=torch.int32))

                # compute pi-weighted circulation using local pi and subgraph outdeg
                # (same functional form as before: w = pi[s] / outdeg_sub[s])
                def _tangential_pi_local(src, dst, od_sub, pi_local, m, tag):
                    n = int(m.sum().item())
                    if n < 1:
                        return {"APPLICABLE": False, "n_edges": 0, "weight_mode": "pi_local", "tag": tag}
                    s = src[m]; d = dst[m]

                    sr = torch.div(s, W, rounding_mode="floor").to(torch.float32)
                    sc = (s - (sr.to(torch.int64) * W)).to(torch.float32)
                    dr = torch.div(d, W, rounding_mode="floor").to(torch.float32)
                    dc = (d - (dr.to(torch.int64) * W)).to(torch.float32)

                    mr = 0.5 * (sr + dr)
                    mc = 0.5 * (sc + dc)

                    dx = (dc - sc)
                    dy = (dr - sr)
                    step = torch.sqrt(dx * dx + dy * dy + 1e-12)

                    x = (mc - float(cx))
                    y = (mr - float(cy))
                    rad = torch.sqrt(x * x + y * y + 1e-12)
                    tx = (-y) / rad
                    ty = (x) / rad
                    comp = dx * tx + dy * ty

                    od = od_sub[s].to(torch.float32)
                    if (od <= 0).any():
                        return {"APPLICABLE": False, "n_edges": n, "weight_mode": "pi_local", "tag": tag, "reason": "subgraph sink(s)"}

                    w = pi_local[s] / od
                    net = torch.sum(w * comp).item()
                    denom = torch.sum(w * step).item() + 1e-18
                    frac = net / denom

                    return {
                        "APPLICABLE": True,
                        "n_edges": n,
                        "net": float(net),
                        "denom_step": float(denom),
                        "frac_net_over_step": float(frac),
                        "weight_mode": "pi_local",
                        "tag": tag,
                    }

                flat_p = _tangential_pi_local(srcF, dstF, odF_sub, pi_local_flat, mF, "flat")
                curv_p = _tangential_pi_local(srcC, dstC, odC_sub, pi_local_curv, mC, "curved")

                if flat_p["APPLICABLE"] and curv_p["APPLICABLE"]:
                    dP = float(curv_p["frac_net_over_step"] - flat_p["frac_net_over_step"])
                    deltas_p.append(dP)
                    if abs(curv_p["frac_net_over_step"]) > float(args.abs_max) or abs(flat_p["frac_net_over_step"]) > float(args.abs_max):
                        abs_fail_p += 1

            except SystemExit as e:
                # record failure but keep going
                hist_flat = {"APPLICABLE": False, "error": str(e)}
                hist_curv = {"APPLICABLE": False, "error": str(e)}

        pi_hist_per_band.append({"band": ring_meta, "flat_local_pi": hist_flat, "curved_local_pi": hist_curv})

        per_band.append({
            "band": ring_meta,
            "common": {"n_common_pairs": n_common, "n_edges_flat": nF, "n_edges_curved": nC, "band_applicable": band_app, "min_common_edges": int(args.min_common_edges)},
            "uniform": {"flat": flat_u, "curved": curv_u, "delta": dU},
            "pi_local": {"flat": flat_p, "curved": curv_p, "delta": dP},
        })

    def _med_abs(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not vals:
            return None, None
        m = float(np.median(vals))
        return m, float(abs(m))

    med_u, medabs_u = _med_abs(deltas_u)
    med_p, medabs_p = _med_abs(deltas_p)

    PASS_delta_u = (medabs_u is not None) and (medabs_u <= float(args.delta_max))
    PASS_delta_p = (medabs_p is not None) and (medabs_p <= float(args.delta_max))
    PASS_abs_u = (len(deltas_u) > 0) and (abs_fail_u == 0)
    PASS_abs_p = (len(deltas_p) > 0) and (abs_fail_p == 0)

    # pi convergence gate: all applicable bands must have l1_last <= pi_l1_last_max (both flat+curved)
    pi_band_ok = True
    n_pi_bands = 0
    for h in pi_hist_per_band:
        bf = h["flat_local_pi"]
        bc = h["curved_local_pi"]
        # only count if band was applicable (we stored band_applicable there in per_band; reuse)
        # safest: check presence of numeric l1_last
        if bf.get("APPLICABLE") and bc.get("APPLICABLE") and ("l1_last" in bf) and ("l1_last" in bc):
            n_pi_bands += 1
            if not (bf["l1_last"] <= float(args.pi_l1_last_max) and bc["l1_last"] <= float(args.pi_l1_last_max)):
                pi_band_ok = False

    ALL_PASS = bool(pi_band_ok and PASS_delta_u and PASS_delta_p and PASS_abs_u and PASS_abs_p)

    out: Dict[str, Any] = {
        "H": H, "W": W, "case": args.case,
        "STRICT_PP": True,
        "device": str(device),
        "inputs": {
            "tau_npz": args.tau_npz,
            "dflat_key": args.dflat_key,
            "mask_key": args.mask_key,
            "edges_flat": args.edges_flat,
            "edges_curved": args.edges_curved,
            "bands": args.bands,
            "min_common_edges": int(args.min_common_edges),
            "abs_max": float(args.abs_max),
            "delta_max": float(args.delta_max),
            "pi_iters": int(args.pi_iters),
            "pi_tol": float(args.pi_tol),
            "pi_l1_last_max": float(args.pi_l1_last_max),
        },
        "mass_center_rc": [float(cy), float(cx)],
        "pi_local_per_band": pi_hist_per_band,
        "per_band": per_band,
        "results": {
            "n_applicable_bands_uniform": int(len(deltas_u)),
            "median_delta_uniform": med_u,
            "median_abs_delta_uniform": medabs_u,
            "n_applicable_bands_pi_local": int(len(deltas_p)),
            "median_delta_pi_local": med_p,
            "median_abs_delta_pi_local": medabs_p,
            "abs_fail_bands_uniform": int(abs_fail_u),
            "abs_fail_bands_pi_local": int(abs_fail_p),
            "n_pi_local_bands_checked": int(n_pi_bands),
        },
        "PASS": {
            "APPLICABLE": True,
            "PASS_pi_local_converged_all_applicable_bands": bool(pi_band_ok),
            "PASS_abs_small_uniform_all_bands": bool(PASS_abs_u),
            "PASS_abs_small_pi_local_all_bands": bool(PASS_abs_p),
            "PASS_delta_uniform_median": bool(PASS_delta_u),
            "PASS_delta_pi_local_median": bool(PASS_delta_p),
            "ALL_PASS_offdiag_ring_circulation_commonpaired_localpi_v7": bool(ALL_PASS),
        },
        "notes": (
            "STRICT PP ring circulation diagnostic (v7). Uses ONLY undirected pairs bidirected in BOTH graphs (common-paired) within each band, "
            "and computes stationary pi LOCALLY on the induced band subgraph (per graph) to prevent global-pi contamination from mass core / global inhomogeneity. "
            "No PDE/Poisson, no smoothing, no GR ansatz, no regression."
        ),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_offdiag_ring_circulation_commonpaired_localpi_v7 =", out["PASS"]["ALL_PASS_offdiag_ring_circulation_commonpaired_localpi_v7"])
    print("median_abs_delta_uniform =", out["results"]["median_abs_delta_uniform"],
          "median_abs_delta_pi_local =", out["results"]["median_abs_delta_pi_local"])
    print("n_applicable_bands_u/pi_local =", out["results"]["n_applicable_bands_uniform"], out["results"]["n_applicable_bands_pi_local"])
    print("pi_local_bands_checked =", out["results"]["n_pi_local_bands_checked"])


if __name__ == "__main__":
    main()

