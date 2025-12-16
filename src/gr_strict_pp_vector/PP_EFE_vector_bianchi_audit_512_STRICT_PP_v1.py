#!/usr/bin/env python3
import argparse, json, os, math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:
    torch = None


# ----------------------------
# Utilities
# ----------------------------
def load_npz(path: str) -> Any:
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: NPZ not found: {path}")
    return np.load(path)

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def parse_bands(s: str) -> List[Tuple[float, float]]:
    # "0.55-0.60,0.60-0.65"
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        qlo = float(a.strip())
        qhi = float(b.strip())
        if not (0.0 <= qlo < qhi <= 1.0):
            raise SystemExit(f"ERROR: bad band: {part}")
        out.append((qlo, qhi))
    if not out:
        raise SystemExit("ERROR: empty --bands")
    return out

def read_edge_txt(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: edges file not found: {path}")
    edges = []
    with open(path, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 2:
                raise SystemExit(f"ERROR: bad edge line {ln} in {path}: {line}")
            u = int(toks[0]); v = int(toks[1])
            edges.append((u, v))
    if not edges:
        raise SystemExit(f"ERROR: no edges parsed from {path}")
    return np.asarray(edges, dtype=np.int64)

def get_device(dev: str) -> "torch.device":
    if torch is None:
        raise SystemExit("ERROR: torch is required for this audit (CPU torch is OK).")
    d = (dev or "gpu").strip().lower()
    if d in ("cpu",):
        return torch.device("cpu")
    if d in ("cuda", "gpu"):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if d in ("hip",):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if ":" in d:
        try:
            return torch.device(d)
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")

def quantile_band_mask(dflat: np.ndarray, qlo: float, qhi: float) -> np.ndarray:
    finite = np.isfinite(dflat)
    if finite.sum() < 10:
        raise SystemExit("ERROR: too few finite d_flat values for quantiles.")
    vals = dflat[finite]
    lo = np.quantile(vals, qlo)
    hi = np.quantile(vals, qhi)
    m = finite & (dflat >= lo) & (dflat <= hi)
    return m

def make_compact_index(nodes: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    nodes_sorted = np.unique(nodes.astype(np.int64))
    mp = {int(n): i for i, n in enumerate(nodes_sorted.tolist())}
    return nodes_sorted, mp

def _band_tag(qlo: float, qhi: float) -> str:
    # stable npz keys: 0.55 -> 0p55
    def f(x: float) -> str:
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")
    return f"{f(qlo)}_{f(qhi)}"


# ----------------------------
# Core: build ring subgraph, stationary pi, flows, divergence(Δ)
# ----------------------------
def restrict_edges_to_nodes(edges: np.ndarray, node_mask: np.ndarray) -> np.ndarray:
    u = edges[:, 0]; v = edges[:, 1]
    m = node_mask[u] & node_mask[v]
    return edges[m]

def require_bidirected(edges: np.ndarray) -> np.ndarray:
    s = set((int(u), int(v)) for u, v in edges.tolist())
    keep = [(u, v) for (u, v) in s if (v, u) in s]
    if not keep:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(keep, dtype=np.int64)

def edge_intersection(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    sb = set((int(u), int(v)) for u, v in B.tolist())
    keep = [(int(u), int(v)) for u, v in A.tolist() if (int(u), int(v)) in sb]
    if not keep:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(keep, dtype=np.int64)

def build_outdeg_compact(edges: np.ndarray, mp: Dict[int, int], n_compact: int) -> np.ndarray:
    outdeg = np.zeros((n_compact,), dtype=np.int64)
    for (u, v) in edges.tolist():
        outdeg[mp[int(u)]] += 1
    return outdeg

def power_iter_pi(
    edges: np.ndarray,
    nodes_sorted: np.ndarray,
    mp: Dict[int, int],
    outdeg: np.ndarray,
    device: "torch.device",
    iters: int,
    tol: float,
    l1_last_max: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if torch is None:
        raise SystemExit("ERROR: torch required")

    n = len(nodes_sorted)
    if n == 0:
        raise SystemExit("ERROR: empty node set for pi")
    if edges.shape[0] == 0:
        raise SystemExit("ERROR: empty edge set for pi")

    src = torch.tensor([mp[int(u)] for u in edges[:, 0].tolist()], dtype=torch.int64, device=device)
    dst = torch.tensor([mp[int(v)] for v in edges[:, 1].tolist()], dtype=torch.int64, device=device)
    outdeg_t = torch.tensor(outdeg, dtype=torch.float64, device=device)

    good = (outdeg_t > 0)
    pi = torch.zeros((n,), dtype=torch.float64, device=device)
    if bool(good.any().item()):
        pi[good] = 1.0 / float(good.sum().item())
    else:
        raise SystemExit("ERROR: all nodes have outdeg=0 in compact graph")

    last_l1 = None
    converged = False

    for t in range(int(iters)):
        pi_new = torch.zeros_like(pi)
        contrib = pi[src] / outdeg_t[src]
        pi_new.index_add_(0, dst, contrib)
        s = pi_new.sum()
        if float(s.item()) <= 0:
            raise SystemExit("ERROR: pi_new sum <= 0 (graph likely degenerate)")
        pi_new = pi_new / s

        l1 = torch.sum(torch.abs(pi_new - pi)).item()
        last_l1 = float(l1)
        pi = pi_new

        if last_l1 < tol or last_l1 <= l1_last_max:
            converged = True
            break

    pi_np = pi.detach().cpu().numpy().astype(np.float64)
    diag = {
        "pi_iters_used": float(t + 1),
        "pi_l1_last": float(last_l1 if last_l1 is not None else math.nan),
        "pi_converged": bool(converged),
    }
    return pi_np, diag

def flow_on_edges(pi: np.ndarray, edges: np.ndarray, mp: Dict[int, int], outdeg: np.ndarray) -> np.ndarray:
    src_idx = np.asarray([mp[int(u)] for u in edges[:, 0].tolist()], dtype=np.int64)
    denom = outdeg[src_idx].astype(np.float64)
    denom = np.where(denom <= 0, 1.0, denom)
    f = pi[src_idx] / denom
    return f.astype(np.float64)

def divergence_from_edge_flow(edges: np.ndarray, flow: np.ndarray, nodes_sorted: np.ndarray, mp: Dict[int, int]) -> np.ndarray:
    n = len(nodes_sorted)
    div = np.zeros((n,), dtype=np.float64)
    for (u, v), f in zip(edges.tolist(), flow.tolist()):
        iu = mp[int(u)]
        iv = mp[int(v)]
        div[iu] += f
        div[iv] -= f
    return div


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP vector-Bianchi proxy audit @512: checks small divergence of Δflow (curved-flat) on common ring subgraph."
    )
    ap.add_argument("--device", default="gpu", help="cpu|gpu|cuda|hip (gpu maps to cuda/hip if available)")
    ap.add_argument("--case", required=True, help="ms080|strong_pf010 (label only)")
    ap.add_argument("--tau_npz", required=True, help="PP_markov_tau_geometry_512_<case>_PPV1.npz (needs d_flat + mass_core_mask)")
    ap.add_argument("--dflat_key", default="d_flat")
    ap.add_argument("--mask_key", default="mass_core_mask")

    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)

    ap.add_argument("--edges_flat", required=True)
    ap.add_argument("--edges_curved", required=True)

    ap.add_argument("--bands", default="0.55-0.60,0.60-0.65,0.65-0.70,0.70-0.75,0.75-0.80")
    ap.add_argument("--min_common_edges", type=int, default=160)
    ap.add_argument("--require_bidirected", action="store_true", help="If set, require bidirected edges before intersection.")

    ap.add_argument("--pi_iters", type=int, default=5000)
    ap.add_argument("--pi_tol", type=float, default=1e-8)
    ap.add_argument("--pi_l1_last_max", type=float, default=5e-4)

    ap.add_argument("--mean_scaled_max", type=float, default=1e-6,
                    help="Gate on mean(|divΔ|)*n_nodes <= this.")
    ap.add_argument("--p99_scaled_max", type=float, default=1e-4,
                    help="Gate on p99(|divΔ|)*n_nodes <= this.")

    ap.add_argument("--npz_out_fields", default="",
                    help="Optional: write per-band compact fields to NPZ (nodes/div/absdiv). No impact on PASS.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if torch is None:
        raise SystemExit("ERROR: torch is required (CPU torch is OK).")

    device = get_device(args.device)

    Z = load_npz(args.tau_npz)
    if args.dflat_key not in Z.files:
        raise SystemExit(f"ERROR: {args.dflat_key} not in tau_npz keys: {sorted(Z.files)}")
    if args.mask_key not in Z.files:
        raise SystemExit(f"ERROR: {args.mask_key} not in tau_npz keys: {sorted(Z.files)}")

    dflat = Z[args.dflat_key].astype(np.float64).reshape(-1)
    mask_core = Z[args.mask_key].astype(bool).reshape(-1)

    N = int(dflat.shape[0])
    if mask_core.shape[0] != N:
        raise SystemExit("ERROR: d_flat and mass_core_mask size mismatch")
    if N != int(args.H) * int(args.W):
        raise SystemExit(f"ERROR: N={N} but H*W={int(args.H)*int(args.W)} (check --H/--W or NPZ)")

    offcore = ~mask_core

    Ef = read_edge_txt(args.edges_flat)
    Ec = read_edge_txt(args.edges_curved)

    bands = parse_bands(args.bands)

    per_band: List[Dict[str, Any]] = []
    n_applicable = 0
    pass_all = True

    # Optional per-band NPZ export
    npz_fields: Dict[str, np.ndarray] = {}

    for (qlo, qhi) in bands:
        band_mask = offcore & quantile_band_mask(dflat, qlo, qhi)
        n_nodes_band = int(band_mask.sum())

        Ef_b = restrict_edges_to_nodes(Ef, band_mask)
        Ec_b = restrict_edges_to_nodes(Ec, band_mask)

        if args.require_bidirected:
            Ef_b = require_bidirected(Ef_b)
            Ec_b = require_bidirected(Ec_b)

        E_common = edge_intersection(Ef_b, Ec_b)
        n_common = int(E_common.shape[0])
        band_app = bool(n_common >= int(args.min_common_edges))

        band_rec: Dict[str, Any] = {
            "band": {"qlo": float(qlo), "qhi": float(qhi)},
            "nodes": {"n_band_offcore": n_nodes_band},
            "common": {
                "n_common_edges": n_common,
                "band_applicable": band_app,
                "min_common_edges": int(args.min_common_edges),
            },
            "pi": {"flat": None, "curved": None},
            "divergence": None,
            "PASS": None,
        }

        if not band_app:
            per_band.append(band_rec)
            continue

        nodes_in_common = np.unique(np.concatenate([E_common[:, 0], E_common[:, 1]]))
        nodes_sorted, mp = make_compact_index(nodes_in_common)
        n_compact = int(len(nodes_sorted))

        node_mask_common = np.zeros((N,), dtype=bool)
        node_mask_common[nodes_sorted] = True

        Ef_c = restrict_edges_to_nodes(Ef_b, node_mask_common)
        Ec_c = restrict_edges_to_nodes(Ec_b, node_mask_common)

        if args.require_bidirected:
            Ef_c = require_bidirected(Ef_c)
            Ec_c = require_bidirected(Ec_c)

        E_common2 = edge_intersection(Ef_c, Ec_c)
        n_common2 = int(E_common2.shape[0])
        band_rec["common"]["n_common_edges_compact"] = n_common2

        if n_common2 < int(args.min_common_edges):
            band_rec["common"]["band_applicable"] = False
            per_band.append(band_rec)
            continue

        outdeg_f = build_outdeg_compact(Ef_c, mp, n_compact)
        outdeg_c = build_outdeg_compact(Ec_c, mp, n_compact)

        pi_f, diag_f = power_iter_pi(
            edges=Ef_c, nodes_sorted=nodes_sorted, mp=mp, outdeg=outdeg_f,
            device=device, iters=args.pi_iters, tol=args.pi_tol, l1_last_max=args.pi_l1_last_max
        )
        pi_c, diag_c = power_iter_pi(
            edges=Ec_c, nodes_sorted=nodes_sorted, mp=mp, outdeg=outdeg_c,
            device=device, iters=args.pi_iters, tol=args.pi_tol, l1_last_max=args.pi_l1_last_max
        )

        f_flow = flow_on_edges(pi_f, E_common2, mp, outdeg_f)
        c_flow = flow_on_edges(pi_c, E_common2, mp, outdeg_c)
        d_flow = c_flow - f_flow

        div = divergence_from_edge_flow(E_common2, d_flow, nodes_sorted, mp)
        abs_div = np.abs(div)

        mean_scaled = float(abs_div.mean() * n_compact)
        p99_scaled = float(np.quantile(abs_div, 0.99) * n_compact)

        pass_mean = bool(mean_scaled <= float(args.mean_scaled_max))
        pass_p99 = bool(p99_scaled <= float(args.p99_scaled_max))
        band_pass = bool(pass_mean and pass_p99 and bool(diag_f["pi_converged"]) and bool(diag_c["pi_converged"]))

        band_rec["pi"]["flat"] = diag_f
        band_rec["pi"]["curved"] = diag_c
        band_rec["divergence"] = {
            "n_compact_nodes": n_compact,
            "mean_abs_div_scaled": mean_scaled,
            "p99_abs_div_scaled": p99_scaled,
            "thresholds": {
                "mean_scaled_max": float(args.mean_scaled_max),
                "p99_scaled_max": float(args.p99_scaled_max),
            },
        }
        band_rec["PASS"] = {
            "PASS_pi_converged_flat": bool(diag_f["pi_converged"]),
            "PASS_pi_converged_curved": bool(diag_c["pi_converged"]),
            "PASS_div_mean_scaled_small": pass_mean,
            "PASS_div_p99_scaled_small": pass_p99,
            "ALL_PASS_band": band_pass,
        }

        per_band.append(band_rec)
        n_applicable += 1
        if not band_pass:
            pass_all = False

        # Optional NPZ export (compact)
        if args.npz_out_fields:
            tag = _band_tag(qlo, qhi)
            npz_fields[f"nodes_{tag}"] = nodes_sorted.astype(np.int64)
            npz_fields[f"div_{tag}"] = div.astype(np.float64)
            npz_fields[f"absdiv_{tag}"] = abs_div.astype(np.float64)

    APPLICABLE = bool(n_applicable > 0)

    mean_scaled_vals = []
    p99_scaled_vals = []
    for b in per_band:
        if b.get("divergence") is not None and b.get("common", {}).get("band_applicable") is True:
            mean_scaled_vals.append(float(b["divergence"]["mean_abs_div_scaled"]))
            p99_scaled_vals.append(float(b["divergence"]["p99_abs_div_scaled"]))

    # Suite-grade PASS naming (match scalar audit style) + keep existing names
    PASS_div_mean_scaled_offcore_small = bool(APPLICABLE and all(
        (b.get("PASS") or {}).get("PASS_div_mean_scaled_small") is True
        for b in per_band if b.get("PASS") is not None
    ))
    PASS_div_p99_scaled_offcore_small = bool(APPLICABLE and all(
        (b.get("PASS") or {}).get("PASS_div_p99_scaled_small") is True
        for b in per_band if b.get("PASS") is not None
    ))
    PASS_pi_converged_all = bool(APPLICABLE and all(
        (b.get("PASS") or {}).get("PASS_pi_converged_flat") is True and (b.get("PASS") or {}).get("PASS_pi_converged_curved") is True
        for b in per_band if b.get("PASS") is not None
    ))

    ALL_PASS_vector_bianchi_proxy_audit_v1 = bool(APPLICABLE and pass_all)
    ALL_PASS_bianchi_proxy_audit_v2 = bool(APPLICABLE and pass_all)

    out = {
        "H": int(args.H),
        "W": int(args.W),
        "N": int(N),
        "STRICT_PP": True,
        "device": str(device),
        "case": args.case,

        "inputs": {
            "tau_npz": args.tau_npz,
            "dflat_key": args.dflat_key,
            "mask_key": args.mask_key,
            "edges_flat": args.edges_flat,
            "edges_curved": args.edges_curved,
            "bands": args.bands,
            "min_common_edges": int(args.min_common_edges),
            "require_bidirected": bool(args.require_bidirected),
            "npz_out_fields": (args.npz_out_fields or None),
        },

        # jq-friendly status block (like other artifacts)
        "status": {
            "APPLICABLE": bool(APPLICABLE),
            "PASS_vector_bianchi_proxy_audit_512_STRICT_PP_v1": bool(ALL_PASS_vector_bianchi_proxy_audit_v1),
            "PASS_bianchi_proxy_audit_v2": bool(ALL_PASS_bianchi_proxy_audit_v2),
            "reason": None,
        },

        # Suite-grade PASS map + legacy keys preserved
        "PASS": {
            "APPLICABLE": bool(APPLICABLE),

            # legacy
            "ALL_PASS_vector_bianchi_proxy_audit_v1": bool(ALL_PASS_vector_bianchi_proxy_audit_v1),
            "PASS_div_mean_scaled_all_applicable_bands": bool(PASS_div_mean_scaled_offcore_small),
            "PASS_div_p99_scaled_all_applicable_bands": bool(PASS_div_p99_scaled_offcore_small),
            "PASS_pi_converged_all_applicable_bands": bool(PASS_pi_converged_all),

            # scalar-style naming (v2-style keys)
            "ALL_PASS_bianchi_proxy_audit_v2": bool(ALL_PASS_bianchi_proxy_audit_v2),
            "PASS_div_mean_scaled_offcore_small": bool(PASS_div_mean_scaled_offcore_small),
            "PASS_div_p99_scaled_offcore_small": bool(PASS_div_p99_scaled_offcore_small),
            "PASS_pi_converged_offcore_all_bands": bool(PASS_pi_converged_all),
        },

        "results": {
            "n_bands_total": int(len(bands)),
            "n_applicable_bands": int(n_applicable),
            "median_mean_abs_div_scaled": float(np.median(mean_scaled_vals)) if mean_scaled_vals else None,
            "median_p99_abs_div_scaled": float(np.median(p99_scaled_vals)) if p99_scaled_vals else None,
        },

        "per_band": per_band,

        "notes": (
            "STRICT PP vector-Bianchi proxy: selects off-core nodes (~mass_core_mask==False) and ring bands by d_flat quantiles, "
            "restricts to common directed edges (optionally bidirected) in that band for flat vs curved. "
            "Defines uniform-RW flow per edge as pi[u]/outdeg[u] using stationary pi of each restricted graph. "
            "Audits local conservation of Δflow = flow_curved - flow_flat via divergence on the compact band subgraph. "
            "Gates on intensive stats mean and p99 of |divΔ| scaled by n_nodes. No PDE/Poisson/smoothing/ansatz/regression."
        ),
    }

    # Optional NPZ export
    if args.npz_out_fields:
        npz_path = str(args.npz_out_fields)
        os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
        if npz_fields:
            np.savez_compressed(npz_path, **npz_fields)
            out.setdefault("refs", {})
            if isinstance(out["refs"], dict):
                out["refs"]["npz_out_fields"] = npz_path
        else:
            out.setdefault("refs", {})
            if isinstance(out["refs"], dict):
                out["refs"]["npz_out_fields"] = npz_path
                out["refs"]["npz_out_fields_note"] = "requested but no applicable bands produced fields"

    save_json(args.output, out)
    print("WROTE", args.output)
    print("ALL_PASS_bianchi_proxy_audit_v2 =", out["PASS"]["ALL_PASS_bianchi_proxy_audit_v2"])
    print("APPLICABLE =", out["PASS"]["APPLICABLE"])
    print("median_mean_scaled =", out["results"]["median_mean_abs_div_scaled"])
    print("median_p99_scaled  =", out["results"]["median_p99_abs_div_scaled"])


if __name__ == "__main__":
    main()
