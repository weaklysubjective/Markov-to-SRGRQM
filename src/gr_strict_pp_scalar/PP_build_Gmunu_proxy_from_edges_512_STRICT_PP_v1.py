#!/usr/bin/env python3
import argparse, os, json
import numpy as np

try:
    import torch
except Exception:
    torch = None


def read_edge_txt(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: edges file not found: {path}")
    edges = []
    with open(path, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a = line.split()
            if len(a) < 2:
                raise SystemExit(f"ERROR: bad edge line {ln}: {line}")
            edges.append((int(a[0]), int(a[1])))
    if not edges:
        raise SystemExit(f"ERROR: no edges parsed from {path}")
    return np.asarray(edges, dtype=np.int64)


def get_device(dev: str):
    if torch is None:
        return None
    d = (dev or "gpu").strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d in ("gpu", "cuda", "hip"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(d)
    except Exception:
        return torch.device("cpu")


def periodic_delta(a: np.ndarray, mod: int) -> np.ndarray:
    half = mod // 2
    x = (a + half) % mod - half
    return x


def stationary_pi_torch(src, dst, outdeg, n, device, iters, tol, alpha, lazy):
    outdeg_f = outdeg.to(torch.float64)
    good = outdeg_f > 0
    pi = torch.zeros((n,), dtype=torch.float64, device=device)
    if bool(good.any().item()):
        pi[good] = 1.0 / float(good.sum().item())
    else:
        raise SystemExit("ERROR: all nodes have outdeg=0")
    uni = torch.ones((n,), dtype=torch.float64, device=device) / float(n)

    last = None
    converged = False
    for t in range(int(iters)):
        contrib = torch.zeros_like(pi)
        contrib.index_add_(0, dst, pi[src] / outdeg_f[src])
        w = lazy * pi + (1.0 - lazy) * contrib
        pi_new = (1.0 - alpha) * w + alpha * uni
        pi_new = pi_new / pi_new.sum()
        l1 = torch.sum(torch.abs(pi_new - pi)).item()
        last = float(l1)
        pi = pi_new
        if last < tol:
            converged = True
            break

    return pi.detach().cpu().numpy().astype(np.float64), {
        "pi_iters_used": int(t + 1),
        "pi_l1_last": float(last if last is not None else np.nan),
        "pi_converged": bool(converged),
        "alpha": float(alpha),
        "lazy": float(lazy),
    }


def moments_from_edges(E: np.ndarray, H: int, W: int, periodic: bool):
    N = H * W
    u = E[:, 0].astype(np.int64)
    v = E[:, 1].astype(np.int64)

    ur, uc = (u // W), (u % W)
    vr, vc = (v // W), (v % W)

    dr = (vr - ur).astype(np.int64)
    dc = (vc - uc).astype(np.int64)
    if periodic:
        dr = periodic_delta(dr, H)
        dc = periodic_delta(dc, W)

    outdeg = np.zeros((N,), dtype=np.int64)
    np.add.at(outdeg, u, 1)

    inv_out = np.zeros((N,), dtype=np.float64)
    inv_out[outdeg > 0] = 1.0 / outdeg[outdeg > 0].astype(np.float64)

    mean_dr = np.zeros((N,), dtype=np.float64)
    mean_dc = np.zeros((N,), dtype=np.float64)
    np.add.at(mean_dr, u, dr.astype(np.float64) * inv_out[u])
    np.add.at(mean_dc, u, dc.astype(np.float64) * inv_out[u])

    m_rr = np.zeros((N,), dtype=np.float64)
    m_rc = np.zeros((N,), dtype=np.float64)
    m_cc = np.zeros((N,), dtype=np.float64)
    np.add.at(m_rr, u, (dr.astype(np.float64) * dr.astype(np.float64)) * inv_out[u])
    np.add.at(m_rc, u, (dr.astype(np.float64) * dc.astype(np.float64)) * inv_out[u])
    np.add.at(m_cc, u, (dc.astype(np.float64) * dc.astype(np.float64)) * inv_out[u])

    return outdeg, mean_dr, mean_dc, m_rr, m_rc, m_cc


def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP 512: build per-node Gmunu proxy fields from edge-only (curved-flat) RW moment deltas."
    )
    ap.add_argument("--case", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--edges_flat", required=True)
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--tau_npz", required=True, help="Needs G00_med, mass_core_mask (and optional kappa_med)")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--periodic", action="store_true")
    ap.add_argument("--pi_iters", type=int, default=20000)
    ap.add_argument("--pi_tol", type=float, default=1e-10)
    ap.add_argument("--teleport_alpha", type=float, default=1e-6)
    ap.add_argument("--lazy", type=float, default=0.10)

    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    H, W = int(args.H), int(args.W)
    N = H * W

    Ef = read_edge_txt(args.edges_flat)
    Ec = read_edge_txt(args.edges_curved)

    if Ef.min() < 0 or Ef.max() >= N or Ec.min() < 0 or Ec.max() >= N:
        raise SystemExit("ERROR: edges out of range")

    # τ NPZ (for core mask + scalar anchor)
    Z = np.load(args.tau_npz)
    if "G00_med" not in Z.files or "mass_core_mask" not in Z.files:
        raise SystemExit(f"ERROR: tau_npz missing required keys. keys={list(Z.files)}")
    G00_med = Z["G00_med"].reshape(-1).astype(np.float64)
    core = Z["mass_core_mask"].reshape(-1).astype(bool)
    if G00_med.shape[0] != N or core.shape[0] != N:
        raise SystemExit("ERROR: tau_npz shape mismatch vs H*W")

    # Local moments (purely combinatorial from edges)
    outdeg_f, mean_dr_f, mean_dc_f, mrr_f, mrc_f, mcc_f = moments_from_edges(Ef, H, W, bool(args.periodic))
    outdeg_c, mean_dr_c, mean_dc_c, mrr_c, mrc_c, mcc_c = moments_from_edges(Ec, H, W, bool(args.periodic))

    # Stationary pi per graph (RW) for density weighting (STRICT PP; Markov-only)
    device = get_device(args.device)
    if torch is None or device is None:
        raise SystemExit("ERROR: torch required (CPU torch OK).")

    # Flat pi
    uf = Ef[:, 0].astype(np.int64); vf = Ef[:, 1].astype(np.int64)
    src_f = torch.tensor(uf, dtype=torch.int64, device=device)
    dst_f = torch.tensor(vf, dtype=torch.int64, device=device)
    outdeg_tf = torch.tensor(outdeg_f, dtype=torch.int64, device=device)
    pi_f, diag_f = stationary_pi_torch(
        src_f, dst_f, outdeg_tf, N, device,
        iters=args.pi_iters, tol=args.pi_tol,
        alpha=float(args.teleport_alpha), lazy=float(args.lazy)
    )

    # Curved pi
    uc = Ec[:, 0].astype(np.int64); vc = Ec[:, 1].astype(np.int64)
    src_c = torch.tensor(uc, dtype=torch.int64, device=device)
    dst_c = torch.tensor(vc, dtype=torch.int64, device=device)
    outdeg_tc = torch.tensor(outdeg_c, dtype=torch.int64, device=device)
    pi_c, diag_c = stationary_pi_torch(
        src_c, dst_c, outdeg_tc, N, device,
        iters=args.pi_iters, tol=args.pi_tol,
        alpha=float(args.teleport_alpha), lazy=float(args.lazy)
    )

    if not (diag_f["pi_converged"] and diag_c["pi_converged"]):
        raise SystemExit("ERROR: pi did not converge (flat or curved)")

    # Build density-weighted drift and 2nd moments per node
    # (These are “flux-like” local tensors in index basis.)
    rho_f = pi_f
    rho_c = pi_c

    J0y_f = rho_f * mean_dr_f
    J0x_f = rho_f * mean_dc_f
    Jyy_f = rho_f * mrr_f
    Jyx_f = rho_f * mrc_f
    Jxx_f = rho_f * mcc_f

    J0y_c = rho_c * mean_dr_c
    J0x_c = rho_c * mean_dc_c
    Jyy_c = rho_c * mrr_c
    Jyx_c = rho_c * mrc_c
    Jxx_c = rho_c * mcc_c

    # Δ = curved - flat (this is the “geometry response” we will treat as G-proxy components)
    d0y = J0y_c - J0y_f
    d0x = J0x_c - J0x_f
    dyy = Jyy_c - Jyy_f
    dyx = Jyx_c - Jyx_f
    dxx = Jxx_c - Jxx_f

    # Scalar anchor scale: match median magnitude of |G00_med| on core
    m_core = core & np.isfinite(G00_med)
    if int(m_core.sum()) < 10:
        raise SystemExit("ERROR: too few core nodes for calibration")

    # Use a deterministic per-case scale that does NOT use regression:
    # scale = median(G00_med) / median(Δtrace2) on core, where Δtrace2 = dxx + dyy
    dtrace = dxx + dyy
    denom = np.median(dtrace[m_core])
    if not np.isfinite(denom) or abs(float(denom)) < 1e-300:
        # fallback: use median absolute
        denom = np.median(np.abs(dtrace[m_core]))
        if not np.isfinite(denom) or abs(float(denom)) < 1e-300:
            raise SystemExit("ERROR: degenerate Δtrace for calibration")

    numer = np.median(G00_med[m_core])
    scale = float(numer / denom)

    # Define G-proxy components (2+1) with that single scale
    G00 = G00_med
    G0x = scale * d0x
    G0y = scale * d0y
    Gxx = scale * dxx
    Gxy = scale * dyx
    Gyy = scale * dyy

    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        G00=G00, G0x=G0x, G0y=G0y,
        Gxx=Gxx, Gxy=Gxy, Gyy=Gyy,
        scale=np.asarray([scale], dtype=np.float64),
    )

    meta = {
        "STRICT_PP": True,
        "case": args.case,
        "H": H, "W": W, "N": N,
        "inputs": {
            "edges_flat": args.edges_flat,
            "edges_curved": args.edges_curved,
            "tau_npz": args.tau_npz,
            "periodic": bool(args.periodic),
        },
        "pi": {"flat": diag_f, "curved": diag_c},
        "calibration": {
            "scale": scale,
            "scale_numer_med_G00_core": float(numer),
            "scale_denom_med_dtrace_core": float(denom),
            "dtrace_key": "dxx+dyy where d**=(curved-flat) density-weighted moments",
        },
        "notes": (
            "Gmunu-proxy fields: G00 from tau geometry; non-00 components from (curved-flat) RW moment deltas on E2 edges, "
            "scaled by a single deterministic core calibration. No PDE/Poisson/smoothing/ansatz/regression."
        ),
    }
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    print("WROTE", args.output_npz)
    print("scale =", scale)
    print("pi_converged_flat =", bool(diag_f["pi_converged"]))
    print("pi_converged_curved =", bool(diag_c["pi_converged"]))


if __name__ == "__main__":
    main()

