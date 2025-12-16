#!/usr/bin/env python3
import argparse, os, json
from typing import Any, Tuple
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
    E = np.asarray(edges, dtype=np.int64)
    return E

def get_device(dev: str):
    if torch is None:
        return None
    d = (dev or "gpu").strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    # GPU-first: ROCm often reports as cuda in torch
    if d in ("gpu","cuda","hip"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(d)
    except Exception:
        return torch.device("cpu")

def periodic_delta(a: np.ndarray, mod: int) -> np.ndarray:
    # map delta into [-mod/2, +mod/2] (integers)
    half = mod // 2
    x = a.copy()
    x = (x + half) % mod - half
    return x

def stationary_pi_torch(
    src: "torch.Tensor",
    dst: "torch.Tensor",
    outdeg: "torch.Tensor",
    n: int,
    device: "torch.device",
    iters: int,
    tol: float,
    alpha: float,
    lazy: float,
) -> Tuple[np.ndarray, dict]:
    """
    Stationary distribution for a lazy random walk with teleportation:
      pi_new = (1-alpha)*W(pi) + alpha*uniform
      W(pi): lazy*(pi) + (1-lazy)*P^T pi  where P is uniform-outneighbors.
    """
    assert 0.0 <= alpha < 1.0
    assert 0.0 <= lazy < 1.0

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
        # scatter for P^T pi
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

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP 512: build per-node Tmunu (2+1 proxy) from E2 edges via RW stationary rho + step moments."
    )
    ap.add_argument("--case", required=True, help="ms080|strong_pf010 (label only)")
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--device", default="gpu", help="gpu|cpu (gpu uses torch cuda/rocm if available)")
    ap.add_argument("--periodic", action="store_true", help="Use periodic wrap for (dr,dc) step vectors")
    ap.add_argument("--pi_iters", type=int, default=20000)
    ap.add_argument("--pi_tol", type=float, default=1e-10)
    ap.add_argument("--teleport_alpha", type=float, default=1e-6, help="Small teleport to ensure ergodicity (0 disables)")
    ap.add_argument("--lazy", type=float, default=0.10, help="Lazy weight (0 disables)")
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    H, W = int(args.H), int(args.W)
    N = H * W

    E = read_edge_txt(args.edges_curved)
    if E.ndim != 2 or E.shape[1] != 2:
        raise SystemExit("ERROR: edges must be (E,2)")
    if E.min() < 0 or E.max() >= N:
        raise SystemExit(f"ERROR: edge endpoints out of range [0,{N})")

    # row/col for each node label
    u = E[:, 0].astype(np.int64)
    v = E[:, 1].astype(np.int64)
    ur, uc = (u // W), (u % W)
    vr, vc = (v // W), (v % W)

    dr = (vr - ur).astype(np.int64)
    dc = (vc - uc).astype(np.int64)
    if bool(args.periodic):
        dr = periodic_delta(dr, H)
        dc = periodic_delta(dc, W)

    # Outdegree
    outdeg = np.zeros((N,), dtype=np.int64)
    np.add.at(outdeg, u, 1)
    if outdeg.max() <= 0:
        raise SystemExit("ERROR: all outdeg=0")

    # Stationary rho (GPU-first if torch available)
    device = get_device(args.device)
    if torch is not None and device is not None:
        src_t = torch.tensor(u, dtype=torch.int64, device=device)
        dst_t = torch.tensor(v, dtype=torch.int64, device=device)
        outdeg_t = torch.tensor(outdeg, dtype=torch.int64, device=device)
        pi, pi_diag = stationary_pi_torch(
            src_t, dst_t, outdeg_t, N, device,
            iters=args.pi_iters, tol=args.pi_tol,
            alpha=float(args.teleport_alpha), lazy=float(args.lazy)
        )
    else:
        # CPU fallback (slower)
        pi = np.zeros((N,), dtype=np.float64)
        good = outdeg > 0
        pi[good] = 1.0 / good.sum()
        alpha = float(args.teleport_alpha)
        lazy = float(args.lazy)
        uni = np.ones((N,), dtype=np.float64) / float(N)
        pi_diag = {"pi_iters_used": 0, "pi_l1_last": None, "pi_converged": False, "alpha": alpha, "lazy": lazy}
        for t in range(int(args.pi_iters)):
            contrib = np.zeros_like(pi)
            np.add.at(contrib, v, pi[u] / np.maximum(outdeg[u], 1))
            w = lazy * pi + (1.0 - lazy) * contrib
            pi_new = (1.0 - alpha) * w + alpha * uni
            pi_new = pi_new / pi_new.sum()
            l1 = float(np.abs(pi_new - pi).sum())
            pi = pi_new
            pi_diag["pi_iters_used"] = int(t + 1)
            pi_diag["pi_l1_last"] = l1
            if l1 < float(args.pi_tol):
                pi_diag["pi_converged"] = True
                break

    # Per-node step moments (uniform over outneighbors)
    inv_out = np.zeros((N,), dtype=np.float64)
    inv_out[outdeg > 0] = 1.0 / outdeg[outdeg > 0].astype(np.float64)

    # mean step
    mean_dr = np.zeros((N,), dtype=np.float64)
    mean_dc = np.zeros((N,), dtype=np.float64)
    np.add.at(mean_dr, u, dr.astype(np.float64) * inv_out[u])
    np.add.at(mean_dc, u, dc.astype(np.float64) * inv_out[u])

    # second moments E[dx dx], E[dx dy], E[dy dy]
    m_rr = np.zeros((N,), dtype=np.float64)
    m_rc = np.zeros((N,), dtype=np.float64)
    m_cc = np.zeros((N,), dtype=np.float64)
    np.add.at(m_rr, u, (dr.astype(np.float64) * dr.astype(np.float64)) * inv_out[u])
    np.add.at(m_rc, u, (dr.astype(np.float64) * dc.astype(np.float64)) * inv_out[u])
    np.add.at(m_cc, u, (dc.astype(np.float64) * dc.astype(np.float64)) * inv_out[u])

    rho = pi.astype(np.float64)

    # Define a STRICT-PP operational Tmunu proxy (2+1):
    #   T00 = rho
    #   T0y = rho * mean_dr   (y=row)
    #   T0x = rho * mean_dc   (x=col)
    #   Tyy = rho * E[dr^2]
    #   Tyx = rho * E[dr*dc]
    #   Txx = rho * E[dc^2]
    T00 = rho
    T0y = rho * mean_dr
    T0x = rho * mean_dc
    Tyy = rho * m_rr
    Tyx = rho * m_rc
    Txx = rho * m_cc

    # Sanity asserts
    assert T00.shape == (N,)
    assert np.isfinite(T00).all()
    assert float(T00.sum()) > 0

    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        T00=T00, T0x=T0x, T0y=T0y,
        Txx=Txx, Tyx=Tyx, Tyy=Tyy,
    )

    meta = {
        "H": H, "W": W, "N": N,
        "STRICT_PP": True,
        "case": args.case,
        "edges_curved": args.edges_curved,
        "periodic": bool(args.periodic),
        "pi": pi_diag,
        "notes": "STRICT PP: Tmunu proxy built from RW stationary rho on E2 edges and local step moments in lattice index basis (no Euclidean distance)."
    }
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    print("WROTE", args.output_npz)
    print("pi_converged =", bool(pi_diag.get("pi_converged")))
    print("T00_sum =", float(T00.sum()))

if __name__ == "__main__":
    main()

