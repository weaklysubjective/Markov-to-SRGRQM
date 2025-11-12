#!/usr/bin/env python3
# CA/MM isotropy via micro-frame symmetrization (per-agent random direction each tick).
# One-step/tick; strict t->t+1; 4-neighbor hops with stochastic axial rounding.
# GPU optional (CUDA/MPS). Isotropy score = |λ1-λ2|/(λ1+λ2) from endpoint covariance.

import argparse, json, math
import numpy as np

try:
    import torch
    TORCH = True
except Exception:
    TORCH = False

def isotropy_score_from_cov(Sxx, Syy, Sxy):
    tr  = Sxx + Syy
    det = Sxx * Syy - Sxy * Sxy
    disc = max(0.0, tr * tr - 4.0 * det)
    lam1 = 0.5 * (tr + math.sqrt(disc))
    lam2 = 0.5 * (tr - math.sqrt(disc))
    ani = 0.0 if (lam1 + lam2) == 0 else abs(lam1 - lam2) / (lam1 + lam2)
    return lam1, lam2, ani

def run_torch(H, W, T, N, seed, device, tol=0.03):
    torch.manual_seed(seed)
    pi = math.pi

    # positions (int on device)
    xs = torch.full((N,), H // 2, dtype=torch.int32, device=device)
    ys = torch.full((N,), W // 2, dtype=torch.int32, device=device)

    for _ in range(T):
        # random micro-frame direction per agent
        thetas = torch.rand(N, device=device) * (2.0 * pi)
        c = torch.cos(thetas)
        s = torch.sin(thetas)

        # positive axial weights → probabilities
        wxp = torch.clamp(c, min=0.0)
        wxn = torch.clamp(-c, min=0.0)
        wyp = torch.clamp(s, min=0.0)
        wyn = torch.clamp(-s, min=0.0)
        Z = wxp + wxn + wyp + wyn
        Z = torch.clamp(Z, min=1e-12)

        pR = wxp / Z
        pL = wxn / Z
        pU = wyp / Z
        pD = wyn / Z

        # categorical sample via uniform + cumulative probs
        r = torch.rand(N, device=device)
        cum1 = pR
        cum2 = cum1 + pL
        cum3 = cum2 + pU
        # indices: 0=R,1=L,2=U,3=D
        idx = torch.where(r < cum1, torch.zeros_like(r, dtype=torch.int64),
              torch.where(r < cum2, torch.ones_like(r, dtype=torch.int64),
              torch.where(r < cum3, torch.full_like(r, 2, dtype=torch.int64),
                          torch.full_like(r, 3, dtype=torch.int64))))

        # map indices → (dx,dy)
        # 0:R(+1,0), 1:L(-1,0), 2:U(0,+1), 3:D(0,-1)
        dx = torch.gather(torch.tensor([1,-1,0,0], device=device, dtype=torch.int32), 0, idx)
        dy = torch.gather(torch.tensor([0,0,1,-1], device=device, dtype=torch.int32), 0, idx)

        xs = torch.clamp(xs + dx, 0, H - 1)
        ys = torch.clamp(ys + dy, 0, W - 1)

    # covariance on device (float64 for stability)
    X = (xs.to(torch.float64) - (H // 2))
    Y = (ys.to(torch.float64) - (W // 2))
    Xc = X - X.mean()
    Yc = Y - Y.mean()
    Sxx = (Xc * Xc).mean().item()
    Syy = (Yc * Yc).mean().item()
    Sxy = (Xc * Yc).mean().item()

    lam1, lam2, ani = isotropy_score_from_cov(Sxx, Syy, Sxy)
    out = {
        "H": H, "W": W, "T": T, "N_agents": N, "seed": seed,
        "device": str(device),
        "cov": {"Sxx": Sxx, "Syy": Syy, "Sxy": Sxy},
        "eigvals": {"lambda_max": lam1, "lambda_min": lam2},
        "isotropy_score_cov_anisotropy": ani,
        "PASS_isotropy": ani <= tol,
        "tol": tol,
        "notes": "One-step/tick; per-tick random direction; stochastic axial rounding; strict t->t+1."
    }
    print(json.dumps(out, indent=2))

def run_numpy(H, W, T, N, seed, tol=0.03):
    rng = np.random.default_rng(seed)
    xs = np.full((N,), H // 2, dtype=np.int32)
    ys = np.full((N,), W // 2, dtype=np.int32)

    for _ in range(T):
        thetas = rng.uniform(0.0, 2.0 * math.pi, size=N)
        c = np.cos(thetas)
        s = np.sin(thetas)
        wxp = np.clip(c, 0.0, None)
        wxn = np.clip(-c,0.0, None)
        wyp = np.clip(s, 0.0, None)
        wyn = np.clip(-s,0.0, None)
        Z = wxp + wxn + wyp + wyn
        Z[Z == 0.0] = 1e-12
        pR = wxp / Z; pL = wxn / Z; pU = wyp / Z; pD = wyn / Z
        r = rng.random(N)
        # indices 0..3 same mapping as torch version
        idx = np.where(r < pR, 0,
              np.where(r < pR + pL, 1,
              np.where(r < pR + pL + pU, 2, 3)))
        dx = np.take(np.array([1,-1,0,0], dtype=np.int32), idx)
        dy = np.take(np.array([0,0,1,-1], dtype=np.int32), idx)
        xs = np.clip(xs + dx, 0, H - 1)
        ys = np.clip(ys + dy, 0, W - 1)

    X = xs.astype(np.float64) - (H // 2)
    Y = ys.astype(np.float64) - (W // 2)
    Xc = X - X.mean(); Yc = Y - Y.mean()
    Sxx = float((Xc * Xc).mean())
    Syy = float((Yc * Yc).mean())
    Sxy = float((Xc * Yc).mean())

    lam1, lam2, ani = isotropy_score_from_cov(Sxx, Syy, Sxy)
    out = {
        "H": H, "W": W, "T": T, "N_agents": N, "seed": seed,
        "device": "numpy",
        "cov": {"Sxx": Sxx, "Syy": Syy, "Sxy": Sxy},
        "eigvals": {"lambda_max": lam1, "lambda_min": lam2},
        "isotropy_score_cov_anisotropy": ani,
        "PASS_isotropy": ani <= tol,
        "tol": tol,
        "notes": "One-step/tick; per-tick random direction; stochastic axial rounding; strict t->t+1."
    }
    print(json.dumps(out, indent=2))

def main():
    ap = argparse.ArgumentParser(description="CA/MM isotropy with micro-frame symmetrization (GPU optional).")
    ap.add_argument("--H", type=int, default=601)
    ap.add_argument("--W", type=int, default=601)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--N", type=int, default=20000, help="number of independent agents")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()

    if TORCH and (not args.cpu_only):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        run_torch(args.H, args.W, args.T, args.N, args.seed, device, tol=args.tol)
    else:
        run_numpy(args.H, args.W, args.T, args.N, args.seed, tol=args.tol)

if __name__ == "__main__":
    main()

