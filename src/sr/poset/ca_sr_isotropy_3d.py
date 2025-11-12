#!/usr/bin/env python3
import argparse, json, math, numpy as np

def step_probs_3d(cx,cy,cz):
    ax,ay,az = abs(cx),abs(cy),abs(cz)
    wxp = max(0.0, cx); wxn = max(0.0, -cx)
    wyp = max(0.0, cy); wyn = max(0.0, -cy)
    wzp = max(0.0, cz); wzn = max(0.0, -cz)
    Z = ax+ay+az if (ax+ay+az)>0 else 1.0
    return (wxp/Z, wxn/Z, wyp/Z, wyn/Z, wzp/Z, wzn/Z)

def run(H,W,D,T,N,seed):
    rng = np.random.default_rng(seed)
    cx,cy,cz = H//2, W//2, D//2
    X = np.zeros((N,3), dtype=np.int32); X[:,0]=cx; X[:,1]=cy; X[:,2]=cz
    for _ in range(T):
        # random unit direction on sphere
        u = rng.normal(size=(N,3)); u /= np.linalg.norm(u,axis=1,keepdims=True)
        r = rng.random(N)
        p = np.zeros((N,6))
        for i in range(N):
            px, nx, py, ny, pz, nz = step_probs_3d(u[i,0], u[i,1], u[i,2])
            p[i] = (px,nx,py,ny,pz,nz)
        # cumulative
        c = np.cumsum(p, axis=1)
        # choose which of ±x,±y,±z
        sel = (r[:,None] < c).argmax(axis=1)
        delta = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.int32)
        X = np.clip(X + delta[sel], [0,0,0], [H-1,W-1,D-1])

    # center positions and compute covariance
    pos = X - np.array([[cx,cy,cz]], dtype=np.int32)
    cov = (pos.T @ pos) / float(N)
    # eigenvalues
    lam, _ = np.linalg.eigh(cov)
    lam = np.sort(lam)  # ascending
    score = float((lam[-1]-lam[0]) / max(1e-9, lam[-1]+lam[0]))
    return cov, lam, score

def main():
    ap=argparse.ArgumentParser(description="3D isotropy via covariance eigenvalues (6-neighbor rounding).")
    ap.add_argument("--H", type=int, default=201)
    ap.add_argument("--W", type=int, default=201)
    ap.add_argument("--D", type=int, default=201)
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--N", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tol", type=float, default=0.04)
    a=ap.parse_args()
    cov, lam, score = run(a.H,a.W,a.D,a.T,a.N,a.seed)
    out = {
      "H":a.H,"W":a.W,"D":a.D,"T":a.T,"N_agents":a.N,"seed":a.seed,
      "cov":{"Sxx":float(cov[0,0]),"Syy":float(cov[1,1]),"Szz":float(cov[2,2]),
             "Sxy":float(cov[0,1]),"Sxz":float(cov[0,2]),"Syz":float(cov[1,2])},
      "eigvals":{"lambda_min":float(lam[0]),"lambda_mid":float(lam[1]),"lambda_max":float(lam[2])},
      "anisotropy_score": score,
      "tol": a.tol,
      "PASS_isotropy_3d": (score <= a.tol),
      "notes":"One-step/tick; per-tick random 3D direction; 6-neighbor stochastic rounding; strict t->t+1."
    }
    print(json.dumps(out, indent=2))

if __name__=="__main__":
    main()

