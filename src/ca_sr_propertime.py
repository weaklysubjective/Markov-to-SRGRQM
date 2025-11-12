#!/usr/bin/env python3
# CA/MM-only SR test: Proper-time proxy & time-dilation from Alexandrov counts.
# Uses experiences + implicit trace operator (von-Neumann 4-nbr) + strict t->t+1 partial order (acyclic).

import argparse, json, math, numpy as np
try:
    import torch
    TORCH = True
except Exception:
    TORCH = False

def apply_S_numpy(v):  # v:[H] bool, 1D line (use Hx1 conceptually)
    H = v.shape[0]
    y = np.zeros_like(v, dtype=bool)
    y[0:H-1] |= v[1:H]
    y[1:H  ] |= v[0:H-1]
    return y

def apply_S_torch(v):  # v:[H] torch.bool
    H = v.shape[0]
    y = torch.zeros_like(v, dtype=torch.bool)
    y[0:H-1] = y[0:H-1] | v[1:H]
    y[1:H  ] = y[1:H  ] | v[0:H-1]
    return y

def future_frames(H, T, x0, use_torch=False, device=None):
    """Return F[t,:] = set of sites reachable from (t=0, x0) within <=t ticks (inclusive)."""
    if use_torch:
        v = torch.zeros((H,), dtype=torch.bool, device=device); v[x0]=True
        acc = v.clone()
        frames = [acc.detach().cpu().numpy()]
        for _ in range(T):
            v = apply_S_torch(v)
            acc = acc | v
            frames.append(acc.detach().cpu().numpy())
        return np.stack(frames, axis=0).astype(bool)
    else:
        v = np.zeros((H,), dtype=bool); v[x0]=True
        acc = v.copy()
        frames = [acc.copy()]
        for _ in range(T):
            v = apply_S_numpy(v)
            acc = np.logical_or(acc, v)
            frames.append(acc.copy())
        return np.stack(frames, axis=0).astype(bool)

def past_frames(H, T, xT, use_torch=False, device=None):
    """Return P[t,:] = set of sites that can reach (T,xT) from time t (i.e., within <=(T-t) ticks)."""
    # Compute future from (time=T, xT) backward by symmetry (same spatial S).
    # Equivalent: run future from xT for T ticks, then read in reverse.
    F_T = future_frames(H, T, xT, use_torch=use_torch, device=device)  # shape [T+1, H]
    # P[t] = set of positions at time t that can reach q by T: that's the same as F_T[T-t]
    return F_T[::-1, :]

def alexandrov_count(F, P, exclude_endpoints=True):
    """
    F[t,x]: reachable from p by <=t. P[t,x]: can reach q within <=(T-t).
    Alexandrov set A = { (t,x) | F[t,x] & P[t,x] }.
    """
    A = np.logical_and(F, P)
    if exclude_endpoints:
        A[0, :]  = False
        A[-1, :] = False
    return int(A.sum()), A

def build_worldline_end(H, T, x0, v):
    """
    Deterministic inertial worldline endpoint at time T:
      x(t) = round(x0 + v * t), with |v| <= 1 (cells/tick).
    """
    if abs(v) > 1.0 + 1e-12:
        raise ValueError("Speed must satisfy |v| <= 1 in this poset.")
    xT = int(round(x0 + v * T))
    if xT < 0 or xT >= H:
        raise ValueError("Endpoint left the lattice; enlarge H or reduce T/|v|.")
    return xT

def run(H, T, v, seed, cpu_only=False, tol=0.05):
    np.random.seed(seed)
    # Ensure no boundary clipping of the lightcone for both rest and moving:
    # Max L1 radius needed = max(T, |xT-x0|). For a centered start, require margins >= T.
    x0 = H // 2
    # Basic box sanity
    if (H - 1) // 2 < T:
        raise ValueError(f"Boundary would clip the cone: need H >= {2*T+1}, got {H}.")
    # Device
    use_torch = TORCH and (not cpu_only)
    if use_torch:
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): device = torch.device("mps")
        else: device = torch.device("cpu")
    else:
        device = None
    # Endpoints
    xT_rest   = build_worldline_end(H, T, x0, 0.0)
    xT_moving = build_worldline_end(H, T, x0, v)
    # Future/past frames
    F_rest = future_frames(H, T, x0, use_torch=use_torch, device=device)
    P_rest = past_frames  (H, T, xT_rest, use_torch=use_torch, device=device)
    F_mov  = F_rest  # same source p
    P_mov  = past_frames(H, T, xT_moving, use_torch=use_torch, device=device)
    # Alexandrov counts (interval proxies)
    N0, _A0 = alexandrov_count(F_rest, P_rest, exclude_endpoints=True)
    Nv, _Av = alexandrov_count(F_mov , P_mov , exclude_endpoints=True)
    # Proper-time proxy kappa = sqrt(N)/T
    kappa0 = math.sqrt(max(N0,0)) / max(T,1)
    kappav = math.sqrt(max(Nv,0)) / max(T,1)
    ratio  = kappav / (kappa0 if kappa0>0 else float('nan'))
    target = math.sqrt(max(0.0, 1.0 - v*v))  # SR prediction in lattice units (c=1)
    abs_err = abs(ratio - target)
    PASS = (abs_err <= tol)
    out = {
        "H": H, "T": T, "v": v, "seed": seed,
        "device": (str(device) if use_torch else "numpy"),
        "N_rest": N0, "N_moving": Nv,
        "kappa_rest": kappa0, "kappa_moving": kappav,
        "ratio_kappa": ratio, "target_sqrt1_minus_v2": target,
        "abs_err": abs_err, "tol": tol,
        "boundary_limited": False,
        "poset_edges": "t->t+1 only",
        "trace_operator": "implicit 1D 2-neighbor (subset of 4-nbr in 2D)",
        "PASS": PASS
    }
    print(json.dumps(out, indent=2, sort_keys=True))

def main():
    ap = argparse.ArgumentParser(description="CA/MM poset SR test: time-dilation via Alexandrov counts (proper-time proxy).")
    ap.add_argument("--H", type=int, default=1201, help="lattice sites along x (choose >= 2*T+1 to avoid boundary)")
    ap.add_argument("--T", type=int, default=400,  help="ticks (duration)")
    ap.add_argument("--v", type=float, default=0.8, help="inertial drift in cells/tick, |v|<=1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--tol", type=float, default=0.05, help="pass tolerance for |ratio - sqrt(1-v^2)|")
    args = ap.parse_args()
    run(args.H, args.T, args.v, args.seed, cpu_only=args.cpu_only, tol=args.tol)

if __name__ == "__main__":
    main()

