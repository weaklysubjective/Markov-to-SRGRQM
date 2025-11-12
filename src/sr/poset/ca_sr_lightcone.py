#!/usr/bin/env python3
# CA/MM-only: experiences + trace operator + strict time partial order (no cycles).
# Outputs JSON by default; optional --save-front saves NPZ derived from the same CA poset.

import argparse, json, numpy as np
try:
    import torch
    TORCH = True
except Exception:
    TORCH = False

def l1_mask(H,W,T):
    yy,xx = np.mgrid[0:H,0:W]; cx,cy=W//2,H//2
    d1 = np.abs(xx-cx)+np.abs(yy-cy)
    return np.stack([(d1<=t) for t in range(T+1)], axis=0)

def metrics(front_bool, ideal_bool):
    assert front_bool.shape == ideal_bool.shape
    T1,H,W = front_bool.shape
    arr, vio = [], []
    for t in range(T1):
        inside  = ideal_bool[t]
        outside = ~inside
        active  = front_bool[t]
        ni = int(inside.sum()); no = int(outside.sum())
        arr.append(float((active & inside).sum())/max(1,ni))
        vio.append(float((active & outside).sum())/max(1,no))
    return float(np.mean(arr)), float(np.mean(vio))

def apply_S_numpy(v):  # v:[H,W] bool
    H,W = v.shape
    y = np.zeros_like(v, dtype=bool)
    y[0:H-1,:] |= v[1:H,:]
    y[1:H,  :] |= v[0:H-1,:]
    y[:,0:W-1] |= v[:,1:W]
    y[:,1:W  ] |= v[:,0:W-1]
    return y

def apply_S_torch(v):  # v:[H,W] torch.bool
    H,W = v.shape
    y = torch.zeros_like(v, dtype=torch.bool)
    y[0:H-1,:] = y[0:H-1,:] | v[1:H,:]
    y[1:H,  :] = y[1:H,  :] | v[0:H-1,:]
    y[:,0:W-1] = y[:,0:W-1] | v[:,1:W]
    y[:,1:W  ] = y[:,1:W  ] | v[:,0:W-1]
    return y

def run(H,W,T,seed,save_front=None,cpu_only=False):
    np.random.seed(seed)
    cx,cy = W//2, H//2

    use_torch = TORCH and (not cpu_only)
    if use_torch:
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): device = torch.device("mps")
        else: device = torch.device("cpu")
        v = torch.zeros((H,W), dtype=torch.bool, device=device); v[cy,cx]=True
        acc = v.clone()
        frames = [acc.detach().cpu().numpy()]
        for _ in range(T):
            y = apply_S_torch(v)
            acc = acc | y
            frames.append(acc.detach().cpu().numpy())
            v = y
        front = np.stack(frames, axis=0).astype(np.uint8)
        dev_str = str(device)
    else:
        v = np.zeros((H,W), dtype=bool); v[cy,cx]=True
        acc = v.copy(); frames=[acc.copy()]
        for _ in range(T):
            y = apply_S_numpy(v)
            acc = np.logical_or(acc, y)
            frames.append(acc.copy()); v=y
        front = np.stack(frames, axis=0).astype(np.uint8)
        dev_str = "numpy"

    ideal = l1_mask(H,W,T)
    arrived, viols = metrics(front.astype(bool), ideal)
    # c_hat = max L1 radius / ticks
    yy,xx = np.mgrid[0:H,0:W]
    d1 = np.abs(xx-cx)+np.abs(yy-cy)
    max_r = int(d1[front[-1].astype(bool)].max()) if front[-1].any() else 0
    c_hat = (max_r)/(T if T>0 else 1)

    PASS = (viols<=1e-15) and (arrived>=1.0-1e-15)

    if save_front:
        # only saved when explicitly requested
        np.savez_compressed(save_front, front=front)

    print(json.dumps({
        "H":H,"W":W,"T":T,"seed":seed,
        "device":dev_str,
        "arrived_fraction":arrived,
        "violations_fraction":viols,
        "c_hat_cells_per_tick":c_hat,
        "max_L1_radius":max_r,
        "poset_edges": "t->t+1 only (acyclic by construction)",
        "trace_operator": "implicit 4-neighbor (von-Neumann), no wrap",
        "saved_front": bool(save_front),
        "PASS": PASS
    }, indent=2, sort_keys=True))

def main():
    ap = argparse.ArgumentParser(description="CA/MM poset light-cone (trace + partial order, no cycles). JSON only by default.")
    ap.add_argument("--H", type=int, default=257)
    ap.add_argument("--W", type=int, default=257)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-front", type=str, default=None, help="Optional path to save NPZ (front); not saved unless set.")
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()
    run(args.H,args.W,args.T,args.seed,save_front=args.save_front,cpu_only=args.cpu_only)

if __name__ == "__main__":
    main()

