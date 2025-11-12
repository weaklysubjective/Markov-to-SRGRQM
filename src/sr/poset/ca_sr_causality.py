#!/usr/bin/env python3
# CA / Experiences → Light-cone Causality (Lorentz space) via trace operator (implicit)
# No dense NxN matrix; we apply S by neighbor shifts. No PDE/morphology libs.
# Writes NPZ [T+1,H,W] and JSON with PASS/FAIL.

import argparse, json, numpy as np, time, sys

try:
    import torch
    TORCH = True
except Exception:
    TORCH = False

def pretty(obj): return json.dumps(obj, indent=2, sort_keys=True)

def l1_lightcone_mask(H, W, T):
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W//2, H//2
    d1 = np.abs(xx-cx)+np.abs(yy-cy)
    return np.stack([(d1<=t) for t in range(T+1)], axis=0)

def causality_metrics(front, ideal):
    assert front.shape == ideal.shape
    T1, H, W = front.shape
    arrived, viols = [], []
    for t in range(T1):
        inside  = ideal[t]
        outside = ~inside
        active  = front[t].astype(bool)
        ni = int(inside.sum()); no = int(outside.sum())
        arrived.append( float((active & inside).sum())  / max(1, ni) )
        viols.append(   float((active & outside).sum()) / max(1, no) )
    return float(np.mean(arrived)), float(np.mean(viols))

# ------- implicit S application (4-neighbor, no wrap) -------
def apply_S_numpy(v):  # v: [H,W] bool
    H, W = v.shape
    y = np.zeros_like(v, dtype=bool)
    # up
    y[0:H-1, :] |= v[1:H, :]
    # down
    y[1:H,   :] |= v[0:H-1, :]
    # left
    y[:, 0:W-1] |= v[:, 1:W]
    # right
    y[:, 1:W  ] |= v[:, 0:W-1]
    return y

def apply_S_torch(v):  # v: [H,W] bool tensor
    H, W = v.shape
    y = torch.zeros_like(v, dtype=torch.bool)
    y[0:H-1, :] = y[0:H-1, :] | v[1:H, :]
    y[1:H,   :] = y[1:H,   :] | v[0:H-1, :]
    y[:, 0:W-1] = y[:, 0:W-1] | v[:, 1:W]
    y[:, 1:W  ] = y[:, 1:W  ] | v[:, 0:W-1]
    return y

def run_generate(H, W, T, seed, out_npz, out_json, cpu_only=False):
    np.random.seed(seed)
    cx, cy = W//2, H//2

    use_torch = (TORCH and (not cpu_only))
    device = None
    if use_torch:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # init v0 (single origin)
    if use_torch:
        v = torch.zeros((H,W), dtype=torch.bool, device=device)
        v[cy, cx] = True
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
        v = np.zeros((H,W), dtype=bool)
        v[cy, cx] = True
        acc = v.copy()
        frames = [acc.copy()]
        for _ in range(T):
            y = apply_S_numpy(v)
            acc = np.logical_or(acc, y)
            frames.append(acc.copy())
            v = y
        front = np.stack(frames, axis=0).astype(np.uint8)
        dev_str = "numpy"

    ideal = l1_lightcone_mask(H, W, T)
    arrived, viols = causality_metrics(front.astype(bool), ideal)
    PASS = (viols <= 0.0 + 1e-15) and (arrived >= 1.0 - 1e-15)

    if out_npz:
        np.savez_compressed(out_npz, front=front)

    result = {
        "H": H, "W": W, "T": T, "seed": seed,
        "device": dev_str,
        "arrived_fraction": arrived,
        "violations_fraction": viols,
        "PASS": PASS,
        "notes": "Implicit 4-neighbor trace (poset t→t+1), no dense S, no wrap."
    }
    if out_json:
        with open(out_json, "w") as f: json.dump(result, f, indent=2)
    print(pretty(result))

def run_test(front_path, out_json, strict=True):
    dat = np.load(front_path)
    if "front" not in dat: raise ValueError("NPZ missing 'front'")
    front = dat["front"].astype(bool)
    T1, H, W = front.shape
    ideal = l1_lightcone_mask(H, W, T1-1)
    arrived, viols = causality_metrics(front, ideal)
    PASS = (viols <= (0.0 if strict else 1e-12)) and (arrived >= (1.0 - (0.0 if strict else 1e-12)))
    res = {"H":H,"W":W,"T":T1-1,"arrived_fraction":arrived,"violations_fraction":viols,"PASS":PASS}
    if out_json:
        with open(out_json,"w") as f: json.dump(res, f, indent=2)
    print(pretty(res))

def main():
    ap = argparse.ArgumentParser(description="CA Experiences → Light-cone via implicit trace operator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--H", type=int, default=257)
    g.add_argument("--W", type=int, default=257)
    g.add_argument("--T", type=int, default=300)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out-npz", type=str, default="front.npz")
    g.add_argument("--out-json", type=str, default="causality_from_ca.json")
    g.add_argument("--cpu-only", action="store_true")

    t = sub.add_parser("test")
    t.add_argument("--front", type=str, required=True)
    t.add_argument("--out-json", type=str, default="causality_check.json")
    t.add_argument("--non-strict", action="store_true")

    args = ap.parse_args()
    if args.cmd=="generate":
        run_generate(args.H,args.W,args.T,args.seed,args.out_npz,args.out_json,cpu_only=args.cpu_only)
    else:
        run_test(args.front,args.out_json,strict=(not args.non_strict))

if __name__=="__main__":
    main()

