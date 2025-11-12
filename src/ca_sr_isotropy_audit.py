#!/usr/bin/env python3
# CA/MM poset isotropy audit: one-step/tick reachability with optional staggered/random neighborhoods.
# Outputs JSON with angular radius stats and an isotropy score (RMS fractional deviation).

import argparse, json, math, numpy as np
try:
    import torch
    TORCH=True
except:
    TORCH=False

def evolve_front(H,W,T,schedule,seed=7,device=None):
    """Return final boolean front mask after <=T ticks (union over time)."""
    cx, cy = H//2, W//2
    if TORCH and device is not None:
        v = torch.zeros((H,W), dtype=torch.bool, device=device); v[cx,cy]=True
        acc = v.clone()
        rng = None
        import numpy as np
        rng = np.random.default_rng(seed)
    else:
        v = np.zeros((H,W), dtype=bool); v[cx,cy]=True
        acc = v.copy()
        rng = np.random.default_rng(seed)

    def step_axial(x):
        if TORCH and device is not None:
            up    = torch.zeros_like(x); up[:-1,:]   |= x[1: ,:]
            down  = torch.zeros_like(x); down[1: ,:] |= x[:-1,:]
            left  = torch.zeros_like(x); left[:,1: ] |= x[:, :-1]
            right = torch.zeros_like(x); right[:,:-1]|= x[:, 1: ]
            return up|down|left|right
        else:
            y = np.zeros_like(x)
            y[:-1,:]  |= x[1: ,:]
            y[1: ,:]  |= x[:-1,:]
            y[:,1: ]  |= x[:, :-1]
            y[:,:-1]  |= x[:, 1: ]
            return y

    def step_diag(x):
        if TORCH and device is not None:
            y = torch.zeros_like(x)
            y[1:, 1:]   |= x[:-1,:-1]
            y[1:, :-1]  |= x[:-1,1:]
            y[:-1, 1:]  |= x[1: ,:-1]
            y[:-1, :-1] |= x[1: , 1:]
            return y
        else:
            y = np.zeros_like(x)
            y[1:, 1:]   |= x[:-1,:-1]
            y[1:, :-1]  |= x[:-1,1:]
            y[:-1, 1:]  |= x[1: ,:-1]
            y[:-1, :-1] |= x[1: , 1:]
            return y

    def step_random(x):
        # choose axial vs diagonal per tick, unbiased
        #return step_diag(x) if (rng.random() < 0.5) else step_axial(x)
        return step_diag(x) if (rng.random() < 0.5) else step_axial(x)

    for t in range(T):
        if schedule == "axial":
            v_next = step_axial(v)
        elif schedule == "staggered":
            v_next = step_axial(v) if (t % 2 == 0) else step_diag(v)
        elif schedule == "random":
            v_next = step_random(v)
        else:
            raise ValueError("schedule must be axial|staggered|random")
        v = v_next
        acc = acc | v

    if TORCH and device is not None:
        return acc.detach().cpu().numpy()
    else:
        return acc

def sample_radii(mask, num_angles=360):
    H,W = mask.shape
    cx, cy = H//2, W//2
    radii = []
    for k in range(num_angles):
        theta = 2*math.pi*k/num_angles
        dx, dy = math.cos(theta), math.sin(theta)
        # Ray march until we leave mask or bounds
        r = 0.0
        x, y = cx + 0.5, cy + 0.5
        # march in small steps to approximate continuous direction
        step = 0.25
        last_inside = True
        while True:
            xi, yi = int(x), int(y)
            if xi<0 or yi<0 or xi>=H or yi>=W:
                break
            inside = mask[xi, yi]
            if not inside and last_inside:
                break
            last_inside = inside
            x += dx*step; y += dy*step; r += step
            if r > max(H,W): break
        radii.append(r)
    return np.array(radii, dtype=float)

def main():
    ap = argparse.ArgumentParser(description="CA/MM isotropy audit for one-step/tick fronts.")
    ap.add_argument("--H", type=int, default=601)
    ap.add_argument("--W", type=int, default=601)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--schedule", type=str, default="staggered", choices=["axial","staggered","random"])
    ap.add_argument("--angles", type=int, default=360)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()

    use_torch = TORCH and (not args.cpu_only)
    device = None
    if use_torch:
        if torch.cuda.is_available(): device=torch.device("cuda")
        elif getattr(torch.backends,"mps",None) and torch.backends.mps.is_available(): device=torch.device("mps")
        else: device=torch.device("cpu")

    mask = evolve_front(args.H, args.W, args.T, args.schedule, seed=args.seed, device=device)
    radii = sample_radii(mask, num_angles=args.angles)

    mean_r = float(np.mean(radii))
    std_r  = float(np.std(radii))
    rms_frac = float(np.sqrt(np.mean(((radii-mean_r)/max(mean_r,1e-9))**2)))

    out = {
        "H": args.H, "W": args.W, "T": args.T, "schedule": args.schedule,
        "angles": args.angles, "device": (str(device) if use_torch else "numpy"),
        "mean_radius": mean_r, "std_radius": std_r,
        "isotropy_score_rms_fraction": rms_frac,
        "PASS_isotropy": rms_frac <= 0.05  # target: ≤5% anisotropy
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

