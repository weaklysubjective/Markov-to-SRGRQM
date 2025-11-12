#!/usr/bin/env python3
# CA/MM-only SR: Length contraction via max-width diamond cross-section (poset simultaneity).
# Experiences + implicit trace (1D 2-neighbor) + strict t->t+1 partial order. No cycles.

import argparse, json, math, numpy as np
try:
    import torch
    TORCH=True
except Exception:
    TORCH=False

# ---- trace operator (implicit), 1D ----
def apply_S_numpy(v):  # v:[H] bool
    H=v.shape[0]; y=np.zeros_like(v, dtype=bool)
    y[0:H-1] |= v[1:H]
    y[1:H  ] |= v[0:H-1]
    return y

def apply_S_torch(v):  # v:[H] torch.bool
    import torch
    H=v.shape[0]; y=torch.zeros_like(v, dtype=torch.bool)
    y[0:H-1] = y[0:H-1] | v[1:H]
    y[1:H  ] = y[1:H  ] | v[0:H-1]
    return y

def future_frames(H, T, x0, use_torch=False, device=None):
    if use_torch:
        import torch
        v=torch.zeros((H,), dtype=torch.bool, device=device); v[x0]=True
        acc=v.clone(); frames=[acc.detach().cpu().numpy()]
        for _ in range(T):
            v=apply_S_torch(v); acc=acc|v; frames.append(acc.detach().cpu().numpy())
        return np.stack(frames, axis=0).astype(bool)
    else:
        v=np.zeros((H,), dtype=bool); v[x0]=True
        acc=v.copy(); frames=[acc.copy()]
        for _ in range(T):
            v=apply_S_numpy(v); acc=np.logical_or(acc, v); frames.append(acc.copy())
        return np.stack(frames, axis=0).astype(bool)

# ---- diamond cross-sections ----
def max_width_cross_section(H, span, x_p0, x_p2, use_torch=False, device=None):
    """
    Build diamond between p0=(0,x_p0) and p2=(span,x_p2).
    For each t in [0,span], cross-section C_t = Future(p0,t) ∩ Future(p2, span - t) mirrored as Past.
    Pick t* with maximal |C_t|. Return xs mask for C_{t*}, chosen t*, and width.
    """
    F0 = future_frames(H, span, x_p0, use_torch=use_torch, device=device)  # [span+1, H]
    F2 = future_frames(H, span, x_p2, use_torch=use_torch, device=device)
    best_t = 0
    best_mask = None
    best_w = -1
    for t in range(span+1):
        Ct = F0[t] & F2[span - t]
        w = int(Ct.sum())
        if w > best_w:
            best_w = w
            best_t = t
            best_mask = Ct
    xs = np.where(best_mask)[0]
    return xs, best_t, best_w

# ---- rod endpoints and measuring along the cross-section ----
def rod_endpoints(H, L0, x_center):
    if L0 % 2 != 0:
        raise ValueError("Use even L0 so endpoints lie on integer sites.")
    xL = x_center - (L0//2)
    xR = x_center + (L0//2)
    if not (0 <= xL < H and 0 <= xR < H):
        raise ValueError("Rod endpoints out of lattice; enlarge H or reduce L0.")
    return xL, xR

def nearest_indices(xs_sorted, target):
    import bisect
    i = bisect.bisect_left(xs_sorted, target)
    if i==0: return 0
    if i==len(xs_sorted): return len(xs_sorted)-1
    before, after = xs_sorted[i-1], xs_sorted[i]
    return i-1 if abs(before-target) <= abs(after-target) else i

def length_along_section(xs_mask, xL, xR):
    xs_sorted = np.sort(xs_mask)
    iL = nearest_indices(xs_sorted, xL)
    iR = nearest_indices(xs_sorted, xR)
    return abs(iR - iL)  # edges between projected endpoints along section

def run(H, T, v, L0, seed, cpu_only=False, tol_frac=0.07):
    np.random.seed(seed)
    # Box sanity: avoid boundary clipping for both rod and diamond
    need = max(T, L0//2)
    if (H - 1)//2 < need:
        raise ValueError(f"Lattice too small; need H >= {2*need+1}, got {H}.")
    use_torch = TORCH and (not cpu_only)
    if use_torch:
        import torch
        if torch.cuda.is_available(): device=torch.device("cuda")
        elif getattr(torch.backends,"mps",None) and torch.backends.mps.is_available(): device=torch.device("mps")
        else: device=torch.device("cpu")
    else:
        device=None

    x_center = H//2
    # Choose anchor span with simple parity repair (span=2*tau or 2*tau+1)
    candidates = []
    base = T//2
    for tau in [base, max(1, base-1)]:
        for extra in [0,1]:
            span = 2*tau + extra
            D = int(round(v * span))
            x_p0 = x_center
            x_p2 = x_center + D
            # Ensure anchor endpoint stays inside lattice
            if not (0 <= x_p2 < H): continue
            xs, tstar, width = max_width_cross_section(H, span, x_p0, x_p2, use_torch=use_torch, device=device)
            if width > 0:
                candidates.append((span, D, xs, tstar, x_p0, x_p2, width))
    if not candidates:
        print(json.dumps({"error":"No valid cross-section found. Try increasing H or adjusting T/L0/v."}, indent=2)); return
    # prefer largest span then widest section
    candidates.sort(key=lambda z: (z[0], z[6]), reverse=True)
    span, D, xs, tstar, x_p0, x_p2, width = candidates[0]

    xL, xR = rod_endpoints(H, L0, x_center)
    Lp_edges = length_along_section(xs, xL, xR)
    Lp = float(Lp_edges)

    v_hat = abs(D)/max(span,1)
    target = L0 * math.sqrt(max(0.0, 1.0 - v_hat*v_hat))
    abs_err = abs(Lp - target)
    tol_cells = max(1.0, tol_frac * L0)
    PASS = (abs_err <= tol_cells)

    print(json.dumps({
        "H": H, "T": T, "L0": L0,
        "v_cli": v, "v_hat": v_hat,
        "anchor_span_ticks": span,
        "cross_section_tstar": tstar,
        "cross_section_width_nodes": int(width),
        "rod_endpoints": [int(xL), int(xR)],
        "L_prime_measured_cells": Lp,
        "L_prime_target_cells": target,
        "abs_err_cells": abs_err,
        "tol_cells": tol_cells,
        "device": (str(device) if use_torch else "numpy"),
        "poset_edges": "t->t+1 only (acyclic)",
        "trace_operator": "implicit 1D 2-neighbor",
        "simultaneity": "max-width diamond cross-section",
        "endpoint_projection": "nearest nodes in cross-section (pure poset)",
        "PASS": PASS
    }, indent=2))

def main():
    ap = argparse.ArgumentParser(description="CA/MM poset SR: length contraction via max-width diamond cross-section.")
    ap.add_argument("--H", type=int, default=2401)
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--v", type=float, default=0.6)
    ap.add_argument("--L0", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--tol-frac", type=float, default=0.07)
    args = ap.parse_args()
    run(args.H, args.T, args.v, args.L0, args.seed, cpu_only=args.cpu_only, tol_frac=args.tol_frac)

if __name__=="__main__":
    main()

