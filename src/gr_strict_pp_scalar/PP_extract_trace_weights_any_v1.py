#!/usr/bin/env python3
import argparse, os, sys, json
import numpy as np

def load_any(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".npy"):
        arr = np.load(path)
        return {"__npy__": arr}

    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}

    raise ValueError("Expected .npy or .npz input")

def pick_array(d, H, W):
    N = H * W
    candidates = []

    for k, a in d.items():
        if not isinstance(a, np.ndarray):
            continue
        candidates.append((k, a))

    if not candidates:
        raise ValueError("No ndarray found in input")

    # Prefer exact (H,W)
    for k, a in candidates:
        if a.ndim == 2 and a.shape == (H, W):
            return k, a

    # Prefer exact flat N
    for k, a in candidates:
        if a.ndim == 1 and a.size == N:
            return k, a.reshape(H, W)

    # If 3D, try to collapse time-like axis
    for k, a in candidates:
        if a.ndim == 3:
            # common patterns: (T,H,W) or (H,W,T)
            if a.shape[1:] == (H, W):
                return k, a.sum(axis=0)
            if a.shape[:2] == (H, W):
                return k, a.sum(axis=2)

    # Last resort: first array that can be reshaped to N
    for k, a in candidates:
        if a.size == N:
            return k, a.reshape(H, W)

    shapes = {k: list(a.shape) for k, a in candidates}
    raise ValueError(f"No array compatible with H*W={N}. Found shapes: {shapes}")

def main():
    ap = argparse.ArgumentParser(description="STRICT PP: extract trace_weights txt from any NPZ/NPY trace-like artifact.")
    ap.add_argument("--in_trace", required=True, help="NPZ/NPY containing trace density/visits at HxW")
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--report_out", required=True)
    ap.add_argument("--normalize", action="store_true", help="Normalize to sum=1")
    args = ap.parse_args()

    H, W = args.H, args.W
    N = H * W

    d = load_any(args.in_trace)
    key, grid = pick_array(d, H, W)

    assert grid.shape == (H, W), "shape mismatch after selection"
    grid = grid.astype(np.float64)

    # Ensure nonnegative
    if np.nanmin(grid) < 0:
        raise ValueError("Trace grid has negative values; expected counts/weights.")

    flat = grid.reshape(-1)
    s = float(np.sum(flat))

    if s <= 0:
        raise ValueError("Trace grid sum <= 0; cannot form weights.")

    if args.normalize:
        flat = flat / s

    # Write txt as one weight per node index
    with open(args.out_txt, "w") as f:
        f.write(f"# STRICT PP trace_weights extracted\n")
        f.write(f"# case={args.case} H={H} W={W} source={args.in_trace} key={key}\n")
        for i, v in enumerate(flat.tolist()):
            f.write(f"{i} {v}\n")

    report = {
        "case": args.case,
        "H": H, "W": W, "N": N,
        "in_trace": args.in_trace,
        "selected_key": key,
        "sum_raw": s,
        "normalize": bool(args.normalize),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "q50": float(np.quantile(flat, 0.50)),
        "nonzero": int(np.count_nonzero(flat)),
        "notes": "STRICT PP: this script only converts an existing trace-derived density/visit artifact into a flat trace_weights txt. No PDE, no Laplacian/Poisson, no GR ansatz, no regression."
    }

    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("WROTE", args.out_txt)
    print("WROTE", args.report_out)
    print("selected_key =", key, "nonzero =", report["nonzero"])

if __name__ == "__main__":
    main()

