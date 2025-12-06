#!/usr/bin/env python3
import argparse, json
import numpy as np

def load_edges(path, N):
    edges = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace(",", " ")
            p = line.split()
            if len(p) < 2:
                continue
            try:
                a = int(p[0]); b = int(p[1])
            except:
                continue
            if 0 <= a < N and 0 <= b < N:
                edges.append((a, b))
    return edges

def order_ring_nodes(or_mask_bool, H, W):
    ids = np.where(or_mask_bool.ravel())[0]
    if ids.size < 3:
        return []
    rr = ids // W
    cc = ids % W
    r0 = (H - 1) / 2.0
    c0 = (W - 1) / 2.0
    ang = np.arctan2(rr - r0, cc - c0)
    order = ids[np.argsort(ang)]
    return order.tolist()

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP vector azimuthal flux bias v1 on a Markov-defined ring band."
    )
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--orbit_mask", required=True)
    ap.add_argument("--mass_mask", required=False, default=None)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--min_ring_nodes", type=int, default=12)
    ap.add_argument("--step_window", type=int, default=3)
    ap.add_argument("--min_edges_used", type=int, default=25)
    ap.add_argument("--bias_threshold", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    H, W = args.H, args.W
    N = H * W

    orbit = np.load(args.orbit_mask).astype(bool).ravel()
    if orbit.size != N:
        raise ValueError("orbit_mask size mismatch")

    order = order_ring_nodes(orbit.reshape(H, W), H, W)
    L = len(order)

    if L < args.min_ring_nodes:
        out = {
            "H": H, "W": W, "N": N,
            "edges_curved": args.edges_curved,
            "orbit_mask": args.orbit_mask,
            "mass_mask": args.mass_mask,
            "n_orbit_nodes": int(np.sum(orbit)),
            "n_ring_ordered": L,
            "VECTOR_AZIMUTHAL_FLUX_APPLICABLE": False,
            "PASS_vector_azimuthal_flux_PP": None,
            "notes": "STRICT PP: ring band too small to label a stable azimuthal ordering."
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print("WROTE", args.output)
        return

    # label-only index map
    idx = {node: i for i, node in enumerate(order)}
    ring_set = set(order)

    edges = load_edges(args.edges_curved, N)

    cw = 0
    ccw = 0
    other = 0
    used = 0

    w = max(1, int(args.step_window))

    for a, b in edges:
        if (a not in ring_set) or (b not in ring_set):
            continue
        ia = idx[a]
        ib = idx[b]
        # signed modular step from a -> b
        d = (ib - ia) % L
        if d == 0:
            other += 1
            continue

        # classify only "near-azimuthal" steps within window
        if 1 <= d <= w:
            cw += 1
            used += 1
        elif 1 <= ((ia - ib) % L) <= w:
            ccw += 1
            used += 1
        else:
            other += 1

    applicable = used >= args.min_edges_used

    if not applicable:
        PASS = None
        bias = None
    else:
        denom = max(1, cw + ccw)
        bias = (cw - ccw) / denom  # in [-1,1]
        PASS = bool(abs(bias) >= args.bias_threshold)

    out = {
        "H": H, "W": W, "N": N,
        "edges_curved": args.edges_curved,
        "orbit_mask": args.orbit_mask,
        "mass_mask": args.mass_mask,
        "step_window": w,
        "bias_threshold": args.bias_threshold,
        "min_edges_used": args.min_edges_used,
        "n_orbit_nodes": int(np.sum(orbit)),
        "n_ring_ordered": L,
        "edges_in_band_total": int(cw + ccw + other),
        "edges_used_near_azimuthal": int(used),
        "cw_count": int(cw),
        "ccw_count": int(ccw),
        "other_count": int(other),
        "azimuthal_bias": bias,
        "VECTOR_AZIMUTHAL_FLUX_APPLICABLE": bool(applicable),
        "PASS_vector_azimuthal_flux_PP": PASS,
        "notes": (
            "STRICT PP vector azimuthal flux bias v1. "
            "Ring ordering uses grid angles as LABELS ONLY. "
            "Observable counts directed edges inside a Markov-defined ring band "
            "that step forward (CW) vs backward (CCW) within a small index window. "
            "PASS is a nonzero azimuthal bias above threshold. "
            "No PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        )
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("VECTOR_AZIMUTHAL_FLUX_APPLICABLE =", out["VECTOR_AZIMUTHAL_FLUX_APPLICABLE"])
    print("PASS_vector_azimuthal_flux_PP =", out["PASS_vector_azimuthal_flux_PP"])
    if bias is not None:
        print("azimuthal_bias =", bias)

if __name__ == "__main__":
    main()
