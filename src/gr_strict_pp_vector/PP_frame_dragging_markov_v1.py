#!/usr/bin/env python3
import argparse, json
import numpy as np
from collections import deque

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

def build_adj(edges, N):
    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
    return adj

def bfs_dist(adj, src):
    N = len(adj)
    dist = np.full(N, -1, dtype=np.int32)
    q = deque([src])
    dist[src] = 0
    while q:
        u = q.popleft()
        du = dist[u]
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = du + 1
                q.append(v)
    return dist

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
        description="STRICT PP frame-dragging proxy v1 via directed Markov hop asymmetry on opposite ring/feeder-band arcs."
    )
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--orbit_mask", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--min_ring_nodes", type=int, default=10)
    ap.add_argument("--half_frac", type=float, default=0.5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    H, W = args.H, args.W
    N = H * W

    orbit = np.load(args.orbit_mask).astype(bool).ravel()
    if orbit.size != N:
        raise ValueError("orbit_mask size mismatch")

    edges = load_edges(args.edges_curved, N)
    adj = build_adj(edges, N)

    order = order_ring_nodes(orbit.reshape(H, W), H, W)
    L = len(order)

    if L < args.min_ring_nodes:
        out = {
            "H": H, "W": W, "N": N,
            "edges_curved": args.edges_curved,
            "orbit_mask": args.orbit_mask,
            "n_orbit_nodes": int(np.sum(orbit)),
            "n_ring_ordered": L,
            "FRAME_DRAGGING_APPLICABLE": False,
            "PASS_frame_dragging_PP": None,
            "notes": "STRICT PP: orbit band too small to define opposite-arc frame-dragging proxy."
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print("WROTE", args.output)
        return

    k = max(1, int(round(args.half_frac * L)))

    pairs = []
    for i in range(L):
        a = order[i]
        b = order[(i + k) % L]
        pairs.append((a, b))

    diffs = []
    used = 0
    for a, b in pairs:
        da = bfs_dist(adj, a)
        dab = da[b]
        if dab < 0:
            continue
        db = bfs_dist(adj, b)
        dba = db[a]
        if dba < 0:
            continue
        used += 1
        diffs.append(int(dab - dba))

    diffs = np.array(diffs, dtype=np.int32)
    applicable = used >= max(5, L // 3)

    if not applicable:
        PASS = None
        med = None
        med_abs = None
    else:
        med = float(np.median(diffs))
        med_abs = float(np.median(np.abs(diffs)))
        PASS = bool(med_abs >= 1.0)

    out = {
        "H": H, "W": W, "N": N,
        "edges_curved": args.edges_curved,
        "orbit_mask": args.orbit_mask,
        "half_frac": args.half_frac,
        "half_k_nodes": k,
        "n_orbit_nodes": int(np.sum(orbit)),
        "n_ring_ordered": L,
        "n_pairs_total": int(len(pairs)),
        "n_pairs_used": int(used),
        "median_diff_hops": med,
        "median_abs_diff_hops": med_abs,
        "FRAME_DRAGGING_APPLICABLE": bool(applicable),
        "PASS_frame_dragging_PP": PASS,
        "notes": (
            "STRICT PP frame-dragging proxy v1. "
            "Band ordering uses grid angles as LABELS ONLY. "
            "PASS uses directed BFS hop asymmetry for opposite-band pairs. "
            "No PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        )
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("FRAME_DRAGGING_APPLICABLE =", out["FRAME_DRAGGING_APPLICABLE"])
    print("PASS_frame_dragging_PP =", out["PASS_frame_dragging_PP"])

if __name__ == "__main__":
    main()
