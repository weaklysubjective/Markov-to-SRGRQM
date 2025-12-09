#!/usr/bin/env python3
import argparse, json, math
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

def build_radj(edges, N):
    radj = [[] for _ in range(N)]
    for a, b in edges:
        radj[b].append(a)
    return radj

def dist_to_mass_multisrc(edges, mass_ids, N):
    # dist[x] = min hops x -> mass in original graph
    radj = build_radj(edges, N)
    dist = np.full(N, -1, dtype=np.int32)
    q = deque()
    for s in mass_ids:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        du = dist[u]
        for v in radj[u]:
            if dist[v] < 0:
                dist[v] = du + 1
                q.append(v)
    return dist

def tarjan_scc_sizes(nodes, adj):
    """
    Tarjan SCC on induced subgraph of 'nodes'.
    Returns list of SCC sizes.
    """
    node_set = set(nodes)
    idx = {}
    low = {}
    stack = []
    onstack = set()
    index = 0
    sizes = []

    def strongconnect(v):
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in adj[v]:
            if w not in node_set:
                continue
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])

        if low[v] == idx[v]:
            sz = 0
            while True:
                w = stack.pop()
                onstack.remove(w)
                sz += 1
                if w == v:
                    break
            sizes.append(sz)

    for v in nodes:
        if v not in idx:
            strongconnect(v)

    return sizes

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP vector scale admissibility probe v1."
    )
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--mass_mask", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--band_q_pairs", nargs="*", default=["0.60,0.70","0.65,0.75","0.70,0.80","0.75,0.85","0.80,0.90"])
    ap.add_argument("--scc_max_nodes", type=int, default=5000,
                    help="Skip SCC analysis for bands larger than this (to keep runtime sane).")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    H, W = args.H, args.W
    N = H * W

    mass = np.load(args.mass_mask).astype(bool).ravel()
    if mass.size != N:
        raise ValueError("mass_mask size mismatch")
    mass_ids = np.where(mass)[0].tolist()
    if not mass_ids:
        raise ValueError("empty mass mask")

    edges = load_edges(args.edges_curved, N)
    adj = build_adj(edges, N)

    # Mass outdegree
    mass_set = set(mass_ids)
    mass_out = 0
    mass_out_to_nonmass = 0
    for u in mass_ids:
        outs = adj[u]
        mass_out += len(outs)
        for v in outs:
            if v not in mass_set:
                mass_out_to_nonmass += 1

    # TO-mass distances
    dist = dist_to_mass_multisrc(edges, mass_ids, N)
    reachable = dist >= 0
    reachable_count = int(reachable.sum())
    pos = dist[reachable]
    pos = pos[pos > 0]

    band_reports = []
    if pos.size > 0:
        for pair in args.band_q_pairs:
            lo_q, hi_q = pair.split(",")
            lo_q = float(lo_q); hi_q = float(hi_q)
            lo = int(np.quantile(pos, lo_q))
            hi = int(np.quantile(pos, hi_q))
            if hi <= lo:
                hi = lo + 1
            band = (dist >= lo) & (dist <= hi)
            ids = np.where(band)[0].tolist()
            sz = len(ids)

            scc_info = None
            if sz > 0 and sz <= args.scc_max_nodes:
                sizes = tarjan_scc_sizes(ids, adj)
                sizes.sort(reverse=True)
                scc_info = {
                    "n_scc": len(sizes),
                    "largest_scc_size": int(sizes[0]) if sizes else 0,
                    "top5_scc_sizes": [int(x) for x in sizes[:5]]
                }
            elif sz > args.scc_max_nodes:
                scc_info = {
                    "skipped": True,
                    "reason": "band too large for SCC pass at this probe setting",
                    "n_band_nodes": sz
                }

            band_reports.append({
                "lo_q": lo_q, "hi_q": hi_q,
                "band_lo": lo, "band_hi": hi,
                "band_size": sz,
                "scc": scc_info
            })

    out = {
        "H": H, "W": W, "N": N,
        "edges_curved": args.edges_curved,
        "mass_mask": args.mass_mask,
        "mass_core_size": len(mass_ids),
        "mass_outdeg_total": int(mass_out),
        "mass_outdeg_to_nonmass": int(mass_out_to_nonmass),
        "reachable_to_mass_count": reachable_count,
        "reachable_to_mass_frac": float(reachable_count) / float(N),
        "has_positive_to_mass_distances": bool(pos.size > 0),
        "band_reports": band_reports,
        "notes": (
            "STRICT PP admissibility probe. Computes directed TO-mass hop distances "
            "and examines band thickness and SCC availability as prerequisites for "
            "vector ring/arc observables. No PDE, no Laplacian/Poisson, no regression."
        )
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("mass_outdeg_to_nonmass =", out["mass_outdeg_to_nonmass"])
    print("reachable_to_mass_frac =", out["reachable_to_mass_frac"])
    if band_reports:
        best = max(band_reports, key=lambda b: b["band_size"])
        print("largest_band_size =", best["band_size"], "at q", best["lo_q"], best["hi_q"])
        if best["scc"] and not best["scc"].get("skipped", False):
            print("largest_scc_size =", best["scc"].get("largest_scc_size"))

if __name__ == "__main__":
    main()
