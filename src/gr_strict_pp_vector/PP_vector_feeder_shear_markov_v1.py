#!/usr/bin/env python3
import argparse, json
import numpy as np
from collections import deque, defaultdict

# -----------------------------
# Strict PP helper functions
# -----------------------------

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

def dist_to_mass_multisrc_rBFS(edges, mass_ids, N):
    """
    dist[x] = min directed hops from x -> mass in ORIGINAL graph
    computed via multi-source BFS on REVERSED adjacency.
    """
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

def angle_labels(ids, H, W):
    rr = ids // W
    cc = ids % W
    r0 = (H - 1) / 2.0
    c0 = (W - 1) / 2.0
    ang = np.arctan2(rr - r0, cc - c0)  # LABELS ONLY
    return ang

def pick_band_from_quantiles(dist, reachable_mask, lo_q=0.60, hi_q=0.70):
    dvals = dist[reachable_mask]
    dvals = dvals[dvals > 0]
    if dvals.size == 0:
        return None
    lo = int(np.quantile(dvals, lo_q))
    hi = int(np.quantile(dvals, hi_q))
    if hi <= lo:
        hi = lo + 1
    return lo, hi

# -----------------------------
# Main observable
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP vector feeder-shear proxy v1 (sink-friendly)."
    )
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--mass_mask", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)

    # Band selection in TO-mass hop distance
    ap.add_argument("--band_lo", type=int, default=None)
    ap.add_argument("--band_hi", type=int, default=None)
    ap.add_argument("--band_lo_q", type=float, default=0.60)
    ap.add_argument("--band_hi_q", type=float, default=0.70)

    # Wedge binning (LABELS ONLY)
    ap.add_argument("--n_sectors", type=int, default=8)

    # Applicability + PASS gates
    ap.add_argument("--min_band_nodes", type=int, default=20)
    ap.add_argument("--min_sector_nodes", type=int, default=5)
    ap.add_argument("--shear_threshold", type=float, default=0.15)

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

    # STRICT PP distance TO mass (sink-friendly)
    dist = dist_to_mass_multisrc_rBFS(edges, mass_ids, N)
    reachable = dist >= 0

    # Determine band
    if args.band_lo is None or args.band_hi is None:
        pick = pick_band_from_quantiles(dist, reachable, args.band_lo_q, args.band_hi_q)
        if pick is None:
            out = {
                "H": H, "W": W, "N": N,
                "edges_curved": args.edges_curved,
                "mass_mask": args.mass_mask,
                "reachable_to_mass_count": int(reachable.sum()),
                "band_lo": None, "band_hi": None,
                "n_sectors": args.n_sectors,
                "FEEDER_SHEAR_APPLICABLE": False,
                "PASS_vector_feeder_shear_PP": None,
                "notes": (
                    "STRICT PP feeder-shear v1 N/A: "
                    "no positive TO-mass distances available on this directed graph."
                )
            }
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2, sort_keys=True)
            print("WROTE", args.output)
            return
        band_lo, band_hi = pick
    else:
        band_lo, band_hi = int(args.band_lo), int(args.band_hi)
        if band_hi <= band_lo:
            band_hi = band_lo + 1

    band = (dist >= band_lo) & (dist <= band_hi)
    band_ids = np.where(band)[0]
    band_size = int(band_ids.size)

    # If band too small: N/A
    if band_size < args.min_band_nodes:
        out = {
            "H": H, "W": W, "N": N,
            "edges_curved": args.edges_curved,
            "mass_mask": args.mass_mask,
            "reachable_to_mass_count": int(reachable.sum()),
            "band_lo": band_lo, "band_hi": band_hi,
            "band_size": band_size,
            "n_sectors": args.n_sectors,
            "FEEDER_SHEAR_APPLICABLE": False,
            "PASS_vector_feeder_shear_PP": None,
            "notes": (
                "STRICT PP feeder-shear v1 N/A: "
                "TO-mass feeder band too thin for stable sector comparison."
            )
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print("WROTE", args.output)
        return

    # LABEL-only angles -> sector bins
    ang = angle_labels(band_ids, H, W)
    # Map angle [-pi, pi) -> sector [0, n_sectors)
    nsec = int(args.n_sectors)
    # shift to [0, 2pi)
    ang2 = (ang + np.pi) % (2 * np.pi)
    sec = np.floor(ang2 / (2 * np.pi / nsec)).astype(int)
    sec = np.clip(sec, 0, nsec - 1)

    # Compute "inflow efficiency": does node have an outgoing edge that reduces dist by 1?
    # This is a directed drift proxy toward mass.
    has_step_in = np.zeros(band_size, dtype=np.int8)
    dist_map = dist  # alias

    # Build a set for fast membership if needed
    # but we only scan outgoing edges from band nodes
    band_index = {int(node): i for i, node in enumerate(band_ids.tolist())}

    for node in band_ids.tolist():
        i = band_index[node]
        dn = int(dist_map[node])
        if dn <= 0:
            continue
        ok = False
        for v in adj[node]:
            dv = int(dist_map[v])
            if dv == dn - 1:
                ok = True
                break
        has_step_in[i] = 1 if ok else 0

    # Aggregate per sector
    sector_nodes = [[] for _ in range(nsec)]
    for i, s in enumerate(sec.tolist()):
        sector_nodes[s].append(i)

    sector_size = []
    sector_eff = []
    sector_stepin_count = []

    for s in range(nsec):
        idxs = sector_nodes[s]
        sz = len(idxs)
        sector_size.append(sz)
        if sz == 0:
            sector_eff.append(None)
            sector_stepin_count.append(0)
        else:
            vals = has_step_in[idxs]
            c = int(vals.sum())
            sector_stepin_count.append(c)
            sector_eff.append(float(c) / float(sz))

    # Compare opposite sectors
    if nsec % 2 != 0:
        raise ValueError("n_sectors must be even for opposite-pair shear.")

    half = nsec // 2
    pairs = []
    for s in range(half):
        t = s + half
        e1 = sector_eff[s]
        e2 = sector_eff[t]
        sz1 = sector_size[s]
        sz2 = sector_size[t]
        if e1 is None or e2 is None:
            shear = None
        else:
            shear = abs(e1 - e2)
        pairs.append({
            "sector_a": s,
            "sector_b": t,
            "size_a": sz1,
            "size_b": sz2,
            "eff_a": e1,
            "eff_b": e2,
            "shear": shear
        })

    # Pick best valid pair by shear
    best = None
    best_val = -1.0
    for p in pairs:
        if p["shear"] is None:
            continue
        if p["size_a"] < args.min_sector_nodes or p["size_b"] < args.min_sector_nodes:
            continue
        if p["shear"] > best_val:
            best_val = p["shear"]
            best = p

    applicable = best is not None

    if not applicable:
        PASS = None
        shear_val = None
    else:
        shear_val = float(best_val)
        PASS = bool(shear_val >= float(args.shear_threshold))

    out = {
        "H": H, "W": W, "N": N,
        "edges_curved": args.edges_curved,
        "mass_mask": args.mass_mask,

        "reachable_to_mass_count": int(reachable.sum()),
        "band_lo": band_lo, "band_hi": band_hi,
        "band_lo_q": args.band_lo_q if args.band_lo is None else None,
        "band_hi_q": args.band_hi_q if args.band_hi is None else None,
        "band_size": band_size,

        "n_sectors": nsec,
        "min_band_nodes": args.min_band_nodes,
        "min_sector_nodes": args.min_sector_nodes,
        "shear_threshold": args.shear_threshold,

        "sector_size": sector_size,
        "sector_stepin_count": sector_stepin_count,
        "sector_inflow_eff": sector_eff,

        "opposite_pairs": pairs,
        "best_opposite_pair": best,
        "best_shear": shear_val,

        "FEEDER_SHEAR_APPLICABLE": bool(applicable),
        "PASS_vector_feeder_shear_PP": PASS,

        "notes": (
            "STRICT PP vector feeder-shear v1. "
            "Distance is directed Markov hop distance TO mass (multi-source reverse BFS). "
            "Feeder band is defined only by that operational distance. "
            "Angles are LABELS ONLY to bin nodes into opposite sectors. "
            "Vector proxy is asymmetry in 'inflow efficiency': fraction of nodes with an outgoing edge "
            "that reduces TO-mass distance by 1. "
            "No PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        )
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("FEEDER_SHEAR_APPLICABLE =", out["FEEDER_SHEAR_APPLICABLE"])
    print("PASS_vector_feeder_shear_PP =", out["PASS_vector_feeder_shear_PP"])
    if shear_val is not None:
        print("best_shear =", shear_val)

if __name__ == "__main__":
    main()
