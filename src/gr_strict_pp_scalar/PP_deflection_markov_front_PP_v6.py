#!/usr/bin/env python3
import argparse
import json
import numpy as np
from collections import deque


def load_edges(path: str, N: int):
    """
    Load directed edges from a text file.

    Allowed line formats:
      i j
      i j w
    Comments (#...) and blank lines are ignored.

    Returns:
      src, dst, w  (1D numpy arrays)
    """
    src = []
    dst = []
    w = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Bad edge line in {path!r}: {line!r}")
            try:
                i = int(parts[0])
                j = int(parts[1])
            except ValueError:
                raise ValueError(f"Non-integer indices in {path!r}: {line!r}")
            if i < 0 or i >= N or j < 0 or j >= N:
                raise ValueError(
                    f"Edge indices out of range in {path!r}: {i} -> {j} with N={N}"
                )
            if len(parts) >= 3:
                try:
                    wij = float(parts[2])
                except ValueError:
                    raise ValueError(f"Non-float weight in {path!r}: {line!r}")
            else:
                wij = 1.0
            src.append(i)
            dst.append(j)
            w.append(wij)

    if not src:
        raise ValueError(f"No edges loaded from {path!r}")

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)

    assert src.shape == dst.shape == w.shape
    assert src.ndim == 1
    return src, dst, w


def load_mask(path: str, N: int):
    """
    Load a boolean mask from .npy or .npz, reshape to (N,) and validate.
    """
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.size != N:
        # Allow (H, W) then flatten
        if arr.ndim == 2 and arr.shape[0] * arr.shape[1] == N:
            arr = arr.reshape(N)
        else:
            raise ValueError(
                f"Mask {path!r} has size {arr.size}, expected {N}. "
                f"Shape was {arr.shape}."
            )
    mask = arr.astype(bool).reshape(N)
    return mask


def build_reverse_adjacency(src: np.ndarray, dst: np.ndarray, N: int):
    """
    Build reverse adjacency lists: for each node j, list of i with edge i->j.
    Used for BFS distances TO mass (reverse graph).
    """
    assert src.shape == dst.shape
    assert src.ndim == 1
    rev = [[] for _ in range(N)]
    for i, j in zip(src, dst):
        rev[j].append(i)
    return rev


def bfs_dist_to_mass_reverse(src: np.ndarray,
                             dst: np.ndarray,
                             mass_mask: np.ndarray,
                             N: int) -> np.ndarray:
    """
    Compute hop distance TO mass using reverse BFS.

    - mass nodes have distance 0.
    - nodes reached in k reverse hops have distance k.
    - unreachable nodes get +inf.

    This uses ONLY the Markov edges and mass_mask, so it is STRICT PP.
    """
    rev = build_reverse_adjacency(src, dst, N)
    dist = np.full(N, np.inf, dtype=np.float64)

    mass_idx = np.nonzero(mass_mask)[0]
    if mass_idx.size == 0:
        raise ValueError("mass_mask has no True entries; cannot define distance-to-mass")

    q = deque()
    for m in mass_idx:
        dist[m] = 0.0
        q.append(m)

    while q:
        j = q.popleft()
        dj = dist[j]
        for i in rev[j]:
            if dist[i] == np.inf:
                dist[i] = dj + 1.0
                q.append(i)

    return dist


def build_sources_from_maxdist(dist_flat: np.ndarray,
                               mass_mask: np.ndarray,
                               source_band_width: int,
                               min_source_nodes: int = 10):
    """
    Source policy (STRICT PP, Markov-only):

    - dist_flat is hop distance TO mass on the **flat** graph.
    - mass nodes (dist=0) are excluded.
    - sources are nodes in the top 'source_band_width' integer distance bands
      closest to max finite distance.

    Returns:
      src_mask (bool[N]), meta (dict)
    """
    N = dist_flat.size
    assert mass_mask.size == N

    finite = np.isfinite(dist_flat)
    finite_nonmass = finite & (dist_flat > 0.0)
    if not finite_nonmass.any():
        return np.zeros(N, dtype=bool), {
            "APPLICABLE": False,
            "reason": "No non-mass nodes with finite Markov distance to mass."
        }

    # dist_flat from BFS is integer-valued in float form; cast safely
    dvals = dist_flat[finite_nonmass]
    dmax = int(dvals.max())
    if dmax <= 0:
        return np.zeros(N, dtype=bool), {
            "APPLICABLE": False,
            "reason": "Max Markov distance to mass is 0 (no outer band)."
        }

    bw = max(1, int(source_band_width))
    # Take the last 'bw' integer distances: [dmax-bw+1 ... dmax]
    lo = max(1, dmax - bw + 1)
    hi = dmax

    dist_int = np.full(N, -1, dtype=np.int64)
    valid = finite_nonmass
    dist_int[valid] = dist_flat[valid].astype(np.int64)

    band_mask = (dist_int >= lo) & (dist_int <= hi) & finite_nonmass & (~mass_mask)
    n_src = int(band_mask.sum())
    meta = {
        "APPLICABLE": n_src >= min_source_nodes,
        "reason": None if n_src >= min_source_nodes else (
            f"Too few source nodes ({n_src}) in Markov max-distance band "
            f"[{lo},{hi}] with min_source_nodes={min_source_nodes}."
        ),
        "N": N,
        "dmax": dmax,
        "band_lo": lo,
        "band_hi": hi,
        "band_width": bw,
        "n_source_nodes": n_src,
    }
    if n_src < min_source_nodes:
        return np.zeros(N, dtype=bool), meta
    return band_mask, meta


def build_edge_probs(src: np.ndarray, dst: np.ndarray, w: np.ndarray, N: int):
    """
    Build per-edge transition probabilities P(i->j) from raw weights w.

    For each node i, normalize outgoing weights so they sum to 1.
    Nodes with zero outdegree retain zero outgoing probability (front mass
    at such nodes becomes absorbing).
    """
    assert src.shape == dst.shape == w.shape
    assert src.ndim == 1
    M = src.size

    # Sum weights per source row
    out_sum = np.zeros(N, dtype=np.float64)
    np.add.at(out_sum, src, w)

    probs = np.zeros(M, dtype=np.float64)
    nonzero_mask = out_sum[src] > 0.0
    probs[nonzero_mask] = w[nonzero_mask] / out_sum[src[nonzero_mask]]

    return probs


def simulate_front(N: int,
                   src_nodes_mask: np.ndarray,
                   src_edges: np.ndarray,
                   dst_edges: np.ndarray,
                   probs: np.ndarray,
                   steps: int,
                   burn_in: int,
                   window: int):
    """
    Simulate a Markov front via repeated multiplications using edge list.

    - src_nodes_mask: boolean mask of sources (True = initial front mass).
    - src_edges, dst_edges, probs: define the Markov transition i->j with prob p.

    Returns:
      avg_front (N,): time-averaged front mass over [burn_in, burn_in+window).
    """
    assert src_edges.shape == dst_edges.shape == probs.shape
    assert src_edges.ndim == 1
    assert src_nodes_mask.size == N

    steps = int(steps)
    burn_in = int(burn_in)
    window = int(window)
    assert steps > 0
    assert burn_in >= 0
    assert window > 0
    assert burn_in + window <= steps

    cur = np.zeros(N, dtype=np.float64)
    # Uniform initial mass over sources (strictly, scale does not matter
    # as we compare relative masses, but we normalize anyway).
    n_src = int(src_nodes_mask.sum())
    if n_src == 0:
        return np.zeros(N, dtype=np.float64)
    cur[src_nodes_mask] = 1.0 / n_src

    avg = np.zeros(N, dtype=np.float64)

    for t in range(steps):
        # edge-based propagation: next[j] += cur[i] * P(i->j)
        mass_on_edges = cur[src_edges] * probs
        nxt = np.zeros(N, dtype=np.float64)
        np.add.at(nxt, dst_edges, mass_on_edges)

        cur = nxt

        if burn_in <= t < burn_in + window:
            avg += cur

    avg /= float(window)
    return avg


def select_markov_ring(dist_flat: np.ndarray,
                       mass_mask: np.ndarray,
                       orbit_mask: np.ndarray,
                       front_flat: np.ndarray,
                       front_curved: np.ndarray,
                       min_front_mass: float,
                       min_ring_nodes: int):
    """
    Adaptive Markov ring selection:

    - dist_flat: hop distance TO mass on the flat graph.
    - mass_mask: True for mass nodes.
    - orbit_mask: True for orbit-ring candidate nodes (outer band).
    - front_flat / front_curved: time-averaged front mass over steps.

    We look for the *smallest* d > 0 such that:

      ring_nodes = { i :
          orbit_mask[i] == True and
          mass_mask[i] == False and
          dist_int_flat[i] == d
      }

      len(ring_nodes) >= min_ring_nodes
      front_mass_flat_ring >= min_front_mass
      front_mass_curved_ring >= min_front_mass

    Returns:
      ring_info dict with APPLICABLE flag and ring fields.
    """
    N = dist_flat.size
    assert mass_mask.size == N
    assert orbit_mask.size == N
    assert front_flat.size == N
    assert front_curved.size == N

    finite = np.isfinite(dist_flat)
    valid = finite & (dist_flat > 0.0)
    if not valid.any():
        return {
            "APPLICABLE": False,
            "reason": "No non-mass nodes with finite Markov distance to mass (for ring).",
            "ring_distance_hops": None,
            "ring_idx": None,
            "front_mass_flat_ring": 0.0,
            "front_mass_curved_ring": 0.0,
        }

    dist_int = np.full(N, -1, dtype=np.int64)
    dist_int[valid] = dist_flat[valid].astype(np.int64)

    dvals = dist_int[valid]
    dmax = int(dvals.max())
    if dmax <= 0:
        return {
            "APPLICABLE": False,
            "reason": "Max Markov distance to mass is 0 (no ring layer).",
            "ring_distance_hops": None,
            "ring_idx": None,
            "front_mass_flat_ring": 0.0,
            "front_mass_curved_ring": 0.0,
        }

    chosen_d = None
    chosen_idx = None
    fm_flat = 0.0
    fm_curved = 0.0

    for d in range(1, dmax + 1):
        ring_mask = (
            (dist_int == d) &
            orbit_mask &
            (~mass_mask)
        )
        idx = np.nonzero(ring_mask)[0]
        n_ring = int(idx.size)
        if n_ring < min_ring_nodes:
            continue

        fm_f = float(front_flat[idx].sum())
        fm_c = float(front_curved[idx].sum())

        if fm_f >= min_front_mass and fm_c >= min_front_mass:
            chosen_d = d
            chosen_idx = idx
            fm_flat = fm_f
            fm_curved = fm_c
            break

    if chosen_d is None:
        # record the last seen flat mass on orbit (for diagnostics)
        # (not strictly necessary to keep, but helpful in JSON)
        return {
            "APPLICABLE": False,
            "reason": "No Markov hop-distance band (d>0) with sufficient nodes and "
                      "nonzero flat/curved front mass.",
            "ring_distance_hops": None,
            "ring_idx": None,
            "front_mass_flat_ring": float(front_flat[(orbit_mask) & (~mass_mask)].sum()),
            "front_mass_curved_ring": float(front_curved[(orbit_mask) & (~mass_mask)].sum()),
        }

    return {
        "APPLICABLE": True,
        "reason": None,
        "ring_distance_hops": int(chosen_d),
        "ring_idx": chosen_idx.tolist(),
        "front_mass_flat_ring": fm_flat,
        "front_mass_curved_ring": fm_curved,
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "STRICT PP deflection via Markov fronts v6. "
            "Sources are from a Markov max-distance band to mass on the flat graph; "
            "ring is defined by Markov hop-distance to mass and orbit_mask; "
            "observable is front-mass difference on that ring."
        )
    )
    ap.add_argument("--edges_flat", required=True)
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--mass_mask", required=True)
    ap.add_argument("--orbit_mask", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--burn_in", type=int, default=64)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--min_front_mass", type=float, default=1e-4)
    ap.add_argument("--min_ring_nodes", type=int, default=100)
    ap.add_argument("--delta_threshold", type=float, default=0.05)
    ap.add_argument("--source_band_width", type=int, default=4)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    H = int(args.H)
    W = int(args.W)
    N = H * W

    # --- Load geometry / masks ---
    src_f, dst_f, w_f = load_edges(args.edges_flat, N)
    src_c, dst_c, w_c = load_edges(args.edges_curved, N)

    assert src_f.size == dst_f.size == w_f.size
    assert src_c.size == dst_c.size == w_c.size

    mass_mask = load_mask(args.mass_mask, N)
    orbit_mask = load_mask(args.orbit_mask, N)

    # --- Markov distance to mass on flat graph (reverse BFS) ---
    dist_flat = bfs_dist_to_mass_reverse(src_f, dst_f, mass_mask, N)

    # --- Source selection: Markov max-distance band on flat graph ---
    src_mask, src_meta = build_sources_from_maxdist(
        dist_flat,
        mass_mask,
        source_band_width=args.source_band_width,
        min_source_nodes=10
    )

    # Prepare base JSON
    out = {
        "H": H,
        "W": W,
        "N": N,
        "edges_flat": args.edges_flat,
        "edges_curved": args.edges_curved,
        "mass_mask": args.mass_mask,
        "orbit_mask": args.orbit_mask,
        "steps": int(args.steps),
        "burn_in": int(args.burn_in),
        "window": int(args.window),
        "min_front_mass": float(args.min_front_mass),
        "min_ring_nodes": int(args.min_ring_nodes),
        "delta_threshold": float(args.delta_threshold),
        "source_band_width": int(args.source_band_width),
        "notes": (
            "STRICT PP deflection v6 with Markov max-distance sources and adaptive "
            "Markov ring. Distance and sources are defined solely via Markov hop "
            "distance to mass on the flat graph. Ring selection is based on "
            "Markov hop distance and orbit_mask. Observable is front-mass "
            "difference on the selected ring. No PDE, no Laplacian/Poisson, "
            "no GR ansatz, no regression."
        ),
        "source_meta": {
            **src_meta,
            "auto_source": True,
        },
        "ring_selection": None,
        "deflection_stats": None,
    }

    # If source policy fails, mark test as NA
    if not src_meta["APPLICABLE"]:
        out["ring_selection"] = {
            "APPLICABLE": False,
            "reason": "Source selection failed: " + (src_meta["reason"] or ""),
            "ring_distance_hops": None,
            "ring_idx": None,
            "front_mass_flat_ring": 0.0,
            "front_mass_curved_ring": 0.0,
        }
        out["deflection_stats"] = {
            "APPLICABLE": False,
            "PASS_deflection_markov_front_PP_v6": False,
            "D_flat": None,
            "D_curved": None,
            "DeltaD": None,
            "reason": "Source selection failed (no sufficient Markov max-distance band)."
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print("WROTE", args.output)
        print("APPLICABLE = False")
        print("reason:", out["deflection_stats"]["reason"])
        return

    # --- Front simulation setup ---
    probs_f = build_edge_probs(src_f, dst_f, w_f, N)
    probs_c = build_edge_probs(src_c, dst_c, w_c, N)

    front_flat = simulate_front(
        N,
        src_mask,
        src_f,
        dst_f,
        probs_f,
        steps=args.steps,
        burn_in=args.burn_in,
        window=args.window,
    )
    front_curved = simulate_front(
        N,
        src_mask,
        src_c,
        dst_c,
        probs_c,
        steps=args.steps,
        burn_in=args.burn_in,
        window=args.window,
    )

    # --- Adaptive Markov ring selection ---
    ring_info = select_markov_ring(
        dist_flat=dist_flat,
        mass_mask=mass_mask,
        orbit_mask=orbit_mask,
        front_flat=front_flat,
        front_curved=front_curved,
        min_front_mass=args.min_front_mass,
        min_ring_nodes=args.min_ring_nodes,
    )
    out["ring_selection"] = ring_info

    if not ring_info["APPLICABLE"]:
        out["deflection_stats"] = {
            "APPLICABLE": False,
            "PASS_deflection_markov_front_PP_v6": False,
            "D_flat": None,
            "D_curved": None,
            "DeltaD": None,
            "reason": ring_info["reason"],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print("WROTE", args.output)
        print("APPLICABLE = False")
        print("reason:", ring_info["reason"])
        return

    # --- Deflection metric on the selected ring ---
    ring_idx = np.asarray(ring_info["ring_idx"], dtype=np.int64)
    fm_flat = float(ring_info["front_mass_flat_ring"])
    fm_curved = float(ring_info["front_mass_curved_ring"])

    D_flat = fm_flat
    D_curved = fm_curved
    DeltaD = D_curved - D_flat
    ref = max(D_flat, 1e-12)
    passed = bool(abs(DeltaD) >= args.delta_threshold * ref)

    out["deflection_stats"] = {
        "APPLICABLE": True,
        "PASS_deflection_markov_front_PP_v6": passed,
        "D_flat": D_flat,
        "D_curved": D_curved,
        "DeltaD": DeltaD,
        "ring_distance_hops": ring_info["ring_distance_hops"],
        "n_ring_nodes": int(ring_idx.size),
        "reason": None if passed else (
            "Deflection signal below threshold: "
            f"|DeltaD|={abs(DeltaD):.6g} < "
            f"delta_threshold*D_flat={args.delta_threshold*ref:.6g}."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("APPLICABLE = True")
    print("PASS_deflection_markov_front_PP_v6 =", passed)


if __name__ == "__main__":
    main()

