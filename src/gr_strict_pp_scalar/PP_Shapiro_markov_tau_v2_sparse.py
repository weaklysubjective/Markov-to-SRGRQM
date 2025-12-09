#!/usr/bin/env python3
import argparse, json, time, os
import numpy as np

def load_edges(path):
    # Accept both "a b" text and comma separated
    # Returns src, dst, w (float)
    data = np.loadtxt(path, dtype=float)
    if data.size == 0:
        raise ValueError(f"Empty edges file: {path}")
    if data.ndim == 1:
        # single line
        if data.shape[0] < 2:
            raise ValueError(f"Unexpected edge format in {path}, shape={data.shape}")
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Unexpected edge format in {path}, shape={data.shape}")

    src = data[:, 0].astype(np.int64)
    dst = data[:, 1].astype(np.int64)
    if data.shape[1] >= 3:
        w = data[:, 2].astype(np.float64)
    else:
        w = np.ones_like(src, dtype=np.float64)
    return src, dst, w

def build_sparse_undirected_laplacian(N, src, dst, w):
    # Build symmetric adjacency from directed edges
    # A_ij = w for i->j plus j->i
    try:
        import scipy.sparse as sp
    except Exception as e:
        raise RuntimeError("scipy is required for sparse Shapiro v2") from e

    # keep only valid bounds
    m = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
    src = src[m]; dst = dst[m]; w = w[m]

    # symmetrize
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    vals = np.concatenate([w, w])

    #A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64).tocsr()
    #deg = np.asarray(A.sum(axis=1)).ravel()
    #L = sp.diags(deg, 0, shape=(N, N), dtype=np.float64) - A

    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64).tocsr()

    # STRICT PP Doyle-resistance hygiene
    A = 0.5 * (A + A.T)
    A.eliminate_zeros()

    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(deg, 0, shape=(N, N), dtype=np.float64) - A

    vol = float(deg.sum())
    return L, vol, deg

def effective_resistance_pair(L, s, t, tol=1e-5, maxiter=200000, eps=1e-10):
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = L.shape[0]
    if s == t:
        return 0.0

    # --- numeric stabilization (STRICT PP-safe) ---
    L = L.tocsr()
    Lr = L + eps * sp.eye(n, format="csr")

    # RHS
    b = np.zeros(n, dtype=np.float64)
    b[s] = 1.0
    b[t] = -1.0

    # Jacobi preconditioner
    d = Lr.diagonal()
    d = np.where(np.abs(d) < 1e-14, 1.0, d)
    M = sp.diags(1.0 / d, format="csr")

    # SciPy API compatibility
    try:
        x, info = spla.cg(Lr, b, M=M, rtol=tol, atol=0.0, maxiter=maxiter)
    except TypeError:
        x, info = spla.cg(Lr, b, M=M, tol=tol, maxiter=maxiter)

    if info != 0:
        # last resort: relax tol ladder
        for r in (3e-5, 1e-4, 3e-4):
            try:
                x, info = spla.cg(Lr, b, M=M, rtol=r, atol=0.0, maxiter=maxiter)
            except TypeError:
                x, info = spla.cg(Lr, b, M=M, tol=r, maxiter=maxiter)
            if info == 0:
                break

    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info}).")

    return float(b @ x)

def compute_commute_pairset(edges_path, H, W, pairs, tol, maxiter):
    N = H * W
    src, dst, w = load_edges(edges_path)
    L, vol, deg = build_sparse_undirected_laplacian(N, src, dst, w)

    results = []
    for (s, t) in pairs:
        R = effective_resistance_pair(L, s, t, tol=tol, maxiter=maxiter)
        C = vol * R
        results.append({
            "src": int(s),
            "dst": int(t),
            "R_eff": R,
            "commute": C
        })

    # summarize
    commutes = np.array([r["commute"] for r in results], dtype=np.float64)
    summary = {
        "edges_file": edges_path,
        "N": N,
        "H": H, "W": W,
        "n_pairs": len(results),
        "vol": vol,
        "deg_mean": float(deg.mean()) if deg.size else None,
        "commute_mean": float(commutes.mean()) if commutes.size else None,
        "commute_median": float(np.median(commutes)) if commutes.size else None,
        "pairs": results
    }
    return summary

def main():
    ap = argparse.ArgumentParser(description="STRICT PP Shapiro Markov tau v2 (sparse).")
    ap.add_argument("--edges_flat", required=True)
    ap.add_argument("--edges_curved", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)

    ap.add_argument("--src_through", type=int, required=True)
    ap.add_argument("--dst_through", type=int, required=True)
    ap.add_argument("--src_around", type=int, required=True)
    ap.add_argument("--dst_around", type=int, required=True)

    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=20000)

    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t0 = time.time()

    pairs_through = [(args.src_through, args.dst_through)]
    pairs_around  = [(args.src_around,  args.dst_around)]

    flat_through = compute_commute_pairset(args.edges_flat, args.H, args.W, pairs_through, args.tol, args.maxiter)
    flat_around  = compute_commute_pairset(args.edges_flat, args.H, args.W, pairs_around,  args.tol, args.maxiter)

    curved_through = compute_commute_pairset(args.edges_curved, args.H, args.W, pairs_through, args.tol, args.maxiter)
    curved_around  = compute_commute_pairset(args.edges_curved, args.H, args.W, pairs_around,  args.tol, args.maxiter)

    flat_tau_through   = flat_through["pairs"][0]["commute"]
    flat_tau_around    = flat_around["pairs"][0]["commute"]
    curved_tau_through = curved_through["pairs"][0]["commute"]
    curved_tau_around  = curved_around["pairs"][0]["commute"]

    d_tau_through = curved_tau_through - flat_tau_through
    d_tau_around  = curved_tau_around  - flat_tau_around

    # STRICT PP pass rule same spirit as v1:
    # "through" should show extra delay vs flat,
    # and typically d_tau_through > d_tau_around.
    PASS = bool(d_tau_through > 0 and d_tau_through > d_tau_around)

    out = {
        "H": args.H, "W": args.W, "N": args.H * args.W,
        "edges_flat": args.edges_flat,
        "edges_curved": args.edges_curved,
        "src_through": args.src_through,
        "dst_through": args.dst_through,
        "src_around": args.src_around,
        "dst_around": args.dst_around,

        "flat_tau_through": flat_tau_through,
        "flat_tau_around": flat_tau_around,
        "curved_tau_through": curved_tau_through,
        "curved_tau_around": curved_tau_around,
        "d_tau_through": d_tau_through,
        "d_tau_around": d_tau_around,

        "PASS_Shapiro_markov_tau": PASS,

        "flat_detail_through": flat_through,
        "flat_detail_around": flat_around,
        "curved_detail_through": curved_through,
        "curved_detail_around": curved_around,

        "runtime_sec": time.time() - t0,
        "notes": (
            "STRICT PP Shapiro Markov tau v2 (sparse). Commute times computed from "
            "graph-effective resistance using sparse symmetric Laplacian and CG solves "
            "for the specified pair(s). No PDE, no Poisson/Laplacian field injection, "
            "no Euclidean radius assumptions, no GR ansatz, no regression."
        )
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"Wrote Shapiro Markov τ report to {args.output}")
    print("PASS_Shapiro_markov_tau =", "TRUE" if PASS else "FALSE")

if __name__ == "__main__":
    main()

