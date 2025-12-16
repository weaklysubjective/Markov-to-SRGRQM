#!/usr/bin/env python3
import argparse, json, os
import numpy as np

def load_npz(p: str):
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: missing npz: {p}")
    return np.load(p)

def robust_rel_mad(x: np.ndarray):
    x = x[np.isfinite(x)]
    if x.size < 10:
        return None, None
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    rel = float(mad / (abs(med) + 1e-300))
    return med, rel

def corr(a: np.ndarray, b: np.ndarray):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return None
    aa = a[m]; bb = b[m]
    if float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0,1])

def main():
    ap = argparse.ArgumentParser(description="STRICT PP 512: field-level align stats for available (Gmunu,Tmunu) components.")
    ap.add_argument("--case", required=True)
    ap.add_argument("--tau_npz", required=True, help="PP_markov_tau_geometry_512_<case>_PPV1.npz (has G00_med,T00,mass_core_mask,kappa_med)")
    ap.add_argument("--T_npz", required=True, help="PP_Tmunu_edges_512_<case>_STRICT_PP_v1.npz")
    ap.add_argument("--output", required=True)
    ap.add_argument("--eps_T", type=float, default=1e-30)
    args = ap.parse_args()

    Z = load_npz(args.tau_npz)
    T = load_npz(args.T_npz)

    required_tau = ["T00","G00_med","kappa_med","mass_core_mask"]
    for k in required_tau:
        if k not in Z.files:
            raise SystemExit(f"ERROR: {k} missing from tau_npz. keys={list(Z.files)}")

    for k in ["T00","T0x","T0y","Txx","Tyx","Tyy"]:
        if k not in T.files:
            raise SystemExit(f"ERROR: {k} missing from T_npz. keys={list(T.files)}")

    T00 = Z["T00"].reshape(-1).astype(np.float64)
    G00 = Z["G00_med"].reshape(-1).astype(np.float64)
    kappa = Z["kappa_med"].reshape(-1).astype(np.float64)
    core = Z["mass_core_mask"].reshape(-1).astype(bool)

    T00b = T["T00"].reshape(-1).astype(np.float64)
    if T00b.shape != T00.shape:
        raise SystemExit("ERROR: T00 shape mismatch between tau_npz and T_npz")

    # Core-only stats (primary)
    m_core = core & np.isfinite(T00) & (np.abs(T00) > float(args.eps_T)) & np.isfinite(G00) & np.isfinite(kappa)

    k_med, k_relmad = robust_rel_mad(kappa[m_core])
    c = corr(G00[m_core], T00[m_core])

    out = {
        "case": args.case,
        "STRICT_PP": True,
        "inputs": {
            "tau_npz": args.tau_npz,
            "T_npz": args.T_npz,
        },
        "status": {
            "APPLICABLE": bool(m_core.sum() >= 10),
        },
        "EFE00_core": {
            "n_core_good": int(m_core.sum()),
            "kappa_med": k_med,
            "kappa_rel_mad": k_relmad,
            "corr_G00_vs_T00": c,
            "sign_flip_frac_vs_kappa_med": (
                None if k_med is None else float(np.mean(np.sign(kappa[m_core]) != np.sign(k_med)))
            ),
        },
        "notes": (
            "This is a field-level report. Today it validates EFE00 via (kappa flatness on core, sign stability). "
            "Full Guv=Tuv requires per-node G0i/Gij artifacts; this script is structured to extend once those exist."
        ),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("APPLICABLE =", out["status"]["APPLICABLE"])
    print("n_core_good =", out["EFE00_core"]["n_core_good"])
    print("kappa_rel_mad =", out["EFE00_core"]["kappa_rel_mad"])

if __name__ == "__main__":
    main()

