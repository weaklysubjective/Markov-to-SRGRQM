#!/usr/bin/env python3
import argparse, json, os
import numpy as np

def load_npz(p: str):
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: missing npz: {p}")
    return np.load(p)

def robust_rel_mad(x):
    x = x[np.isfinite(x)]
    if x.size < 10:
        return None, None
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    rel = float(mad / (abs(med) + 1e-300))
    return med, rel

def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return None
    aa = a[m]; bb = b[m]
    if float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0,1])

def main():
    ap = argparse.ArgumentParser(description="STRICT PP 512: component-wise align for (Gproxy vs T) on core.")
    ap.add_argument("--case", required=True)
    ap.add_argument("--tau_npz", required=True, help="Provides mass_core_mask and kappa_med if present")
    ap.add_argument("--T_npz", required=True, help="PP_Tmunu_edges_512_<case>_STRICT_PP_v1.npz")
    ap.add_argument("--G_npz", required=True, help="PP_Gmunu_proxy_edges_512_<case>_STRICT_PP_v1.npz")
    ap.add_argument("--output", required=True)
    ap.add_argument("--eps_T", type=float, default=1e-30)
    args = ap.parse_args()

    Z = load_npz(args.tau_npz)
    T = load_npz(args.T_npz)
    G = load_npz(args.G_npz)

    if "mass_core_mask" not in Z.files:
        raise SystemExit("ERROR: tau_npz missing mass_core_mask")
    core = Z["mass_core_mask"].reshape(-1).astype(bool)

    # Use kappa_med if available; else estimate core kappa from G00/T00 (from tau_npz’s T00 if present; else from T_npz T00)
    kappa = None
    if "kappa_med" in Z.files:
        kappa = Z["kappa_med"].reshape(-1).astype(np.float64)

    # Required components
    needT = ["T00","T0x","T0y","Txx","Tyx","Tyy"]
    needG = ["G00","G0x","G0y","Gxx","Gxy","Gyy"]
    for k in needT:
        if k not in T.files: raise SystemExit(f"ERROR: missing {k} in T_npz")
    for k in needG:
        if k not in G.files: raise SystemExit(f"ERROR: missing {k} in G_npz")

    T00 = T["T00"].reshape(-1).astype(np.float64)
    T0x = T["T0x"].reshape(-1).astype(np.float64)
    T0y = T["T0y"].reshape(-1).astype(np.float64)
    Txx = T["Txx"].reshape(-1).astype(np.float64)
    Tyx = T["Tyx"].reshape(-1).astype(np.float64)
    Tyy = T["Tyy"].reshape(-1).astype(np.float64)

    G00 = G["G00"].reshape(-1).astype(np.float64)
    G0x = G["G0x"].reshape(-1).astype(np.float64)
    G0y = G["G0y"].reshape(-1).astype(np.float64)
    Gxx = G["Gxx"].reshape(-1).astype(np.float64)
    Gxy = G["Gxy"].reshape(-1).astype(np.float64)
    Gyy = G["Gyy"].reshape(-1).astype(np.float64)

    # Define core-good mask for ratios using T00 (density) as support
    m_core = core & np.isfinite(T00) & (np.abs(T00) > float(args.eps_T)) & np.isfinite(G00)

    # Core kappa estimate (median of G00/T00) for scaling the T side
    k_est = float(np.median((G00[m_core] / T00[m_core]).astype(np.float64)))

    def comp_metrics(name, Gc, Tc):
        # Compare Gc vs k_est * Tc on core (robust)
        A = Gc[m_core]
        B = (k_est * Tc)[m_core]
        diff = A - B
        med, relmad = robust_rel_mad(diff)
        return {
            "corr_G_vs_kT": corr(A, B),
            "diff_med": med,
            "diff_rel_mad": relmad,
        }

    out = {
        "case": args.case,
        "STRICT_PP": True,
        "inputs": {"tau_npz": args.tau_npz, "T_npz": args.T_npz, "G_npz": args.G_npz},
        "status": {"APPLICABLE": bool(m_core.sum() >= 10)},
        "core": {"n_core_good": int(m_core.sum()), "kappa_est_med_G00_over_T00": k_est},
        "components": {
            "00": comp_metrics("00", G00, T00),
            "0x": comp_metrics("0x", G0x, T0x),
            "0y": comp_metrics("0y", G0y, T0y),
            "xx": comp_metrics("xx", Gxx, Txx),
            "xy": comp_metrics("xy", Gxy, Tyx),  # note: Tyx is the symmetric mixed component from your builder
            "yy": comp_metrics("yy", Gyy, Tyy),
        },
        "notes": (
            "Component-wise strict PP alignment report: compares Gproxy components to kappa_est*T components on the mass core. "
            "This is the first per-node multi-component Guv~kTuv report at 512 using only E2 edges + tau geometry anchor."
        ),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("APPLICABLE =", out["status"]["APPLICABLE"])
    print("n_core_good =", out["core"]["n_core_good"])
    print("kappa_est =", out["core"]["kappa_est_med_G00_over_T00"])

if __name__ == "__main__":
    main()

