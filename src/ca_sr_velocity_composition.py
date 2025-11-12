#!/usr/bin/env python3
import argparse, json, math

def N_rest(T):
    m = T//2
    return (2*m*(m+1)-1) if (T%2==0) else ((m+1)**2 + m**2)

def N_moving(T, D):
    N=0
    for t in range(1, T):
        L=max(-t, D-(T-t)); R=min(t, D+(T-t))
        if L<=R: N += (R-L+1)
    return N

def gamma_hat_from_counts(T, v):
    D = int(round(v*T))
    N0 = N_rest(T)
    Nm = N_moving(T, D)
    return (N0/max(Nm,1))**0.5, D, N0, Nm

def main():
    ap = argparse.ArgumentParser(description="Boost composition from poset counts (rapidity additivity).")
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--u", type=float, default=0.4)
    ap.add_argument("--v", type=float, default=0.6)
    ap.add_argument("--tol_eta", type=float, default=0.02)
    args = ap.parse_args()

    T = args.T
    gh_u, D_u, N0u, Nmu = gamma_hat_from_counts(T, args.u)
    gh_v, D_v, N0v, Nmv = gamma_hat_from_counts(T, args.v)

    # Compose analytically, then test via counts at 2T
    w = (args.u + args.v) / (1.0 + args.u*args.v)
    gh_w, D_w, N0w, Nm_w = gamma_hat_from_counts(2*T, w)  # longer interval for the composed boost

    # Rapidities from gamma (order+counts only)
    eta_u = math.acosh(max(1.0, gh_u))
    eta_v = math.acosh(max(1.0, gh_v))
    eta_w = math.acosh(max(1.0, gh_w))
    lhs = eta_w
    rhs = eta_u + eta_v
    abs_err = abs(lhs - rhs)

    out = {
      "T": T,
      "u": args.u, "v": args.v, "w_einstein": w,
      "D_u": D_u, "D_v": D_v, "D_w": D_w,
      "gamma_hat": {"u": gh_u, "v": gh_v, "w": gh_w},
      "rapidity_hat": {"u": eta_u, "v": eta_v, "w": eta_w},
      "abs_err_eta": abs_err,
      "tol_eta": args.tol_eta,
      "PASS_rapidity_additivity": abs_err <= args.tol_eta,
      "notes": "All gammas from Alexandrov counts; no metric assumed. Checks η(w)≈η(u)+η(v)."
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

