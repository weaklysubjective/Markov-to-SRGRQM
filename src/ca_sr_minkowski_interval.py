#!/usr/bin/env python3
import argparse, json, math

def N_rest(T):
    m=T//2
    return (2*m*(m+1)-1) if T%2==0 else ((m+1)**2 + m**2)

def N_moving(T,D):
    N=0
    for t in range(1,T):
        L=max(-t, D-(T-t)); R=min(t, D+(T-t))
        if L<=R: N += (R-L+1)
    return N

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--v", type=float, default=0.8)
    a=ap.parse_args()
    T=a.T; v=a.v
    D=round(v*T)
    N0=N_rest(T)
    Nm=N_moving(T,D)

    # Calibrate alpha so that at rest: s^2 = T^2
    # We want alpha*N0 = T^2  ⇒ alpha = T^2 / N0
    alpha = (T*T)/N0

    # Interval estimator from counts:
    # s_hat^2 = alpha * Nm  ≈  T^2 - D^2  (discrete poset version of Δt^2 - Δx^2)
    s2_hat = alpha * Nm
    minkowski_target = T*T - D*D

    out = {
      "T":T, "v":v, "D":D,
      "N_rest":N0, "N_moving":Nm, "alpha":alpha,
      "s2_hat": s2_hat,
      "s2_target": minkowski_target,
      "abs_err": abs(s2_hat - minkowski_target),
      "rel_err": abs(s2_hat - minkowski_target)/max(1.0, abs(minkowski_target)),
      "notes": "s^2 from order+counts; no metric assumed; one calibration constant from rest case."
    }
    print(json.dumps(out, indent=2))
if __name__=="__main__": main()

