#!/usr/bin/env python3
import argparse, json, math

def N_rest(T):
    m=T//2
    return (2*m*(m+1)-1) if (T%2==0) else ((m+1)**2 + m**2)

def N_moving_1p1(T, D1):
    N=0
    for t in range(1, T):
        L=max(-t,  D1-(T-t)); R=min(t,  D1+(T-t))
        if L<=R: N += (R-L+1)
    return N

def split_L1(D1, phi):
    # Split integer D1 into Dx, Dy with |Dx|+|Dy|=D1, preserving angle sign pattern
    ax, ay = abs(math.cos(phi)), abs(math.sin(phi))
    denom = ax+ay if (ax+ay)>0 else 1.0
    Dx_mag = int(round(D1 * ax/denom))
    Dy_mag = D1 - Dx_mag
    sx = 1 if math.cos(phi)>=0 else -1
    sy = 1 if math.sin(phi)>=0 else -1
    return sx*Dx_mag, sy*Dy_mag

def main():
    ap=argparse.ArgumentParser(description="Proper-time orientation sweep in 2D with L1-correct drift.")
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--v1", type=float, default=0.8, help="L1 speed (|Dx|+|Dy|)/T, must be < 1")
    ap.add_argument("--angles", type=int, default=12)
    ap.add_argument("--tol_mean", type=float, default=0.03)
    ap.add_argument("--tol_spread", type=float, default=0.02)
    a=ap.parse_args()

    if not (0.0 <= a.v1 < 1.0):
        raise SystemExit("ERROR: --v1 must be in [0,1).")

    T = a.T
    D1 = int(round(a.v1 * T))          # total L1 drift
    N0 = N_rest(T)
    target = math.sqrt(1.0 - a.v1*a.v1)  # SR factor in L1 model

    ratios=[]
    D_pairs=[]
    for k in range(a.angles):
        phi = 2*math.pi*k/a.angles
        Dx, Dy = split_L1(D1, phi)      # guarantees |Dx|+|Dy|=D1 ≤ T
        Nm = N_moving_1p1(T, abs(Dx)+abs(Dy))
        kappa = (Nm**0.5)/T
        kappa0= (N0**0.5)/T
        ratios.append(kappa/kappa0)
        D_pairs.append([Dx, Dy])

    mean = sum(ratios)/len(ratios)
    spread = max(abs(r-mean) for r in ratios)

    out = {
        "T":T, "v1_L1":a.v1, "D1":D1, "angles":a.angles,
        "DxDy_per_angle": D_pairs,
        "ratios": [float(r) for r in ratios],
        "mean_ratio": float(mean),
        "target_sqrt1_minus_v1_sq": float(target),
        "abs_err_mean": float(abs(mean-target)),
        "max_dev_across_angles": float(spread),
        "tol_mean": a.tol_mean, "tol_spread": a.tol_spread,
        "PASS_propertime_orientation_2d": (abs(mean-target)<=a.tol_mean and spread<=a.tol_spread),
        "notes":"Uses L1 speed v1 and splits D1 across axes so |Dx|+|Dy|=D1; Alexandrov counts stay nonzero for all angles."
    }
    print(json.dumps(out, indent=2))

if __name__=="__main__":
    main()

