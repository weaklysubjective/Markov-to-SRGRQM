#!/usr/bin/env python3
import argparse, json

def N_between(dt, dx):
    """Alexandrov count for 1D lattice between two events with separations (dt>=0, integer), (dx integer)."""
    T = abs(int(dt)); D = int(abs(dx))
    if T <= 0: return 0
    N=0
    for t in range(1, T):
        L = max(-t,  D-(T-t))
        R = min( t,  D+(T-t))
        if L<=R: N += (R-L+1)
    return N

def main():
    ap = argparse.ArgumentParser(description="Relativity of simultaneity via poset counts.")
    ap.add_argument("--Tau", type=int, default=200, help="half-span of moving-frame anchors")
    ap.add_argument("--v", type=float, default=0.6, help="moving frame speed")
    ap.add_argument("--L", type=int, default=200, help="rest-simultaneous spatial separation")
    ap.add_argument("--tol_zero", type=int, default=0)  # for exact integer sign checks
    args = ap.parse_args()

    Tau = args.Tau
    D = int(round(args.v * Tau))  # drift for anchors
    # Anchors on moving worldline
    p_minus = (-Tau, -D)
    p_plus  = (+Tau, +D)
    # Rest-simultaneous events at t=0
    EL = (0, -args.L//2)
    ER = (0, +args.L//2)

    # Helper: count from A->B given coords (t,x)
    def N_AB(A,B):
        t1,x1 = A; t2,x2 = B
        dt = t2 - t1; dx = x2 - x1
        if dt < 0:
            # swap to ensure dt>=0 (counts symmetric under reversal)
            return N_between(-dt, -dx)
        return N_between(dt, dx)

    # Poset-native "moving time" imbalance:
    def Delta(E):
        return N_AB(p_minus, E) - N_AB(E, p_plus)

    dL = Delta(EL)
    dR = Delta(ER)

    # Signs (allow tol_zero=0 for strictness)
    sign = lambda z: (1 if z>args.tol_zero else (-1 if z<-args.tol_zero else 0))

    out = {
      "Tau": Tau, "v": args.v, "D": D, "L": args.L,
      "anchors": {"p_minus": p_minus, "p_plus": p_plus},
      "events": {"E_L": EL, "E_R": ER},
      "Delta": {"E_L": dL, "E_R": dR},
      "signs": {"E_L": sign(dL), "E_R": sign(dR)},
      "PASS_simultaneity_flip": (sign(dL) * sign(dR) == -1),
      "notes": "Δ(E)=N(p-,E)-N(E,p+). Opposite signs ⇒ events that were simultaneous at rest are not simultaneous in moving frame."
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

