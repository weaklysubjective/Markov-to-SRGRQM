#!/usr/bin/env python3
import argparse, json, math

def N_rest(T):
    m=T//2
    return (2*m*(m+1)-1) if T%2==0 else ((m+1)**2 + m**2)

def N_moving(T, D):
    N=0
    for t in range(1, T):
        L=max(-t, D-(T-t)); R=min(t, D+(T-t))
        if L<=R: N += (R-L+1)
    return N

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--vlist", type=str, default="0.0,0.2,0.4,0.6,0.8")
    a=ap.parse_args()
    T=a.T; Nv=[]
    N0=N_rest(T)
    for v in map(float, a.vlist.split(",")):
        D=round(v*T)
        Nm=N_moving(T,D)
        gamma_hat=(N0/max(Nm,1))**0.5
        Nv.append({"v":v,"gamma_hat":gamma_hat,"gamma_target":1.0/math.sqrt(max(1e-12,1-v*v)),
                   "abs_err":abs(gamma_hat-(1.0/math.sqrt(max(1e-12,1-v*v))))})
    out={"T":T,"results":Nv,"max_abs_err":max(x["abs_err"] for x in Nv)}
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()

