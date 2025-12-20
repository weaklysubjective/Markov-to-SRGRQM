#!/usr/bin/env python3
import argparse, json, os

def loadj(p):
    with open(p,"r") as f: return json.load(f)

def get_bool_any(j, keys):
    # keys can be "a.b.c"
    for k in keys:
        cur = j
        ok = True
        for part in k.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and isinstance(cur, bool):
            return cur
    return None

def passish(j):
    # Heuristic: prefer explicit ALL_PASS keys, else any PASS boolean
    # 1) explicit
    for path in [
        "PASS.ALL_PASS_real_EFE_strict_PP_512_v1",
        "PASS.ALL_PASS_real_EFE_strict_PP_512_v2",
        "PASS.ALL_PASS",
        "PASS.ALL_PASS_status",
    ]:
        b = get_bool_any(j,[path])
        if isinstance(b,bool): return b
    # 2) scan PASS dict
    p = j.get("PASS", {})
    if isinstance(p, dict):
        for k,v in p.items():
            if isinstance(v,bool) and ("ALL_PASS" in k or k.startswith("PASS_")):
                # don't return a partial PASS_*, but accept ALL_PASS*
                if "ALL_PASS" in k:
                    return bool(v)
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scalar_a2b", default="src/gr_strict_pp_scalar/PP_EFE_scalar_A2b_status_512_STRICT_PP_v1.json")
    ap.add_argument("--offdiag", default="src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v3.json")
    ap.add_argument("--bianchi", default="src/gr_strict_pp_scalar/PP_EFE_bianchi_status_512_STRICT_PP_v1.json")
    ap.add_argument("--closure", default="src/gr_strict_pp_scalar/PP_EFE_tensor_closure_status_512_STRICT_PP_v1.json")
    ap.add_argument("--tensor3p1", default="src/gr_strict_pp_scalar/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.json")
    ap.add_argument("--output", default="src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v2.json")
    ap.add_argument("--require_invariance", type=int, default=0,
                    help="If 1, require tensor3p1 PASS_invariance==true (otherwise treated as 'reported-only').")
    args=ap.parse_args()

    sa = loadj(args.scalar_a2b)
    od = loadj(args.offdiag)
    bi = loadj(args.bianchi)
    cl = loadj(args.closure)
    t3 = loadj(args.tensor3p1)

    pass_scalar = bool(sa.get("PASS",{}).get("ALL_PASS_scalar_A2b_512_STRICT_PP_v1", False))
    pass_offdiag = bool(od.get("offdiag_flags",{}).get("PASS", False)) and bool(od.get("offdiag_flags",{}).get("APPLICABLE", False))
    pass_bianchi = bool(bi.get("PASS",{}).get("ALL_PASS_bianchi_512_STRICT_PP_v1", False))

    # closure pass: accept whatever explicit ALL_PASS exists under PASS
    pass_closure = get_bool_any(cl, [
        "PASS.overall_PASS_tensor_closure_evidence_512_strict_PP_v1",
        "PASS.ALL_PASS_tensor_closure_512_STRICT_PP_v1",
        "PASS.ALL_PASS_tensor_closure_strict_PP_512_v1",
        "PASS.ALL_PASS_tensor_closure",
        "PASS.ALL_PASS",
    ])
    if pass_closure is None:
        # fallback: any ALL_PASS boolean inside PASS
        pass_closure = False
        p = cl.get("PASS",{})
        if isinstance(p, dict):
            for k,v in p.items():
                if isinstance(v,bool) and "ALL_PASS" in k:
                    pass_closure = bool(v)
                    break

    pass_tensor3p1 = bool(t3.get("PASS",{}).get("ALL_PASS_EFE_tensor_3p1_strict_PP_512_v1", False))

    # invariance is reported in tensor3p1 status as PASS_invariance (can be null)
    inv_val = t3.get("PASS",{}).get("PASS_invariance", None)
    if isinstance(inv_val, bool):
        pass_invariance = inv_val
    else:
        pass_invariance = None  # unknown / not asserted

    if int(args.require_invariance) == 1:
        inv_gate = (pass_invariance is True)
    else:
        inv_gate = True  # reported-only by default

    ALL = bool(pass_scalar and pass_offdiag and pass_bianchi and pass_closure and pass_tensor3p1 and inv_gate)

    out = {
      "STRICT_PP": True,
      "H": 512, "W": 512,
      "inputs": {
        "scalar_A2b": args.scalar_a2b,
        "offdiag_2p1": args.offdiag,
        "bianchi": args.bianchi,
        "tensor_closure": args.closure,
        "tensor_3p1": args.tensor3p1,
        "require_invariance": bool(int(args.require_invariance)==1),
      },
      "PASS_components": {
        "PASS_scalar_A2b": pass_scalar,
        "PASS_offdiag_2p1": pass_offdiag,
        "PASS_bianchi": pass_bianchi,
        "PASS_tensor_closure": bool(pass_closure),
        "PASS_tensor_3p1": pass_tensor3p1,
        "PASS_invariance": pass_invariance,
      },
      "PASS": {
        "ALL_PASS_real_EFE_strict_PP_512_v2": ALL
      }
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output,"w") as f: json.dump(out,f,indent=2,sort_keys=True)
    print("WROTE", args.output)
    print("ALL_PASS_real_EFE_strict_PP_512_v2 =", out["PASS"]["ALL_PASS_real_EFE_strict_PP_512_v2"])

if __name__=="__main__":
    main()
