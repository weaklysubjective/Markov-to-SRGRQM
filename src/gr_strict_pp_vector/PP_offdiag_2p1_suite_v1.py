#!/usr/bin/env python3
import argparse, json

def load(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser(description="STRICT PP off-diagonal 2+1 suite v1 (aggregates per-case toy blocks).")
    ap.add_argument("--blocks", nargs="+", required=True, help="PP_offdiag_Einstein_block_2p1_*_v1.json files")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    blocks_out = []
    all_pass = True

    for p in args.blocks:
        P = load(p)
        ok = bool(P.get("result", {}).get("PASS_offdiag_2p1_toy_diag"))
        applicable = bool(P.get("result", {}).get("APPLICABLE"))
        case = P.get("case")

        blocks_out.append({
            "case": case,
            "block": p,
            "APPLICABLE": applicable,
            "PASS_offdiag_2p1_toy_diag": ok,
            "corr_rt": (P.get("M_toy", {}) or {}).get("corr_rt"),
            "lam2": ((P.get("M_toy", {}) or {}).get("eig", {}) or {}).get("lam2"),
            "v_r_from_ring_grad": (P.get("v_toy", {}) or {}).get("v_r_from_ring_grad"),
            "v_t_from_azimuthal_bias": (P.get("v_toy", {}) or {}).get("v_t_from_azimuthal_bias"),
        })

        if not (applicable and ok):
            all_pass = False

    out = {
        "n_blocks": len(args.blocks),
        "blocks": blocks_out,
        "ALL_PASS_offdiag_2p1_toy_diag_v1": all_pass,
        "notes": (
            "STRICT PP off-diagonal 2+1 suite v1. Aggregates per-case ring-local toy blocks "
            "produced by PP_offdiag_Einstein_block_2p1_v1.py. No new physics is introduced here."
        )
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_offdiag_2p1_toy_diag_v1 =", all_pass)

if __name__ == "__main__":
    main()

