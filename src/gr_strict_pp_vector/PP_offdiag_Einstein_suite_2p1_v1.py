#!/usr/bin/env python3
import argparse, json

def load(path: str):
    with open(path, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser(description="STRICT PP off-diagonal 2+1 Einstein-block suite v1.")
    ap.add_argument("--blocks", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    blocks_out = []
    all_pass = True

    for p in args.blocks:
        P = load(p)
        ok = bool(P.get("PASS_offdiag_2p1_toy_diag_v2", False))
        blocks_out.append({
            "block": p,
            "case": P.get("case"),
            "APPLICABLE": bool(P.get("APPLICABLE", False)),
            "PASS_offdiag_2p1_toy_diag_v2": ok,
        })
        if not ok:
            all_pass = False

    out = {
        "n_blocks": len(args.blocks),
        "blocks": blocks_out,
        "ALL_PASS_offdiag_2p1_toy_diag_v2": all_pass,
        "notes": (
            "STRICT PP off-diagonal 2+1 Einstein-block suite v1. "
            "Aggregates per-case block artifacts; no new physics introduced."
        )
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_offdiag_2p1_toy_diag_v2 =", all_pass)

if __name__ == "__main__":
    main()

