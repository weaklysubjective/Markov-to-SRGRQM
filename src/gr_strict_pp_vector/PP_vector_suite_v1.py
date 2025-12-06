#!/usr/bin/env python3
import argparse, json

def load(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser(description="STRICT PP vector suite v1. Aggregates vector packs.")
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    packs_out = []
    all_pass = True

    for p in args.packs:
        P = load(p)
        ok = bool(P.get("overall_PASS_vector_strict_PP_v1"))
        packs_out.append({
            "pack": p,
            "overall_PASS_vector_strict_PP_v1": ok,
            "case": P.get("case")
        })
        if not ok:
            all_pass = False

    out = {
        "n_packs": len(args.packs),
        "packs": packs_out,
        "ALL_PASS_vector_strict_PP_v1": all_pass,
        "notes": (
            "STRICT PP vector suite v1. Aggregates vector packs built from "
            "azimuthal flux and frame-dragging proxies with admissibility gates. "
            "No new physics is introduced here."
        )
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_vector_strict_PP_v1 =", all_pass)

if __name__ == "__main__":
    main()
