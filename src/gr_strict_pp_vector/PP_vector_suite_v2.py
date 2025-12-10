#!/usr/bin/env python3
import argparse
import json


def load(path):
    with open(path, "r") as f:
        return json.load(f)


def pick_overall_key(P):
    # Prefer v2 if present, otherwise fall back to v1
    if "overall_PASS_vector_strict_PP_v2" in P:
        return "overall_PASS_vector_strict_PP_v2"
    if "overall_PASS_vector_strict_PP_v1" in P:
        return "overall_PASS_vector_strict_PP_v1"
    return ""


def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP vector suite v2. Aggregates vector packs."
    )
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    packs_out = []
    all_pass = True

    for p in args.packs:
        P = load(p)
        key = pick_overall_key(P)

        if key:
            ok = bool(P.get(key))
        else:
            # If a pack does not expose an overall key, treat as FAIL.
            ok = False

        entry = {
            "pack": p,
            "case": P.get("case"),
        }

        if key:
            entry["overall_key_used"] = key
            entry[key] = ok
        else:
            entry["overall_key_used"] = None
            entry["missing_overall_key"] = True

        packs_out.append(entry)

        if not ok:
            all_pass = False

    out = {
        "n_packs": len(args.packs),
        "packs": packs_out,
        "ALL_PASS_vector_strict_PP_v2": all_pass,
        "notes": (
            "STRICT PP vector suite v2. Aggregates vector packs. "
            "Accepts pack-level overall keys v1 or v2 "
            "(v2 may be 512-aware). No new physics is introduced here."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_vector_strict_PP_v2 =", all_pass)


if __name__ == "__main__":
    main()

