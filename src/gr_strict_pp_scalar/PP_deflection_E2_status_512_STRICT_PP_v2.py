#!/usr/bin/env python3
import argparse, json, os, sys
from typing import Any, Dict

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def pick_pass_block(J: Dict[str, Any]) -> Dict[str, Any]:
    # v7 format: top-level has "deflection_stats"
    if "deflection_stats" in J and isinstance(J["deflection_stats"], dict):
        return J["deflection_stats"]
    # fallback
    return J

def get_bool(d: Dict[str, Any], key: str):
    v = d.get(key)
    if isinstance(v, bool):
        return v
    return None

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP deflection E2 status (512x512). Pure aggregation; no new physics."
    )
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)

    ap.add_argument("--ms080_json", default="src/gr_strict_pp_scalar/PP_deflection_markov_front_512_ms080_v7_PPV1_E2_late.json")
    ap.add_argument("--pf010_json", default="src/gr_strict_pp_scalar/PP_deflection_markov_front_512_strong_pf010_v7_PPV1_E2_late.json")

    ap.add_argument("--output", default="src/gr_strict_pp_scalar/PP_deflection_E2_status_512_STRICT_PP_v1.json")
    args = ap.parse_args()

    ms = load_json(args.ms080_json)
    pf = load_json(args.pf010_json)

    msS = pick_pass_block(ms)
    pfS = pick_pass_block(pf)

    # Pull canonical fields (best-effort)
    def summarize(case: str, J: Dict[str, Any], S: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "case": case,
            "json": None,
            "APPLICABLE": S.get("APPLICABLE"),
            "PASS": None,
            "D_flat": S.get("D_flat"),
            "D_curved": S.get("D_curved"),
            "reduction_fraction": S.get("reduction_fraction"),
            "reason": S.get("reason"),
            "notes": J.get("notes"),
            "edges_flat": J.get("edges_flat"),
            "edges_curved": J.get("edges_curved"),
            "mass_mask": J.get("mass_mask"),
            "orbit_mask": J.get("orbit_mask"),
        }
        out["json"] = args.ms080_json if case == "ms080" else args.pf010_json

        # v7 pass key name
        # (some runs might use PASS_deflection_markov_front_PP_v7; tolerate variants)
        p = (
            S.get("PASS_deflection_markov_front_PP_v7")
            if "PASS_deflection_markov_front_PP_v7" in S
            else S.get("PASS")
        )
        out["PASS"] = bool(p) if isinstance(p, bool) else False
        return out

    ms_out = summarize("ms080", ms, msS)
    pf_out = summarize("strong_pf010", pf, pfS)

    # Strict gating: both must be applicable + pass
    overall = bool(ms_out["APPLICABLE"] and ms_out["PASS"] and pf_out["APPLICABLE"] and pf_out["PASS"])

    out = {
        "H": int(args.H),
        "W": int(args.W),
        "notes": (
            "STRICT PP deflection E2 status @512x512. Aggregates v7 Markov-front deflection reports "
            "(E2 edge policy). No PDE/Poisson/GR ansatz/regression; no parameter fitting; "
            "this script only summarizes existing JSONs."
        ),
        "cases": {
            "ms080": ms_out,
            "strong_pf010": pf_out,
        },
        "overall_PASS_deflection_E2_strict_PP_v1": overall,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("overall_PASS_deflection_E2_strict_PP_v1 =", overall)

if __name__ == "__main__":
    main()

