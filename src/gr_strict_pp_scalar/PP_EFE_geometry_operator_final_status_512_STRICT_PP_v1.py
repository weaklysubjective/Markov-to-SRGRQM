#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from typing import Dict, Any, List, Tuple

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def must_exist(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.isdir(path):
        raise IsADirectoryError(path)
    if os.path.getsize(path) <= 0:
        raise ValueError(f"Empty file: {path}")

def file_meta(path: str, do_hash: bool) -> Dict[str, Any]:
    st = os.stat(path)
    d = {
        "path": path,
        "bytes": int(st.st_size),
        "mtime_epoch": float(st.st_mtime),
    }
    if do_hash:
        d["sha256"] = sha256_file(path)
    return d

def get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def main():
    ap = argparse.ArgumentParser(
        description="Canonical STRICT-PP final status binder for Doyle/Steiner length→curvature operator + adversarial v6."
    )
    ap.add_argument("--adversarial_json", default="src/gr_strict_pp_scalar/PP_EFE_adversarial_status_512_STRICT_PP_v6_diag.json")
    ap.add_argument("--R_A_npz", default="src/gr_strict_pp_geom/PP_RicciVolDist_R_512_strong_pf010_STRICT_PP_v1.npz")
    ap.add_argument("--R_B_npz", default="src/gr_strict_pp_geom/PP_RicciVolDist_R_512_ms080_STRICT_PP_v1.npz")
    ap.add_argument("--lengths_A_npz", default="src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_strong_pf010_STRICT_PP_v1.npz")
    ap.add_argument("--lengths_B_npz", default="src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_ms080_STRICT_PP_v1.npz")
    ap.add_argument("--T_A_npz", default="src/gr_strict_pp_scalar/PP_Tmunu_tracew_edges_512_strong_pf010_STRICT_PP_v3.npz")
    ap.add_argument("--T_B_npz", default="src/gr_strict_pp_scalar/PP_Tmunu_tracew_edges_512_ms080_STRICT_PP_v3.npz")
    ap.add_argument("--flow_A_matched_npz",
                    default="src/gr_strict_pp_align/runs/finalleg_twgeom_q9995_v1/PP_T00_flow_fixedGeom_lengths_512_strong_pf010_STRICT_PP_v1.npz")
    ap.add_argument("--flow_B_matched_npz",
                    default="src/gr_strict_pp_align/runs/finalleg_twgeom_q9995_v1/PP_T00_flow_fixedGeom_lengths_512_ms080_STRICT_PP_v1.npz")

    ap.add_argument("--require_flows", action="store_true",
                    help="If set, require the matched flow NPZs exist (recommended once stable).")
    ap.add_argument("--hash", action="store_true", help="Compute sha256 for all canonical artifacts.")
    ap.add_argument("--output", required=True, help="Output JSON path.")
    ap.add_argument("--strict_pp_tag", default="STRICT_PP", help="Tag name recorded in JSON.")
    args = ap.parse_args()

    t0 = time.time()

    # ---- Required artifacts (canonical geometry+matter inputs) ----
    required = [
        ("adversarial_json", args.adversarial_json),
        ("R_A_npz", args.R_A_npz),
        ("R_B_npz", args.R_B_npz),
        ("lengths_A_npz", args.lengths_A_npz),
        ("lengths_B_npz", args.lengths_B_npz),
        ("T_A_npz", args.T_A_npz),
        ("T_B_npz", args.T_B_npz),
    ]
    for _, p in required:
        must_exist(p)

    # Optional flows (used by v6 adversarial; in your run they exist under RUN/)
    flow_items = [
        ("flow_A_matched_npz", args.flow_A_matched_npz),
        ("flow_B_matched_npz", args.flow_B_matched_npz),
    ]
    flows_present = True
    for _, p in flow_items:
        if not os.path.exists(p):
            flows_present = False
            break
    if args.require_flows:
        for _, p in flow_items:
            must_exist(p)

    # ---- Load adversarial diag ----
    with open(args.adversarial_json, "r") as f:
        adv = json.load(f)

    # Primary PASS flag in the diag JSON (what we freeze)
    all_pass = bool(adv.get("ALL_PASS_adversarial_512_STRICT_PP_v6", False))

    # Pull out key numbers (these are the “load-bearing” adversarial stats)
    cross_A_abs = get(adv, ["crossFlow", "gated_abs_corr_A_same_mask"], None)
    cross_B_abs = get(adv, ["crossFlow", "gated_abs_corr_B_same_mask"], None)

    ratioA = get(adv, ["qPermTopK", "A", "ratio_worst_over_matched"], None)
    ratioB = get(adv, ["qPermTopK", "B", "ratio_worst_over_matched"], None)

    worstA = get(adv, ["qPermTopK", "A", "worst_abs_corr"], None)
    worstB = get(adv, ["qPermTopK", "B", "worst_abs_corr"], None)

    matchedA = get(adv, ["qPermTopK", "A", "matched_abs_corr"], None)
    matchedB = get(adv, ["qPermTopK", "B", "matched_abs_corr"], None)

    # Note: in your summary, adv["mask"] and ["alt_mask"] are null. We keep them, but don't depend on them.
    mask = adv.get("mask", None)
    alt_mask = adv.get("alt_mask", None)

    # ---- Build report ----
    artifacts = {}
    for name, path in required:
        artifacts[name] = file_meta(path, args.hash)
    # Flows: record if present + optionally hash
    for name, path in flow_items:
        if os.path.exists(path):
            artifacts[name] = file_meta(path, args.hash)
        else:
            artifacts[name] = {"path": path, "present": False}

    rep: Dict[str, Any] = {
        args.strict_pp_tag: True,
        "script": os.path.basename(__file__),
        "timestamp_epoch": float(time.time()),
        "FINAL_PASS_EFE_geometry_operator_512_STRICT_PP_v1": bool(all_pass),
        "inputs_present": True,  # if required() would have thrown otherwise
        "flows_present": bool(flows_present),
        "adversarial_v6": {
            "adversarial_json": args.adversarial_json,
            "ALL_PASS_adversarial_512_STRICT_PP_v6": bool(all_pass),
            "mask": mask,
            "alt_mask": alt_mask,
            "crossFlow": {
                "gated_abs_corr_A_same_mask": cross_A_abs,
                "gated_abs_corr_B_same_mask": cross_B_abs,
            },
            "qPermTopK": {
                "A": {
                    "matched_abs_corr": matchedA,
                    "worst_abs_corr": worstA,
                    "ratio_worst_over_matched": ratioA,
                },
                "B": {
                    "matched_abs_corr": matchedB,
                    "worst_abs_corr": worstB,
                    "ratio_worst_over_matched": ratioB,
                },
            },
        },
        "canonical_artifacts": artifacts,
        "elapsed_s": float(time.time() - t0),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(rep, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("FINAL_PASS_EFE_geometry_operator_512_STRICT_PP_v1 =", rep["FINAL_PASS_EFE_geometry_operator_512_STRICT_PP_v1"])

if __name__ == "__main__":
    main()

