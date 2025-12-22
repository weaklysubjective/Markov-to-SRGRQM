#!/usr/bin/env python3
"""
PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.py

Purpose
-------
A small "freeze binder" that produces ONE canonical JSON stating whether the
FULL 512 STRICT_PP "REAL EFE" claim is PASS, by reading an existing
PP_EFE_real_EFE_status_512_STRICT_PP_v*.json produced by the suite.

This binder is intentionally non-invasive:
- No recomputation.
- No edits to existing pipelines.
- Just a deterministic summary + optional file hashing.

Usage
-----
python src/gr_strict_pp_scalar/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.py \
  --real_efe_json src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v1.json \
  --output src/gr_strict_pp_scalar/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.json

Optional:
  --hash   include sha256 of the input json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict, Tuple


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def infer_final_pass(status: Dict[str, Any]) -> Tuple[bool, Dict[str, bool], str]:
    """
    Infer FINAL_PASS from the existing real-EFE status JSON.

    Priority:
      1) If PASS dict contains a single overall all-pass key like "ALL_PASS_*", use it.
      2) Else if PASS dict exists: final = all boolean values inside PASS dict.
      3) Else: scan top-level for ALL_PASS_* booleans and use all of them.
      4) Else: final=False with reason.

    Returns:
      (final_pass, pass_flags_used, reasoning)
    """
    pass_flags_used: Dict[str, bool] = {}

    # 1) Prefer PASS dict if present
    if isinstance(status.get("PASS"), dict):
        P = status["PASS"]
        # collect boolean entries
        bool_items = {k: bool(v) for k, v in P.items() if isinstance(v, bool)}
        if bool_items:
            # if there is a single "overall" key, use that
            overall_keys = [k for k in bool_items.keys() if k.startswith("ALL_PASS")]
            if len(overall_keys) == 1:
                k = overall_keys[0]
                pass_flags_used[k] = bool_items[k]
                return bool_items[k], pass_flags_used, f"Used PASS['{k}'] as the canonical overall gate."
            # otherwise use all PASS booleans
            pass_flags_used.update(bool_items)
            final_pass = all(bool_items.values())
            return final_pass, pass_flags_used, "Used conjunction of all boolean entries in PASS dict."

    # 3) fall back to top-level ALL_PASS* booleans
    top_all = {k: bool(v) for k, v in status.items() if k.startswith("ALL_PASS") and isinstance(v, bool)}
    if top_all:
        pass_flags_used.update(top_all)
        final_pass = all(top_all.values())
        return final_pass, pass_flags_used, "Used conjunction of top-level ALL_PASS* booleans."

    return False, pass_flags_used, "No PASS dict booleans and no top-level ALL_PASS* booleans found; cannot infer FINAL_PASS."


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze binder: REAL EFE final PASS summary (512 STRICT_PP).")
    ap.add_argument(
        "--real_efe_json",
        type=str,
        default="src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v1.json",
        help="Input status JSON from the REAL-EFE suite/binder.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="src/gr_strict_pp_scalar/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.json",
        help="Output canonical final-status JSON.",
    )
    ap.add_argument("--hash", action="store_true", help="Include sha256 of the input status JSON.")
    args = ap.parse_args()

    t0 = time.time()

    if not os.path.exists(args.real_efe_json):
        raise FileNotFoundError(f"Missing --real_efe_json: {args.real_efe_json}")

    status = load_json(args.real_efe_json)
    final_pass, pass_flags_used, reasoning = infer_final_pass(status)

    out: Dict[str, Any] = {
        "script": os.path.basename(__file__),
        "STRICT_PP": True,
        "N_expected": 512 * 512,
        "input_real_efe_status_json": args.real_efe_json,
        "FINAL_PASS_real_EFE_512_STRICT_PP_v2": bool(final_pass),
        "pass_flags_used": pass_flags_used,
        "reasoning": reasoning,
        "elapsed_s": float(time.time() - t0),
    }

    if args.hash:
        out["input_real_efe_status_sha256"] = sha256_file(args.real_efe_json)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"WROTE {args.output}")
    print(f"FINAL_PASS_real_EFE_512_STRICT_PP_v2 = {out['FINAL_PASS_real_EFE_512_STRICT_PP_v2']}")


if __name__ == "__main__":
    main()

