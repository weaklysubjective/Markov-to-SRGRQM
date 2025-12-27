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
  --real_efe_json runs/status/PP_EFE_real_EFE_status_512_STRICT_PP_v2.json \
  --output runs/status/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.json

Optional:
  --hash   include sha256 of the input json

Notes
-----
This script writes the final PASS flag BOTH:
- as a top-level boolean key: FINAL_PASS_real_EFE_512_STRICT_PP_v2
- and mirrored inside PASS dict: PASS[FINAL_PASS_real_EFE_512_STRICT_PP_v2]

This makes downstream tooling robust regardless of whether it expects root keys
or PASS dict keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict, Tuple


FINAL_KEY = "FINAL_PASS_real_EFE_512_STRICT_PP_v2"


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
      1) If PASS dict contains exactly one ALL_PASS* boolean key, use it.
      2) Else if PASS dict has booleans: final = conjunction of all PASS booleans.
      3) Else: scan top-level for ALL_PASS* booleans and use conjunction.
      4) Else: final=False (cannot infer).

    Returns:
      (final_pass, pass_flags_used, reasoning)
    """
    pass_flags_used: Dict[str, bool] = {}

    P = status.get("PASS")
    if isinstance(P, dict):
        bool_items = {k: bool(v) for k, v in P.items() if isinstance(v, bool)}
        if bool_items:
            overall_keys = [k for k in bool_items.keys() if k.startswith("ALL_PASS")]
            if len(overall_keys) == 1:
                k = overall_keys[0]
                pass_flags_used[k] = bool_items[k]
                return bool_items[k], pass_flags_used, f"Used PASS['{k}'] as the canonical overall gate."
            pass_flags_used.update(bool_items)
            final_pass = all(bool_items.values())
            return final_pass, pass_flags_used, "Used conjunction of all boolean entries in PASS dict."

    top_all = {k: bool(v) for k, v in status.items() if k.startswith("ALL_PASS") and isinstance(v, bool)}
    if top_all:
        pass_flags_used.update(top_all)
        final_pass = all(top_all.values())
        return final_pass, pass_flags_used, "Used conjunction of top-level ALL_PASS* booleans."

    return False, pass_flags_used, "No PASS dict booleans and no top-level ALL_PASS* booleans found; cannot infer FINAL_PASS."


def main() -> int:
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

    real_efe_json_path = os.path.abspath(args.real_efe_json)
    if not os.path.exists(real_efe_json_path):
        raise FileNotFoundError(f"Missing --real_efe_json: {real_efe_json_path}")

    status = load_json(real_efe_json_path)
    final_pass, pass_flags_used, reasoning = infer_final_pass(status)

    final_val = bool(final_pass)
    elapsed_s = float(time.time() - t0)
    N_expected = 512 * 512

    out: Dict[str, Any] = {
        "script": os.path.basename(__file__),
        "STRICT_PP": True,
        "N_expected": N_expected,
        "input_real_efe_status_json": real_efe_json_path,
        FINAL_KEY: final_val,
        "PASS": {FINAL_KEY: final_val},
        "pass_flags_used": pass_flags_used,
        "reasoning": reasoning,
        "elapsed_s": elapsed_s,
    }

    if args.hash:
        out["input_real_efe_status_sha256"] = sha256_file(real_efe_json_path)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"WROTE {args.output}")
    print(f"{FINAL_KEY} = {out[FINAL_KEY]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

