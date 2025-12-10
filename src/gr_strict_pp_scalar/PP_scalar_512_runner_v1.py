#!/usr/bin/env python3
"""
PP_scalar_512_runner_v1.py  (REWRITE)

STRICT PP Scalar 512 Runner

Runs (per case):
  1) PP_markov_tau_geometry_v4_multiShellIntersect.py
  2) PP_deflection_markov_front_PP_v3_sparse_only.py
  3) PP_Shapiro_markov_tau_v2_sparse.py

Design goals:
  - 512-scale safe (no dense NxN).
  - STRICT PP: runners only orchestrate existing strict PP scripts.
  - Tri-state PASS with explicit reasons.
  - Shapiro is pair-sensitive at 512; by default it is NOT required
    for ALL_PASS unless --require_shapiro is set.
  - Manual Shapiro pair override supported.

Conventions (override via flags if needed):
  edges_flat:     edges_ca_v3_flat_512x512_PPV1.txt
  edges_curved:   edges_ca_v3_{case}_512x512_PPV1.txt
  trace_weights:  trace_weights_ca_v3_{case}_512x512.txt
  mass_mask:      PP_mass_mask_512x512_{case}_PPV1.npy
  ring_mask:      PP_orbit_ring_mask_512x512_{case}_PPV1.npy

Cases default:
  mass_ms080,strong_pf010
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple, List

import numpy as np


# -----------------------------
# Tri-state logic helpers
# -----------------------------

def tri_and(values: List[Optional[bool]]) -> Optional[bool]:
    """AND over {True, False, None} with None = INDETERMINATE."""
    any_none = False
    for v in values:
        if v is False:
            return False
        if v is None:
            any_none = True
    return None if any_none else True


# -----------------------------
# File helpers
# -----------------------------

def need(path: str, label: str, errors: List[str]) -> bool:
    if path and os.path.exists(path):
        return True
    errors.append(f"missing_{label}:{path}")
    return False


def default_paths(case: str, H: int, W: int) -> Dict[str, str]:
    hw = f"{H}x{W}"
    return {
        "edges_flat":     f"edges_ca_v3_flat_{hw}_PPV1.txt",
        "edges_curved":   f"edges_ca_v3_{case}_{hw}_PPV1.txt",
        "trace_weights":  f"trace_weights_ca_v3_{case}_{hw}.txt",
        "mass_mask":      f"PP_mass_mask_{hw}_{case}_PPV1.npy",
        "ring_mask":      f"PP_orbit_ring_mask_{hw}_{case}_PPV1.npy",
    }


# -----------------------------
# Shapiro pair policy
# -----------------------------

def shapiro_auto_pairs_from_ring(ring_mask_path: str) -> Optional[Tuple[int, int, int, int, int]]:
    """
    STRICT PP auto pair selection using ONLY ring membership.

    Policy (ring_local_first_and_mid_arcs):
      - "around": first ring node and 11th ring node (local arc)
      - "through": mid ring node and mid+10 ring node (different segment)

    Returns:
      (src_through, dst_through, src_around, dst_around, ring_size)
    or None if ring too small.
    """
    ring = np.load(ring_mask_path).reshape(-1)
    idxs = np.where(ring > 0)[0].astype(int)
    if idxs.size < 12:
        return None
    idxs = np.sort(idxs)
    ring_size = int(idxs.size)

    src_around = int(idxs[0])
    dst_around = int(idxs[min(10, ring_size - 1)])

    mid = ring_size // 2
    src_through = int(idxs[mid])
    dst_through = int(idxs[min(mid + 10, ring_size - 1)])

    return src_through, dst_through, src_around, dst_around, ring_size


def pick_shapiro_pairs(
    args: argparse.Namespace,
    ring_mask_path: str,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Dict[str, Any]]:
    """
    Decide Shapiro pairs with priority:
      1) --skip_shapiro  -> skipped
      2) manual override -> if any of the 4 is provided, all 4 must be provided
      3) auto ring-local policy

    Returns:
      (st, dt, sa, da, meta)
    where meta includes:
      skipped, skip_reason, policy, ring_size, and chosen indices.
    """
    meta: Dict[str, Any] = {"skipped": False}

    if args.skip_shapiro:
        meta["skipped"] = True
        meta["policy"] = "skipped_by_flag"
        return None, None, None, None, meta

    manual = (
        args.shapiro_src_through is not None or
        args.shapiro_dst_through is not None or
        args.shapiro_src_around  is not None or
        args.shapiro_dst_around  is not None
    )

    if manual:
        st = args.shapiro_src_through
        dt = args.shapiro_dst_through
        sa = args.shapiro_src_around
        da = args.shapiro_dst_around

        if None in (st, dt, sa, da):
            raise ValueError(
                "If any manual Shapiro pair is provided, all four must be set:\n"
                "  --shapiro_src_through --shapiro_dst_through\n"
                "  --shapiro_src_around  --shapiro_dst_around"
            )

        # ring_size is informational only
        try:
            ring = np.load(ring_mask_path).reshape(-1)
            ring_size = int(np.count_nonzero(ring > 0))
        except Exception:
            ring_size = None

        meta.update({
            "policy": "manual_pairs",
            "ring_size": ring_size,
            "src_through": int(st),
            "dst_through": int(dt),
            "src_around": int(sa),
            "dst_around": int(da),
        })
        return int(st), int(dt), int(sa), int(da), meta

    auto = shapiro_auto_pairs_from_ring(ring_mask_path)
    if auto is None:
        meta["skipped"] = True
        meta["skip_reason"] = "ring_too_small_for_auto_pairs"
        meta["policy"] = "auto_ring_local_first_and_mid_arcs"
        return None, None, None, None, meta

    st, dt, sa, da, ring_size = auto
    meta.update({
        "policy": "ring_local_first_and_mid_arcs",
        "ring_size": ring_size,
        "src_through": st,
        "dst_through": dt,
        "src_around": sa,
        "dst_around": da,
    })
    return st, dt, sa, da, meta


# -----------------------------
# Subprocess runner
# -----------------------------

def run_cmd(cmd: List[str]) -> Tuple[int, float, str]:
    t0 = time.time()
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, time.time() - t0, out
    except subprocess.CalledProcessError as e:
        return e.returncode, time.time() - t0, e.output


def py() -> str:
    return sys.executable


def tail_lines(txt: str, n: int = 20) -> List[str]:
    lines = txt.splitlines()
    return lines[-n:] if lines else []


# -----------------------------
# Extract PASS from reports
# -----------------------------

def load_json_safe(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def extract_tau_pass(rep: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not rep:
        return None
    a = rep.get("PASS_G00_sign_attractive", None)
    b = rep.get("PASS_kappa_median_sign", None)
    return tri_and([a, b])


def extract_deflection_pass(rep: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not rep:
        return None
    return rep.get("PASS_deflection_markov_front_PP", None)


def extract_shapiro_pass(rep: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not rep:
        return None
    return rep.get("PASS_Shapiro_markov_tau", None)


# -----------------------------
# Main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="STRICT PP scalar 512 runner (ms080 + strong_pf010).")

    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument(
        "--cases",
        type=str,
        default="mass_ms080,strong_pf010",
        help="Comma-separated cases (default: mass_ms080,strong_pf010)",
    )

    # Optional overrides for base files (single-case manual runs)
    ap.add_argument("--edges_flat", type=str, default="")
    ap.add_argument("--edges_curved", type=str, default="")
    ap.add_argument("--trace_weights", type=str, default="")
    ap.add_argument("--mass_mask", type=str, default="")
    ap.add_argument("--ring_mask", type=str, default="")

    # Tau-geometry params
    ap.add_argument("--mass_topk", type=int, default=500)
    ap.add_argument("--shell_bands", type=str, default="2:3,3:4,4:5,5:6")
    ap.add_argument("--ht_tol", type=float, default=1e-5)
    ap.add_argument("--ht_maxiter", type=int, default=40000)

    # Deflection params
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--defl_ht_tol", type=float, default=1e-5)
    ap.add_argument("--defl_ht_maxiter", type=int, default=20000)
    ap.add_argument("--src_auto", type=str, default="ring_mid", choices=["ring_mid", "none"])

    # Shapiro params
    ap.add_argument("--shapiro_tol", type=float, default=1e-5)
    ap.add_argument("--shapiro_maxiter", type=int, default=200000)
    ap.add_argument("--require_shapiro", action="store_true")
    ap.add_argument("--skip_shapiro", action="store_true")

    # Manual Shapiro override (all-or-nothing)
    ap.add_argument("--shapiro_src_through", type=int, default=None)
    ap.add_argument("--shapiro_dst_through", type=int, default=None)
    ap.add_argument("--shapiro_src_around",  type=int, default=None)
    ap.add_argument("--shapiro_dst_around",  type=int, default=None)

    # Output
    ap.add_argument(
        "--output",
        type=str,
        default="src/gr_strict_pp_scalar/PP_scalar_512_suite_report.json",
    )

    return ap.parse_args()


def main():
    args = parse_args()
    H, W = args.H, args.W
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive.")
    N = H * W

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    if not cases:
        raise ValueError("No cases specified.")

    # Validate shell band string lightly
    # (we still pass the raw string through to the tau script)
    if not args.shell_bands or ":" not in args.shell_bands:
        raise ValueError("shell_bands looks malformed; expected e.g. 2:3,3:4,...")

    suite: Dict[str, Any] = {
        "H": H, "W": W, "N": N,
        "cases": {},
        "require_shapiro": bool(args.require_shapiro),
        "skip_shapiro": bool(args.skip_shapiro),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": (
            "STRICT PP scalar 512 runner. "
            "Runs tau-geometry v4, deflection v3 sparse-only, "
            "and Shapiro v2 sparse with manual-or-auto ring-local pairs. "
            "By default Shapiro is not required for ALL_PASS "
            "due to known pair sensitivity at 512."
        ),
    }

    all_case_passes: List[Optional[bool]] = []

    for case in cases:
        errs: List[str] = []
        paths = default_paths(case, H, W)

        # Per-run overrides (useful for one-off manual runs)
        edges_flat = args.edges_flat or paths["edges_flat"]
        edges_curved = args.edges_curved or paths["edges_curved"]
        trace_weights = args.trace_weights or paths["trace_weights"]
        mass_mask = args.mass_mask or paths["mass_mask"]
        ring_mask = args.ring_mask or paths["ring_mask"]

        # Basic existence checks
        need(edges_flat, "edges_flat", errs)
        need(edges_curved, "edges_curved", errs)
        need(trace_weights, "trace_weights", errs)
        need(mass_mask, "mass_mask", errs)

        # ring_mask is required for:
        #   - deflection src_auto ring_mid
        #   - shapiro auto policy (unless skip_shapiro or manual pairs used)
        manual_shapiro = (
            args.shapiro_src_through is not None or
            args.shapiro_dst_through is not None or
            args.shapiro_src_around  is not None or
            args.shapiro_dst_around  is not None
        )

        if args.src_auto == "ring_mid":
            need(ring_mask, "ring_mask", errs)
        else:
            if (not args.skip_shapiro) and (not manual_shapiro):
                need(ring_mask, "ring_mask", errs)

        case_out: Dict[str, Any] = {
            "case": case,
            "inputs": {
                "edges_flat": edges_flat,
                "edges_curved": edges_curved,
                "trace_weights": trace_weights,
                "mass_mask": mass_mask,
                "ring_mask": ring_mask,
            },
            "errors": errs[:],
            "runs": {},
            "PASS_components": {},
            "PASS": None,
        }

        if errs:
            case_out["PASS"] = False
            suite["cases"][case] = case_out
            all_case_passes.append(False)
            continue

        # -------------------------
        # 1) Tau geometry v4
        # -------------------------
        tau_report = f"src/gr_strict_pp_scalar/PP_markov_tau_geometry_512_{case}_topk{args.mass_topk}.json"
        tau_cmd = [
            py(), "src/gr_strict_pp_scalar/PP_markov_tau_geometry_v4_multiShellIntersect.py",
            "--edges_flat", edges_flat,
            "--edges_curved", edges_curved,
            "--trace_weights", trace_weights,
            "--H", str(H), "--W", str(W),
            "--mass_topk", str(args.mass_topk),
            "--shell_bands", args.shell_bands,
            "--ht_tol", str(args.ht_tol),
            "--ht_maxiter", str(args.ht_maxiter),
            "--report", tau_report,
        ]

        rc, rt, out_txt = run_cmd(tau_cmd)
        case_out["runs"]["tau_geometry_v4"] = {
            "cmd": " ".join(tau_cmd),
            "returncode": rc,
            "runtime_sec": rt,
            "report": tau_report,
            "stdout_tail": tail_lines(out_txt),
        }

        tau_rep = load_json_safe(tau_report)
        PASS_tau = extract_tau_pass(tau_rep)

        # -------------------------
        # 2) Deflection v3 sparse-only
        # -------------------------
        defl_report = f"src/gr_strict_pp_scalar/PP_deflection_front_512_{case}_PPV1.json"
        defl_cmd = [
            py(), "src/gr_strict_pp_scalar/PP_deflection_markov_front_PP_v3_sparse_only.py",
            "--edges_flat", edges_flat,
            "--edges_curved", edges_curved,
            "--mass_mask", mass_mask,
            "--H", str(H), "--W", str(W),
            "--steps", str(args.steps),
            "--ht_tol", str(args.defl_ht_tol),
            "--ht_maxiter", str(args.defl_ht_maxiter),
            "--output", defl_report,
        ]
        if args.src_auto != "none":
            defl_cmd += ["--ring_mask", ring_mask, "--src_auto", args.src_auto]

        rc, rt, out_txt = run_cmd(defl_cmd)
        case_out["runs"]["deflection_v3_sparse"] = {
            "cmd": " ".join(defl_cmd),
            "returncode": rc,
            "runtime_sec": rt,
            "report": defl_report,
            "stdout_tail": tail_lines(out_txt),
        }

        defl_rep = load_json_safe(defl_report)
        PASS_defl = extract_deflection_pass(defl_rep)

        # -------------------------
        # 3) Shapiro v2 sparse
        # -------------------------
        shapiro_report = f"src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_512_{case}_PPV1.json"
        PASS_shap: Optional[bool] = None
        shapiro_meta: Dict[str, Any] = {}

        st = dt = sa = da = None

        if args.skip_shapiro:
            st, dt, sa, da, shapiro_meta = (None, None, None, None, {"skipped": True, "policy": "skipped_by_flag"})
            PASS_shap = None
        else:
            st, dt, sa, da, shapiro_meta = pick_shapiro_pairs(args, ring_mask)

            if shapiro_meta.get("skipped", False):
                PASS_shap = None
            else:
                shap_cmd = [
                    py(), "src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_v2_sparse.py",
                    "--edges_flat", edges_flat,
                    "--edges_curved", edges_curved,
                    "--H", str(H), "--W", str(W),
                    "--src_through", str(st),
                    "--dst_through", str(dt),
                    "--src_around", str(sa),
                    "--dst_around", str(da),
                    "--tol", str(args.shapiro_tol),
                    "--maxiter", str(args.shapiro_maxiter),
                    "--output", shapiro_report,
                ]

                rc, rt, out_txt = run_cmd(shap_cmd)
                case_out["runs"]["shapiro_v2_sparse"] = {
                    "cmd": " ".join(shap_cmd),
                    "returncode": rc,
                    "runtime_sec": rt,
                    "report": shapiro_report,
                    "stdout_tail": tail_lines(out_txt),
                }

                shap_rep = load_json_safe(shapiro_report)
                PASS_shap = extract_shapiro_pass(shap_rep)

        case_out["shapiro_pair_meta"] = shapiro_meta

        # -------------------------
        # Per-case aggregation
        # -------------------------
        case_out["PASS_components"] = {
            "PASS_tau_geometry": PASS_tau,
            "PASS_deflection": PASS_defl,
            "PASS_shapiro": PASS_shap,
        }

        comps = [PASS_tau, PASS_defl]
        if args.require_shapiro:
            comps.append(PASS_shap)

        PASS_case = tri_and(comps)

        case_out["PASS"] = PASS_case
        suite["cases"][case] = case_out
        all_case_passes.append(PASS_case)

    suite["ALL_PASS"] = tri_and(all_case_passes)

    # Write suite JSON
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(suite, f, indent=2)

    print(f"WROTE suite report: {args.output}")
    print(f"ALL_PASS = {suite['ALL_PASS']}")

    # Compact human summary
    print("\n=== STRICT PP Scalar 512 Summary ===")
    print("H,W,N =", H, W, N)
    print("cases  =", ", ".join(cases))
    print("require_shapiro =", bool(args.require_shapiro))
    print("skip_shapiro    =", bool(args.skip_shapiro))
    print("ALL_PASS        =", suite["ALL_PASS"])
    for case in cases:
        c = suite["cases"].get(case, {})
        pc = c.get("PASS_components", {})
        print(f"\n[{case}] PASS = {c.get('PASS')}")
        print("  tau_geometry =", pc.get("PASS_tau_geometry"))
        print("  deflection   =", pc.get("PASS_deflection"))
        print("  shapiro      =", pc.get("PASS_shapiro"))


if __name__ == "__main__":
    main()

