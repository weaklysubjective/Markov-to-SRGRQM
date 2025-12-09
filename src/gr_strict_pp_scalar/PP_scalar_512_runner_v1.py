#!/usr/bin/env python3
"""
PP_scalar_512_runner_v1.py

STRICT PP Scalar 512 Runner

Runs (per case):
  1) PP_markov_tau_geometry_v4_multiShellIntersect.py
  2) PP_deflection_markov_front_PP_v3_sparse_only.py
  3) PP_Shapiro_markov_tau_v2_sparse.py  (auto ring-local pairs)

Design goals:
  - 512-scale safe (no dense NxN).
  - GPU-first for torch scripts when --device auto.
  - Tri-state PASS with explicit reasons.
  - Shapiro is pair-sensitive at 512; by default it is NOT required
    for ALL_PASS unless --require_shapiro is set.

Conventions (override via flags if needed):
  edges_flat:  edges_ca_v3_flat_512x512_PPV1.txt
  edges_curved: edges_ca_v3_{case}_512x512_PPV1.txt
  trace_weights: trace_weights_ca_v3_{case}_512x512.txt
  mass_mask:  PP_mass_mask_512x512_{case}_PPV1.npy
  ring_mask:  PP_orbit_ring_mask_512x512_{case}_PPV1.npy

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
        "edges_flat":   f"edges_ca_v3_flat_{hw}_PPV1.txt",
        "edges_curved": f"edges_ca_v3_{case}_{hw}_PPV1.txt",
        "trace_weights": f"trace_weights_ca_v3_{case}_{hw}.txt",
        "mass_mask":    f"PP_mass_mask_{hw}_{case}_PPV1.npy",
        "ring_mask":    f"PP_orbit_ring_mask_{hw}_{case}_PPV1.npy",
    }


# -----------------------------
# Shapiro auto-pair policy
# -----------------------------

def shapiro_auto_pairs_from_ring(ring_mask_path: str) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Use ONLY ring membership for auto pair selection (STRICT PP friendly).
    We pick two short-arc pairs:
      - "around": first and 11th ring node (local arc)
      - "through": mid and mid+10 ring node (also local arc, different segment)

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


def extract_tau_pass(rep: Dict[str, Any]) -> Optional[bool]:
    if not rep:
        return None
    a = rep.get("PASS_G00_sign_attractive", None)
    b = rep.get("PASS_kappa_median_sign", None)
    return tri_and([a, b])


def extract_deflection_pass(rep: Dict[str, Any]) -> Optional[bool]:
    if not rep:
        return None
    return rep.get("PASS_deflection_markov_front_PP", None)


def extract_shapiro_pass(rep: Dict[str, Any]) -> Optional[bool]:
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

    # Optional overrides for base files
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
    N = H * W

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    if not cases:
        raise ValueError("No cases specified.")

    # Parse shell bands
    bands = []
    for tok in args.shell_bands.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split(":")
        bands.append((int(a), int(b)))
    if not bands:
        raise ValueError("No shell bands parsed.")

    suite: Dict[str, Any] = {
        "H": H, "W": W, "N": N,
        "cases": {},
        "require_shapiro": bool(args.require_shapiro),
        "skip_shapiro": bool(args.skip_shapiro),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": (
            "STRICT PP scalar 512 runner. "
            "Runs tau-geometry v4, deflection v3 sparse-only, "
            "and Shapiro v2 sparse with ring-local auto pairs. "
            "By default Shapiro is not required for ALL_PASS "
            "due to known pair sensitivity at 512."
        ),
    }

    all_case_passes: List[Optional[bool]] = []

    for case in cases:
        errs: List[str] = []
        paths = default_paths(case, H, W)

        # Apply per-run overrides if provided (useful for single-case manual runs)
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
        # ring_mask is required for src_auto ring_mid + shapiro auto
        if args.src_auto == "ring_mid" or (not args.skip_shapiro):
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
            "--ht_tol", str(args.ht_tol),
            "--ht_maxiter", str(args.ht_maxiter),
            "--report", tau_report,
        ]
        # expand shell bands
        for rmin, rmax in bands:
            tau_cmd += ["--shell_bands", f"{rmin}:{rmax}"] if False else []
        # The v4 script expects --shell_bands as repeated tokens in your earlier usage:
        # We'll pass them as a flat list at the end:
        tau_cmd += ["--shell_bands"] + [f"{rmin}:{rmax}" for rmin, rmax in bands]

        rc, rt, out_txt = run_cmd(tau_cmd)
        case_out["runs"]["tau_geometry_v4"] = {
            "cmd": " ".join(tau_cmd),
            "returncode": rc,
            "runtime_sec": rt,
            "report": tau_report,
            "stdout_tail": out_txt.splitlines()[-20:],
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
            "--ring_mask", ring_mask,
            "--H", str(H), "--W", str(W),
            "--src_auto", args.src_auto,
            "--steps", str(args.steps),
            "--ht_tol", str(args.defl_ht_tol),
            "--ht_maxiter", str(args.defl_ht_maxiter),
            "--output", defl_report,
        ]
        rc, rt, out_txt = run_cmd(defl_cmd)
        case_out["runs"]["deflection_v3_sparse"] = {
            "cmd": " ".join(defl_cmd),
            "returncode": rc,
            "runtime_sec": rt,
            "report": defl_report,
            "stdout_tail": out_txt.splitlines()[-20:],
        }

        defl_rep = load_json_safe(defl_report)
        PASS_defl = extract_deflection_pass(defl_rep)

        # -------------------------
        # 3) Shapiro v2 sparse (auto pairs)
        # -------------------------
        shapiro_report = f"src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_512_{case}_PPV1.json"
        PASS_shap: Optional[bool] = None
        shapiro_meta: Dict[str, Any] = {"skipped": False}

        if args.skip_shapiro:
            shapiro_meta["skipped"] = True
            PASS_shap = None
        else:
            auto = shapiro_auto_pairs_from_ring(ring_mask)
            if auto is None:
                shapiro_meta["skipped"] = True
                shapiro_meta["skip_reason"] = "ring_too_small_for_auto_pairs"
                PASS_shap = None
            else:
                src_through, dst_through, src_around, dst_around, ring_size = auto
                shapiro_meta.update({
                    "ring_size": ring_size,
                    "src_through": src_through,
                    "dst_through": dst_through,
                    "src_around": src_around,
                    "dst_around": dst_around,
                    "policy": "ring_local_first_and_mid_arcs",
                })

                shap_cmd = [
                    py(), "src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_v2_sparse.py",
                    "--edges_flat", edges_flat,
                    "--edges_curved", edges_curved,
                    "--H", str(H), "--W", str(W),
                    "--src_through", str(src_through),
                    "--dst_through", str(dst_through),
                    "--src_around", str(src_around),
                    "--dst_around", str(dst_around),
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
                    "stdout_tail": out_txt.splitlines()[-20:],
                }

                shap_rep = load_json_safe(shapiro_report)
                PASS_shap = extract_shapiro_pass(shap_rep)

        case_out["shapiro_auto_meta"] = shapiro_meta

        # -------------------------
        # Per-case aggregation
        # -------------------------
        case_out["PASS_components"] = {
            "PASS_tau_geometry": PASS_tau,
            "PASS_deflection": PASS_defl,
            "PASS_shapiro": PASS_shap,
        }

        # Default scalar-case pass policy:
        #   tau AND deflection AND (shapiro if require_shapiro)
        comps = [PASS_tau, PASS_defl]
        if args.require_shapiro:
            comps.append(PASS_shap)

        PASS_case = tri_and(comps)

        case_out["PASS"] = PASS_case
        suite["cases"][case] = case_out
        all_case_passes.append(PASS_case)

    suite["ALL_PASS"] = tri_and(all_case_passes)

    # Write suite JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(suite, f, indent=2)

    print(f"WROTE suite report: {args.output}")
    print(f"ALL_PASS = {suite['ALL_PASS']}")


if __name__ == "__main__":
    main()

