#!/usr/bin/env python3
"""
PP_EFE00_markov_scalar_pack_v2.py

STRICT PP scalar-EFE evidence pack (v2):

Aggregates:
  - Geometry / G00 vs rho            (PP_markov_tau_geometry_v2_multiShellIntersect_*.json)
  - Kappa / EFE-00 block stats       (PP_EFE00_markov_block_kappa_*.json)
  - Perihelion C1 loop τ             (PP_perihelion_markov_v1 curved + flat)
  - Perihelion C3 radial τ profile   (PP_perihelion_markov_C3_v1)
  - Shapiro Markov τ                 (PP_Shapiro_markov_tau_v1.json)

All are STRICT PP:
  - Markov + trace + Doyle commute time
  - No PDE, no GR ansatz, no regression added here
"""

import argparse
import json
import math
import sys


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def safe_get(dct, key, default=None):
    return dct.get(key, default)


def summarize_geom(geom_report):
    """
    Geometry summary.

    We try to read:
      - rho_mean
      - G00_mean
      - corr_G00_rho

    If not present, they remain None. No new computations here.
    """
    summary = {}

    rho_mean = safe_get(geom_report, "rho_mean", None)
    G00_mean = safe_get(geom_report, "G00_mean", None)
    corr_G00_rho = safe_get(geom_report, "corr_G00_rho", None)

    summary["rho_mean"] = rho_mean
    summary["G00_mean"] = G00_mean
    summary["corr_G00_rho"] = corr_G00_rho

    # Sign-based PASS: G00_mean negative => attractive
    pass_sign = None
    if G00_mean is not None:
        try:
            pass_sign = (G00_mean < 0.0)
        except TypeError:
            pass_sign = None

    summary["PASS_G00_sign_attractive"] = pass_sign

    return summary


def summarize_efe00(efe_report):
    """
    EFE-00 / kappa summary.

    We attempt to read:
      - kappa_median_core
      - kappa_min_core
      - kappa_max_core
      - kappa_q1_core
      - kappa_q3_core

    If not there, values remain None. No new fits.
    """
    summary = {}

    kappa_med = safe_get(efe_report, "kappa_median_core", None)
    kappa_min = safe_get(efe_report, "kappa_min_core", None)
    kappa_max = safe_get(efe_report, "kappa_max_core", None)
    kappa_q1 = safe_get(efe_report, "kappa_q1_core", None)
    kappa_q3 = safe_get(efe_report, "kappa_q3_core", None)

    summary["kappa_median_core"] = kappa_med
    summary["kappa_min_core"] = kappa_min
    summary["kappa_max_core"] = kappa_max
    summary["kappa_q1_core"] = kappa_q1
    summary["kappa_q3_core"] = kappa_q3

    # Sign-based PASS: median kappa negative (attractive scalar EFE-like)
    pass_sign = None
    if kappa_med is not None:
        try:
            pass_sign = (kappa_med < 0.0)
        except TypeError:
            pass_sign = None

    # Band consistency: same sign, not huge spread in magnitude
    pass_band = None
    if (kappa_min is not None) and (kappa_max is not None):
        try:
            same_sign = (kappa_min < 0.0 and kappa_max < 0.0) or (kappa_min > 0.0 and kappa_max > 0.0)
            ratio = abs(kappa_max / kappa_min) if kappa_min != 0 else None
            pass_band = bool(same_sign and (ratio is not None) and (ratio < 10.0))
        except TypeError:
            pass_band = None

    summary["PASS_kappa_median_sign"] = pass_sign
    summary["PASS_kappa_band_consistent"] = pass_band

    return summary


def summarize_peri_C1(curved, flat):
    """
    Perihelion C1 summary (loop τ).

    Expected keys:
      - tau_flat
      - tau_curved
      - tau_ratio_curved_over_flat
    """

    out = {}

    tau_flat_curved = safe_get(curved, "tau_flat", None)
    tau_curved_curved = safe_get(curved, "tau_curved", None)
    ratio_curved = safe_get(curved, "tau_ratio_curved_over_flat", None)

    tau_flat_flat = safe_get(flat, "tau_flat", None)
    tau_curved_flat = safe_get(flat, "tau_curved", None)
    ratio_flat = safe_get(flat, "tau_ratio_curved_over_flat", None)

    out["curved_tau_flat"] = tau_flat_curved
    out["curved_tau_curved"] = tau_curved_curved
    out["curved_tau_ratio"] = ratio_curved

    out["flat_tau_flat"] = tau_flat_flat
    out["flat_tau_curved"] = tau_curved_flat
    out["flat_tau_ratio"] = ratio_flat

    pass_flat_sanity = None
    pass_curved_attraction = None

    if ratio_flat is not None:
        try:
            pass_flat_sanity = (abs(ratio_flat - 1.0) < 1e-6)
        except TypeError:
            pass_flat_sanity = None

    if ratio_curved is not None:
        try:
            pass_curved_attraction = (ratio_curved < 1.0)
        except TypeError:
            pass_curved_attraction = None

    out["PASS_flat_tau_ratio_1"] = pass_flat_sanity
    out["PASS_curved_tau_ratio_lt1"] = pass_curved_attraction

    return out


def summarize_peri_C3(peri_C3):
    """
    Perihelion C3 summary (radial τ-ratio profile).

    Uses:
      - shells[*]['tau_ratio_curved_over_flat']
      - radial_gradients[*]['d_tau_ratio_over_dr']
    """

    shells = safe_get(peri_C3, "shells", [])
    radial_gradients = safe_get(peri_C3, "radial_gradients", [])

    out = {
        "n_shells": len(shells),
        "n_radial_gradients": len(radial_gradients),
        "shell_tau_ratios": [],
        "radial_d_tau_ratio_over_dr": [],
    }

    # Shell τ ratios
    for sh in shells:
        inner = safe_get(sh, "shell_inner", None)
        outer = safe_get(sh, "shell_outer", None)
        tau_ratio = safe_get(sh, "tau_ratio_curved_over_flat", None)
        out["shell_tau_ratios"].append(
            {
                "shell_inner": inner,
                "shell_outer": outer,
                "tau_ratio_curved_over_flat": tau_ratio,
            }
        )

    # Radial gradients
    for rg in radial_gradients:
        r1 = safe_get(rg, "r1", None)
        r2 = safe_get(rg, "r2", None)
        dr = safe_get(rg, "dr", None)
        dtr = safe_get(rg, "d_tau_ratio_over_dr", None)
        out["radial_d_tau_ratio_over_dr"].append(
            {
                "r1": r1,
                "r2": r2,
                "dr": dr,
                "d_tau_ratio_over_dr": dtr,
            }
        )

    # PASS: all d_tau_ratio_over_dr <= 0 (non-positive) => attractive monotone
    pass_monotone = None
    if radial_gradients:
        try:
            vals = [rg.get("d_tau_ratio_over_dr", 0.0) for rg in radial_gradients]
            vals = [v if v is not None else 0.0 for v in vals]
            pass_monotone = all(v <= 1e-8 for v in vals)
        except Exception:
            pass_monotone = None

    out["PASS_radial_attractive_monotone"] = pass_monotone

    return out


def summarize_shapiro(shapiro_report):
    """
    Shapiro Markov τ summary from PP_Shapiro_markov_tau_v1.json.

    Expected keys at top level:
      - d_tau_through
      - d_tau_around
      - PASS_Shapiro_markov_tau
      - flat: { tau_through, tau_around }
      - curved: { tau_through, tau_around }
    """

    out = {}

    d_tau_through = safe_get(shapiro_report, "d_tau_through", None)
    d_tau_around = safe_get(shapiro_report, "d_tau_around", None)
    pass_shapiro = safe_get(shapiro_report, "PASS_Shapiro_markov_tau", None)

    flat = safe_get(shapiro_report, "flat", {})
    curved = safe_get(shapiro_report, "curved", {})

    out["flat_tau_through"] = safe_get(flat, "tau_through", None)
    out["flat_tau_around"] = safe_get(flat, "tau_around", None)
    out["curved_tau_through"] = safe_get(curved, "tau_through", None)
    out["curved_tau_around"] = safe_get(curved, "tau_around", None)

    out["d_tau_through"] = d_tau_through
    out["d_tau_around"] = d_tau_around
    out["PASS_Shapiro_markov_tau"] = pass_shapiro

    return out


def main():
    p = argparse.ArgumentParser(
        description="Aggregate STRICT-PP scalar EFE evidence (v2) into a single JSON pack."
    )
    p.add_argument(
        "--geom_report",
        required=True,
        help="JSON report from PP_markov_tau_geometry_v2_multiShellIntersect (G00/rho).",
    )
    p.add_argument(
        "--efe00_report",
        required=True,
        help="JSON report from PP_EFE00_markov_block_kappa_v1 (kappa stats).",
    )
    p.add_argument(
        "--peri_curved",
        required=True,
        help="JSON from PP_perihelion_markov_v1 (curved mass case).",
    )
    p.add_argument(
        "--peri_flat",
        required=True,
        help="JSON from PP_perihelion_markov_v1 (flat control, edges_curved=edges_flat).",
    )
    p.add_argument(
        "--peri_C3",
        required=True,
        help="JSON from PP_perihelion_markov_C3_v1 (radial/sector profile).",
    )
    p.add_argument(
        "--shapiro_report",
        required=True,
        help="JSON from PP_Shapiro_markov_tau_v1 (Shapiro Markov τ).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSON path for aggregated scalar-EFE evidence pack (v2).",
    )

    args = p.parse_args()

    geom_rep = load_json(args.geom_report)
    efe_rep = load_json(args.efe00_report)
    peri_curved = load_json(args.peri_curved)
    peri_flat = load_json(args.peri_flat)
    peri_C3 = load_json(args.peri_C3)
    shapiro_rep = load_json(args.shapiro_report)

    geom_summary = summarize_geom(geom_rep)
    efe_summary = summarize_efe00(efe_rep)
    peri_C1_summary = summarize_peri_C1(peri_curved, peri_flat)
    peri_C3_summary = summarize_peri_C3(peri_C3)
    shapiro_summary = summarize_shapiro(shapiro_rep)

    # Overall PASS: all available sector flags that are not None must be True.
    sector_flags = []

    for summary in [geom_summary, efe_summary, peri_C1_summary, peri_C3_summary, shapiro_summary]:
        for key, val in summary.items():
            if key.startswith("PASS_") and val is not None:
                sector_flags.append(val)

    if sector_flags:
        overall_pass = all(sector_flags)
    else:
        overall_pass = None

    pack = {
        "geom_summary": geom_summary,
        "efe00_summary": efe_summary,
        "peri_C1_summary": peri_C1_summary,
        "peri_C3_summary": peri_C3_summary,
        "shapiro_summary": shapiro_summary,
        "overall_PASS_scalar_EFE_strict_PP": overall_pass,
        "notes": (
            "Strict-PP scalar EFE evidence pack v2. Uses only Markov/trace-derived G00, "
            "kappa over mass core, perihelion-like τ observables (C1/C3), and Shapiro Markov τ. "
            "No PDE, no GR ansatz, no regression added here; pure aggregation of existing PP diagnostics."
        ),
    }

    with open(args.output, "w") as f:
        json.dump(pack, f, indent=2)

    print(f"Wrote strict-PP scalar EFE evidence pack v2 to {args.output}")
    if overall_pass is not None:
        print(f"overall_PASS_scalar_EFE_strict_PP = {overall_pass}")
    else:
        print("overall_PASS_scalar_EFE_strict_PP = None (no sector flags available)")


if __name__ == "__main__":
    main()

