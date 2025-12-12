#!/usr/bin/env python3
import argparse
import json
import math
from typing import Any, Dict, Optional, Tuple, List

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def is_finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False

def find_first_key(d: Dict[str, Any], keys: List[str]) -> Tuple[Optional[str], Any]:
    for k in keys:
        if k in d:
            return k, d.get(k)
    return None, None

def dig(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

def extract_offdiag_cov(P: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected keys from PP_offdiag_ring_cov_toy_v1.py:
      var_r, var_t, cov_rt, corr_rt, APPLICABLE, n_edges_used
    May be at top-level.
    """
    out: Dict[str, Any] = {}
    # Primary expected
    for k in ["APPLICABLE", "n_edges_used", "var_r", "var_t", "cov_rt", "corr_rt"]:
        if k in P:
            out[k] = P[k]

    # Fallback: sometimes these might be nested
    if "var_r" not in out:
        cand = dig(P, ["stats", "var_r"])
        if cand is not None:
            out["var_r"] = cand
    if "var_t" not in out:
        cand = dig(P, ["stats", "var_t"])
        if cand is not None:
            out["var_t"] = cand
    if "cov_rt" not in out:
        cand = dig(P, ["stats", "cov_rt"])
        if cand is not None:
            out["cov_rt"] = cand
    if "corr_rt" not in out:
        cand = dig(P, ["stats", "corr_rt"])
        if cand is not None:
            out["corr_rt"] = cand
    if "n_edges_used" not in out:
        cand = dig(P, ["stats", "n_edges_used"])
        if cand is not None:
            out["n_edges_used"] = cand
    if "APPLICABLE" not in out:
        cand = dig(P, ["stats", "APPLICABLE"])
        if cand is not None:
            out["APPLICABLE"] = cand

    return out

def extract_azimuthal_bias(P: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Expected keys from PP_vector_azimuthal_flux_markov_v1.py:
      azimuthal_bias, PASS_vector_azimuthal_flux_PP, VECTOR_AZIMUTHAL_FLUX_APPLICABLE
    Possibly nested under 'azimuthal_flux' if you pass a pack, but this script expects the raw azimuthal json.
    """
    meta: Dict[str, Any] = {}
    key, val = find_first_key(P, ["azimuthal_bias", "bias", "cw_ccw_bias"])
    if key is None:
        # fallback nested
        val = dig(P, ["azimuthal_flux", "azimuthal_bias"])
        if val is not None:
            key = "azimuthal_flux.azimuthal_bias"
    if key is not None and is_finite(val):
        meta["azimuthal_bias_key"] = key
        meta["azimuthal_bias"] = float(val)
        return float(val), meta
    return None, meta

def extract_ring_grad_bias(P: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Your PP_vector_ring_grad_corr_v1.json isn’t shown here, so we extract robustly.
    We look for a signed inward bias / correlation-like scalar.

    If we can't find it, we return None but still write a complete report.
    """
    meta: Dict[str, Any] = {}

    # Candidate keys (add more as your schema evolves)
    candidates = [
        "radial_bias",
        "inward_bias",
        "bias_inward",
        "ring_radial_bias",
        "grad_bias",
        "corr",
        "corr_rg",
        "corr_r_grad",
        "corr_ring_grad",
        "ring_grad_corr",
    ]
    key, val = find_first_key(P, candidates)
    if key is None:
        # common nested patterns
        for path in [
            ["ring_grad_corr", "radial_bias"],
            ["ring_grad_corr", "inward_bias"],
            ["ring_grad_corr", "corr"],
            ["stats", "radial_bias"],
            ["stats", "corr"],
        ]:
            v = dig(P, path)
            if v is not None:
                key = ".".join(path)
                val = v
                break

    if key is not None and is_finite(val):
        meta["ring_grad_key"] = key
        meta["ring_grad_value"] = float(val)
        return float(val), meta
    return None, meta

def eig_2x2(a: float, b: float, c: float) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
    """
    Matrix [[a, b],[b, c]]
    Returns (lam1>=lam2), and unit eigenvectors (v1, v2) as tuples.
    """
    tr = a + c
    det = a * c - b * b
    disc = tr * tr - 4.0 * det
    disc = max(disc, 0.0)
    s = math.sqrt(disc)
    lam1 = 0.5 * (tr + s)
    lam2 = 0.5 * (tr - s)

    # eigenvector for lam1: solve (A - lam I)v=0
    def unit(vx: float, vy: float) -> Tuple[float, float]:
        n = math.hypot(vx, vy)
        if n == 0:
            return (1.0, 0.0)
        return (vx / n, vy / n)

    # pick stable formula
    if abs(b) > 1e-18:
        v1 = unit(b, lam1 - a)
        v2 = unit(b, lam2 - a)
    else:
        # already diagonal
        v1 = (1.0, 0.0) if a >= c else (0.0, 1.0)
        v2 = (0.0, 1.0) if a >= c else (1.0, 0.0)

    return lam1, lam2, v1, v2

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP 2+1 off-diagonal Einstein-block toy aggregator (orbit ring)."
    )
    ap.add_argument("--case", required=True, help="e.g. ms080 or strong_pf010 (for labeling only)")
    ap.add_argument("--offdiag_cov_json", required=True, help="PP_offdiag_ring_cov_*_toy_v1.json")
    ap.add_argument("--azimuthal_json", required=True, help="PP_vector_azimuthal_*_bt001.json (raw azimuthal output)")
    ap.add_argument("--ring_grad_corr_json", required=False, default=None, help="PP_vector_ring_grad_corr_*.json")
    ap.add_argument("--scalar_status_json", required=False, default=None,
                    help="Optional scalar status summary JSON (PASS flags recorded; no new physics).")
    ap.add_argument("--corr_small_threshold", type=float, default=0.1,
                    help="|corr_rt| <= this counts as 'diagonal enough' for the toy block.")
    ap.add_argument("--min_edges_used", type=int, default=1000,
                    help="Require at least this many edges used in the offdiag cov estimator.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cov_raw = load_json(args.offdiag_cov_json)
    azi_raw = load_json(args.azimuthal_json)
    rg_raw = load_json(args.ring_grad_corr_json) if args.ring_grad_corr_json else {}

    cov = extract_offdiag_cov(cov_raw)
    az_bias, az_meta = extract_azimuthal_bias(azi_raw)
    rg_val, rg_meta = extract_ring_grad_bias(rg_raw) if rg_raw else (None, {"ring_grad_key": None})

    # Build toy block M = [[var_r, cov_rt],[cov_rt, var_t]]
    var_r = cov.get("var_r", None)
    var_t = cov.get("var_t", None)
    cov_rt = cov.get("cov_rt", None)
    corr_rt = cov.get("corr_rt", None)

    applicable = True
    reasons: List[str] = []

    if cov.get("APPLICABLE") is False:
        applicable = False
        reasons.append("offdiag_cov marked APPLICABLE=false by producer script")

    if not (is_finite(var_r) and is_finite(var_t) and is_finite(cov_rt) and is_finite(corr_rt)):
        applicable = False
        reasons.append("missing or non-finite var_r/var_t/cov_rt/corr_rt")

    n_edges_used = cov.get("n_edges_used", None)
    if n_edges_used is not None:
        try:
            n_edges_used_i = int(n_edges_used)
            if n_edges_used_i < args.min_edges_used:
                applicable = False
                reasons.append(f"n_edges_used={n_edges_used_i} < min_edges_used={args.min_edges_used}")
        except Exception:
            # don't hard-fail; just record
            reasons.append("n_edges_used present but not parseable as int")

    # SPD + diagonal-ish checks
    PASS_diag = False
    lam1 = lam2 = None
    v1 = v2 = None
    M_rt_abs_over_diag = None

    if applicable:
        a = float(var_r)
        c = float(var_t)
        b = float(cov_rt)
        lam1, lam2, v1, v2 = eig_2x2(a, b, c)

        # “mixing ratio”
        denom = max(abs(a) + abs(c), 1e-18)
        M_rt_abs_over_diag = abs(b) / denom

        spd = (lam2 is not None) and (lam2 > 0.0)
        diagish = abs(float(corr_rt)) <= args.corr_small_threshold

        PASS_diag = bool(spd and diagish)

    # Optional scalar status (record only)
    scalar_status = None
    if args.scalar_status_json:
        S = load_json(args.scalar_status_json)
        scalar_status = {
            "scalar_status_json": args.scalar_status_json,
            "overall_PASS_scalar_EFE00_strict_PP_v1": S.get("overall_PASS_scalar_EFE00_strict_PP_v1"),
            "overall_PASS_scalar_EFE00_strict_PP_v2": S.get("overall_PASS_scalar_EFE00_strict_PP_v2"),
            "notes": "Loaded for bookkeeping only; no new computations performed."
        }

    out: Dict[str, Any] = {
        "case": args.case,
        "notes": (
            "STRICT PP 2+1 off-diagonal Einstein-block toy (ring). "
            "This script introduces no new physics: it aggregates existing STRICT PP JSONs "
            "into a single ring-local 2x2 block M (r,t) plus a toy vector v (radial, azimuthal). "
            "No PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        ),
        "inputs": {
            "offdiag_cov_json": args.offdiag_cov_json,
            "azimuthal_json": args.azimuthal_json,
            "ring_grad_corr_json": args.ring_grad_corr_json,
            "scalar_status_json": args.scalar_status_json,
        },
        "v_toy": {
            "v_r_from_ring_grad": rg_val,
            "v_t_from_azimuthal_bias": az_bias,
            "extraction_meta": {
                **az_meta,
                **rg_meta,
            }
        },
        "M_toy": {
            "M_rr": float(var_r) if is_finite(var_r) else None,
            "M_tt": float(var_t) if is_finite(var_t) else None,
            "M_rt": float(cov_rt) if is_finite(cov_rt) else None,
            "corr_rt": float(corr_rt) if is_finite(corr_rt) else None,
            "n_edges_used": cov.get("n_edges_used"),
            "eig": {
                "lam1": lam1,
                "lam2": lam2,
                "v1": v1,
                "v2": v2,
            },
            "M_rt_abs_over_diag": M_rt_abs_over_diag,
        },
        "criteria": {
            "corr_small_threshold": args.corr_small_threshold,
            "min_edges_used": args.min_edges_used,
            "requires_SPD": True,
        },
        "result": {
            "APPLICABLE": applicable,
            "reasons": reasons if not applicable else [],
            "PASS_offdiag_2p1_toy_diag": bool(PASS_diag) if applicable else False,
        },
        "scalar_bookkeeping": scalar_status,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("APPLICABLE =", out["result"]["APPLICABLE"])
    if not out["result"]["APPLICABLE"]:
        print("reasons:", "; ".join(out["result"]["reasons"]))
    print("PASS_offdiag_2p1_toy_diag =", out["result"]["PASS_offdiag_2p1_toy_diag"])

if __name__ == "__main__":
    main()

