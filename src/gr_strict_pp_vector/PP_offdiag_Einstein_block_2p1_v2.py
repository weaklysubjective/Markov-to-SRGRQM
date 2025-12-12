#!/usr/bin/env python3
import argparse, json, math
from typing import Any, Optional

def load(path: str):
    with open(path, "r") as f:
        return json.load(f)

def _find_key_recursive(obj: Any, key: str) -> Optional[Any]:
    """Depth-first search for a key inside nested dict/list JSON."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            out = _find_key_recursive(v, key)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for it in obj:
            out = _find_key_recursive(it, key)
            if out is not None:
                return out
    return None

def _get_bool(P: Any, key: str, default: bool = False) -> bool:
    v = _find_key_recursive(P, key)
    if v is None:
        return default
    return bool(v)

def _get_num(P: Any, key: str, default: Optional[float] = None) -> Optional[float]:
    v = _find_key_recursive(P, key)
    if v is None:
        return default
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    # Allow numeric strings (rare but happens)
    try:
        vv = float(v)
        if math.isfinite(vv):
            return vv
    except Exception:
        pass
    return default

def main():
    ap = argparse.ArgumentParser(
        description="STRICT PP off-diagonal 2+1 Einstein-block (toy) v2: combines (i) ring cov diag, (ii) ring momentum-flux smallness, (iii) radial/azimuthal context."
    )
    ap.add_argument("--case", required=True)
    ap.add_argument("--offdiag_cov_json", required=True)
    ap.add_argument("--momentum_flux_json", required=True)
    ap.add_argument("--azimuthal_json", required=True)
    ap.add_argument("--ring_grad_corr_json", required=True)
    ap.add_argument("--deltaJ_small_threshold", type=float, default=0.01,
                    help="Static/non-rotating expectation: PASS if |deltaJ| <= this.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    C = load(args.offdiag_cov_json)
    M = load(args.momentum_flux_json)
    A = load(args.azimuthal_json)
    R = load(args.ring_grad_corr_json)

    # --- Extract gates/fields robustly ---
    cov_app = _get_bool(C, "APPLICABLE", False)
    cov_corr = _get_num(C, "corr_rt", None)

    # cov pass key variants
    cov_pass_diag = (
        _get_bool(C, "PASS_offdiag_toy_diagonal", False)
        or _get_bool(C, "PASS_offdiag_toy_diag", False)
        or _get_bool(C, "PASS_offdiag_toy_diagonal_v1", False)
    )

    mom_app = _get_bool(M, "APPLICABLE", False)
    deltaJ = _get_num(M, "deltaJ", None)

    if mom_app and (deltaJ is not None):
        mom_pass_small = (abs(deltaJ) <= float(args.deltaJ_small_threshold))
    else:
        mom_pass_small = False

    ring_app = _get_bool(R, "VECTOR_RING_GRAD_CORR_APPLICABLE", _get_bool(R, "APPLICABLE", False))
    ring_pass = _get_bool(R, "PASS_vector_ring_grad_corr_PP", False)

    az_app = _get_bool(A, "VECTOR_AZIMUTHAL_FLUX_APPLICABLE", _get_bool(A, "APPLICABLE", False))
    az_bias = _get_num(A, "azimuthal_bias", None)
    az_pass = _get_bool(A, "PASS_vector_azimuthal_flux_PP", False)

    applicable = bool(cov_app and mom_app and ring_app and az_app)

    # Core “offdiag small” block pass (toy 2+1)
    pass_offdiag_2p1_toy_diag = bool(applicable and cov_pass_diag and mom_pass_small)

    out = {
        "case": args.case,
        "notes": (
            "STRICT PP off-diagonal Einstein-block (toy) v2. "
            "Combines ring covariance diagonalization with ring tangential momentum-flux *smallness* "
            "for static/non-rotating cases. No PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        ),
        "inputs": {
            "offdiag_cov_json": args.offdiag_cov_json,
            "momentum_flux_json": args.momentum_flux_json,
            "azimuthal_json": args.azimuthal_json,
            "ring_grad_corr_json": args.ring_grad_corr_json,
        },
        "APPLICABLE": applicable,
        "gates": {
            "cov_applicable": cov_app,
            "momentum_applicable": mom_app,
            "azimuthal_applicable": az_app,
            "ring_grad_corr_applicable": ring_app,
        },
        "offdiag_cov": {
            "corr_rt": cov_corr,
            "PASS_offdiag_toy_diagonal": cov_pass_diag,
        },
        "momentum_flux": {
            "deltaJ": deltaJ,
            "deltaJ_small_threshold": float(args.deltaJ_small_threshold),
            "PASS_momentum_flux_small_static": mom_pass_small,
        },
        "context": {
            "azimuthal_bias": az_bias,
            "PASS_vector_azimuthal_flux_PP": az_pass,
            "PASS_vector_ring_grad_corr_PP": ring_pass,
        },
        "PASS_offdiag_2p1_toy_diag_v2": pass_offdiag_2p1_toy_diag,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("APPLICABLE =", out["APPLICABLE"])
    print("PASS_offdiag_2p1_toy_diag_v2 =", out["PASS_offdiag_2p1_toy_diag_v2"])
    if applicable:
        print("corr_rt =", cov_corr, "deltaJ =", deltaJ)

if __name__ == "__main__":
    main()

