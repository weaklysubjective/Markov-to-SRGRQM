#!/usr/bin/env python3
import argparse, json, os, subprocess, sys
from typing import Optional, Dict, Any, List

def _run(cmd: List[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def _script_accepts_flag(py: str, script: str, flag: str) -> bool:
    try:
        p = subprocess.run([py, script, "-h"], capture_output=True, text=True, check=True)
        return flag in (p.stdout + p.stderr)
    except Exception:
        return False

def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _get_nested(d: Dict[str, Any], path: List[str]) -> Optional[Any]:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def main():
    ap = argparse.ArgumentParser(description="STRICT PP 512: one-shot runner for vector/offdiag -> master v3 -> tensor_3p1 -> strongfield -> real_EFE.")
    ap.add_argument("--py", default=sys.executable, help="Python interpreter to use.")
    ap.add_argument("--device", default="gpu", help="Passed only to scripts that support --device.")
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)

    # Scripts (paths)
    ap.add_argument("--offdiag_binder", default="src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v2.py")
    ap.add_argument("--master",        default="src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.py")
    ap.add_argument("--tensor_3p1",    default="src/gr_strict_pp_scalar/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.py")
    ap.add_argument("--strongfield",   default="src/gr_strict_pp_scalar/PP_EFE_strongfield_suite_status_512_STRICT_PP_v1.py")
    ap.add_argument("--real_efe",      default="src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v1.py")

    # Outputs (JSONs)
    ap.add_argument("--offdiag_json",   default="src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v2.json")
    ap.add_argument("--master_json",    default="src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.json")
    ap.add_argument("--tensor_3p1_json",default="src/gr_strict_pp_scalar/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.json")
    ap.add_argument("--strongfield_json",default="src/gr_strict_pp_scalar/PP_EFE_strongfield_suite_status_512_STRICT_PP_v1.json")
    ap.add_argument("--real_efe_json",  default="src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v1.json")

    ap.add_argument("--output", default="src/gr_strict_pp_scalar/PP_REAL_EFE_chain_runner_512_STRICT_PP_v1.json")
    args = ap.parse_args()

    for s in [args.offdiag_binder, args.master, args.tensor_3p1, args.strongfield, args.real_efe]:
        if not os.path.exists(s):
            raise SystemExit(f"Missing script: {s}")

    py = args.py

    # 1) offdiag binder (no --device assumed; but harmless if supported)
    cmd = [py, args.offdiag_binder, "--output", args.offdiag_json]
    if _script_accepts_flag(py, args.offdiag_binder, "--device"):
        cmd = [py, args.offdiag_binder, "--device", args.device, "--output", args.offdiag_json]
    _run(cmd)

    # 2) master v3
    cmd = [py, args.master]
    if _script_accepts_flag(py, args.master, "--device"):
        cmd = [py, args.master, "--device", args.device]
    _run(cmd)

    # 3) tensor 3p1 binder
    cmd = [py, args.tensor_3p1]
    if _script_accepts_flag(py, args.tensor_3p1, "--device"):
        cmd = [py, args.tensor_3p1, "--device", args.device]
    _run(cmd)

    # 4) strongfield suite binder
    cmd = [py, args.strongfield]
    if _script_accepts_flag(py, args.strongfield, "--device"):
        cmd = [py, args.strongfield, "--device", args.device]
    _run(cmd)

    # 5) real EFE binder
    cmd = [py, args.real_efe]
    if _script_accepts_flag(py, args.real_efe, "--device"):
        cmd = [py, args.real_efe, "--device", args.device]
    _run(cmd)

    # --- Evaluate PASS bits with known schemas ---
    off = _load_json(args.offdiag_json)
    m   = _load_json(args.master_json)
    t3  = _load_json(args.tensor_3p1_json)
    sf  = _load_json(args.strongfield_json)
    re  = _load_json(args.real_efe_json)

    off_pass = bool(_get_nested(off, ["offdiag_flags","PASS"]) is True)
    master_pass = bool(_get_nested(m, ["overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_strict_PP_512_v3"]) is True)
    t3_pass = bool(_get_nested(t3, ["PASS","ALL_PASS_EFE_tensor_3p1_strict_PP_512_v1"]) is True)
    sf_pass = bool(_get_nested(sf, ["PASS","ALL_PASS_strongfield_suite_strict_PP_512_v1"]) is True)
    re_pass = bool(_get_nested(re, ["PASS","ALL_PASS_real_EFE_strict_PP_512_v1"]) is True)

    overall = bool(off_pass and master_pass and t3_pass and sf_pass and re_pass)

    out = {
        "H": int(args.H),
        "W": int(args.W),
        "device_requested": str(args.device),
        "scripts": {
            "offdiag_binder": args.offdiag_binder,
            "master": args.master,
            "tensor_3p1": args.tensor_3p1,
            "strongfield": args.strongfield,
            "real_efe": args.real_efe,
        },
        "json_outputs": {
            "offdiag_json": args.offdiag_json,
            "master_json": args.master_json,
            "tensor_3p1_json": args.tensor_3p1_json,
            "strongfield_json": args.strongfield_json,
            "real_efe_json": args.real_efe_json,
        },
        "PASS": {
            "PASS_offdiag_binder": off_pass,
            "PASS_master_v3": master_pass,
            "PASS_tensor_3p1": t3_pass,
            "PASS_strongfield_suite": sf_pass,
            "PASS_real_EFE": re_pass,
            "ALL_PASS_chain_runner_strict_PP_512_v1": overall,
        },
        "notes": "Runner is STRICT_PP-safe: it only executes existing scripts and reads their reported PASS flags. It auto-detects --device support per script via -h.",
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("ALL_PASS_chain_runner_strict_PP_512_v1 =", overall)

if __name__ == "__main__":
    main()

