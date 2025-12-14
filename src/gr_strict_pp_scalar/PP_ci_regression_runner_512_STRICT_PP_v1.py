#!/usr/bin/env python3
import argparse, json, os, sys, subprocess

def load_json(p: str):
    with open(p, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser(description="STRICT_PP 512 regression runner (status-only; no heavy generators).")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--structural_py", default="src/gr_strict_pp_scalar/PP_EFE_structural_status_512_STRICT_PP_v1.py")
    ap.add_argument("--master_py", default="src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.py")
    ap.add_argument("--structural_json", default="src/gr_strict_pp_scalar/PP_EFE_structural_status_512_STRICT_PP_v1.json")
    ap.add_argument("--master_json", default="src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.json")
    ap.add_argument("--key", default="overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_strict_PP_512_v3")
    ap.add_argument("--output", default="src/gr_strict_pp_scalar/PP_ci_regression_runner_512_STRICT_PP_v1.json")
    args = ap.parse_args()

    # Run structural + master (parsers only; rely on existing artifacts)
    cmds = [
        [args.python, args.structural_py, "--output", args.structural_json],
        [args.python, args.master_py, "--structural_json", args.structural_json, "--output", args.master_json],
    ]
    logs = []
    for c in cmds:
        r = subprocess.run(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logs.append({"cmd": c, "rc": r.returncode, "out": r.stdout})
        if r.returncode != 0:
            out = {
                "PASS": False,
                "reason": "subcommand failed",
                "failed_cmd": c,
                "logs": logs,
            }
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2, sort_keys=True)
            print("FAIL: subcommand failed; wrote", args.output)
            raise SystemExit(r.returncode)

    j = load_json(args.master_json)
    ok = bool(j.get(args.key))

    out = {
        "PASS": ok,
        "key": args.key,
        "master_json": args.master_json,
        "structural_json": args.structural_json,
        "deflection": {
            "APPLICABLE": (j.get("deflection_flags") or {}).get("APPLICABLE"),
            "PASS": (j.get("deflection_flags") or {}).get("PASS"),
        },
        "structural": {
            "APPLICABLE": (j.get("structural_flags") or {}).get("APPLICABLE"),
            "PASS": (j.get("structural_flags") or {}).get("PASS"),
        },
        "logs": logs,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("CI_SANITY:", args.key, "=", ok)
    raise SystemExit(0 if ok else 2)

if __name__ == "__main__":
    main()

