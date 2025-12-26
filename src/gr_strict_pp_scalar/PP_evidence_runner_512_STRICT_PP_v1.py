#!/usr/bin/env python3
"""
PP_evidence_runner_512_STRICT_PP_v1.py

DISCOVER + AUDIT ONLY (NO GENERATION):
- Discovers T00-flow NPZ artifacts in a PASSDIR for cases {ms080,strong_pf010}.
- Runs PP_math_audit_harness_v3.py on each discovered artifact (or skips if --skip_existing and audit json exists).
- Ingests an existing PP_EFE_adversarial_status_512_STRICT_PP_v8*.json near the run directory (no rerun unless you wire it).
- Emits a single consolidated evidence JSON with ALL_PASS_EVIDENCE_512_STRICT_PP_v1.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def utc_ts() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def die(msg: str, code: int = 2) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    raise SystemExit(code)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def resolve_existing_path(candidates: List[str]) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    die("Could not resolve script path from candidates:\n  " + "\n  ".join(candidates))


def parse_seed_range(s: str) -> List[int]:
    """
    Accept: "0-4" or "0,1,2,3,4" or "0-4,7,9-12"
    """
    s = s.strip()
    if not s:
        return []
    out: List[int] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a = int(a.strip()); b = int(b.strip())
            if b < a:
                die(f"Bad seed range '{p}' (b<a).")
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(p))
    out = sorted(set(out))
    return out


def pick_latest_by_mtime(paths: List[str]) -> Optional[str]:
    if not paths:
        return None
    paths2 = [p for p in paths if os.path.exists(p)]
    if not paths2:
        return None
    paths2.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths2[0]


def find_adversarial_json(passdir: str, pattern: str) -> Optional[str]:
    """
    Search near PASSDIR for adversarial status JSON(s).
    We search:
      PASSDIR
      dirname(PASSDIR)
      parent(dirname(PASSDIR))
      parent(parent(...))  (up to 3 levels)
    """
    roots = []
    p = os.path.abspath(passdir)
    roots.append(p)
    roots.append(os.path.dirname(p))
    roots.append(os.path.dirname(os.path.dirname(p)))
    roots.append(os.path.dirname(os.path.dirname(os.path.dirname(p))))
    seen = set()
    hits: List[str] = []
    for r in roots:
        if not r or r in seen:
            continue
        seen.add(r)
        hits.extend(glob.glob(os.path.join(r, pattern)))
    return pick_latest_by_mtime(hits)


def extract_pass_flag_from_audit(audit_json: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    """
    Audit harness keys have drifted across v1/v2/v3. Normalize here.
    Returns: (pass_bool_or_None, key_used)
    """
    p = audit_json.get("PASS", {})
    for k in ["ALL_PASS_math_audit_v3", "ALL_PASS_math_audit_v2", "ALL_PASS_math_audit_v1", "ALL_PASS_math_audit"]:
        if k in p:
            return bool(p[k]), f"PASS.{k}"
    return None, "PASS.(missing)"


def extract_pass_flag_from_adversarial(adv_json: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    """
    Adversarial status v8 tends to contain either:
      top-level ALL_PASS_adversarial_512_STRICT_PP_v8
      or PASS.ALL_PASS_adversarial_512_STRICT_PP_v8
    """
    if "ALL_PASS_adversarial_512_STRICT_PP_v8" in adv_json:
        return bool(adv_json["ALL_PASS_adversarial_512_STRICT_PP_v8"]), "ALL_PASS_adversarial_512_STRICT_PP_v8"
    p = adv_json.get("PASS", {})
    if "ALL_PASS_adversarial_512_STRICT_PP_v8" in p:
        return bool(p["ALL_PASS_adversarial_512_STRICT_PP_v8"]), "PASS.ALL_PASS_adversarial_512_STRICT_PP_v8"
    return None, "(missing ALL_PASS_adversarial_512_STRICT_PP_v8)"


@dataclass
class Artifact:
    case: str
    seed: int
    npz_path: str
    audit_out: str


def discover_artifacts(passdir: str, cases: List[str], seed_range: Optional[List[int]]) -> List[Artifact]:
    """
    Discovers NPZs matching:
      PP_T00_flow_fixedGeom_lengths_512_{case}_qPermTopK_s{seed}_STRICT_PP_v1.npz
    If seed_range is provided, only include those seeds (if present).
    """
    out: List[Artifact] = []
    for case in cases:
        pat = os.path.join(
            passdir,
            f"PP_T00_flow_fixedGeom_lengths_512_{case}_qPermTopK_s*_STRICT_PP_v1.npz",
        )
        hits = sorted(glob.glob(pat))
        for npz in hits:
            m = re.search(r"_s(\d+)_STRICT_PP_v1\.npz$", npz)
            if not m:
                continue
            s = int(m.group(1))
            if seed_range is not None and s not in seed_range:
                continue
            # audit output path filled later by caller (needs audit_dir)
            out.append(Artifact(case=case, seed=s, npz_path=npz, audit_out=""))
    out.sort(key=lambda a: (a.case, a.seed))
    return out


def run_one_audit(
    python_exe: str,
    audit_script: str,
    npz: str,
    edges_undirected: str,
    lengths_npz: str,
    device: str,
    q_expect_topk: int,
    output: str,
    skip_existing: bool,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if skip_existing and os.path.exists(output):
        return {"status": "skipped_existing", "output": output}

    cmd = [
        python_exe,
        audit_script,
        "--npz", npz,
        "--edges_undirected", edges_undirected,
        "--lengths_npz", lengths_npz,
        "--device", device,
        "--q_expect_topk", str(q_expect_topk),
        "--output", output,
    ]
    # run and forward stdout/stderr live to preserve audit logs
    p = subprocess.run(cmd)
    return {"status": "ran", "returncode": p.returncode, "cmd": cmd, "output": output}


def main() -> int:
    ap = argparse.ArgumentParser(description="STRICT PP 512 evidence runner: discover + audit only (no generation).")
    ap.add_argument("--passdir", required=True, help="Directory containing PP_T00_flow_fixedGeom_lengths_512_*_qPermTopK_s*_STRICT_PP_v1.npz artifacts.")
    ap.add_argument("--runbase", default="src/gr_strict_pp_align/runs/finalleg_traceOnly_q9995_v2_mix0.7", help="Base dir for edges (default matches your current pipeline).")
    ap.add_argument("--audit_dir", default="runs/audits", help="Where to write per-artifact audit JSONs.")
    ap.add_argument("--output", default="runs/evidence/evidence_512_STRICT_PP_v1.json", help="Consolidated evidence JSON output.")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"], help="Device passed into audit harness.")
    ap.add_argument("--q_expect_topk", type=int, default=80, help="Expected topk for q field.")
    ap.add_argument("--cases", default="ms080,strong_pf010", help="Comma-separated cases to include.")
    ap.add_argument("--seed_range", default="", help='Optional seed constraint, e.g. "0-4" or "0-15". Empty = discover all present.')
    ap.add_argument("--expected_seeds", default="", help='Optional policy: seeds you expected to exist, e.g. "0-4" or "0-15".')
    ap.add_argument("--strict_missing", action="store_true", help="If set with --expected_seeds, missing artifacts become a hard FAIL.")
    ap.add_argument("--skip_existing", action="store_true", help="Skip audit runs if audit JSON already exists.")
    ap.add_argument("--audit_script", default="", help="Override path to PP_math_audit_harness_v3.py.")
    ap.add_argument("--adversarial_json", default="", help="If provided, use exactly this adversarial status JSON (do not search).")
    ap.add_argument("--adversarial_glob", default="PP_EFE_adversarial_status_512_STRICT_PP_v8*.json", help="Glob pattern searched near PASSDIR if --adversarial_json not given.")
    ap.add_argument("--python", default=sys.executable, help="Python executable to invoke sub-audits.")
    args = ap.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    if not cases:
        die("No cases parsed from --cases.")

    seed_range = parse_seed_range(args.seed_range) if args.seed_range.strip() else None
    expected = parse_seed_range(args.expected_seeds) if args.expected_seeds.strip() else None

    # Resolve audit script path
    audit_script = args.audit_script.strip() or resolve_existing_path([
        "src/gr_strict_pp_scalar/PP_math_audit_harness_v3.py",
        "src/gr_strict_pp_scalar/PP_math_audit_harness_v3_fixed.py",
        "src/gr_strict_pp_scalar/PP_math_audit_harness_v3_hardened.py",
    ])

    passdir = args.passdir.rstrip("/")
    if not os.path.isdir(passdir):
        die(f"--passdir not a directory: {passdir}")

    # Resolve per-case edges/lengths using your current convention
    runbase = args.runbase.rstrip("/")
    edges_map = {
        "ms080": os.path.join(runbase, "edges_undirected_ms080_512x512_E2.txt"),
        "strong_pf010": os.path.join(runbase, "edges_undirected_strong_pf010_512x512_E2.txt"),
    }
    lengths_map = {
        "ms080": "src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_ms080_STRICT_PP_v1.npz",
        "strong_pf010": "src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_strong_pf010_STRICT_PP_v1.npz",
    }

    for c in cases:
        if c not in edges_map or c not in lengths_map:
            die(f"Unsupported case '{c}' (edges/lengths mapping missing). Add it explicitly in this script.")
        if not os.path.exists(edges_map[c]):
            die(f"Missing edges file for case '{c}': {edges_map[c]}")
        if not os.path.exists(lengths_map[c]):
            die(f"Missing lengths npz for case '{c}': {lengths_map[c]}")

    artifacts = discover_artifacts(passdir, cases, seed_range)

    # Expected seed policy bookkeeping
    expected_missing: Dict[str, List[int]] = {}
    if expected is not None:
        present_by_case: Dict[str, set] = {c: set() for c in cases}
        for a in artifacts:
            present_by_case[a.case].add(a.seed)
        for c in cases:
            missing = [s for s in expected if s not in present_by_case[c]]
            expected_missing[c] = missing

    # Run audits
    audits: List[Dict[str, Any]] = []
    all_audit_pass = True

    for a in artifacts:
        audit_out = os.path.join(args.audit_dir, f"audit_{a.case}_s{a.seed}_math_v3.json")
        a.audit_out = audit_out

        run_res = run_one_audit(
            python_exe=args.python,
            audit_script=audit_script,
            npz=a.npz_path,
            edges_undirected=edges_map[a.case],
            lengths_npz=lengths_map[a.case],
            device=args.device,
            q_expect_topk=args.q_expect_topk,
            output=audit_out,
            skip_existing=args.skip_existing,
        )
        if run_res.get("status") == "ran" and run_res.get("returncode", 0) != 0:
            all_audit_pass = False

        # load audit json if it exists
        audit_rec: Dict[str, Any] = {
            "case": a.case,
            "seed": a.seed,
            "npz": a.npz_path,
            "audit_json": audit_out,
            "run": run_res,
        }
        if os.path.exists(audit_out):
            aj = read_json(audit_out)
            pflag, key_used = extract_pass_flag_from_audit(aj)
            audit_rec["pass_key_used"] = key_used
            audit_rec["pass"] = pflag
            if pflag is not True:
                all_audit_pass = False
        else:
            audit_rec["pass"] = None
            all_audit_pass = False

        audits.append(audit_rec)

    # Missing policy
    missing_hard_fail = False
    if expected is not None and args.strict_missing:
        for c, miss in expected_missing.items():
            if miss:
                missing_hard_fail = True

    # Ingest adversarial JSON
    adv_path = args.adversarial_json.strip()
    if not adv_path:
        adv_path = find_adversarial_json(passdir, args.adversarial_glob) or ""
    adv_obj: Optional[Dict[str, Any]] = None
    adv_pass: Optional[bool] = None
    adv_key: str = ""
    if adv_path and os.path.exists(adv_path):
        adv_obj = read_json(adv_path)
        adv_pass, adv_key = extract_pass_flag_from_adversarial(adv_obj)
    else:
        adv_path = ""

    # Consolidated pass
    all_pass_evidence = True
    if not all_audit_pass:
        all_pass_evidence = False
    if missing_hard_fail:
        all_pass_evidence = False
    if adv_pass is not True:
        all_pass_evidence = False

    out: Dict[str, Any] = {
        "timestamp_utc": utc_ts(),
        "version": "PP_evidence_runner_512_STRICT_PP_v1",
        "args": {
            "passdir": args.passdir,
            "runbase": args.runbase,
            "audit_dir": args.audit_dir,
            "device": args.device,
            "q_expect_topk": args.q_expect_topk,
            "cases": cases,
            "seed_range": args.seed_range,
            "expected_seeds": args.expected_seeds,
            "strict_missing": bool(args.strict_missing),
            "skip_existing": bool(args.skip_existing),
            "audit_script": audit_script,
            "adversarial_json": args.adversarial_json,
            "adversarial_glob": args.adversarial_glob,
        },
        "discovery": {
            "n_artifacts": len(artifacts),
            "artifacts": [
                {"case": a.case, "seed": a.seed, "npz": a.npz_path, "audit_json": a.audit_out}
                for a in artifacts
            ],
            "expected_missing": expected_missing if expected is not None else None,
        },
        "math_audits": {
            "all_pass": bool(all_audit_pass),
            "records": audits,
        },
        "adversarial": {
            "status_json": adv_path or None,
            "pass_key_used": adv_key or None,
            "pass": adv_pass,
        },
        "PASS": {
            "ALL_PASS_EVIDENCE_512_STRICT_PP_v1": bool(all_pass_evidence),
            "PASS_math_audits": bool(all_audit_pass),
            "PASS_missing_policy": (not missing_hard_fail),
            "PASS_adversarial": (adv_pass is True),
        },
        "ALL_PASS_EVIDENCE_512_STRICT_PP_v1": bool(all_pass_evidence),
        "notes": [
            "This runner does NOT generate any NPZs; it only audits existing artifacts and ingests an existing adversarial status JSON.",
            "If you want missing seeds to fail evidence, pass --expected_seeds ... --strict_missing.",
        ],
    }

    write_json(args.output, out)
    print(f"WROTE {args.output}")
    print(f"ALL_PASS_EVIDENCE_512_STRICT_PP_v1 = {out['ALL_PASS_EVIDENCE_512_STRICT_PP_v1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

