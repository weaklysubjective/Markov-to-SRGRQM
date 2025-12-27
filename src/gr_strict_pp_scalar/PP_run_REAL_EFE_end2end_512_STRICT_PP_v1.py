#!/usr/bin/env python3
"""
PP_run_REAL_EFE_end2end_512_STRICT_PP_v1.py

Purpose
-------
End-to-end (within a defined boundary) runner that:
  1) Discovers and runs PP_math_audit_harness_v3.py on existing NPZs (no generation).
  2) Runs tensor_3p1 binder.
  3) Runs REAL EFE binder.
  4) Runs FINAL wrapper binder.
  5) Writes an audit-grade MANIFEST JSON that includes:
       - run_dir
       - inputs (paths + sha256)
       - outputs (paths + sha256)
       - commands (exact shell commands executed + stderr/stdout tails + rc)
       - FINAL flags

STRICT_PP boundary
------------------
This runner does NOT regenerate physics kernels, edges, lengths, or Tmunu NPZs.
It validates correctness/consistency via:
  - math audit harness (NPZ structural + operator consistency checks)
  - status-chain propagation (tensor3p1 -> real_efe -> final)

Usage
-----
python src/gr_strict_pp_scalar/PP_run_REAL_EFE_end2end_512_STRICT_PP_v1.py \
  --repo_root . \
  --npz_glob "runs/adversarial_knobs_v8_search/inner/trial_seed0_pass_try/PP_T00_flow_fixedGeom_lengths_512_*_qPermTopK_s*_STRICT_PP_v1.npz" \
  --edges_ms080 "src/gr_strict_pp_align/runs/finalleg_traceOnly_q9995_v2_mix0.7/edges_undirected_ms080_512x512_E2.txt" \
  --lengths_ms080 "src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_ms080_STRICT_PP_v1.npz" \
  --edges_strong "src/gr_strict_pp_align/runs/finalleg_traceOnly_q9995_v2_mix0.7/edges_undirected_strong_pf010_512x512_E2.txt" \
  --lengths_strong "src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_strong_pf010_STRICT_PP_v1.npz" \
  --device gpu \
  --q_expect_topk 80 \
  --allow_na_invariance \
  --require_invariance false
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Basic utilities
# ----------------------------

def die(msg: str, code: int = 2) -> None:
    raise SystemExit(code if code is not None else 2)

def now_tag_utc() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def file_meta(path: str) -> Dict[str, Any]:
    path = abspath(path)
    st = os.stat(path)
    return {
        "path": path,
        "bytes": int(st.st_size),
        "mtime": float(st.st_mtime),
        "sha256": sha256_file(path),
    }

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def git_meta(repo_root: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"git_head": None, "git_dirty": None}
    try:
        out["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        pass
    try:
        s = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        ).strip()
        out["git_dirty"] = (s != "")
    except Exception:
        pass
    return out


# ----------------------------
# Command runner (string cmd)
# ----------------------------

def run_cmd(cmd: str, cwd: str, timeout_s: Optional[int] = None) -> Dict[str, Any]:
    """
    Run a *shell* command string. Capture limited stdout/stderr tail for the manifest.
    """
    t0 = time.time()
    p = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )
    t1 = time.time()

    def tail(s: str, n: int = 4000) -> str:
        if s is None:
            return ""
        return s[-n:] if len(s) > n else s

    return {
        "returncode": int(p.returncode),
        "elapsed_s": float(t1 - t0),
        "stdout_tail": tail(p.stdout or ""),
        "stderr_tail": tail(p.stderr or ""),
    }


# ----------------------------
# Paths
# ----------------------------

@dataclass
class Paths:
    repo_root: str
    run_dir: str
    audits_dir: str
    status_dir: str
    logs_dir: str
    manifest_json: str

    # scripts
    math_audit_v3_py: str
    tensor_3p1_binder_py: str
    real_efe_binder_v2_py: str
    real_efe_final_v2_py: str

def make_paths(repo_root: str, base_runs_dir: str) -> Paths:
    tag = now_tag_utc()
    run_dir = abspath(os.path.join(base_runs_dir, tag))
    audits_dir = os.path.join(run_dir, "audits")
    status_dir = os.path.join(run_dir, "status")
    logs_dir = os.path.join(run_dir, "logs")
    manifest_json = os.path.join(run_dir, "MANIFEST_end2end_real_efe_512_STRICT_PP_v1.json")

    # script absolute paths (repo-relative)
    math_audit_v3_py = abspath(os.path.join(repo_root, "src/gr_strict_pp_scalar/PP_math_audit_harness_v3.py"))
    tensor_3p1_binder_py = abspath(os.path.join(repo_root, "src/gr_strict_pp_scalar/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.py"))
    real_efe_binder_v2_py = abspath(os.path.join(repo_root, "src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v2.py"))
    real_efe_final_v2_py = abspath(os.path.join(repo_root, "src/gr_strict_pp_scalar/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.py"))

    return Paths(
        repo_root=abspath(repo_root),
        run_dir=run_dir,
        audits_dir=audits_dir,
        status_dir=status_dir,
        logs_dir=logs_dir,
        manifest_json=manifest_json,
        math_audit_v3_py=math_audit_v3_py,
        tensor_3p1_binder_py=tensor_3p1_binder_py,
        real_efe_binder_v2_py=real_efe_binder_v2_py,
        real_efe_final_v2_py=real_efe_final_v2_py,
    )


# ----------------------------
# PASS extraction helpers
# ----------------------------

def extract_boolish(d: Any, key: str) -> Optional[bool]:
    if not isinstance(d, dict):
        return None
    v = d.get(key)
    if isinstance(v, bool):
        return v
    return None

def extract_final_pass_flag(final_json_obj: Any) -> Tuple[Optional[str], Optional[bool]]:
    """
    FINAL wrapper writes FINAL_PASS_real_EFE_512_STRICT_PP_v2 at top-level in your observed output.
    Also support PASS dict if present.
    """
    if not isinstance(final_json_obj, dict):
        return None, None

    # Prefer top-level canonical
    v = final_json_obj.get("FINAL_PASS_real_EFE_512_STRICT_PP_v2")
    if isinstance(v, bool):
        return "FINAL_PASS_real_EFE_512_STRICT_PP_v2", v

    # Optional PASS dict (if present)
    P = final_json_obj.get("PASS")
    if isinstance(P, dict):
        v2 = P.get("FINAL_PASS_real_EFE_512_STRICT_PP_v2")
        if isinstance(v2, bool):
            return "PASS.FINAL_PASS_real_EFE_512_STRICT_PP_v2", v2

    # Last resort: any boolean key containing FINAL_PASS
    for k, vv in final_json_obj.items():
        if isinstance(vv, bool) and ("FINAL_PASS" in str(k)):
            return str(k), vv

    return None, None


# ----------------------------
# Core pipeline steps
# ----------------------------

def case_from_npz_name(npz_path: str) -> str:
    b = os.path.basename(npz_path)
    if "_ms080_" in b:
        return "ms080"
    if "_strong_pf010_" in b or "_strong_" in b:
        return "strong_pf010"
    # fallback: treat as strong unless explicitly ms080
    return "strong_pf010"

def run_step(manifest: Dict[str, Any], paths: Paths, name: str, cmd: str) -> Dict[str, Any]:
    """
    Run and record a pipeline step in manifest["commands"] (not build_cmds).
    """
    r = run_cmd(cmd, cwd=paths.repo_root)
    rec = {"name": name, "cmd": cmd, **r}
    manifest["commands"].append(rec)

    # write manifest early for forensic value on failure
    write_json(paths.manifest_json, manifest)

    if r["returncode"] != 0:
        die(f"Step failed: {name}\nCMD: {cmd}\nSTDERR:\n{r['stderr_tail']}", code=3)
    return rec

def run_math_audits(
    manifest: Dict[str, Any],
    paths: Paths,
    npz_paths: List[str],
    edges_ms080: str,
    lengths_ms080: str,
    edges_strong: str,
    lengths_strong: str,
    device: str,
    q_expect_topk: int,
) -> Tuple[bool, List[str]]:
    ensure_dir(paths.audits_dir)

    audit_jsons: List[str] = []
    all_pass = True

    for npz in npz_paths:
        c = case_from_npz_name(npz)
        if c == "ms080":
            edg = edges_ms080
            leng = lengths_ms080
        else:
            edg = edges_strong
            leng = lengths_strong

        out = os.path.join(
            paths.audits_dir,
            f"audit_{os.path.basename(npz).replace('.npz','')}_math_v3.json",
        )
        cmd = (
            f'python "{paths.math_audit_v3_py}" '
            f'--npz "{abspath(npz)}" '
            f'--edges_undirected "{abspath(edg)}" '
            f'--lengths_npz "{abspath(leng)}" '
            f'--device {device} '
            f'--q_expect_topk {int(q_expect_topk)} '
            f'--output "{abspath(out)}"'
        )
        run_step(manifest, paths, name=f"math_audit_{c}_{os.path.basename(npz)}", cmd=cmd)
        audit_jsons.append(out)

        try:
            aj = load_json(out)
            P = aj.get("PASS", {})
            # v3 harness might still write v2 key; accept any:
            v = (
                P.get("ALL_PASS_math_audit_v3")
                if isinstance(P, dict) else None
            )
            if not isinstance(v, bool):
                v = P.get("ALL_PASS_math_audit_v2") if isinstance(P, dict) else None
            if not isinstance(v, bool):
                v = P.get("ALL_PASS_math_audit_v1") if isinstance(P, dict) else None
            if not isinstance(v, bool):
                v = P.get("ALL_PASS_math_audit") if isinstance(P, dict) else None
            if v is not True:
                all_pass = False
        except Exception:
            all_pass = False

    return all_pass, audit_jsons

def run_real_efe_status_chain(
    manifest: Dict[str, Any],
    paths: Paths,
    allow_na_invariance: bool,
    require_invariance: bool,
) -> Dict[str, str]:
    """
    Run:
      tensor_3p1 binder -> real EFE binder -> final wrapper

    Returns a dict of output paths.
    """
    ensure_dir(paths.status_dir)

    # Inputs: checked-in component status JSONs (these are the "witness artifacts")
    scalar_a2b = abspath(os.path.join(paths.repo_root, "src/gr_strict_pp_scalar/PP_EFE_scalar_A2b_status_512_STRICT_PP_v1.json"))
    offdiag = abspath(os.path.join(paths.repo_root, "src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v3.json"))
    bianchi = abspath(os.path.join(paths.repo_root, "src/gr_strict_pp_scalar/PP_EFE_bianchi_status_512_STRICT_PP_v1.json"))
    closure = abspath(os.path.join(paths.repo_root, "src/gr_strict_pp_scalar/PP_EFE_tensor_closure_status_512_STRICT_PP_v1.json"))

    master = abspath(os.path.join(paths.repo_root, "src/gr_strict_pp_scalar/PP_EFE_master_status_512_STRICT_PP_v3.json"))
    tensor2p1 = abspath(os.path.join(paths.repo_root, "src/gr_strict_pp_scalar/PP_EFE_tensor_2p1_status_512_STRICT_PP_v1.json"))

    tensor3p1_out = abspath(os.path.join(paths.status_dir, "PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.json"))
    real_efe_out = abspath(os.path.join(paths.status_dir, "PP_EFE_real_EFE_status_512_STRICT_PP_v2.json"))
    final_out = abspath(os.path.join(paths.status_dir, "PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.json"))

    # 1) tensor_3p1 binder (master + tensor2p1 + optional invariance)
    cmd1 = (
        f'python "{paths.tensor_3p1_binder_py}" '
        f'--device gpu '
        f'--master_status_json "{master}" '
        f'--tensor_2p1_status_json "{tensor2p1}" '
        f'{"--allow_na_invariance " if allow_na_invariance else ""}'
        f'--output "{tensor3p1_out}"'
    )
    run_step(manifest, paths, name="binder_tensor_3p1", cmd=cmd1)

    # Gate immediately on binder output
    t3 = load_json(tensor3p1_out)
    P3 = t3.get("PASS", {})
    k3 = "ALL_PASS_EFE_tensor_3p1_strict_PP_512_v1"
    v3 = P3.get(k3) if isinstance(P3, dict) else None
    if v3 is not True:
        write_json(paths.manifest_json, manifest)
        die(f"FATAL: tensor_3p1 binder did not PASS (key={k3}, val={v3}). See: {tensor3p1_out}", code=4)

    # 2) REAL EFE binder v2
    # NOTE: this binder expects --require_invariance as int (0/1).
    req_inv_int = int(bool(require_invariance))
    cmd2 = (
        f'python "{paths.real_efe_binder_v2_py}" '
        f'--scalar_a2b "{scalar_a2b}" '
        f'--offdiag "{offdiag}" '
        f'--bianchi "{bianchi}" '
        f'--closure "{closure}" '
        f'--tensor3p1 "{tensor3p1_out}" '
        f'--output "{real_efe_out}" '
        f'--require_invariance {req_inv_int}'
    )
    run_step(manifest, paths, name="binder_real_efe_status_v2", cmd=cmd2)

    # Gate binder
    re = load_json(real_efe_out)
    Pr = re.get("PASS", {})
    kr = "ALL_PASS_real_EFE_strict_PP_512_v2"
    vr = Pr.get(kr) if isinstance(Pr, dict) else None
    if vr is not True:
        write_json(paths.manifest_json, manifest)
        die(f"FATAL: REAL EFE binder did not PASS (key={kr}, val={vr}). See: {real_efe_out}", code=5)

    # 3) FINAL wrapper (this writes FINAL_PASS_real_EFE_512_STRICT_PP_v2 at top-level)
    cmd3 = (
        f'python "{paths.real_efe_final_v2_py}" '
        f'--real_efe_json "{real_efe_out}" '
        f'--output "{final_out}" '
        f'--hash'
    )
    run_step(manifest, paths, name="binder_final_wrapper_v2", cmd=cmd3)

    fj = load_json(final_out)
    fk, fv = extract_final_pass_flag(fj)
    if fv is not True:
        write_json(paths.manifest_json, manifest)
        die(f"FATAL: FINAL wrapper did not PASS (key={fk}, val={fv}). See: {final_out}", code=6)

    return {
        "scalar_a2b": scalar_a2b,
        "offdiag_2p1": offdiag,
        "bianchi": bianchi,
        "closure": closure,
        "master": master,
        "tensor_2p1": tensor2p1,
        "tensor_3p1_out": tensor3p1_out,
        "real_efe_out": real_efe_out,
        "final_out": final_out,
    }


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="STRICT_PP end2end runner for REAL EFE 512 (status-chain + math audits).")
    ap.add_argument("--repo_root", type=str, default=".", help="Repo root containing src/ ...")
    ap.add_argument("--npz_glob", type=str, required=True, help="Glob for NPZs to audit (no generation).")

    ap.add_argument("--edges_ms080", type=str, required=True)
    ap.add_argument("--lengths_ms080", type=str, required=True)
    ap.add_argument("--edges_strong", type=str, required=True)
    ap.add_argument("--lengths_strong", type=str, required=True)

    ap.add_argument("--device", type=str, default="gpu", help="Forwarded to math audit harness (gpu/cpu/cuda/hip).")
    ap.add_argument("--q_expect_topk", type=int, default=80)
    ap.add_argument("--allow_na_invariance", action="store_true")
    ap.add_argument("--require_invariance", type=str, default="false", help="true/false forwarded to REAL EFE binder (as int 0/1).")

    ap.add_argument("--base_runs_dir", type=str, default="runs/end2end_real_efe_512", help="Base directory for timestamped run dirs.")

    # Optional: record build commands supplied by user
    ap.add_argument("--enable_build_cmds", action="store_true", help="If set, execute and record user-specified build commands.")
    ap.add_argument("--cmds_json", type=str, default=None, help="Optional JSON {name: cmd} to run if --enable_build_cmds.")
    ap.add_argument("--cmd", action="append", default=[], help="Repeatable shell commands to run if --enable_build_cmds.")

    args = ap.parse_args()

    repo_root = abspath(args.repo_root)
    paths = make_paths(repo_root=repo_root, base_runs_dir=args.base_runs_dir)
    ensure_dir(paths.run_dir)
    ensure_dir(paths.audits_dir)
    ensure_dir(paths.status_dir)
    ensure_dir(paths.logs_dir)

    # Discover NPZs
    npz_paths = sorted(glob.glob(args.npz_glob))
    if not npz_paths:
        die(f"No NPZs matched --npz_glob: {args.npz_glob}", code=2)

    # Parse require_invariance safely
    req_inv = (args.require_invariance.strip().lower() == "true")

    # Manifest skeleton
    manifest: Dict[str, Any] = {
        "script": os.path.basename(__file__),
        "STRICT_PP": True,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": paths.run_dir,
        "repo_root": paths.repo_root,
        "git": git_meta(paths.repo_root),

        "args": {
            "npz_glob": args.npz_glob,
            "edges_ms080": abspath(args.edges_ms080),
            "lengths_ms080": abspath(args.lengths_ms080),
            "edges_strong": abspath(args.edges_strong),
            "lengths_strong": abspath(args.lengths_strong),
            "device": args.device,
            "q_expect_topk": int(args.q_expect_topk),
            "allow_na_invariance": bool(args.allow_na_invariance),
            "require_invariance": bool(req_inv),
            "base_runs_dir": abspath(args.base_runs_dir),
        },

        # Two distinct command streams:
        "build_cmds": [],   # optional, user-provided
        "commands": [],     # pipeline internal steps (audits + binders)

        "inputs": {},
        "outputs": {},
        "FINAL": {},
    }

    # Record inputs with hashes (audit-grade)
    manifest["inputs"] = {
        "npz_files": [file_meta(p) for p in npz_paths],
        "edges_ms080": file_meta(args.edges_ms080),
        "lengths_ms080": file_meta(args.lengths_ms080),
        "edges_strong": file_meta(args.edges_strong),
        "lengths_strong": file_meta(args.lengths_strong),
    }
    write_json(paths.manifest_json, manifest)

    # Optional: run build commands (your existing behavior, preserved)
    if args.enable_build_cmds:
        cmds: List[Tuple[str, str]] = []
        if args.cmds_json:
            cj = load_json(abspath(args.cmds_json))
            if not isinstance(cj, dict):
                die("--cmds_json must be a JSON object {name: cmd}")
            for k, v in cj.items():
                cmds.append((str(k), str(v)))
        for i, c in enumerate(args.cmd):
            cmds.append((f"cmd_{i:03d}", c))

        for name, cmd in cmds:
            r = run_cmd(cmd, cwd=paths.repo_root)
            rec = {"name": name, "cmd": cmd, **r}
            manifest["build_cmds"].append(rec)
            write_json(paths.manifest_json, manifest)
            if r["returncode"] != 0:
                die(f"Build command failed: {name}\nCMD: {cmd}\nSTDERR:\n{r['stderr_tail']}", code=10)

    # 1) Math audits
    all_pass_math, audit_jsons = run_math_audits(
        manifest=manifest,
        paths=paths,
        npz_paths=npz_paths,
        edges_ms080=args.edges_ms080,
        lengths_ms080=args.lengths_ms080,
        edges_strong=args.edges_strong,
        lengths_strong=args.lengths_strong,
        device=args.device,
        q_expect_topk=args.q_expect_topk,
    )
    manifest["outputs"]["audit_jsons"] = [file_meta(p) for p in audit_jsons]
    manifest["FINAL"]["ALL_PASS_math_audits"] = bool(all_pass_math)
    write_json(paths.manifest_json, manifest)

    if not all_pass_math:
        die(f"FATAL: math audits did not all PASS. See: {paths.audits_dir}", code=20)

    # 2) Status-chain (tensor3p1 -> realEfe -> final)
    outs = run_real_efe_status_chain(
        manifest=manifest,
        paths=paths,
        allow_na_invariance=bool(args.allow_na_invariance),
        require_invariance=bool(req_inv),
    )

    # Record outputs + hashes
    manifest["outputs"]["status_chain"] = {k: file_meta(v) for k, v in outs.items() if os.path.exists(v)}

    # Final flags (explicit)
    final_obj = load_json(outs["final_out"])
    fk, fv = extract_final_pass_flag(final_obj)
    manifest["FINAL"]["FINAL_PASS_real_EFE_512_STRICT_PP_v2"] = bool(fv is True)
    manifest["FINAL"]["final_key_used"] = fk

    # Also include sha256 for the checked-in witness JSONs used by the chain
    witness = [
        outs["scalar_a2b"],
        outs["offdiag_2p1"],
        outs["bianchi"],
        outs["closure"],
        outs["master"],
        outs["tensor_2p1"],
    ]
    manifest["inputs"]["component_status_jsons"] = [file_meta(p) for p in witness]

    # Write manifest (final)
    write_json(paths.manifest_json, manifest)

    # Final console summary
    print(f"WROTE {paths.manifest_json}")
    print(f"OK: FINAL_PASS_real_EFE_512_STRICT_PP_v2 = {manifest['FINAL']['FINAL_PASS_real_EFE_512_STRICT_PP_v2']}")
    print(f"Manifest: {paths.manifest_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

