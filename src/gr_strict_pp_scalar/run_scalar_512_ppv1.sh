#!/usr/bin/env bash
# run_scalar_512_ppv1.sh
#
# STRICT PP Scalar 512 one-command runner wrapper.
#
# Default behavior (UNCHANGED):
#   Calls:
#     src/gr_strict_pp_scalar/PP_scalar_512_runner_v1.py
#
# Optional behavior via ENV:
#   If SHAPIRO_POLICY is set (e.g., "sweep"), we will prefer:
#     src/gr_strict_pp_scalar/PP_scalar_512_runner_v2.py
#   and pass:
#     --shapiro_policy <value>
#
# Defaults:
#   cases = mass_ms080,strong_pf010
#   H=W=512
#   output = src/gr_strict_pp_scalar/PP_scalar_512_suite_report.json
#
# Options:
#   --cases <csv>
#   --require_shapiro
#   --skip_shapiro
#   --output <path>
#   --H <int> --W <int>
#   --steps <int>
#   --mass_topk <int>
#
# Notes:
# - No heredocs.
# - Uses python to summarize JSON (no jq dependency).

set -euo pipefail

PY_BIN="${PY_BIN:-python}"

# Optional ENV to switch runner:
#   SHAPIRO_POLICY=sweep|auto|manual|skip
SHAPIRO_POLICY="${SHAPIRO_POLICY:-}"

# Sweep knobs (only used if v2 + sweep)
SHAPIRO_SWEEP_N_SEEDS="${SHAPIRO_SWEEP_N_SEEDS:-16}"
SHAPIRO_SWEEP_AROUND_DELTA="${SHAPIRO_SWEEP_AROUND_DELTA:-10}"
SHAPIRO_SWEEP_THROUGH_DELTA="${SHAPIRO_SWEEP_THROUGH_DELTA:-10}"
SHAPIRO_SWEEP_PREFER_PASS="${SHAPIRO_SWEEP_PREFER_PASS:-0}"

# Manual pair ENV (only used if v2 + manual)
SHAPIRO_SRC_THROUGH="${SHAPIRO_SRC_THROUGH:-}"
SHAPIRO_DST_THROUGH="${SHAPIRO_DST_THROUGH:-}"
SHAPIRO_SRC_AROUND="${SHAPIRO_SRC_AROUND:-}"
SHAPIRO_DST_AROUND="${SHAPIRO_DST_AROUND:-}"

H=512
W=512
CASES="mass_ms080,strong_pf010"
OUTPUT="src/gr_strict_pp_scalar/PP_scalar_512_suite_report.json"
REQUIRE_SHAPIRO=0
SKIP_SHAPIRO=0
STEPS=800
MASS_TOPK=500

# Pass-through advanced params (optional)
HT_TOL="1e-5"
HT_MAXITER="40000"
DEFL_HT_TOL="1e-5"
DEFL_HT_MAXITER="20000"
SHAPIRO_TOL="1e-5"
SHAPIRO_MAXITER="200000"
SRC_AUTO="ring_mid"
SHELL_BANDS="2:3,3:4,4:5,5:6"

usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --cases <csv>              e.g. mass_ms080,strong_pf010"
  echo "  --H <int>                  default 512"
  echo "  --W <int>                  default 512"
  echo "  --steps <int>              deflection steps (default 800)"
  echo "  --mass_topk <int>          tau-geometry mass_topk (default 500)"
  echo "  --require_shapiro          include Shapiro in ALL_PASS gate"
  echo "  --skip_shapiro             skip Shapiro entirely"
  echo "  --output <path>            suite JSON output"
  echo ""
  echo "Env:"
  echo "  PY_BIN=<python>            default 'python'"
  echo "  SHAPIRO_POLICY=sweep|auto|manual|skip"
  echo "    If set, wrapper will prefer runner v2 and pass --shapiro_policy."
  echo ""
  echo "  Sweep-only env:"
  echo "    SHAPIRO_SWEEP_N_SEEDS=<int>          default 16"
  echo "    SHAPIRO_SWEEP_AROUND_DELTA=<int>    default 10"
  echo "    SHAPIRO_SWEEP_THROUGH_DELTA=<int>   default 10"
  echo "    SHAPIRO_SWEEP_PREFER_PASS=1         optional"
  echo ""
  echo "  Manual-only env (requires all 4):"
  echo "    SHAPIRO_SRC_THROUGH=<int>"
  echo "    SHAPIRO_DST_THROUGH=<int>"
  echo "    SHAPIRO_SRC_AROUND=<int>"
  echo "    SHAPIRO_DST_AROUND=<int>"
}

# Simple arg parse
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cases) CASES="$2"; shift 2;;
    --H) H="$2"; shift 2;;
    --W) W="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --mass_topk) MASS_TOPK="$2"; shift 2;;
    --require_shapiro) REQUIRE_SHAPIRO=1; shift 1;;
    --skip_shapiro) SKIP_SHAPIRO=1; shift 1;;
    --output) OUTPUT="$2"; shift 2;;

    # advanced passthrough knobs
    --ht_tol) HT_TOL="$2"; shift 2;;
    --ht_maxiter) HT_MAXITER="$2"; shift 2;;
    --defl_ht_tol) DEFL_HT_TOL="$2"; shift 2;;
    --defl_ht_maxiter) DEFL_HT_MAXITER="$2"; shift 2;;
    --shapiro_tol) SHAPIRO_TOL="$2"; shift 2;;
    --shapiro_maxiter) SHAPIRO_MAXITER="$2"; shift 2;;
    --src_auto) SRC_AUTO="$2"; shift 2;;
    --shell_bands) SHELL_BANDS="$2"; shift 2;;

    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ "$REQUIRE_SHAPIRO" -eq 1 && "$SKIP_SHAPIRO" -eq 1 ]]; then
  echo "[ERROR] Cannot set both --require_shapiro and --skip_shapiro"
  exit 1
fi

RUNNER_V1="src/gr_strict_pp_scalar/PP_scalar_512_runner_v1.py"
RUNNER_V2="src/gr_strict_pp_scalar/PP_scalar_512_runner_v2.py"

RUNNER="$RUNNER_V1"
RUNNER_MODE="v1_default"

# If SHAPIRO_POLICY is set, prefer v2 if present
if [[ -n "$SHAPIRO_POLICY" ]]; then
  if [[ -f "$RUNNER_V2" ]]; then
    RUNNER="$RUNNER_V2"
    RUNNER_MODE="v2_env_shapiro_policy"
  else
    echo "[WARN] SHAPIRO_POLICY set but v2 runner not found: $RUNNER_V2"
    echo "[WARN] Falling back to v1."
  fi
fi

CMD=(
  "$PY_BIN" "$RUNNER"
  "--H" "$H" "--W" "$W"
  "--cases" "$CASES"
  "--steps" "$STEPS"
  "--mass_topk" "$MASS_TOPK"
  "--shell_bands" "$SHELL_BANDS"
  "--ht_tol" "$HT_TOL"
  "--ht_maxiter" "$HT_MAXITER"
  "--defl_ht_tol" "$DEFL_HT_TOL"
  "--defl_ht_maxiter" "$DEFL_HT_MAXITER"
  "--src_auto" "$SRC_AUTO"
  "--shapiro_tol" "$SHAPIRO_TOL"
  "--shapiro_maxiter" "$SHAPIRO_MAXITER"
  "--output" "$OUTPUT"
)

if [[ "$REQUIRE_SHAPIRO" -eq 1 ]]; then
  CMD+=("--require_shapiro")
fi
if [[ "$SKIP_SHAPIRO" -eq 1 ]]; then
  CMD+=("--skip_shapiro")
fi

# v2-only shapiro policy extras
if [[ "$RUNNER" == "$RUNNER_V2" ]]; then
  CMD+=("--shapiro_policy" "$SHAPIRO_POLICY")

  if [[ "$SHAPIRO_POLICY" == "sweep" ]]; then
    CMD+=(
      "--shapiro_sweep_n_seeds" "$SHAPIRO_SWEEP_N_SEEDS"
      "--shapiro_sweep_around_delta" "$SHAPIRO_SWEEP_AROUND_DELTA"
      "--shapiro_sweep_through_delta" "$SHAPIRO_SWEEP_THROUGH_DELTA"
    )
    if [[ "$SHAPIRO_SWEEP_PREFER_PASS" == "1" ]]; then
      CMD+=("--shapiro_sweep_prefer_pass")
    fi
  elif [[ "$SHAPIRO_POLICY" == "manual" ]]; then
    if [[ -z "$SHAPIRO_SRC_THROUGH" || -z "$SHAPIRO_DST_THROUGH" || -z "$SHAPIRO_SRC_AROUND" || -z "$SHAPIRO_DST_AROUND" ]]; then
      echo "[ERROR] SHAPIRO_POLICY=manual requires all 4 env vars:"
      echo "  SHAPIRO_SRC_THROUGH, SHAPIRO_DST_THROUGH, SHAPIRO_SRC_AROUND, SHAPIRO_DST_AROUND"
      exit 1
    fi
    CMD+=(
      "--shapiro_src_through" "$SHAPIRO_SRC_THROUGH"
      "--shapiro_dst_through" "$SHAPIRO_DST_THROUGH"
      "--shapiro_src_around"  "$SHAPIRO_SRC_AROUND"
      "--shapiro_dst_around"  "$SHAPIRO_DST_AROUND"
    )
  fi
fi

echo "[MODE] $RUNNER_MODE"
echo "[RUN] ${CMD[*]}"
"${CMD[@]}"

# Summary printer (no jq)
"$PY_BIN" - <<PY
import json, sys, os
path = "${OUTPUT}"
if not os.path.exists(path):
    print("[ERROR] suite report missing:", path)
    sys.exit(1)
with open(path, "r") as f:
    rep = json.load(f)

print("")
print("=== STRICT PP Scalar 512 Summary ===")
print("H,W,N =", rep.get("H"), rep.get("W"), rep.get("N"))
print("cases  =", ", ".join(rep.get("cases", {}).keys()))
print("require_shapiro =", rep.get("require_shapiro"))
print("skip_shapiro    =", rep.get("skip_shapiro"))
print("ALL_PASS        =", rep.get("ALL_PASS"))

cases = rep.get("cases", {})
for name, c in cases.items():
    pc = c.get("PASS_components", {})
    print("")
    print(f"[{name}] PASS =", c.get("PASS"))
    print("  tau_geometry =", pc.get("PASS_tau_geometry"))
    print("  deflection   =", pc.get("PASS_deflection"))
    print("  shapiro      =", pc.get("PASS_shapiro"))

print("")
print("report =", path)
PY

echo "[OK]"

