#!/usr/bin/env bash
# run_scalar_512_ppv1.sh
#
# STRICT PP Scalar 512 one-command runner wrapper.
# Calls:
#   src/gr_strict_pp_scalar/PP_scalar_512_runner_v1.py
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

    # advanced passthrough knobs if you want them later
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

CMD=(
  "$PY_BIN" "src/gr_strict_pp_scalar/PP_scalar_512_runner_v1.py"
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

echo "[OK] Done."

