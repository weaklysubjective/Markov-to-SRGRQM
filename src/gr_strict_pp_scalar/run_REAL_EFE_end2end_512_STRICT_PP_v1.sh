#!/usr/bin/env bash
set -euo pipefail

# One-command, end-to-end runner.
# Default is audit+bind using existing artifacts.
# You control what gets audited via NPZ_GLOB.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Edit these defaults if desired:
RUNBASE="src/gr_strict_pp_align/runs/finalleg_traceOnly_q9995_v2_mix0.7"
PASSDIR="runs/adversarial_knobs_v8_search/inner/trial_seed0_pass_try"

EDG_MS="$RUNBASE/edges_undirected_ms080_512x512_E2.txt"
LEN_MS="src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_ms080_STRICT_PP_v1.npz"
EDG_STR="$RUNBASE/edges_undirected_strong_pf010_512x512_E2.txt"
LEN_STR="src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_strong_pf010_STRICT_PP_v1.npz"

# Audit these NPZs (discover-only; no missing seed spam):
NPZ_GLOB="$PASSDIR/PP_T00_flow_fixedGeom_lengths_512_*_qPermTopK_s*_STRICT_PP_v1.npz"

python "$REPO_ROOT/src/gr_strict_pp_scalar/PP_run_REAL_EFE_end2end_512_STRICT_PP_v1.py" \
  --repo_root "$REPO_ROOT" \
  --npz_glob "$NPZ_GLOB" \
  --edges_ms080 "$EDG_MS" \
  --lengths_ms080 "$LEN_MS" \
  --edges_strong "$EDG_STR" \
  --lengths_strong "$LEN_STR" \
  --device gpu \
  --q_expect_topk 80 \
  --allow_na_invariance


