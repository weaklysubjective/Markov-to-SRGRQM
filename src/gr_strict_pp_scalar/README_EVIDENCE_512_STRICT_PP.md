# STRICT PP 512 Evidence Checkpoint (v1)

This checkpoint is designed to let any reviewer reproduce the core validation with ONE command.

## What this evidence runner does (and does NOT do)

### Does:
1) Discovers existing artifacts:
   `PP_T00_flow_fixedGeom_lengths_512_{case}_qPermTopK_s{seed}_STRICT_PP_v1.npz`
2) Runs `PP_math_audit_harness_v3.py` on each discovered artifact:
   - finite checks, nonneg checks, sum-to-1 checks (where applicable)
   - q-topk expectation check (q field matches expected topk)
   - edges index sanity (in-range)
   - lengths finiteness + nonneg
3) Ingests an existing adversarial status JSON:
   `PP_EFE_adversarial_status_512_STRICT_PP_v8*.json`
4) Emits a single consolidated JSON:
   `runs/evidence/evidence_512_STRICT_PP_v1.json`
   with `ALL_PASS_EVIDENCE_512_STRICT_PP_v1`.

### Does NOT:
- Generate any NPZs, edges, lengths, or new flows.
- Re-run the adversarial sweep itself (it only ingests the already-produced status JSON).

## Directory conventions assumed
- `RUNBASE=src/gr_strict_pp_align/runs/finalleg_traceOnly_q9995_v2_mix0.7`
  contains:
  - edges_undirected_ms080_512x512_E2.txt
  - edges_undirected_strong_pf010_512x512_E2.txt
- `src/gr_strict_pp_geom/` contains:
  - PP_lengths_doyle_edges_512_ms080_STRICT_PP_v1.npz
  - PP_lengths_doyle_edges_512_strong_pf010_STRICT_PP_v1.npz
- `PASSDIR` contains the flow artifacts for the seeds you audited.

## One-command reproduction

```bash
export RUNBASE="src/gr_strict_pp_align/runs/finalleg_traceOnly_q9995_v2_mix0.7"
export PASSDIR="runs/adversarial_knobs_v8_search/inner/trial_seed0_pass_try"

python src/gr_strict_pp_scalar/PP_evidence_runner_512_STRICT_PP_v1.py \
  --passdir "$PASSDIR" \
  --runbase "$RUNBASE" \
  --device gpu \
  --expected_seeds "0-4" \
  --skip_existing \
  --output runs/evidence/evidence_512_STRICT_PP_v1.json

jq '.PASS' runs/evidence/evidence_512_STRICT_PP_v1.json

