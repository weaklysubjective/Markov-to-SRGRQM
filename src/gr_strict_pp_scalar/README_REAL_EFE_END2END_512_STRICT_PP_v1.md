````markdown
# REAL EFE (512, STRICT_PP) — End-to-End Reproduction + Manifest (v1)

This document describes the **end-to-end reproduction driver** for the 512×512 STRICT_PP “REAL EFE” result, including the **run manifest** semantics and how an external reviewer can verify the run.

**Scope:** This is an *end-to-end within the current repo artifacts* driver: it re-runs the verification chain (audits + binders) and writes a manifest that records inputs, outputs, executed commands, hashes, and final PASS gates.

---

## What “end-to-end” means (in this repo)

The end-to-end runner executes the following, in a new timestamped run directory under `runs/end2end_real_efe_512/`:

1) **Math audits** for a selected witness set of NPZ files (both ms080 and strong_pf010), using:
   - `src/gr_strict_pp_scalar/PP_math_audit_harness_v3.py`

2) **Tensor 3+1 binder** (status-only aggregation of:
   - master status
   - tensor 2p1 status
   - invariance status (optional)
   ) using:
   - `src/gr_strict_pp_scalar/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.py`

3) **REAL EFE binder** (status-only aggregation of:
   - scalar A2b status
   - offdiag 2p1 status
   - bianchi status
   - tensor closure status
   - tensor 3p1 binder output
   ) using:
   - `src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v2.py`

4) **Final wrapper** that writes a single canonical PASS flag and hashes the REAL EFE binder JSON using:
   - `src/gr_strict_pp_scalar/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.py`

The run writes a **manifest**:
- `runs/end2end_real_efe_512/<UTCSTAMP>/MANIFEST_end2end_real_efe_512_STRICT_PP_v1.json`

---

## What “end-to-end” does NOT mean (important)

This driver does **not** rebuild the entire physics pipeline from raw traces unless you add those generators explicitly.

Specifically, this end-to-end driver **does not**:
- regenerate traces
- rebuild E2 edges
- recompute Doyle/Steiner lengths
- regenerate the NPZ witness files (flows / T00 / etc.)
- rerun any heavy kernel that produced the precomputed **component status JSONs**

Instead, it assumes those inputs exist (and records their **sha256** so reviewers can confirm they match the claimed run).

---

## One command

### Recommended entrypoint (wrapper)
```bash
bash src/gr_strict_pp_scalar/run_REAL_EFE_end2end_512_STRICT_PP_v1.sh
````

### Direct entrypoint (Python driver)

Example (matches the typical STRICT_PP 512 setup you’ve been using):

```bash
python src/gr_strict_pp_scalar/PP_run_REAL_EFE_end2end_512_STRICT_PP_v1.py \
  --repo_root . \
  --npz_glob "runs/adversarial_knobs_v8_search/inner/trial_seed0_pass_try/PP_T00_flow_fixedGeom_lengths_512_*_qPermTopK_s*_STRICT_PP_v1.npz" \
  --edges_ms080 "src/gr_strict_pp_align/runs/finalleg_traceOnly_q9995_v2_mix0.7/edges_undirected_ms080_512x512_E2.txt" \
  --lengths_ms080 "src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_ms080_STRICT_PP_v1.npz" \
  --edges_strong "src/gr_strict_pp_align/runs/finalleg_traceOnly_q9995_v2_mix0.7/edges_undirected_strong_pf010_512x512_E2.txt" \
  --lengths_strong "src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_strong_pf010_STRICT_PP_v1.npz" \
  --device gpu \
  --q_expect_topk 80 \
  --allow_na_invariance
```

Notes:

* `--allow_na_invariance` means the tensor 3p1 binder will accept `PASS_invariance=null` as allowed (instead of treating it as a failure).
* The runner forwards `require_invariance` as an integer (`0`/`1`) to `PP_EFE_real_EFE_status_512_STRICT_PP_v2.py` via `--require_invariance 0|1`.

---

## Inputs assumed precomputed

The driver expects these to already exist:

### A) Component status JSONs (witness metrics + gates)

These are the “real content” metrics/gates that the binders propagate:

* `src/gr_strict_pp_scalar/PP_EFE_scalar_A2b_status_512_STRICT_PP_v1.json`
* `src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v3.json`
* `src/gr_strict_pp_scalar/PP_EFE_bianchi_status_512_STRICT_PP_v1.json`
* `src/gr_strict_pp_scalar/PP_EFE_tensor_closure_status_512_STRICT_PP_v1.json`
* `src/gr_strict_pp_scalar/PP_EFE_master_status_512_STRICT_PP_v3.json`
* `src/gr_strict_pp_scalar/PP_EFE_tensor_2p1_status_512_STRICT_PP_v1.json`

### B) Graph artifacts

* E2 edges (undirected):

  * `--edges_ms080  <path>/edges_undirected_ms080_512x512_E2.txt`
  * `--edges_strong <path>/edges_undirected_strong_pf010_512x512_E2.txt`

* Doyle/Steiner lengths NPZ:

  * `--lengths_ms080  src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_ms080_STRICT_PP_v1.npz`
  * `--lengths_strong src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_strong_pf010_STRICT_PP_v1.npz`

### C) NPZ witness set

* Provided via `--npz_glob ...` (commonly the adversarial-passing witness set):

  * `PP_T00_flow_fixedGeom_lengths_512_*_qPermTopK_s*_STRICT_PP_v1.npz`

These NPZs are audited for sanity/consistency via the math audit harness and recorded in the manifest.

---

## Outputs produced (per run)

Each end-to-end run writes a new timestamped directory:

* `runs/end2end_real_efe_512/<UTCSTAMP>/`

Key outputs:

### A) Math audit JSONs

* `runs/end2end_real_efe_512/<UTCSTAMP>/audits/audit_<npz_basename>_math_v3.json`

### B) Status chain outputs (generated by this run)

* `runs/end2end_real_efe_512/<UTCSTAMP>/status/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.json`
* `runs/end2end_real_efe_512/<UTCSTAMP>/status/PP_EFE_real_EFE_status_512_STRICT_PP_v2.json`
* `runs/end2end_real_efe_512/<UTCSTAMP>/status/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.json`

### C) Manifest

* `runs/end2end_real_efe_512/<UTCSTAMP>/MANIFEST_end2end_real_efe_512_STRICT_PP_v1.json`

---

## What is verified (end-to-end claim)

This end-to-end driver verifies:

1. **Math audit PASS** across the NPZ witness set for both ms080 and strong_pf010:

   * `ALL_PASS_math_audits == true` in the manifest `FINAL`

2. **Tensor 3p1 binder PASS**:

   * `ALL_PASS_EFE_tensor_3p1_strict_PP_512_v1 == true` (in the tensor_3p1 binder output JSON)

3. **REAL EFE binder PASS**:

   * `ALL_PASS_real_EFE_strict_PP_512_v2 == true` (in the REAL EFE binder output JSON)

4. **Final wrapper PASS**:

   * `FINAL_PASS_real_EFE_512_STRICT_PP_v2 == true` (in the final wrapper output JSON)

---

## Manifest: meaning and schema

The manifest is a “receipt” that captures:

### Top-level fields

* `STRICT_PP`: boolean
* `script`: driver script name
* `created_utc`: ISO timestamp
* `repo_root`: absolute path
* `run_dir`: absolute path to this run directory

### `args`

Echo of the CLI arguments used for the run:

* edges/lengths paths
* npz_glob
* device
* q_expect_topk
* allow_na_invariance
* require_invariance

### `commands`

A list of executed commands, each with:

* `name`: logical label
* `cmd`: full command line string
* `returncode`
* `elapsed_s`
* `stdout_tail`, `stderr_tail`

This is the authoritative trace of what was executed during the run.

### `inputs`

Structured inventory of inputs (with size/mtime/sha256), including:

* component status JSONs
* edges
* lengths
* NPZ witness files

### `outputs`

Structured inventory of outputs (with size/mtime/sha256), including:

* audit JSONs
* status chain outputs

### `FINAL`

Final gates and the key used:

* `ALL_PASS_math_audits`
* `FINAL_PASS_real_EFE_512_STRICT_PP_v2`
* `final_key_used` (should be `FINAL_PASS_real_EFE_512_STRICT_PP_v2`)

### `git`

* `git_head`
* `git_dirty`

Reviewers should treat a dirty working tree as: “the run occurred with uncommitted changes”; the manifest still captures hashes of artifacts used.

---

## Verification checklist (reviewer workflow)

Assume a run directory:

```bash
RUN="runs/end2end_real_efe_512/<UTCSTAMP>"
```

### 1) Confirm final PASS gates

```bash
jq '.FINAL' "$RUN/MANIFEST_end2end_real_efe_512_STRICT_PP_v1.json"
```

You should see:

* `ALL_PASS_math_audits: true`
* `FINAL_PASS_real_EFE_512_STRICT_PP_v2: true`

### 2) Confirm final wrapper JSON is PASS

```bash
jq '.FINAL_PASS_real_EFE_512_STRICT_PP_v2, .PASS.FINAL_PASS_real_EFE_512_STRICT_PP_v2' \
  "$RUN/status/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.json"
```

Expected: `true` (either at top-level or inside `.PASS`).

### 3) Confirm REAL EFE binder PASS and input wiring

```bash
jq '.PASS, .inputs' "$RUN/status/PP_EFE_real_EFE_status_512_STRICT_PP_v2.json"
```

Expected: `ALL_PASS_real_EFE_strict_PP_512_v2: true` and correct paths under `.inputs`.

### 4) Confirm tensor 3p1 binder PASS

```bash
jq '.PASS, .resolved, .inputs, .failures' \
  "$RUN/status/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.json"
```

Expected: `ALL_PASS_EFE_tensor_3p1_strict_PP_512_v1: true`.

### 5) Confirm the executed commands list is populated

```bash
jq '.commands | length' "$RUN/MANIFEST_end2end_real_efe_512_STRICT_PP_v1.json"
jq -r '.commands[] | [.name,.returncode,.elapsed_s] | @tsv' \
  "$RUN/MANIFEST_end2end_real_efe_512_STRICT_PP_v1.json" | column -t
```

### 6) Confirm sha256 hashes correspond to on-disk files (spot check)

Pick a file path and compare to manifest:

```bash
FILE="src/gr_strict_pp_scalar/PP_EFE_scalar_A2b_status_512_STRICT_PP_v1.json"
jq -r --arg f "$FILE" '
  (.inputs.component_status_jsons[]? | select(.path==$f) | .sha256) // empty
' "$RUN/MANIFEST_end2end_real_efe_512_STRICT_PP_v1.json"

python - <<'PY'
import hashlib,sys
p=sys.argv[1]
h=hashlib.sha256(open(p,'rb').read()).hexdigest()
print(h)
PY "$FILE"
```

---

## Relationship to “Gμν = κ Tμν”

Within this STRICT_PP operational/graph framework, the repo’s “REAL EFE” claim is validated by **component suites** whose status JSONs contain the metrics/gates (correlations, residual/ratio tests, divergence stats, closure checks, invariance logic where applicable), and a binder chain that produces a single PASS verdict for the full set.

This end-to-end driver:

* re-checks the NPZ witness set via math audits,
* re-runs the binder chain deterministically,
* records everything in a manifest for independent verification.

If you later want a **full rebuild-from-raw-traces** single-command pipeline, that is a separate “build chain” which must explicitly add trace→edges/lengths/NPZ generation steps and record them similarly.

---

## Troubleshooting

### `--require_invariance` parsing

`PP_EFE_real_EFE_status_512_STRICT_PP_v2.py` expects `--require_invariance` as an integer (0/1).
The end-to-end driver should pass `--require_invariance 0` unless you intentionally require invariance.

### `PASS_invariance` is null

If invariance is not supplied, `PASS_invariance` may be `null`.
Use `--allow_na_invariance` for the tensor 3p1 binder if null is acceptable in this release.

---

## Files related to this doc

* Driver:

  * `src/gr_strict_pp_scalar/PP_run_REAL_EFE_end2end_512_STRICT_PP_v1.py`

* Wrapper:

  * `src/gr_strict_pp_scalar/run_REAL_EFE_end2end_512_STRICT_PP_v1.sh`

* Status-chain-only doc (binder reproduction):

  * `src/gr_strict_pp_scalar/README_REAL_EFE_STATUS_512_STRICT_PP.md`

* This doc (end-to-end + manifest):

  * `src/gr_strict_pp_scalar/README_REAL_EFE_END2END_512_STRICT_PP_v1.md`

