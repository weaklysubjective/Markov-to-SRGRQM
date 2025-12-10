# STRICT PP Scalar Gravity — 512×512 (PPV1)

This folder contains the **STRICT PP scalar** pipeline at **512×512** for:
- **mass_ms080**
- **strong_pf010**

This is a **Markov + trace only** derivation of scalar GR observables at scale.

## STRICT PP commitments (scalar)

All results here are compliant with STRICT PP constraints:

- No Laplacian field injection
- No Poisson
- No PDE
- No Euclidean radius
- No GR ansatz
- No regression

**Combined experiences are allowed and should be used** (STRICT PP policy).

**Time and distance observables** are operational:
- Markov counters / hitting times
- Doyle/Steiner effective-resistance where explicitly stated by the script

---

## 0) Canonical entrypoint (one command)

Default policy:
- Runs **τ-geometry** + **deflection**
- Runs **Shapiro** with **ring-local auto pairs**
- **Shapiro is not required** for `ALL_PASS` at 512 unless explicitly requested

```bash
bash src/gr_strict_pp_scalar/run_scalar_512_ppv1.sh
```

Require Shapiro in the ALL_PASS gate:

```bash
bash src/gr_strict_pp_scalar/run_scalar_512_ppv1.sh --require_shapiro
```

Skip Shapiro (smoke):

```bash
bash src/gr_strict_pp_scalar/run_scalar_512_ppv1.sh --skip_shapiro --cases mass_ms080
```

### Optional: stabilize Shapiro pair sensitivity via sweep

If your wrapper supports the env policy:

```bash
SHAPIRO_POLICY=sweep \
bash src/gr_strict_pp_scalar/run_scalar_512_ppv1.sh --require_shapiro
```

Optional extra sweep breadth:

```bash
SHAPIRO_POLICY=sweep \
SHAPIRO_SWEEP_N_SEEDS=16 \
bash src/gr_strict_pp_scalar/run_scalar_512_ppv1.sh --require_shapiro
```

---

## 1) STRICT PP scalar 512 input chain

**Scalar 512 PPV1 pipeline (canonical):**

**combined experiences → case trace NPZ → trace weights → PPV1 edges →**
**mass core (top-k) → flat-graph hop-shell ring → observables**
(τ-geometry, deflection, Shapiro)

Key ring definition:

> **ring = hop-shell band around the trace-defined mass core on the flat PPV1 graph.**

---

## 2) Core canonical artifacts (512×512)

### Flat baseline

- `edges_ca_v3_flat_512x512_PPV1.txt`

### mass_ms080

- `trace_weights_ca_v3_mass_ms080_512x512.txt`
- `edges_ca_v3_mass_ms080_512x512_PPV1.txt`
- `PP_mass_mask_512x512_mass_ms080_PPV1.npy`
- `PP_orbit_ring_mask_512x512_mass_ms080_PPV1.npy`

### strong_pf010

- `trace_weights_ca_v3_strong_pf010_512x512.txt`
- `edges_ca_v3_strong_pf010_512x512_PPV1.txt`
- `PP_mass_mask_512x512_strong_pf010_PPV1.npy`
- `PP_orbit_ring_mask_512x512_strong_pf010_PPV1.npy`

---

## 3) What is the case trace NPZ?

The case trace NPZ is the **raw, combined-experience trace intensity per node**
**before** it is exported to a `.txt` weights file.

It should contain one array of length `N = H×W` (or reshapeable to `H×W`).

Recommended canonical naming:

- `trace_ca_v3_mass_ms080_512x512_PPV1.npz`
- `trace_ca_v3_strong_pf010_512x512_PPV1.npz`

Inside each NPZ, store one of:
- `trace` as shape `(N,)`, or
- `trace_grid` as shape `(H, W)`.

### If you only have the weights txt (fallback)

You can wrap the existing txt weights into a compatible NPZ:

```bash
python - <<'PY'
import numpy as np

def txt_to_npz(txt_path, npz_path):
    w = np.loadtxt(txt_path).astype(np.float64)
    if w.ndim > 1:
        w = w[:, 0]
    np.savez_compressed(npz_path, trace=w)

txt_to_npz("trace_weights_ca_v3_mass_ms080_512x512.txt",
           "trace_ca_v3_mass_ms080_512x512_PPV1.npz")

txt_to_npz("trace_weights_ca_v3_strong_pf010_512x512.txt",
           "trace_ca_v3_strong_pf010_512x512_PPV1.npz")

print("Wrote two trace NPZ files.")
PY
```

---

## 4) Trace weights extraction (NPZ → txt)

This is the corrected, explicit invocation:

```bash
python src/gr_strict_pp_scalar/PP_extract_trace_weights_any_v1.py \
  --in_trace trace_ca_v3_mass_ms080_512x512_PPV1.npz \
  --H 512 --W 512 \
  --case mass_ms080 \
  --out_txt trace_weights_ca_v3_mass_ms080_512x512.txt \
  --report_out src/gr_strict_pp_scalar/trace_weights_mass_ms080_512_report.json \
  --normalize
```

```bash
python src/gr_strict_pp_scalar/PP_extract_trace_weights_any_v1.py \
  --in_trace trace_ca_v3_strong_pf010_512x512_PPV1.npz \
  --H 512 --W 512 \
  --case strong_pf010 \
  --out_txt trace_weights_ca_v3_strong_pf010_512x512.txt \
  --report_out src/gr_strict_pp_scalar/trace_weights_strong_pf010_512_report.json \
  --normalize
```

Notes:
- `--normalize` is recommended for stable cross-run comparability.

---

## 5) PPV1 edges (from trace weights)

Curved PPV1 edges are built from trace weights under the PPV1 rule-set:

```bash
python src/gr_strict_pp_scalar/PP_build_edges_from_trace_v1.py \
  --trace_weights trace_weights_ca_v3_mass_ms080_512x512.txt \
  --case mass_ms080 --H 512 --W 512 \
  --neighbors 4 --rule uphill_topk --k_out 2 \
  --edges_out edges_ca_v3_mass_ms080_512x512_PPV1.txt \
  --report_out src/gr_strict_pp_scalar/edges_mass_ms080_512x512_PPV1.json
```

```bash
python src/gr_strict_pp_scalar/PP_build_edges_from_trace_v1.py \
  --trace_weights trace_weights_ca_v3_strong_pf010_512x512.txt \
  --case strong_pf010 --H 512 --W 512 \
  --neighbors 4 --rule uphill_topk --k_out 2 \
  --edges_out edges_ca_v3_strong_pf010_512x512_PPV1.txt \
  --report_out src/gr_strict_pp_scalar/edges_strong_pf010_512x512_PPV1.json
```

Flat baseline is reused:

- `edges_ca_v3_flat_512x512_PPV1.txt`

---

## 6) Mass core + orbit ring masks (STRICT PP)

We define the mass core as **top-k** trace-weight nodes.
We define the orbit ring using a **graph-only hop-shell** around that core
on the **flat PPV1** graph.

Ring definition:

> **ring = hop-shell band around the trace-defined mass core on the flat PPV1 graph.**

Canonical commands used in the recent 512 runs:

```bash
python src/gr_strict_pp_scalar/PP_build_mass_orbit_masks_from_trace_v2.py \
  --case mass_ms080 \
  --H 512 --W 512 \
  --mass_topk 500 \
  --ring_min_hops 2 --ring_max_hops 6
```

```bash
python src/gr_strict_pp_scalar/PP_build_mass_orbit_masks_from_trace_v2.py \
  --case strong_pf010 \
  --H 512 --W 512 \
  --mass_topk 500 \
  --ring_min_hops 2 --ring_max_hops 6
```

Expected outputs:

- `PP_mass_mask_512x512_mass_ms080_PPV1.npy`
- `PP_orbit_ring_mask_512x512_mass_ms080_PPV1.npy`
- `PP_mass_mask_512x512_strong_pf010_PPV1.npy`
- `PP_orbit_ring_mask_512x512_strong_pf010_PPV1.npy`

---

## 7) The three scalar observables (512)

### 7.1 τ-geometry (STRICT PP, multi-shell)

```bash
python src/gr_strict_pp_scalar/PP_markov_tau_geometry_v4_multiShellIntersect.py \
  --edges_flat    edges_ca_v3_flat_512x512_PPV1.txt \
  --edges_curved  edges_ca_v3_<case>_512x512_PPV1.txt \
  --trace_weights trace_weights_ca_v3_<case>_512x512.txt \
  --H 512 --W 512 \
  --mass_topk 500 \
  --shell_bands 2:3 3:4 4:5 5:6 \
  --ht_tol 1e-5 \
  --ht_maxiter 40000 \
  --report src/gr_strict_pp_scalar/PP_markov_tau_geometry_512_<case>_topk500.json
```

Expected pass flags:
- `PASS_G00_sign_attractive = true`
- `PASS_kappa_median_sign = true`

---

### 7.2 Deflection (STRICT PP, sparse-only — canonical)

```bash
python src/gr_strict_pp_scalar/PP_deflection_markov_front_PP_v3_sparse_only.py \
  --edges_flat   edges_ca_v3_flat_512x512_PPV1.txt \
  --edges_curved edges_ca_v3_<case>_512x512_PPV1.txt \
  --mass_mask    PP_mass_mask_512x512_<case>_PPV1.npy \
  --ring_mask    PP_orbit_ring_mask_512x512_<case>_PPV1.npy \
  --H 512 --W 512 \
  --src_auto ring_mid \
  --steps 800 \
  --ht_tol 1e-5 --ht_maxiter 20000 \
  --output src/gr_strict_pp_scalar/PP_deflection_front_512_<case>_PPV1.json
```

This version:
- Uses sparse Markov kernels only (no dense NxN)
- Evolves row-distributions via `P^T @ v`
- Auto-selects source via the ring
- Has a tighter PASS/INDETERMINATE policy

---

### 7.3 Shapiro delay (STRICT PP, sparse)

Manual invocation:

```bash
python src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_v2_sparse.py \
  --edges_flat   edges_ca_v3_flat_512x512_PPV1.txt \
  --edges_curved edges_ca_v3_<case>_512x512_PPV1.txt \
  --H 512 --W 512 \
  --src_through <int> --dst_through <int> \
  --src_around  <int> --dst_around  <int> \
  --tol 1e-5 --maxiter 200000 \
  --output src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_512_<case>_PPV1.json
```

At 512:
- Shapiro can be **pair-sensitive**.
- The runner supports:
  - ring-local auto pairs by default
  - optional sweep stabilization via `SHAPIRO_POLICY=sweep`

---

## 8) 512 runner (ms080 + strong_pf010)

The canonical scalar 512 orchestrator:

```bash
python src/gr_strict_pp_scalar/PP_scalar_512_runner_v1.py \
  --H 512 --W 512 \
  --cases mass_ms080,strong_pf010 \
  --steps 800 \
  --mass_topk 500 \
  --shell_bands 2:3,3:4,4:5,5:6 \
  --ht_tol 1e-5 --ht_maxiter 40000 \
  --defl_ht_tol 1e-5 --defl_ht_maxiter 20000 \
  --src_auto ring_mid \
  --shapiro_tol 1e-5 --shapiro_maxiter 200000 \
  --output src/gr_strict_pp_scalar/PP_scalar_512_suite_report.json
```

Require Shapiro:

```bash
python src/gr_strict_pp_scalar/PP_scalar_512_runner_v1.py \
  --H 512 --W 512 \
  --cases mass_ms080,strong_pf010 \
  --require_shapiro \
  --output src/gr_strict_pp_scalar/PP_scalar_512_suite_report_REQUIRE_SHAPIRO.json
```

---

## 9) PASS policy (scalar 512 PPV1)

We consider the scalar 512 strict PP story healthy when:

### Required
1) **τ-geometry v4**:
   - attractive `G00` sign
   - consistent `kappa` sign

2) **Deflection v3 sparse-only**

### Optional / hardening target
3) **Shapiro v2 sparse**
   - required only when `--require_shapiro` is set
   - recommended to use sweep policy at 512 for stability

---

## 10) Summary

At 512×512 PPV1 we now have:

- **STRICT PP τ-geometry sign-consistent** (mass_ms080 + strong_pf010)
- **STRICT PP deflection PASS** using **v3 sparse-only** + `ring_mid`
- **STRICT PP Shapiro** supported with:
  - ring-local auto pairs
  - optional sweep stabilization at scale

Primary entrypoint:

```bash
bash src/gr_strict_pp_scalar/run_scalar_512_ppv1.sh
```

Optional stability enhancement:

```bash
SHAPIRO_POLICY=sweep \
bash src/gr_strict_pp_scalar/run_scalar_512_ppv1.sh --require_shapiro
```

---

## Scope note

This README documents the **scalar** STRICT PP 512 PPV1 pipeline only.

The **vector** pipeline should be documented in a separate README
to keep inputs, observables, and PASS criteria unambiguous.

