---

## Proposed tightened README (overwrite content)

````markdown
# STRICT PP Scalar Gravity — 512×512 (PPV1)

This folder contains the **STRICT PP scalar** pipeline at **512×512** for:
- **mass_ms080**
- **strong_pf010**

All results here are **Markov + trace only**:
- **No Laplacian field injection**
- **No Poisson**
- **No PDE**
- **No Euclidean radius**
- **No GR ansatz**
- **No regression**
- **Combined experiences are allowed and should be used** (STRICT PP policy)

Time and distance observables are **Markov-counter / hitting-time** and **Doyle/Steiner effective-resistance** where explicitly stated.

---

## 0) One-command run (recommended)

This is the **current canonical** 512 runner:

```bash
bash run_scalar_512_ppv1.sh
````

It should:

1. ensure / generate trace weights for each case
2. build curved edges (flat edges are reused)
3. build masks (mass + orbit ring)
4. run **τ-geometry v4 multi-shell**
5. run **Shapiro τ v2 sparse** (case-dependent pair policy)
6. run **deflection front v3 sparse-only** using `--src_auto ring_mid`
7. emit per-case JSONs and a small summary

---

## 1) Core artifacts (512×512)

### Flat baseline

* `edges_ca_v3_flat_512x512_PPV1.txt`

### mass_ms080

* `trace_weights_ca_v3_mass_ms080_512x512.txt`
* `edges_ca_v3_mass_ms080_512x512_PPV1.txt`
* `PP_mass_mask_512x512_mass_ms080_PPV1.npy`
* `PP_orbit_ring_mask_512x512_mass_ms080_PPV1.npy`

### strong_pf010

* `trace_weights_ca_v3_strong_pf010_512x512.txt`
* `edges_ca_v3_strong_pf010_512x512_PPV1.txt`
* `PP_mass_mask_512x512_strong_pf010_PPV1.npy`
* `PP_orbit_ring_mask_512x512_strong_pf010_PPV1.npy`

---

## 2) Manual step-by-step (exact recent 512 path)

### 2.1 Generate trace weights + edges + masks

Use the 512 “min” orchestrator:

```bash
python src/gr_strict_pp_vector/run_pf010_512_ppv1_min.py --case mass_ms080 --H 512 --W 512
python src/gr_strict_pp_vector/run_pf010_512_ppv1_min.py --case strong_pf010 --H 512 --W 512
```

Under the hood this runs:

* trace GPU:

  ```bash
  python ca_trace_to_poset_v3_trace_gpu_v1.py \
    --X 262144 --T 2000000 --eps-trace 0.1 --seed 0 --device auto \
    --out-trace trace_weights_ca_v3_<case>_512x512.txt \
    --report CA_POSV3_TRACE_<case>_512x512_eps010_GPU.json
  ```
* edges:

  ```bash
  python src/gr_strict_pp_scalar/PP_build_edges_from_trace_v1.py \
    --trace_weights trace_weights_ca_v3_<case>_512x512.txt \
    --case <case> --H 512 --W 512 \
    --neighbors 4 --rule uphill_topk --k_out 2 \
    --edges_out edges_ca_v3_<case>_512x512_PPV1.txt \
    --report_out src/gr_strict_pp_scalar/edges_<case>_512x512_PPV1.json
  ```
* masks:

  ```bash
  python src/gr_strict_pp_scalar/make_pfv1_masks.py \
    --edges edges_ca_v3_<case>_512x512_PPV1.txt \
    --trace_weights trace_weights_ca_v3_<case>_512x512.txt \
    --H 512 --W 512 \
    --mass_topk 80 \
    --auto_band --min_ring_nodes 12 \
    --mass_out  PP_mass_mask_512x512_<case>_PPV1.npy \
    --orbit_out PP_orbit_ring_mask_512x512_<case>_PPV1.npy \
    --report_out src/gr_strict_pp_scalar/masks_<case>_512x512_PPV1.json
  ```

---

## 3) τ-geometry (STRICT PP)

This is the **production** scalar geometry check at 512:

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

Expected **STRICT PP pass flags**:

* `PASS_G00_sign_attractive = true`
* `PASS_kappa_median_sign = true`

Notes:

* This is **Markov τ-geometry only**:
  `G00_k(i)=tau_curved_k/tau_flat_k - 1`, `kappa_med=G00_med/T00`.

---

## 4) Shapiro delay (STRICT PP, sparse)

Use the sparse implementation at 512:

```bash
python src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_v2_sparse.py \
  --edges_flat   edges_ca_v3_flat_512x512_PPV1.txt \
  --edges_curved edges_ca_v3_<case>_512x512_PPV1.txt \
  --H 512 --W 512 \
  --src_through <auto_or_manual> --dst_through <auto_or_manual> \
  --src_around  <auto_or_manual> --dst_around  <auto_or_manual> \
  --tol 1e-5 --maxiter 200000 \
  --output src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_512_<case>_PPV1.json
```

**Important recent lesson**:

* Shapiro PASS is **pair-sensitive** at 512.
* Some ring-derived around pairs can fail even when the scalar τ-geometry is healthy.
* The runner should treat Shapiro as:

  * **Required** for strong_pf010 if the auto-pair policy is stable,
  * **Advisory/soft** for ms080 until we lock a robust pair-selection rule.

(We can harden this in the runner by sampling ring pairs and selecting a “flat-R-through matched” around pair.)

---

## 5) Deflection (STRICT PP, sparse-only — canonical)

This is now the **correct** 512 deflection script:

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

* **Deletes dense loaders**
* Uses sparse `P`
* Evolves row-distributions via **`P^T @ v`**
* Adds `--src_auto ring_mid` using the orbit ring
* Has a tighter PASS/INDETERMINATE policy

Your latest runs show:

* `PASS_deflection_markov_front_PP = True` for **mass_ms080**
* `PASS_deflection_markov_front_PP = True` for **strong_pf010**

---

## 6) Current PASS policy at 512 (PPV1)

We consider the **scalar 512 strict-PP story** healthy when:

### Required

1. **τ-geometry v4**:

   * `PASS_G00_sign_attractive = true`
   * `PASS_kappa_median_sign = true`

2. **Deflection v3 sparse-only**:

   * `PASS_deflection_markov_front_PP = true`

### Advisory / case-dependent

3. **Shapiro v2 sparse**:

   * PASS if pair-selection stable
   * otherwise flagged for pair-policy hardening

---

## 7) Suggested git freeze set (512)

```bash
git add \
  trace_weights_ca_v3_mass_ms080_512x512.txt \
  trace_weights_ca_v3_strong_pf010_512x512.txt \
  edges_ca_v3_mass_ms080_512x512_PPV1.txt \
  edges_ca_v3_strong_pf010_512x512_PPV1.txt \
  src/gr_strict_pp_scalar/edges_mass_ms080_512x512_PPV1.json \
  src/gr_strict_pp_scalar/edges_strong_pf010_512x512_PPV1.json \
  PP_mass_mask_512x512_mass_ms080_PPV1.npy \
  PP_orbit_ring_mask_512x512_mass_ms080_PPV1.npy \
  PP_mass_mask_512x512_strong_pf010_PPV1.npy \
  PP_orbit_ring_mask_512x512_strong_pf010_PPV1.npy \
  src/gr_strict_pp_scalar/masks_mass_ms080_512x512_PPV1.json \
  src/gr_strict_pp_scalar/masks_strong_pf010_512x512_PPV1.json \
  src/gr_strict_pp_scalar/PP_markov_tau_geometry_512_mass_ms080_topk500.json \
  src/gr_strict_pp_scalar/PP_markov_tau_geometry_512_strong_pf010_topk500.json \
  src/gr_strict_pp_scalar/PP_deflection_front_512_mass_ms080_PPV1.json \
  src/gr_strict_pp_scalar/PP_deflection_front_512_strong_pf010_PPV1.json \
  src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_512_mass_ms080_PPV1.json \
  src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_512_strong_pf010_PPV1.json \
  src/gr_strict_pp_scalar/PP_deflection_markov_front_PP_v3_sparse_only.py \
  run_scalar_512_ppv1.sh \
  src/gr_strict_pp_scalar/README_STRICT_PP_SCALAR_512_PPV1.md
```

Then:

```bash
git commit -m "STRICT PP scalar 512 PPV1: ms080+pf010 tau-geometry, deflection v3 sparse, Shapiro sparse"
```

---

## 8) Next hardening item

**Shapiro 512 pair-policy**:

* Add an internal ring-pair sampler:

  * match `R_through_flat` within tolerance
  * choose around pair with similar flat resistance
  * then require curved Δτ to exceed flat Δτ

This will remove the current “PASS depends on which ring pair you pick” issue.

---

## 9) Summary

At 512×512 PPV1 we now have:

* **STRICT PP τ-geometry sign-consistent** (ms080 + strong_pf010)
* **STRICT PP deflection PASS** using **v3 sparse-only** with `ring_mid`
* **STRICT PP Shapiro** working with sparse solver but still needs a robust
  **pair-selection policy** at scale

The canonical “today” entrypoint is:

```bash
bash run_scalar_512_ppv1.sh
```

```

---

