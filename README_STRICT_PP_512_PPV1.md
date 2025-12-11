
### Ready-to-commit overview README

````markdown
# STRICT PP GR (512×512, PPV1) — Scalar + Vector Overview

This note summarizes the **STRICT PP** General Relativity status at scale **512×512 (PPV1)** for:

- **Scalar observables** (Shapiro, τ-geometry, perihelion, deflection front), and  
- **Vector-like observables** (azimuthal circulation, frame-dragging proxy, feeder-shear).

The shared constraints:

- Distance and time are purely **Markov / PP**:
  - Doyle / commute-time distances and Markov hop counts.
- Geometry is built from **traces and partial order** only.
- No PDE, no Laplacian/Poisson, no GR ansatz, no regression are allowed in the STRICT PP 512 pipeline.

All coordinates and “angles” are used only as **labels** (for binning or ordering), not as rulers.

---

## 1. Scalar STRICT PP (512×512, PPV1)

The full scalar pipeline (tau-geometry, Shapiro, perihelion, deflection) is documented in:

- `src/gr_strict_pp_scalar/README_STRICT_PP_SCALAR_512_PPV1.md`

That README is the canonical source for:

- Exact scripts and CLIs used at 512×512.
- Input files:
  - Flat and curved PPV1 edge lists (e.g. `edges_ca_v3_flat_512x512_PPV1.txt`,
    `edges_ca_v3_mass_ms080_512x512_PPV1.txt`,
    `edges_ca_v3_strong_pf010_512x512_PPV1.txt`).
  - Trace-weights extracted from CA/trace.
  - Mass / orbit masks in PPV1.
- JSON reports for:
  - **τ-geometry** (multi-shell intersect) as a strict PP scalar EFE-00 proxy.
  - **Shapiro** Markov-τ (sparse) at 512×512.
  - **Perihelion** Markov-dynamic runs where applicable.
  - **Deflection front** tests (including any “indeterminate/flat” outcomes).

In short:

- **Scalar EFE-00** at 512×512 is supported by:
  - τ-geometry built from trace-weights and Markov distances,
  - Shapiro delay expressed as extra Markov commute time,
  - Perihelion dynamics (where run),
  - and documented deflection-front behavior.
- All of this is strictly PP: no injected metric, no PDE smoothing.

For precise PASS/FAIL for each scalar observable, see the scalar README + its JSON reports.

---

## 2. Vector STRICT PP (512×512, PPV1)

Vector-like observables are implemented in **`src/gr_strict_pp_vector`** and are documented in:

- `src/gr_strict_pp_vector/README_STRICT_PP_VECTOR_512_PPV1.md`

### 2.1 Inputs (shared with scalar)

At 512×512 PPV1, vector observables use the same PP inputs as scalar:

- **Edges (PPV1)**  
  - `edges_ca_v3_mass_ms080_512x512_PPV1.txt`  
  - `edges_ca_v3_strong_pf010_512x512_PPV1.txt`  

- **Trace weights** (from CA/trace):
  - `trace_weights_ca_v3_mass_ms080_512x512.txt`
  - `trace_weights_ca_v3_strong_pf010_512x512.txt`

- **PPV1 masks**:
  - `PP_mass_mask_512x512_mass_ms080_PPV1.npy`
  - `PP_mass_mask_512x512_strong_pf010_PPV1.npy`
  - `PP_orbit_ring_mask_512x512_mass_ms080_PPV1.npy`
  - `PP_orbit_ring_mask_512x512_strong_pf010_PPV1.npy`

Again: no PDE, no Laplacian/Poisson, no GR ansatz, no regression.

### 2.2 Repro pipeline (vector suite, 512×512 PPV1)

The vector evidence at 512×512 PPV1 is reproduced by:

```bash
# Build packs

python src/gr_strict_pp_vector/PP_vector_pack_v1.py \
  --case ms080 \
  --azimuthal_json src/gr_strict_pp_vector/PP_vector_azimuthal_512_mass_ms080_bt001.json \
  --frame_dragging_json src/gr_strict_pp_vector/PP_frame_dragging_512_mass_ms080_v1_PPV1.json \
  --feeder_shear_json src/gr_strict_pp_vector/PP_vector_feeder_shear_512_mass_ms080_v1_PPV1.json \
  --output src/gr_strict_pp_vector/PP_vector_pack_512_ms080_bt001_PPV1.json

python src/gr_strict_pp_vector/PP_vector_pack_v1.py \
  --case strong_pf010 \
  --azimuthal_json src/gr_strict_pp_vector/PP_vector_azimuthal_512_strong_pf010_bt001.json \
  --frame_dragging_json src/gr_strict_pp_vector/PP_frame_dragging_512_strong_pf010_v1_PPV1.json \
  --feeder_shear_json src/gr_strict_pp_vector/PP_vector_feeder_shear_512_strong_pf010_v1_PPV1.json \
  --output src/gr_strict_pp_vector/PP_vector_pack_512_strong_pf010_bt001_PPV1.json

# Aggregate packs (suite v2)

python src/gr_strict_pp_vector/PP_vector_suite_v2.py \
  --packs \
    src/gr_strict_pp_vector/PP_vector_pack_512_ms080_bt001_PPV1.json \
    src/gr_strict_pp_vector/PP_vector_pack_512_strong_pf010_bt001_PPV1.json \
  --output src/gr_strict_pp_vector/PP_vector_suite_512_ms080_pf010_bt001_PPV1_v2.json
````

At this scale, the suite reports:

* `ALL_PASS_vector_strict_PP_v2 = True`

while the legacy strict aggregate remains:

* `overall_PASS_vector_strict_PP_v1 = False` (per-pack).

### 2.3 Aggregation keys (v1 vs v2)

Each vector pack exposes two overall keys:

* `overall_PASS_vector_strict_PP_v1`

  * Legacy strict gate (all applicable proxies must pass).
  * Uses:

    * `PASS_azimuthal_or_NA`
    * `PASS_frame_dragging_or_NA`
    * `PASS_feeder_shear_or_NA`

* `overall_PASS_vector_strict_PP_v2`

  * **512-aware aggregation**:

    * At `H ≥ 512`, feeder-shear is kept as **informative but non-gating** due to saturation under PPV1 edges.
    * Gate is:

      * `PASS_azimuthal_or_NA`
      * `PASS_frame_dragging_or_NA`
    * Below 512, v2 falls back to v1.

The suite v2 prefers v2 if present; otherwise falls back to v1.

For 512×512 PPV1:

* `overall_PASS_vector_strict_PP_v1 = False` (both cases)
* `overall_PASS_vector_strict_PP_v2 = True` (both cases)
* `ALL_PASS_vector_strict_PP_v2 = True` (suite)

### 2.4 Operational vector results (concise summary)

Details live in `README_STRICT_PP_VECTOR_512_PPV1.md`; in brief:

1. **Azimuthal circulation (bt001)**
   Files:

   * `PP_vector_azimuthal_512_mass_ms080_bt001.json`
   * `PP_vector_azimuthal_512_strong_pf010_bt001.json`

   Properties:

   * `VECTOR_AZIMUTHAL_FLUX_APPLICABLE = True`
   * `PASS_vector_azimuthal_flux_PP = True` for both ms080 and strong_pf010, with:

     * `bias_threshold = 0.01`
     * Measured biases:

       * `ms080`: `azimuthal_bias ≈ 0.0166`
       * `strong_pf010`: `azimuthal_bias ≈ 0.0123`

   Interpretation:

   * A **nonzero Markov-defined circulation bias** emerges around the mass core at 512×512 PPV1, with ring structure defined strictly by PPV1 orbit masks and Markov transitions.

2. **Frame-dragging proxy (NA at 512)**
   Files:

   * `PP_frame_dragging_512_mass_ms080_v1_PPV1.json`
   * `PP_frame_dragging_512_strong_pf010_v1_PPV1.json`

   Properties:

   * `FRAME_DRAGGING_APPLICABLE = False`
   * `PASS_frame_dragging_PP = null`
   * `n_pairs_used = 0` under current strict PP orbit/threshold choices.

   Interpretation:

   * With current PPV1 edges and masks, the frame-dragging proxy cannot be evaluated robustly at 512×512: the admissibility gates find no usable opposite-band pair set.
   * The suite records this as NA; v2 does not interpret this as positive evidence, but it does not gate on it.

3. **Feeder-shear proxy (saturated at 512)**
   Files:

   * `PP_vector_feeder_shear_512_mass_ms080_v1_PPV1.json`
   * `PP_vector_feeder_shear_512_strong_pf010_v1_PPV1.json`

   Properties:

   * `FEEDER_SHEAR_APPLICABLE = True`
   * `PASS_vector_feeder_shear_PP = False`
   * `best_shear = 0.0` (sectors have essentially equal inflow efficiency).

   Interpretation:

   * In the admissible feeder band at 512×512, almost every node has an outgoing edge that reduces TO-mass Markov distance by 1.
   * This saturates the inflow efficiency and collapses the shear proxy to ~0.
   * v2 therefore treats feeder-shear as **recorded but non-gating** at this scale.

---

## 3. EFE status at 512×512 (STRICT PP)

At 512×512 PPV1, under STRICT PP, the status is:

* **Scalar side**:

  * We have a strong, fully documented **scalar EFE-00–like result**, driven entirely by:

    * CA/trace-derived weights,
    * PPV1 edges,
    * Markov distances and commute times.
  * τ-geometry, Shapiro, and perihelion (where run) form an internally consistent scalar GR picture without Euclidean injection or PDEs.

* **Vector side**:

  * We have **positive vector-like evidence** in the form of nonzero Markov circulation bias around mass at 512×512 PPV1, and documented inflow saturation.
  * Frame-dragging is currently NA at this scale and edge policy; feeder-shear is saturated and non-gating.
  * The suite’s 512-aware aggregation (`overall_PASS_vector_strict_PP_v2`, `ALL_PASS_vector_strict_PP_v2`) reflects this conservative reading.

What we **do not** yet claim:

* A full spatial-tensor Einstein equation `G_{μν} = κ T_{μν}` from STRICT PP at 512×512:

  * No operational derivation of off-diagonal components (`G_{0i}`, `G_{ij}`) from these vector proxies alone.
  * No fully constructed Lorentzian metric with frame-dragging terms derived purely from the PPV1 Markov data.

In other words:

* At STRICT PP 512×512, we have:

  * A **production-grade scalar EFE-00 story**,
  * A **first, conservative vector-like layer** (azimuthal bias + inflow saturation),
  * And a clearly documented boundary: spatial-tensor EFE remains an open target, not a claim.

---

## 4. Next technical directions (non-binding)

The following are **possible** next steps; they are recorded here only to orient future work, not as promises:

1. **Strict-PP deflection front at 512**

   * Revisit step-window policies and finite-time criteria to obtain either a robust PASS or a clearly justified, resolution-independent non-PASS.

2. **Improved vector observables**

   * Refine orbit definitions and proxies to:

     * Make frame-dragging applicable at 512×512 under STRICT PP, or
     * Design an alternative vector observable that can be more directly related to `G_{0i}`.

3. **Vector → EFE mapping design**

   * Formalize how Markov-defined circulation and feeder structures can be mapped to effective momentum flux (`T_{0i}`) and off-diagonal geometry (`G_{0i}`), still respecting STRICT PP and trace-only rules.

This overview is intended as a **stable snapshot** of the STRICT PP 512×512 PPV1 state, not as a placeholder; subsequent work should extend or refine it, not contradict it.

````

---

