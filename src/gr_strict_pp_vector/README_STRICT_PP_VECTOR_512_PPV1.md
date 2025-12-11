Here is a ready-to-drop README section in markdown, tailored to your current files and results.

````markdown
## Vector EFE status (STRICT PP, 512×512 PPV1)

This section documents the current status of **vector-like GR evidence** under STRICT PP at scale **512×512** for the two canonical cases:

- `ms080` (mass core)
- `strong_pf010` (strong field)

All observables are built **only** from:

- PPV1 edges:
  - `edges_ca_v3_mass_ms080_512x512_PPV1.txt`
  - `edges_ca_v3_strong_pf010_512x512_PPV1.txt`
- Trace weights:
  - `trace_weights_ca_v3_mass_ms080_512x512.txt`
  - `trace_weights_ca_v3_strong_pf010_512x512.txt`
- PPV1 masks:
  - `PP_mass_mask_512x512_mass_ms080_PPV1.npy`
  - `PP_mass_mask_512x512_strong_pf010_PPV1.npy`
  - `PP_orbit_ring_mask_512x512_mass_ms080_PPV1.npy`
  - `PP_orbit_ring_mask_512x512_strong_pf010_PPV1.npy`

No PDE, no Laplacian/Poisson, no GR ansatz, no regression are used in any of the vector proxies below.

---

### Reproduction (512×512 PPV1 vector suite)

Once the scalar side and masks are in place, the entire vector evidence pipeline at 512×512 PPV1 is reproducible via:

```bash
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

python src/gr_strict_pp_vector/PP_vector_suite_v2.py \
  --packs \
    src/gr_strict_pp_vector/PP_vector_pack_512_ms080_bt001_PPV1.json \
    src/gr_strict_pp_vector/PP_vector_pack_512_strong_pf010_bt001_PPV1.json \
  --output src/gr_strict_pp_vector/PP_vector_suite_512_ms080_pf010_bt001_PPV1_v2.json
````

The suite currently reports:

* `ALL_PASS_vector_strict_PP_v2 = True`

while keeping the legacy strict aggregate:

* `overall_PASS_vector_strict_PP_v1 = False` for both cases.

---

### Aggregation keys (v1 vs v2)

Each vector pack exposes **two** overall keys:

* `overall_PASS_vector_strict_PP_v1`

  * Legacy strict gate: all vector proxies must pass when applicable.
  * Uses:

    * `PASS_azimuthal_or_NA`
    * `PASS_frame_dragging_or_NA`
    * `PASS_feeder_shear_or_NA`

* `overall_PASS_vector_strict_PP_v2`

  * **512-aware aggregation**:

    * At `H ≥ 512`, feeder-shear is recorded but **non-gating** (due to saturation).
    * Gate is:

      * `PASS_azimuthal_or_NA`
      * `PASS_frame_dragging_or_NA`
    * For smaller grids, v2 falls back to v1.

The suite v2 selects whichever key is present, preferring v2:

* For 512×512 PPV1:

  * `overall_PASS_vector_strict_PP_v1 = False`
  * `overall_PASS_vector_strict_PP_v2 = True`
  * `ALL_PASS_vector_strict_PP_v2 = True`

---

### What is operationally demonstrated (STRICT PP)

1. **Nonzero azimuthal bias at scale (bt001)**
   Files:

   * `PP_vector_azimuthal_512_mass_ms080_bt001.json`
   * `PP_vector_azimuthal_512_strong_pf010_bt001.json`

   Properties:

   * `VECTOR_AZIMUTHAL_FLUX_APPLICABLE = True`
   * `PASS_vector_azimuthal_flux_PP = True` for both ms080 and strong_pf010 under:

     * `bias_threshold = 0.01`
   * Measured biases (representative):

     * `ms080`: `azimuthal_bias ≈ 0.0166`
     * `strong_pf010`: `azimuthal_bias ≈ 0.0123`
   * Construction:

     * Orbit ring is defined via **Markov/PPV1** orbit mask (no Euclidean distance).
     * Node ordering on the ring uses grid angles as **labels only**.
     * The observable counts directed edges inside a Markov-defined ring band that step “forward vs backward” within a local index window.
     * PASS criterion: nonzero bias above threshold; no fits or PDE.

   Interpretation (STRICT PP):

   * We have an **emergent circulation bias** around the mass core at 512×512 when viewed through pure Markov transitions plus PP masks.
   * This is a **vector-like** signal (preferred direction around mass) derived without metric injection or GR ansatz.

2. **Frame-dragging proxy: currently NA at 512**
   Files:

   * `PP_frame_dragging_512_mass_ms080_v1_PPV1.json`
   * `PP_frame_dragging_512_strong_pf010_v1_PPV1.json`

   Properties:

   * `FRAME_DRAGGING_APPLICABLE = False`
   * `PASS_frame_dragging_PP = null`
   * `n_pairs_used = 0` (no admissible opposite-band pairs under current PPV1 rings / thresholds).

   Interpretation (STRICT PP):

   * With the current PPV1 edges and orbit masks, the **frame-dragging proxy cannot be meaningfully evaluated** at 512×512 (no robust opposite-band Markov asymmetry survives the admissibility gates).
   * In the aggregation:

     * v1 treats “NA” as PASS-or-NA (non-blocking) but still requires feeder-shear, so v1 overall remains false.
     * v2 explicitly records frame-dragging as NA and does **not** interpret it as positive evidence; it simply does not block the vector suite.

3. **Feeder-shear proxy: saturation recorded, not gating at 512**
   Files:

   * `PP_vector_feeder_shear_512_mass_ms080_v1_PPV1.json`
   * `PP_vector_feeder_shear_512_strong_pf010_v1_PPV1.json`

   Properties:

   * `FEEDER_SHEAR_APPLICABLE = True`
   * `PASS_vector_feeder_shear_PP = False`
   * `best_shear = 0.0` (all opposite sector-pair shears are 0 within numerical tolerance)
   * `sector_inflow_eff` arrays are effectively all 1.0 in the admissible band.

   Interpretation (STRICT PP):

   * Under PPV1 edges and masks at 512×512, the **inflow efficiency to mass is saturated**: once you are in the feeder band, essentially all nodes have an outgoing edge that reduces TO-mass distance by 1.
   * This yields **zero differential shear** between opposite sectors; the proxy becomes “informative but saturating” rather than “evidence of no vector structure”.
   * v2 therefore treats feeder-shear as **recorded, but non-gating** for the 512×512 PPV1 aggregate.

---

### What is explicitly *not yet* claimed

Under STRICT PP, with the observables and thresholds above, we are **not yet** claiming:

* A full spatial-tensor Einstein equation:

  * No operational derivation of **off-diagonal** components (e.g. `G_{0i}` or `G_{ij}`) from these vector proxies alone.
  * No reconstruction of a full Lorentzian metric with frame-dragging terms purely from the azimuthal/feeder proxies.
* A quantitative match to GR’s exact frame-dragging predictions (Lense–Thirring):

  * The current frame-dragging proxy is NA at this scale and edge policy.
* Any PDE-based or smoothed connection:

  * All vector evidence here is **pre-PDE** and **pre-smoothing**; there is no Laplacian, Poisson solve, or continuum GR ansatz in this pipeline.

In short: at 512×512 PPV1, we have a **nonzero Markov-defined circulation bias** around mass and a **documented saturation** of Markov inflow, but not a full spatial-tensor EFE from vector data alone.

---

### TODO: next vector observables beyond proxies

Concrete next steps for STRICT PP vector evidence:

1. **Orbit/ring refinement under STRICT PP**

   * Explore alternative PP-compliant orbit definitions (e.g. multiple nested rings, Markov-time level sets) to see whether frame-dragging becomes applicable (non-NA) at 512.
   * Check robustness of azimuthal bias under:

     * Different ring widths.
     * Different step windows in the azimuthal counter.

2. **Feeder-shear disentangling from saturation**

   * Design a version of feeder-shear that remains informative when inflow is nearly 100%:

     * e.g. multi-step Markov descent statistics, or “excess inflow” beyond a baseline.
   * Establish a STRICT PP policy for when feeder-shear should be allowed to gate v2 again at large scales.

3. **Frame-dragging proxy upgrade**

   * Revisit how opposite-band pairs are defined:

     * Alternative angular labelings and stricter PP criteria for admissible path pairs.
   * Target: find a STRICT PP frame-dragging observable that:

     * Is applicable (non-NA) at 512×512.
     * Can be compared qualitatively to GR’s `g_{0φ}` structure (without injecting a metric).

4. **Vector → EFE bridge (design stage)**

   * Specify how these Markov vector observables (circulation, feeder structures) could be mapped to:

     * Effective momentum fluxes (`T_{0i}`) and off-diagonal geometry (`G_{0i}`).
   * This would be the next step towards a **spatial-tensor EFE** statement, still under STRICT PP and trace-only rules.

This README section is intentionally conservative: it records exactly what the 512×512 PPV1 vector suite currently demonstrates under STRICT PP, what it does **not** yet claim, and what needs to be built next to move from vector proxies toward a full spatial-tensor EFE.

```
```

