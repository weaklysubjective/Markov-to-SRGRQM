
````markdown
# STRICT_PP Vector / Off-Diag (512×512) — Repro Notes

This document covers the **vector / off-diagonal** STRICT_PP checks at **512×512**, specifically:

1) **Ring circulation / momentum flux** (per-case artifact; `PP_offdiag_ring_momentum_flux_v3.py`)
2) **Offdiag 2+1 status binder** (aggregates cases; `PP_EFE_offdiag_2p1_status_512_STRICT_PP_v2.py`)
3) How these feed the **master v3** → **tensor_3p1** → **strongfield** → **real_EFE** binders

No new observables are introduced by binders; they only read PASS flags from existing artifacts.

---

## 0) Inputs (canonical)

### Canonical E2 edges (512×512)
You should have:
- `edges_ca_v3_flat_512x512_PPV1_E2.txt`
- `edges_ca_v3_mass_ms080_512x512_PPV1_E2.txt`
- `edges_ca_v3_strong_pf010_512x512_PPV1_E2.txt`

Quick check:
```bash
ls -lh edges_ca_v3_*_512x512_PPV1_E2.txt
````

### Masks (example: strong_pf010)

* Mass mask:

  * `./PP_mass_mask_512x512_strong_pf010_PPV1.npy`
* Orbit/ring mask:

  * `./PP_orbit_ring_mask_512x512_strong_pf010_PPV1.npy`

(Analogous ms080 512×512 PPV1 masks must exist for ms080 runs.)

---

## 1) Ring circulation / momentum flux (per-case)

### Important: this script has NO `--device` flag

Run it **without** `--device`. Backend selection (or CPU execution) is internal by design.

Script:

* `src/gr_strict_pp_vector/PP_offdiag_ring_momentum_flux_v3.py`

It writes:

* `.status.APPLICABLE`
* `.status.PASS_offdiag_ring_momentum_flux_PP_v3`
* `.flux.J_net_over_abs`

### 1.1 strong_pf010 run

```bash
python src/gr_strict_pp_vector/PP_offdiag_ring_momentum_flux_v3.py \
  --edges_curved edges_ca_v3_mass_ms080_512x512_PPV1_E2.txt \
  --mass_mask  ./PP_mass_mask_512x512_mass_ms080_PPV1.npy \
  --orbit_mask ./PP_orbit_ring_mask_512x512_mass_ms080_PPV1.npy \
  --H 512 --W 512 \
  --ring_mode quantile_band \
  --edges_ring_ref edges_ca_v3_flat_512x512_PPV1_E2.txt \
  --output src/gr_strict_pp_scalar/PP_offdiag_ring_momentum_flux_ms080_512_STRICT_PP_v7.json

```

Sanity:

```bash
jq -r '.status.APPLICABLE, .status.PASS_offdiag_ring_momentum_flux_PP_v3, .flux.J_net_over_abs' \
  src/gr_strict_pp_scalar/PP_offdiag_ring_momentum_flux_strong_pf010_512_STRICT_PP_v7.json
```

### 1.2 ms080 run (fill in your ms080 mask filenames)

```bash
python src/gr_strict_pp_vector/PP_offdiag_ring_momentum_flux_v3.py \
  --edges_curved edges_ca_v3_mass_ms080_512x512_PPV1_E2.txt \
  --mass_mask  ./PP_mass_mask_512x512_mass_ms080_PPV1.npy \
  --orbit_mask ./PP_orbit_ring_mask_512x512_mass_ms080_PPV1.npy \
  --H 512 --W 512 \
  --ring_mode quantile_band \
  --edges_ring_ref edges_ca_v3_flat_512x512_PPV1_E2.txt \
  --output src/gr_strict_pp_scalar/PP_offdiag_ring_momentum_flux_ms080_512_STRICT_PP_v7.json
```

Sanity:

```bash
jq -r '.status.APPLICABLE, .status.PASS_offdiag_ring_momentum_flux_PP_v3, .flux.J_net_over_abs' \
  src/gr_strict_pp_scalar/PP_offdiag_ring_momentum_flux_ms080_512_STRICT_PP_v7.json
```

### Optional: combined-experience weights

If you have combined-experience weights you want the tool to use, pass:

* `--trace_weights <PATH>`

(Do not invent/derive weights here; use your canonical combined-experience tensor.)

---

## 2) Offdiag 2+1 status binder (aggregates cases)

Binder:

* `src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v2.py`

Run:

```bash
python src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v2.py \
  --output src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v2.json
```

### Important: this binder does NOT have a `.PASS` top-level key

PASS is stored at:

* `.offdiag_flags.APPLICABLE`
* `.offdiag_flags.PASS`

Sanity:

```bash
jq -r '.offdiag_flags.APPLICABLE, .offdiag_flags.PASS' \
  src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v2.json
```

---

## 3) End-to-end chain verification (master → tensor_3p1 → strongfield → real_EFE)

After offdiag is PASS, the standard chain should remain PASS:

```bash
python src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.py
python src/gr_strict_pp_scalar/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.py
python src/gr_strict_pp_scalar/PP_EFE_strongfield_suite_status_512_STRICT_PP_v1.py
python src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v1.py

jq -r '.PASS.ALL_PASS_real_EFE_strict_PP_512_v1' \
  src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v1.json
```

Expected: `true`

---

## 4) Output hygiene / repo policy

* Do **not** commit generated heavy artifacts (`.npy`, `.npz`, `.txt`) unless explicitly intended.
* Status JSONs are lightweight and can be committed **if** your workflow treats them as part of the reproducible report trail.

---

## 5) Common schema keys (avoid jq “null” confusion)

* Flux artifact:

  * PASS: `.status.PASS_offdiag_ring_momentum_flux_PP_v3`
  * Metric: `.flux.J_net_over_abs`

* Offdiag binder:

  * PASS: `.offdiag_flags.PASS`

* Tensor 3p1 binder:

  * PASS bit: `.PASS.ALL_PASS_EFE_tensor_3p1_strict_PP_512_v1`

* Real EFE binder:

  * PASS bit: `.PASS.ALL_PASS_real_EFE_strict_PP_512_v1`

```

```

