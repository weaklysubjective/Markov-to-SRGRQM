# Canonical Geometry Operator Freeze (512, STRICT_PP)

This folder contains the canonical, STRICT-PP “geometry operator” stack for 512×512:
Doyle/Steiner edge-lengths → curvature operator (Ricci/volume-distance proxy) and the associated adversarial robustness checks.

This README documents the *geometry operator finalization* step only.
It is intentionally distinct from “REAL EFE” (scalar+vector+tensor) binders.

## What is finalized here

**Canonical operator inputs (artifacts):**
- Curvature proxy:
  - `src/gr_strict_pp_geom/PP_RicciVolDist_R_512_strong_pf010_STRICT_PP_v1.npz`
  - `src/gr_strict_pp_geom/PP_RicciVolDist_R_512_ms080_STRICT_PP_v1.npz`
- Doyle/Steiner lengths:
  - `src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_strong_pf010_STRICT_PP_v1.npz`
  - `src/gr_strict_pp_geom/PP_lengths_doyle_edges_512_ms080_STRICT_PP_v1.npz`
- Matter proxy (trace-weighted):
  - `src/gr_strict_pp_scalar/PP_Tmunu_tracew_edges_512_strong_pf010_STRICT_PP_v3.npz`
  - `src/gr_strict_pp_scalar/PP_Tmunu_tracew_edges_512_ms080_STRICT_PP_v3.npz`
- Fixed-geometry matched flows (used by adversarial v6):
  - `src/gr_strict_pp_align/runs/finalleg_twgeom_q9995_v1/PP_T00_flow_fixedGeom_lengths_512_strong_pf010_STRICT_PP_v1.npz`
  - `src/gr_strict_pp_align/runs/finalleg_twgeom_q9995_v1/PP_T00_flow_fixedGeom_lengths_512_ms080_STRICT_PP_v1.npz`

**Canonical robustness check (result):**
- Adversarial suite:
  - `src/gr_strict_pp_scalar/PP_EFE_adversarial_status_512_STRICT_PP_v6_diag.json`
  - Expected: `ALL_PASS_adversarial_512_STRICT_PP_v6 = true`

**Final “freeze” binder (this is what we check-in as the canonical stamp):**
- Script:
  - `src/gr_strict_pp_scalar/PP_EFE_geometry_operator_final_status_512_STRICT_PP_v1.py`
- Output JSON:
  - `src/gr_strict_pp_scalar/PP_EFE_geometry_operator_final_status_512_STRICT_PP_v1.json`
  - Expected: `FINAL_PASS_EFE_geometry_operator_512_STRICT_PP_v1 = true`

## How to reproduce the FINAL status JSON

### A) Run the final status binder
```bash
python src/gr_strict_pp_scalar/PP_EFE_geometry_operator_final_status_512_STRICT_PP_v1.py \
  --output src/gr_strict_pp_scalar/PP_EFE_geometry_operator_final_status_512_STRICT_PP_v1.json

jq '.FINAL_PASS_EFE_geometry_operator_512_STRICT_PP_v1, .adversarial_v6.crossFlow, .adversarial_v6.qPermTopK, .flows_present' \
  src/gr_strict_pp_scalar/PP_EFE_geometry_operator_final_status_512_STRICT_PP_v1.json

