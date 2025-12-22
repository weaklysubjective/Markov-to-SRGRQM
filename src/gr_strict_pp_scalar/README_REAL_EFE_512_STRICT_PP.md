# REAL EFE (512, STRICT_PP) — Finalization Notes

This README documents the *full* “REAL EFE” claim for the 512×512 STRICT_PP pipeline:
scalar + vector/offdiag + tensor closure / Bianchi (as implemented in the repo’s existing suites/binders).

Important:
- This is distinct from the **geometry-operator freeze** README:
  `src/gr_strict_pp_scalar/README_GEOM_OPERATOR_512_STRICT_PP.md`
- The geometry-operator README certifies only: lengths→curvature operator + adversarial robustness.
- This README certifies the *full* claim (where the suite’s REAL-EFE status JSON is the source of truth).

## Canonical status artifact (source of truth)

The suite/binder produces:

- `src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v1.json`

This file is the canonical “REAL EFE” status report used to assert PASS/FAIL.

## Freeze binder (one-line final stamp)

To produce a single, stable “final stamp” JSON that depends only on the canonical status file:

```bash
python src/gr_strict_pp_scalar/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.py \
  --real_efe_json src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v1.json \
  --output src/gr_strict_pp_scalar/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.json

jq '.FINAL_PASS_real_EFE_512_STRICT_PP_v2, .pass_flags_used, .reasoning' \
  src/gr_strict_pp_scalar/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.json

Optional: add an input hash for release-grade freezing:

python src/gr_strict_pp_scalar/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.py \
  --hash \
  --real_efe_json src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v1.json \
  --output src/gr_strict_pp_scalar/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2_hashed.json

Folder naming does not imply subsystem scope

Some orchestrator/binder scripts live under src/gr_strict_pp_scalar/ even when they validate
vector/tensor artifacts produced elsewhere. The source of truth is always the status JSON(s),
not the folder name.

Relationship to geometry-operator freeze

The geometry-operator freeze is certified by:

src/gr_strict_pp_scalar/PP_EFE_geometry_operator_final_status_512_STRICT_PP_v1.json

and documented in:

src/gr_strict_pp_scalar/README_GEOM_OPERATOR_512_STRICT_PP.md

REAL EFE depends on additional subsystems beyond that operator.
