#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs/status

python src/gr_strict_pp_scalar/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.py \
  --master_status_json src/gr_strict_pp_scalar/PP_EFE_master_status_512_STRICT_PP_v3.json \
  --tensor_2p1_status_json src/gr_strict_pp_scalar/PP_EFE_tensor_2p1_status_512_STRICT_PP_v1.json \
  --allow_na_invariance \
  --output runs/status/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.json

python src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v2.py \
  --scalar_a2b  src/gr_strict_pp_scalar/PP_EFE_scalar_A2b_status_512_STRICT_PP_v1.json \
  --offdiag     src/gr_strict_pp_scalar/PP_EFE_offdiag_2p1_status_512_STRICT_PP_v3.json \
  --bianchi     src/gr_strict_pp_scalar/PP_EFE_bianchi_status_512_STRICT_PP_v1.json \
  --closure     src/gr_strict_pp_scalar/PP_EFE_tensor_closure_status_512_STRICT_PP_v1.json \
  --tensor3p1   runs/status/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.json \
  --output      runs/status/PP_EFE_real_EFE_status_512_STRICT_PP_v2.json

python src/gr_strict_pp_scalar/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.py \
  --real_efe_json runs/status/PP_EFE_real_EFE_status_512_STRICT_PP_v2.json \
  --output runs/status/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.json

echo "==== REAL EFE (binder) ===="
jq '.PASS, .inputs' runs/status/PP_EFE_real_EFE_status_512_STRICT_PP_v2.json

echo "==== REAL EFE (final) ===="
cat runs/status/PP_EFE_real_EFE_final_status_512_STRICT_PP_v2.json

