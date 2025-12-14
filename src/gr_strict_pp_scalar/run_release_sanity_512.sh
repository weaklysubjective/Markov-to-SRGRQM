#!/usr/bin/env bash
set -euo pipefail

python src/gr_strict_pp_scalar/PP_EFE_structural_status_512_STRICT_PP_v1.py \
  --output src/gr_strict_pp_scalar/PP_EFE_structural_status_512_STRICT_PP_v1.json

python src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.py \
  --structural_json src/gr_strict_pp_scalar/PP_EFE_structural_status_512_STRICT_PP_v1.json \
  --output src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.json

python - <<'PY'
import json
p="src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v1.json"
j=json.load(open(p))
k="overall_PASS_EFE_scalar_vector_offdiag_deflection_closure_bianchi_strict_PP_512_v3"
ok=bool(j.get(k))
print("SANITY:", k, "=", ok)
print("deflection:", j.get("deflection_flags",{}).get("APPLICABLE"), j.get("deflection_flags",{}).get("PASS"))
print("structural:", j.get("structural_flags",{}).get("APPLICABLE"), j.get("structural_flags",{}).get("PASS"))
raise SystemExit(0 if ok else 2)
PY
