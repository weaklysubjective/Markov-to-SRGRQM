#!/usr/bin/env bash
set -euo pipefail

python src/gr_strict_pp_scalar/PP_EFE_real_EFE_status_512_STRICT_PP_v2.py --output src/gr_strict_pp_scalar/PP_EFE_master_status_512_STRICT_PP_v3.json

python src/gr_strict_pp_scalar/PP_EFE_strongfield_suite_status_512_STRICT_PP_v1.py --device gpu --output src/gr_strict_pp_scalar/PP_EFE_strongfield_suite_status_512_STRICT_PP_v1.json

python -c 'import json,time,os
paths={
"real_EFE_master":"src/gr_strict_pp_scalar/PP_EFE_master_status_512_STRICT_PP_v3.json",
"strongfield_suite":"src/gr_strict_pp_scalar/PP_EFE_strongfield_suite_status_512_STRICT_PP_v1.json",
"tensor_3p1_status":"src/gr_strict_pp_scalar/PP_EFE_tensor_3p1_status_512_STRICT_PP_v1.json",
"scalar_vector_status_v2":"src/gr_strict_pp_scalar/PP_EFE_scalar_vector_status_512_STRICT_PP_v2.json",
"bianchi_status":"src/gr_strict_pp_scalar/PP_EFE_bianchi_status_512_STRICT_PP_v1.json",
"structural_status":"src/gr_strict_pp_scalar/PP_EFE_structural_status_512_STRICT_PP_v1.json",
"scalar_A2b_status":"src/gr_strict_pp_scalar/PP_EFE_scalar_A2b_status_512_STRICT_PP_v1.json",
"scalar_vector_A2b_offdiag":"src/gr_strict_pp_scalar/PP_EFE_scalar_vector_A2b_offdiag_status_512_STRICT_PP_v1.json",
}
out={"timestamp_unix":time.time(),"artifacts":paths}
missing=[k for k,v in paths.items() if not os.path.exists(v)]
if missing: out["ERROR_missing"]=missing
else:
  def load(p): 
    with open(p,"r") as f: return json.load(f)
  out["PASS"]={k:load(v).get("PASS",{}) for k,v in paths.items()}
  out["ALL_PASS"]=True
  for blk in out["PASS"].values():
    if isinstance(blk,dict):
      for v in blk.values():
        if v is False: out["ALL_PASS"]=False
with open("src/gr_strict_pp_scalar/PP_EFE_FREEZE_512_STRICT_PP_v1.json","w") as f:
  json.dump(out,f,indent=2,sort_keys=True)
print("WROTE src/gr_strict_pp_scalar/PP_EFE_FREEZE_512_STRICT_PP_v1.json")
print("ALL_PASS =",out.get("ALL_PASS"))'

jq ".ALL_PASS, .ERROR_missing? // empty" src/gr_strict_pp_scalar/PP_EFE_FREEZE_512_STRICT_PP_v1.json
