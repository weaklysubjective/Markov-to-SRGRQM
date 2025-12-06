```

---

### How to Run Your New Project (CLI)

This is a **single-script project**. You just need to run this one file. It performs the *entire* unified experiment, including the new momentum calculation.

**Step 1: Save the File**
* Save the code above as `project_unified_final.py`.

**Step 2: Install Prerequisites**
* Make sure you have `numpy`, `networkx`, and `scipy` installed:
    ```bash
    pip install numpy networkx scipy
    ```

**Step 3: Run the "Dust" Test (w=0.0)**
* This validates the `G_00` vs `T_00` alignment using your "golden" metric and the new unified momentum build. This should give $R^2 \approx 0.99$.
* (Make sure all your .npz and .txt files are in the same directory, or provide full paths).

```bash
python project_unified_final.py \
  --metric_npz phi_metric_from_experience_mass_blob_v2metric.npz \
  --edges edges_ca_mass_blob_40x40.txt \
  --rho_input_npz mass_blob_40x40.npz \
  --w 0.0 \
  --R2_min 0.95 \
  --json_out results_unified_w0.0.json
```

**Step 4: Run the "Pressure" Test (w=0.3)**
* This is the real test. It checks if the spatial components (`G_11`, `G_22`) from your "golden" metric align with the new, clean pressure components (`T_11`, `T_22`).

```bash
python project_unified_final.py \
  --metric_npz phi_metric_from_experience_mass_blob_v2metric.npz \
  --edges edges_ca_mass_blob_40x40.txt \
  --rho_input_npz mass_blob_40x40.npz \
  --w 0.3 \
  --R2_min 0.95 \
  --json_out results_unified_w0.3.json

## GR — STRICT PP Scalar Robustness

Module mirror:
- `src/gr_strict_pp_scalar/`

Stage README:
- `README_STRICT_PP_SCALAR_ROBUSTNESS_ms080_plus_strong_pf010.md`
