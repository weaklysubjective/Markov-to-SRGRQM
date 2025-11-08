````markdown
# CA/MM → Special Relativity (SR) — Release Notes

**Status:** physics-grade checks wired and reproducible  
**Driver:** `mmca_sr_suitev2.py`  
**Requires:** Python 3.10+; NumPy/Matplotlib (see `pkgs/SR/*/requirements.txt`)

## What this pack verifies
- **Light-cone bound** from a growing front stack
- **Causality** against a binary front mask
- **Dispersion (phase-slope)** on a standing-wave stack (CI) and a physics-grade fitter

---

## 0) Environment
```bash
python -m pip install -r pkgs/SR/*/requirements.txt
````

---

## 1) Light-cone ✅ (needs NPZ with key `frames`)

**Generate stack (provided script):**

```bash
# run inside pkgs/SR/CA_MM_SR_Suite_*/
python make_front_npzv2.py
# → Wrote front.npz with frames (256, 257, 257) dtype float32
```

**Run light-cone:**

```bash
python mmca_sr_suitev2.py lightcone \
  --stack front.npz \
  --dx 1 --dy 1 --dt 1 \
  --save-every 1 \
  --json lightcone.json
```

**Outputs:** `lightcone.json` (+ optional PNGs if enabled)

---

## 2) Causality ✅ (expects NPZ with key `front`)

**Run causality:**

```bash
# Option A: using front.npz (if your version accepts 'frames' directly)
python mmca_sr_suitev2.py causality \
  --stack front.npz \
  --dx 1 --dy 1 --dt 1 \
  --json causality.json
```

If it errors about a missing `front` key, convert once and use Option B:

```bash
python - <<'PY'
import numpy as np
d = np.load("front.npz")
frames = d["frames"]
np.savez_compressed("front_for_causality.npz",
                    front=(frames > 0).astype("uint8"))
print("Wrote front_for_causality.npz with key 'front'")
PY

# Option B: using converted front
python mmca_sr_suitev2.py causality \
  --stack front_for_causality.npz \
  --dx 1 --dy 1 --dt 1 \
  --json causality.json
```

**Outputs:** `causality.json`

---

## 3) Dispersion ✅

### A) CI standing-wave (quick sanity; yields high R², near-zero c_fit)

**Generator (provided):**

```bash
python make_dispersion_npz_pass.py
# → writes disp_candidates/waves_standing_T1024_u1-6.npz (key: frames)
```

**CLI:**

```bash
python mmca_sr_suitev2.py dispersion \
  --stack disp_candidates/waves_standing_T1024_u1-6.npz \
  --dx 1 --dy 1 --dt 1 \
  --save-every 1 \
  --method phase \
  --kmax-frac 0.028 \
  --json dispersion.json
```

**Expected JSON:** `R2 ≈ 0.9999`, `"PASS": true`, with near-zero `c_fit` (standing content collapses slope — acceptable for CI only).

---

### B) Physics-grade dispersion (meaningful (c, m))

**Generator (CA/MM linear update; 5-point Laplacian, periodic BCs):**

```bash
python make_waveeq_npz_phys.py \
  --out waves_phys.npz \
  --T 1024 --H 257 --W 257 \
  --c 0.60 \
  --demean
```

**Analyzer (complex time FFT per spatial bin):**

```bash
python dispersion_physfit.py \
  --stack waves_phys.npz \
  --kmax-frac 0.12 \
  --min-power-frac 1e-5 \
  --demean-per-frame \
  --out-json dispersion_phys.json \
  --out-png  dispersion_phys.png
```

**Outputs:** `dispersion_phys.json`, `dispersion_phys.png`

---

## Expected JSON keys (typical)

* `lightcone.json`: fitted front speed, bound checks, `PASS`
* `causality.json`: `arrived_fraction`, `violations_fraction`, `PASS`
* `dispersion.json` / `dispersion_phys.json`: small-k fit metrics, `R²`/errors, `PASS`

**Exit codes:** `0` if PASS, non-zero if any check fails

---

## Paths & tips

* Run **inside** `pkgs/SR/CA_MM_SR_Suite_*/`
* Stacks can be large → ensure free disk
* If you change `T/H/W`, keep `dx/dy/dt` consistent between generation and analysis

```
