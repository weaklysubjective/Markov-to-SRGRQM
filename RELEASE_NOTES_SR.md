````markdown
## CA/MM SR — Correct CLI (required flags shown)

### 1) Physics-grade dispersion (REQUIRED: stack, spacings, dt, method, k-window, output)
Run on a precomputed standing-wave stack `[T,H,W]`:
```bash
python mmca_sr_suitev2.py dispersion \
  --stack disp_candidates/waves_standing_T1024_u1-6.npz \
  --dx 1 --dy 1 --dt 1 --save-every 1 \
  --method phase \
  --kmax-frac 0.028 \
  --json disp_try.json
````

* `--stack` **REQUIRED**: path to the standing-wave NPZ.
* `--dx/--dy/--dt` **REQUIRED**: physical spacings and tick.
* `--method` **REQUIRED**: usually `phase` (two-tick phase-slope estimator).
* `--kmax-frac` **REQUIRED**: small-k window as a fraction of Nyquist.
* `--json` **REQUIRED**: output metrics file.
* `--save-every` optional: keep per-tick intermediates.

**PASS (typical):** small-k slope ( \hat c ) within tol, ( \hat m \approx 0 ), and high (R^2).

---

### 2) CI dispersion (quick check; flags may differ by rev)

Often runs an internal standing-wave routine. If your build still expects a stack, mirror the flags above but use the `ci_dispersion` subcommand and the stack it expects:

```bash
python mmca_sr_suitev2.py ci_dispersion \
  --stack disp_candidates/waves_standing_T512_ci.npz \
  --dx 1 --dy 1 --dt 1 \
  --kmax-frac 0.04 \
  --json ci_dispersion.json
```

**PASS:** meets CI (R^2) threshold (sanity-only).

---

### 3) Light-cone bound (internal sim)

If your revision simulates internally:

```bash
python mmca_sr_suitev2.py lightcone \
  --dx 1 --dy 1 --dt 1 \
  --T 1024 \
  --json lightcone.json
```

**PASS:** measured front speed ≤ cone speed (within tol).

---

### 4) Causality (with external front stack)

If your revision expects a front `[T,H,W]` NPZ:

```bash
# Build a synthetic front (example; adjust to your grid/time)
python make_front_npzv2.py \
  --H 513 --W 513 --T 1024 \
  --dx 1 --dy 1 --dt 1 \
  --out fronts/front_T1024_H513_W513.npz

# Run causality check against that front
python mmca_sr_suitev2.py causality \
  --stack fronts/front_T1024_H513_W513.npz \
  --dx 1 --dy 1 --dt 1 \
  --json causality.json
```

**PASS:** violations_fraction ≈ 0 (no early arrivals).

```

