# CA/MM QM — Release Notes (CLI Cheat Sheet)

This summarizes CLIs, key flags (with defaults), outputs, and pass gates for the QM repro pack.

---

## 1) mm_qm_suite_v5c.py (MM core)

**Run**
```

python mm_qm_suite_v5c.py <subcmd> [flags]

```

**Subcommands & key flags**
- **unitary_1d** — `--N INT --dt FLOAT --T INT --t-hop FLOAT --x0 INT --k0 FLOAT --sigma FLOAT --norm-tol FLOAT`
- **two_slit** — `--Nx INT --Ny INT --barrier-y INT --slit-sep INT --slit-hw INT --alpha FLOAT --V0 FLOAT --dt FLOAT --T INT --vis-min FLOAT --flux-tol FLOAT`
- **chsh** — *(no flags)*
- **exchange** — `--stats {boson,fermion}`
- **spinor** (SSQW demo) — `--N INT --T INT --theta FLOAT --k0 FLOAT`
- **smatrix** — `--N INT --t-hop FLOAT --V0 FLOAT --k0 FLOAT --dt FLOAT --T INT --unitarity-tol FLOAT`

**What it does (at a glance)**
- `unitary_1d` — Build Hermitian 1D tight-binding \(H\); step \(U=e^{-iH\Delta t}\); check norm stability (GPU-first; complex64 on GPU, complex128 on CPU).
- `two_slit` — 2D two-slit via Split-Step Fourier; report fringe visibility and flux conservation (incoming ≈ transmitted + reflected).
- `chsh` — Analytic CHSH; expect \(S>2\).
- `exchange` — HOM-style sanity for bosons/fermions (idealized).
- `spinor` — 1D split-step coin+shift (demo) with simple zitter metric.
- `smatrix` — 1D open-chain S-matrix via integrated bond currents; reports \(R,T\) and unitarity error \(|1-(R+T)|\).

**Pass gates (essentials)**
- `unitary_1d`: max \(|\|\psi\|-1|\) < `norm-tol`
- `two_slit`: visibility ≥ `vis-min`, flux error ≤ `flux-tol`
- `chsh`: \(S>2\)
- `exchange`: stats-appropriate bunching (boson) / antibunching (fermion)
- `smatrix`: \(|1-(R+T)| < \) `unitarity-tol`

**CPU/GPU note**
- Uses GPU automatically if available. To force CPU:
  - CUDA: `CUDA_VISIBLE_DEVICES="" python mm_qm_suite_v5c.py <subcmd> ...`
  - ROCm: `HIP_VISIBLE_DEVICES="" python mm_qm_suite_v5c.py <subcmd> ...`

---

## 2) mm_qm_smatrix_comparev3.py (MM precision S-matrix vs analytic)

**Run**
```

python mm_qm_smatrix_comparev3.py [FLAGS]

```

**Flags & defaults**
- `--N INT (2048)` — lattice length  
- `--V0 FLOAT (2.0)` — on-site defect (center)  
- `--k0 FLOAT (1.0)` — incident wavenumber (0<k0<π)  
- `--dt FLOAT (0.05)` — time step for \(U=\exp(-iH\Delta t)\)  
- `--T INT (12000)` — total steps (ensure packet reaches & clears windows)  
- `--x0_frac FLOAT (0.20)` — initial center \(x_0 = x0\_frac·N\)  
- `--sig_frac FLOAT (0.08)` — width \(\sigma = sig\_frac·N\) (wider ⇒ sharper k)  
- `--left_win INT (180)`, `--right_win INT (180)` — absorber widths  
- `--proj_L INT (512)` — window length (Hann) for projections & flux  
- `--margin INT (128)` — distance from absorber edges to window start  
- `--d INT (520)` — distance from defect to each window center  
- `--avg_frames INT (8)` — average last K frames for flux smoothing  
- `--mask_power INT (6)` — absorber taper power (cos^p)  
- `--device {auto,cpu,cuda} (auto)`

**Outputs (JSON to stdout)**
- `ok`, `test`, args, device/dtypes  
- `projections`: complex \(\hat r,\hat t\); phase-referenced \(r_\text{rel}, t_\text{rel}\)  
- `flux`: `R_flux_raw`, `T_flux_raw`, `unitarity_err_raw`, and calibrated `R_flux_cal`, `T_flux_cal`, `unitarity_err_cal`  
- `analytic`: complex \(r,t\) and \(|r|^2,|t|^2\) (lattice single-site defect)  
- `diff`: \(|\,|\hat r|^2-|r|^2|\), \(|\,|\hat t|^2-|t|^2|\), plus phase-error metrics  
- `PASS` iff:
  - `unitarity_err_cal < 0.05`
  - `abs(|r_proj|^2 - |r|^2) < 0.03`
  - `abs(|t_proj|^2 - |t|^2) < 0.03`

**Good baseline**
```

python mm_qm_smatrix_comparev3.py 
--N 2048 --V0 2.0 --k0 1.0 --dt 0.05 --T 12000 
--proj_L 512 --margin 128 --d 520 
--left_win 180 --right_win 180 --avg_frames 8 --mask_power 6

```

---

## 3) ca_dirac_demo_v3_2d.py (CA/SSQW Dirac demo)

**Run**
```

python ca_dirac_demo_v3_2d.py [flags...]

```

**Flags (with defaults)**
- `--Lx INT (512)`, `--Ly INT (512)` — lattice sizes  
- `--n_agents INT (4000)` — synthetic CA trajectories for ring-estimation  
- `--steps INT (2000)` — steps per trajectory  
- `--p_flip_x FLOAT (0.05)`, `--p_flip_y FLOAT (0.05)` — flip probabilities used to generate logs  
- `--dt FLOAT (1.0)` — tick duration (freezes \(c_x,c_y\))  
- `--tol FLOAT (0.10)` — relative tolerance for checks (gap, dispersion, cone, ZB)  
- `--seed INT (42)` — RNG seed  
- `--out PATH (./results_2d)` — output directory (auto-created)  
- `--u1_q FLOAT (0.0)` — charge for optional U(1) links (0 disables)  
- `--u1_B FLOAT (0.0)` — uniform \(B\) in Landau gauge for U(1) links  

> **Note:** `--u1_q/--u1_B` are defined but not currently threaded into the evolution calls, so they don’t affect the run yet.

**Pipeline**
1) Generate synthetic 2D CA transitions on a torus using `n_agents × steps` with `p_flip_x, p_flip_y`.  
2) Ring-estimate hop sizes and flip probabilities.  
3) Freeze walk params: \(\theta_x=\tfrac12 p_{\text{flip}x}\), \(\theta_y=\tfrac12 p_{\text{flip}y}\), \(m=\theta_x+\theta_y\), \(c_x=\Delta x_x/dt\), \(c_y=\Delta x_y/dt\).  
4) Physics-grade checks:
   - Gap at \(k=0\) vs \(2m\)
   - Dispersion small-k cuts vs \(\sqrt{(c k)^2 + m^2}\) (x & y)
   - Zitterbewegung frequency \(\approx 2m\) from ⟨x(t)⟩/⟨y(t)⟩
   - Light-cone bound: measured front speed ≤ (1+`tol`)·\(v_{\max}\) from dispersion surface  
5) Writes plots + JSON; returns non-zero on failure.

**Outputs (in `--out`)**
- `log.txt` — verbose log  
- `dispersion_cuts.png` — \(E(k)\) vs prediction (x & y cuts)  
- `zitter_xy.png` — ⟨x(t)⟩ and ⟨y(t)⟩ traces  
- `lightcone.png` — front radius vs time with fitted speed  
- `report.json` — summary with:
  - `ALL_PASS` (bool)
  - `frozen_params` (\(\theta_x,\theta_y,m,c_x,c_y,dt\))
  - `estimates` (\(p_{\text{flip}x}, p_{\text{flip}y}, \Delta x\_x, \Delta x\_y\))
  - `checks` (gap, dispersion, zb, cone; pass flags + metrics)

**Exit codes**
- `0` — all checks pass (`ALL_PASS: true`)  
- `2` — any check fails

---

## Tips
- Prefer mid-band \(k_0\) (e.g., 0.8–1.2) in S-matrix to avoid band-edge issues.
- If S-matrix fails unitarity, increase `--T`, widen `--sig_frac`, move windows via `--d`, and/or strengthen absorbers (`--mask_power`, window widths).

