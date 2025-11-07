```bash
# CA/MM GR — Release Notes (CLI Cheat Sheet)

This summarizes the **General Relativity** validations in the CA/MM framework.

## Scripts (expected in the pack)
- `gr_markov_suite_v2.py` — main GR runner (`all` executes the full battery).
- `gr_markov_suite_v4.py` — operators/audits (used by the suite; usually not called directly).

## What’s included
- **Gravitational redshift** — Clock-rate shift vs potential.
- **Shapiro delay** — Extra time of flight vs analytic prediction.
- **Light deflection / lensing** — Ray deflection vs weak-field \(4GM/b\) scaling.
- **Perihelion precession** — 1PN-style scaling check.
- **Field/Poisson consistency** — Discrete Poisson solve with near machine-precision residuals.

## Typical runs
> Use `-h/--help` for flags available in your revision. The `all` target runs everything with packaged defaults.

```bash
# Full suite (recommended)
python gr_markov_suite_v2.py all

# Or individual tests (names may vary slightly by revision)
python gr_markov_suite_v2.py redshift
python gr_markov_suite_v2.py shapiro
python gr_markov_suite_v2.py lens      # deflection/lensing
python gr_markov_suite_v2.py perihelion
python gr_markov_suite_v2.py field     # Poisson/field consistency
````

## Defaults & scale (for the packaged configs)

* Analytic-field grid often uses **large H×W** (e.g., 4097×4097), **dx=dy≈0.10**, **GM/c²≈0.03**.
* Ray-tracing deflection runs commonly sweep **impact parameters** (b \in {8,12,16,24}) with long RK trajectories (e.g., `steps≈32000`, `step≈0.01`) and asymptotic corrections.

## Pass criteria (high level)

* **Redshift**: measured Δν/ν vs predicted potential shift within tolerance.
* **Shapiro**: excess delay matches analytic within tolerance across path lengths.
* **Deflection**: mean deflection (\langle S\rangle) ~ (4GM/b) with small relative error (≈1–2% typical in packaged runs).
* **Perihelion**: precession rate follows 1PN scaling within suite tolerance.
* **Field/Poisson**: residuals near machine precision; boundary/solver audits pass.

## Notes

* GR suite **requires PyTorch** (CPU or ROCm/CUDA). It will run on CPU if no GPU is available.
* Long ray/Poisson runs can be compute-intensive; prefer GPU when possible.
  EOF

```
