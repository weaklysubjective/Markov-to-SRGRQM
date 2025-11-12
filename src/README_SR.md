````markdown
# Special Relativity (SR) from CA/MM + Poset

Self-contained SR kinematics suite derived from:
- **Conscious Agents / Markov Matrix (CA/MM)** local updates
- **Trace logic**: one step per tick, **no** backward-time edges
- **Partial order (poset)** of influence
- **Counting only** (no metric assumed)

We do **not** assume coordinates, Minkowski metric, or Lorentz transforms.  
We evolve one tick forward (`t→t+1`) and **read geometry from causal order + counts**.

---

## Quickstart

Requirements: Python 3.10+, `numpy`

```bash
python -m pip install -U pip
pip install numpy
````

Run the fast battery:

```bash
cd src
chmod +x sr_poset_all.sh
./sr_poset_all.sh --fast
cat sr_all_summary.json   # expect: "ALL_PASS": true
```

Run the full battery:

```bash
cd src
./sr_poset_all.sh
cat sr_all_summary.json
```

You can also run each script directly; each prints compact PASS/FAIL JSON.

---

## 1) Causality & Light-Cone

**Scripts:** `ca_sr_causality.py`, `ca_sr_lightcone.py`

```bash
python src/ca_sr_causality.py generate --H 257 --W 257 --T 300 --seed 7 --out-json causality.json
python src/ca_sr_lightcone.py --H 257 --W 257 --T 256
```

**Checks:** no backward edges; front expands with `ĉ = 1` cell/tick.
**Math:** after `T` ticks, `|dx|+|dy| ≤ T` ⇒ diamond future ⇒ universal speed bound.
**Intuition:** if anything arrives earlier than allowed, physics (here) is broken.

---

## 2) Proper Time (Time Dilation)

**Script:** `ca_sr_propertime.py`

```bash
python src/ca_sr_propertime.py --H 1201 --T 400 --v 0.8
```

**Checks:** Alexandrov count ratio ≈ `√(1−v²)`.
**Math:** counts along a worldline scale like proper-time²; ratio (moving/rest) ⇒ `√(1−v²)`.
**Intuition:** fewer internal causal links for movers ⇒ less proper time (twin effect).

---

## 3) Proper Time — 2D Orientation Sweep

**Script:** `ca_sr_propertime_orientation_2d.py`

```bash
python src/ca_sr_propertime_orientation_2d.py --T 400 --v1 0.8 --angles 12
```

**Checks:** same dilation for many directions at fixed L1 speed `v1`.
**Math/Intuition:** counts depend on `|Δx|+|Δy|`, not heading ⇒ rotational consistency.

---

## 4) Length Contraction

**Script:** `ca_sr_length_contraction.py`

```bash
python src/ca_sr_length_contraction.py --H 2401 --T 400 --v 0.6 --L0 200
```

**Checks:** measured `L′ ≈ L0 √(1−v²)` using **poset simultaneity**.
**Math:** Δ(E)=N(p−,E)−N(E,p+) ; Δ=0 antichain = moving-frame “now”; intersect rod.
**Intuition:** simultaneity from order (not clocks) ⇒ moving rods come out shorter.

---

## 5) Length — Orientation Wrapper

**Script:** `ca_sr_length_orientation.py`

```bash
python src/ca_sr_length_orientation.py --T 400 --v 0.6 --L0 200 --angles 8
```

**Checks:** same contraction across orientations/signs.
**Math/Intuition:** repeats §4 with flipped headings; results agree.

---

## 6) Invariant Interval from Counts

**Script:** `ca_sr_minkowski_interval.py`

```bash
python src/ca_sr_minkowski_interval.py --T 1200 --v 0.8
```

**Checks:** reconstructs `s²` behavior from counts (one rest calibration).
**Math:** rest counts fix a scale α; moving counts give `α·counts ≈ T²−D²`.
**Intuition:** “manufacture” Minkowski interval purely from order + counting.

---

## 7) Velocity Composition (Einstein)

**Script:** `ca_sr_velocity_composition.py`

```bash
python src/ca_sr_velocity_composition.py --T 400 --u 0.4 --v 0.6
```

**Checks:** **rapidity additivity**: η(w) ≈ η(u)+η(v).
**Math:** get γ from counts ⇒ η=arccosh(γ) ≈ artanh(v); test additivity.
**Intuition:** velocities don’t add linearly; rapidities do—recovered from counts.

---

## 8) Relativity of Simultaneity

**Script:** `ca_sr_simultaneity_flip.py`

```bash
python src/ca_sr_simultaneity_flip.py --Tau 200 --v 0.6 --L 200
```

**Checks:** events simultaneous at rest aren’t simultaneous when boosted (Δ signs flip).
**Math/Intuition:** same Δ(E) as §4; sign structure changes across frames.

---

## 9) Isotropy (2D statistics)

**Scripts:** `ca_sr_isotropy_symmetrized_v1.py`, `ca_sr_isotropy_audit.py`

```bash
python src/ca_sr_isotropy_symmetrized_v1.py --H 1001 --W 1001 --T 300 --N 80000
python src/ca_sr_isotropy_audit.py --H 601 --W 601 --T 300 --schedule staggered
```

**Checks:** near-circular endpoint covariance; low angular radius RMS.
**Intuition:** random micro-directions each tick ⇒ statistically round swarms/fronts.

---

## 10) Isotropy (3D)

**Script:** `ca_sr_isotropy_3d.py`

```bash
python src/ca_sr_isotropy_3d.py --H 201 --W 201 --D 201 --T 400 --N 60000
```

**Checks:** 3×3 covariance eigenvalues nearly equal.
**Intuition:** no preferred axis in 3D statistics.

---

## 11) Myrheim–Meyer Order Fractions (Dimension Signal)

**Scripts:** `ca_sr_mm_order_fraction.py`, `ca_sr_mm_dimension_fit.py`

```bash
python src/ca_sr_mm_order_fraction.py --T 400 --n_points 20000 --n_pairs 40000
python src/ca_sr_mm_dimension_fit.py --T 400 --n_points 20000 --n_pairs 40000
```

**Checks:** order-fraction ≈ 0.5 (1+1) vs ≈ 0.35 (2+1).
**Intuition:** dimension shows up in “how often A can influence B.”

---

## 12) (Optional) Circular Union Cone (continuum demo)

**Script:** `ca_sr_poisson_cone.py`

```bash
python src/ca_sr_poisson_cone.py --T 200 --density 4
```

**Checks:** with Poisson sprinkling + Euclidean hop ≤1, union future is a disk of radius T.
**Note:** not required for SR kinematics above.

---

## Full Battery Runner

**Files:** `sr_poset_all.sh`, `sr_poset_all.py` (in `src/`)

```bash
cd src
chmod +x sr_poset_all.sh
./sr_poset_all.sh --fast   # smoke
./sr_poset_all.sh          # full
cat sr_all_summary.json
```

Aggregates all JSONs and reports `"ALL_PASS": true` when everything succeeds.

---

## TL;DR Claim

With only **one-step causal updates** and a **poset of influence**, simple **counts** recover:

* invariant max speed,
* time dilation,
* length contraction,
* relativity of simultaneity,
* Einstein velocity addition,
* an invariant interval,
* statistical isotropy (2D/3D),
* and causal-dimension signals.

No metric was assumed; SR kinematics are **read off** the network of experiences.

```


