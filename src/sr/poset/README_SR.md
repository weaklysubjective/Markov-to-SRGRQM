````markdown
# Special Relativity (SR) from CA/MM + Poset

This directory is a self-contained SR kinematics test suite.

Everything here is derived from:
- **Conscious Agents / Markov Matrix (CA/MM)** updates
- **Trace logic**: one step per tick, no backwards time edges
- **Partial order (poset)** between events (who can influence who)
- **Counting** (no metric assumed up front)

We do **not** assume spacetime, coordinates, Minkowski metric, Lorentz transforms, etc.  
We just evolve local “experience updates” forward one tick at a time, and then read off geometry from the causal structure.

You can run each test directly (each script prints JSON with PASS/FAIL),
or run the whole battery with `sr_poset_all.sh`.

---

## 0. Quickstart

```bash
cd src/sr/poset
chmod +x sr_poset_all.sh

# quick CI-sized run
./sr_poset_all.sh --fast
cat sr_all_summary.json  # look for "ALL_PASS": true

# full run (bigger grids, nicer stats)
./sr_poset_all.sh
cat sr_all_summary.json
````

Every individual script below can also be run on its own.

---

## 1. Causality / Light-Cone Speed

### Script(s):

* `ca_sr_causality.py`
* `ca_sr_lightcone.py`

### How to run:

```bash
python ca_sr_causality.py generate --H 257 --W 257 --T 300 --seed 7 --out-json causality.json
python ca_sr_lightcone.py --H 257 --W 257 --T 256
```

### What it checks:

* That influence only flows one tick forward (`t→t+1`), never backwards.
* That the “front” of possible influence expands at speed `c_hat = 1` cell per tick.
* That there are **zero** causal violations.

PASS means:

* `violations_fraction == 0.0`
* `c_hat_cells_per_tick == 1.0`

### Math:

We’re effectively building a causal poset.
An update can only go to direct neighbors (up, down, left, right) in the next tick.
That means after `T` ticks, the farthest L1 radius you can reach is exactly `T`.
So the causal future of an event is a diamond (`|dx|+|dy| ≤ T`).
This enforces a universal limiting speed `c_hat = 1`.

### Intuition:

This is your “speed of light.”
Nothing is allowed to jump ahead faster than 1 cell per tick.
If you see any cell light up that could not have been reached in that many ticks, physics is broken.
We check and it doesn’t break.

---

## 2. Proper Time (Time Dilation)

### Script:

* `ca_sr_propertime.py`

### How to run:

```bash
python ca_sr_propertime.py --H 1201 --T 400 --v 0.8
```

### What it checks:

* That a “moving worldline” experiences fewer internal causal relations than a “rest worldline.”
* That the ratio matches the relativistic factor (\sqrt{1 - v^2}).

PASS means:

* Reported ratio `kappa_moving / kappa_rest` is within tolerance of `sqrt(1 - v^2)`.

### Math:

For a given worldline, we count how many pairs of its own events are timelike-related (“Alexandrov counts”).
That count scales like the square of proper time.

If you compare:

* a rest worldline, and
* a worldline drifting at velocity (v),
  the ratio of those counts behaves like (\sqrt{1-v^2}).
  That’s exactly the SR time dilation factor.

We never insert (\sqrt{1-v^2}) anywhere. We measure counts and see it emerge.

### Intuition:

Think of a traveler and a stay-home twin.
The traveler’s causal history is “thinner” — fewer internal cause-effect links.
We’re literally counting “how much happened for you.”
Fewer links = less proper time.
That’s twin paradox time dilation, but done with counting instead of clocks.

---

## 3. Proper Time Orientation in 2D (angle sweep)

### Script:

* `ca_sr_propertime_orientation_2d.py`

### How to run:

```bash
python ca_sr_propertime_orientation_2d.py --T 400 --v1 0.8 --angles 12
```

Note: here `--v1` is the L1 speed, i.e. total drift per tick in Manhattan norm.

### What it checks:

* Rotate the motion direction through many angles in 2D.
* Check that all directions give the same dilation factor (\sqrt{1-v_1^2}).
* Check that the mean matches (\sqrt{1-v_1^2}) and the spread across angles is tiny.

PASS means:

* `PASS_propertime_orientation_2d: true`
* small angle-to-angle spread.

### Math:

On the lattice, causal reach is L1: `|dx|+|dy|`.
We enforce a fixed L1 speed `v1`, split across x and y with different angles, and compute the same Alexandrov counts as above.
The counts depend only on the total `|dx|+|dy|`, not on which direction we pointed it.

So the dilation relation (\sqrt{1-v_1^2}) should be rotationally invariant in this discrete sense.

### Intuition:

We’re asking: “Does time dilation care which direction you go, or only how fast you go?”
Answer: only speed matters.
That’s the seed of isotropy / no preferred direction.

---

## 4. Length Contraction

### Script:

* `ca_sr_length_contraction.py`

### How to run:

```bash
python ca_sr_length_contraction.py --H 2401 --T 400 --v 0.6 --L0 200
```

### What it checks:

* A rod of rest length `L0` is observed in a frame moving at velocity `v`.
* We determine “simultaneous endpoints” for that moving frame using pure causal logic.
* We measure the rod length in that frame and expect (L' = L_0 \sqrt{1 - v^2}).

PASS means:

* `PASS: true`
* error within `tol_cells`.

### Math:

Relativity of simultaneity is built from counts:
We define Δ(E) = N(p-,E) - N(E,p+),
where p- and p+ are two reference events at the ends of the rod’s worldtube.
Δ(E)=0 is the “events simultaneous in the moving frame” surface (an antichain in the poset).
We intersect the rod worldtube with that antichain and read off its spatial size.

That measured size matches (L_0 \sqrt{1-v^2}).

### Intuition:

Different frames disagree on “at the same time.”
We don’t assume that — we *measure* it from who can causally reach who.
Then we see that moving rods come out shorter.
So classic SR length contraction appears just from “which events are simultaneous for you.”

---

## 5. Length Orientation Wrapper

### Script:

* `ca_sr_length_orientation.py`

### How to run:

```bash
python ca_sr_length_orientation.py --T 400 --v 0.6 --L0 200 --angles 8
```

### What it checks:

* Runs the length contraction logic repeatedly, flipping direction/sign.
* Confirms you always get the same contracted length, not something angle-dependent.

PASS means:

* `PASS_length_orientation: true`
* max_abs_err_cells ~ 0.

### Math:

Same Δ(E)=0 simultaneity trick as above, just tested across different orientations/signs.

### Intuition:

Does a moving meter stick shrink the same way no matter which way you point it?
Answer in this poset: yes.

---

## 6. Minkowski Interval from Counts

### Script:

* `ca_sr_minkowski_interval.py`

### How to run:

```bash
python ca_sr_minkowski_interval.py --T 1200 --v 0.8
```

### What it checks:

* Computes an invariant quantity (s^2) that should match ((\Delta t)^2 - (\Delta x)^2).
* But we do it without ever assuming ((+,-)) metric.
* We build it from:

  * the rest worldline count (used once to get a scale factor α),
  * the moving worldline count.

PASS means:

* `rel_err` is small (≲1%).

### Math:

For a worldline segment, the Alexandrov count tells you “how much proper time squared went by.”
We take:

* rest segment → define calibration α,
* moving segment → predict proper-time-like measure using α.

Then we compare that to the SR invariant (s^2 = T^2 - D^2) where (D = vT).
They match up to small error.

### Intuition:

This is where “spacetime interval” drops out of pure causal order + counting.
We’re basically manufacturing Minkowski’s (s^2) from nothing but: who can talk to who, and how often.

---

## 7. Velocity Composition (Einstein Addition / Rapidity Additivity)

### Script:

* `ca_sr_velocity_composition.py`

### How to run:

```bash
python ca_sr_velocity_composition.py --T 400 --u 0.4 --v 0.6
```

### What it checks:

* Combine two velocities (`u`, then `v`) and get a resulting velocity `w`.
* Check that rapidities add: η(w) ≈ η(u)+η(v).

PASS means:

* `PASS_rapidity_additivity: true`
* `abs_err_eta` below tolerance.

### Math:

From the counts we already use for proper time, we back out an effective gamma:
(\gamma = \frac{1}{\sqrt{1-v^2}}).
Then define rapidity:
(\eta = \text{arccosh}(\gamma)) (which is also (\text{artanh}(v)) in continuous SR).

In SR, velocities don’t add linearly, but rapidities add linearly.
We check that property here using only the causal-count-derived gammas.

### Intuition:

This is “how fast + how fast = still < light speed.”
We’re showing that the poset’s notion of speed composes the same way Einstein’s does.

---

## 8. Relativity of Simultaneity

### Script:

* `ca_sr_simultaneity_flip.py`

### How to run:

```bash
python ca_sr_simultaneity_flip.py --Tau 200 --v 0.6 --L 200
```

### What it checks:

* Two spatially separated events that are simultaneous in one frame.
* When viewed from a moving frame, they are *not* simultaneous.
* We detect that by the sign of Δ(E) = N(p-,E) - N(E,p+).

PASS means:

* `PASS_simultaneity_flip: true`
* opposite signs for the two test events.

### Math:

Same Δ(E) object from length contraction.
If Δ(E) changes sign when you go to a boosted frame, simultaneity got skewed.
That’s literally the relativity of simultaneity.

### Intuition:

One observer says, “Those two things happened at the same time.”
Another observer says, “Nope, first A, then B.”
We’re not declaring that in advance; we’re *reading* it out of pure causal order.

---

## 9. Isotropy (2D statistical)

### Script:

* `ca_sr_isotropy_symmetrized_v1.py`

### How to run:

```bash
python ca_sr_isotropy_symmetrized_v1.py --H 1001 --W 1001 --T 300 --N 80000
```

### What it checks:

* Many independent agents start at center.
* Each tick, each agent picks a random direction and takes exactly one step (causal, speed=1).
* We look at the covariance of final positions.
* We expect that covariance to be almost isotropic (same spread in x and y).

PASS means:

* `PASS_isotropy: true`
* anisotropy score ≪ tolerance.

### Math:

For each agent we track position ((x,y)).
After T ticks, we compute the covariance matrix
(\Sigma = E[(\vec{r}-\bar{r})(\vec{r}-\bar{r})^T]).
If (\Sigma) has nearly equal eigenvalues and tiny off-diagonals, the cloud is statistically circular.

### Intuition:

Even though the raw per-tick rule is “hop to neighbors on a grid,”
if you rotate the local move direction randomly every tick,
the swarm “looks like a circle” on average.
This is how we get rotational symmetry statistically, not by forcing the lattice to be round.

---

## 10. Isotropy Front Audit (2D contour)

### Script:

* `ca_sr_isotropy_audit.py`

### How to run:

```bash
python ca_sr_isotropy_audit.py --H 601 --W 601 --T 300 --schedule staggered
```

### What it checks:

* Evolve a reachable-front mask with a symmetrized/staggered rule.
* Sample radius vs angle at a fixed propagation depth.
* Compute RMS variation across 360 angles.

PASS means:

* `PASS_isotropy: true`
* `isotropy_score_rms_fraction` below tolerance.

### Math:

We’re basically measuring “how round is the wavefront” in polar coordinates.
Angle-to-angle deviations go down when we alternate/rotate the micro rule.

### Intuition:

Instead of a diamond, we’re getting an octagon that’s almost a circle.
It’s not exact, but it’s close enough that you don’t see a preferred axis at coarse scale.

---

## 11. Isotropy (3D)

### Script:

* `ca_sr_isotropy_3d.py`

### How to run:

```bash
python ca_sr_isotropy_3d.py --H 201 --W 201 --D 201 --T 400 --N 60000
```

### What it checks:

* Same idea as 2D, but now in 3D.
* Each agent picks a random 3D direction each tick, we stochastically “round” it to ±x/±y/±z.
* After T ticks we compute the 3×3 covariance of final (x,y,z).

PASS means:

* `PASS_isotropy_3d: true`
* anisotropy score small (λ_max ≈ λ_mid ≈ λ_min).

### Math:

Covariance eigenvalues should match.
If they’re near-equal, space is statistically isotropic in 3D at coarse scale.

### Intuition:

We don’t hardcode a sphere. We get a sphere-shaped cloud anyway.
That’s “no preferred axis” emerging from the rule.

---

## 12. Myrheim–Meyer Order Fraction (Dimension Signal)

### Script:

* `ca_sr_mm_order_fraction.py`

### How to run:

```bash
python ca_sr_mm_order_fraction.py --T 400 --n_points 20000 --n_pairs 40000
```

### What it checks:

* Pick random events from a causal slab.
* Ask: what fraction of random pairs are causally comparable?
  (i.e., one can influence the other)
* In 1+1D this lands near ~0.5.

PASS means:

* `PASS_mm_order_fraction: true`
* deviation from target within tolerance.

### Math:

In causal set theory, that order fraction is related to spacetime dimension.
Near 0.5 is the “it looks like 1+1D” signature.
We’re checking that our causal structure behaves like 1+1 dimension in that sense.

### Intuition:

This is a dimension detector that doesn’t look at coordinates.
It just asks: “How often can A affect B?”
That alone already tells you “this feels like 1+1D.”

---

## 13. Myrheim–Meyer Dimension Fit (1+1 vs 2+1)

### Script:

* `ca_sr_mm_dimension_fit.py`

### How to run:

```bash
python ca_sr_mm_dimension_fit.py --T 400 --n_points 20000 --n_pairs 40000
```

### What it checks:

* We sample uniformly from:

  * A 1+1 slab (`|x| ≤ t`)
  * A 2+1 slab (`|x|+|y| ≤ t`)
* We compute order fractions separately.
* We compare those to expected “1+1-like” (~0.5) and “2+1-like” (~0.35).

PASS means:

* `PASS_mm_1p1: true`
* `PASS_mm_2p1: true`

### Math:

Same order-fraction idea, just extended.
The chance of two random points being comparable depends on dimension.
We check that the 2+1 slab number is clearly different from the 1+1 slab number.

### Intuition:

We’re literally asking the poset:
“Hey, how many spatial directions do you think you have?”
It answers with different fractions and we interpret that as dimension.

---

## 14. (Optional) Circular Union Cone

### Script:

* `ca_sr_poisson_cone.py`

### How to run:

```bash
python ca_sr_poisson_cone.py --T 200 --density 4
```

### What it checks:

* Instead of using a fixed grid, we sprinkle points in continuous 2D discs each tick with max radius growing at `c=1`.
* We then audit whether the union of all reachable points after T ticks looks like a disk (i.e. circular light-cone).

PASS means:

* `PASS_circular_union: true`
* missing coverage in the ideal disk ≤ ~2%.

### Math:

On the lattice, the “union future” after T ticks is a diamond/octagon, not a perfect circle.
With Poisson sprinkling and Euclidean hop limit ≤ 1 per tick, the union future is literally a disk of radius T.
This is the continuum limit behavior.

### Intuition:

This is the “if you hate polygons, here’s your circle.”
It’s optional. SR kinematics (time dilation, simultaneity, etc.) do **not** require this step.

---

## 15. Full battery runner

### Files:

* `sr_poset_all.sh`
* `sr_poset_all.py`

### How to run:

```bash
cd src/sr/poset
chmod +x sr_poset_all.sh
./sr_poset_all.sh --fast   # CI / smoke
./sr_poset_all.sh          # full
cat sr_all_summary.json
```

### What it does:

* Runs all tests above with known good params.
* Collects the JSON outputs.
* Aggregates into `sr_all_summary.json` with `"ALL_PASS": true` when every check passes.

### Intuition:

This is the one-button “show me SR from pure CA/MM poset math” proof.

---

## TL;DR Claim

1. We enforce only:

   * one-step-per-tick causal updates
   * partial order (no backward edges)

2. We count causal relations between events.

3. From those counts alone, we recover:

   * invariant max speed (our `c`)
   * time dilation
   * length contraction
   * relativity of simultaneity
   * Einstein velocity addition
   * an invariant interval (s^2)
   * statistical isotropy (no preferred direction)
   * causal-dimension signals

No spacetime metric was assumed.
We read SR kinematics off the network of experiences / influence.

That’s the point of this directory.

````


## 2. GitHub landing text (top-level README.md or Release description)

Put this in your repo `README.md` (top level), or in the GitHub Release body for your `v1.0.0-sr-poset` tag:

```markdown
# SR from CA/MM (poset-based)

This repo shows you can get special relativity kinematics out of nothing but:
- local “experiences” updating one tick forward,
- a partial order of causal influence (`t→t+1`, no backward edges),
- and counting.

We do **not** assume:
- coordinates,
- Minkowski metric,
- Lorentz transforms,
- continuous spacetime.

We just run local update rules and then read off structure from the causal graph.

The suite lives in `src/sr/poset/`.  
Each test is a standalone Python script that prints PASS/FAIL JSON.

What we demonstrate:

- **Causality / Light-Cone Speed**
  - Show an invariant max influence speed `ĉ = 1`.
- **Time Dilation**
  - Proper time from pure Alexandrov counts matches \(\sqrt{1-v^2}\).
- **Length Contraction**
  - Using simultaneity surfaces defined purely by causal order.
- **Relativity of Simultaneity**
  - Two events “same time” in one frame are not “same time” in a boosted frame.
- **Einstein Velocity Addition**
  - Velocities don’t add linearly, but rapidities add linearly. We recover that.
- **Invariant Interval**
  - We reconstruct something that behaves like \(s^2 = (\Delta t)^2 - (\Delta x)^2\) by counting relations, not by assuming that formula.
- **Isotropy (2D/3D)**
  - After many ticks, the swarm of possible endpoints is statistically round (no preferred axis).
- **Dimension Signal**
  - Causal order fractions distinguish 1+1 vs 2+1 just from “who can influence who.”

Run everything at once:
```bash
cd src/sr/poset
chmod +x sr_poset_all.sh
./sr_poset_all.sh --fast
cat sr_all_summary.json    # expect "ALL_PASS": true
````

This is intended to be a falsifiable physics-style deliverable.
If any of those tests stops passing with the agreed tolerances, the claim fails.

```

---
