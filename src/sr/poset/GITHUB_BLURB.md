```markdown
# SR from CA/MM (poset-based)

This repo shows you can get special relativity kinematics out of nothing but:
- local “experiences” updating one tick forward,
- a partial order of causal influence (`t→t+1`, no backward edges),
- and counting.

Following are not assumed:
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
