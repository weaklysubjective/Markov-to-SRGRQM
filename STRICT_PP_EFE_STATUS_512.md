# STRICT PP EFE status at 512×512 (PPV1)

This note summarizes the **Einstein field equation (EFE)** status for the STRICT PP
512×512 PPV1 program built from CA/MM traces.

It is not a marketing blurb. It is a physics verdict on what is, and is not,
currently supported by the code and reports in this repo.

---

## 1. Primitives and constraints (STRICT PP)

All results here obey the following constraints:

1. **Primitives**
   - Traces / experiences (CA events).
   - Partial order / causal graph.
   - Markov kernels on that graph.
   - Trace-derived weights on nodes (“how much experience passes here”).

2. **Operational distance & time**
   - Distance is **Doyle / commute-time style**, from Markov transitions.
   - Time is **counter steps** of random walks / Markov processes.
   - No Euclidean metric is injected; angles and grid indices are **labels only**.

3. **STRICT PP bans at 512**
   - No external PDEs, no Laplacian/Poisson solves, no GR ansatz.
   - No regression of “geometry vs mass” to force Einstein-like fits.
   - No coordinates treated as rulers.

Under these rules, we still talk about “Shapiro”, “perihelion”, and “EFE-00”, but
they are **operational** statements about Markov distances and trace weights,
not assumptions of continuum GR.

---

## 2. Scalar EFE-00 status (512×512 PPV1)

The scalar side is implemented under `src/gr_strict_pp_scalar/` and documented in:

- `src/gr_strict_pp_scalar/README_STRICT_PP_SCALAR_512_PPV1.md`

Key ingredients at 512×512 PPV1:

- Flat and curved edges:
  - `edges_ca_v3_flat_512x512_PPV1.txt`
  - `edges_ca_v3_mass_ms080_512x512_PPV1.txt`
  - `edges_ca_v3_strong_pf010_512x512_PPV1.txt`
- Trace weights:
  - `trace_weights_ca_v3_mass_ms080_512x512.txt`
  - `trace_weights_ca_v3_strong_pf010_512x512.txt`
- Mass / orbit masks (PPV1):
  - `PP_mass_mask_512x512_mass_ms080_PPV1.npy`
  - `PP_mass_mask_512x512_strong_pf010_PPV1.npy`
  - (plus orbit masks where needed)

### 2.1 τ-geometry (multi-shell) as scalar EFE-00 proxy

Representative reports (names may vary slightly by run, but this is the pattern):

- `src/gr_strict_pp_scalar/PP_markov_tau_geometry_512_mass_ms080_*.json`
- `src/gr_strict_pp_scalar/PP_markov_tau_geometry_512_strong_pf010_*.json`

What these do:

- Build a **Markov resistance / commute-time based τ-geometry** from:
  - Flat edges vs curved edges (mass present),
  - Trace weights, and
  - Multi-shell mass cores (e.g. top-k mass selection).
- Construct a scalar “curvature” proxy from τ and its discrete structure.
- Compare this scalar curvature proxy to the **trace-defined mass distribution**.

Physics content:

- For distinct mass patterns (e.g. ms080 vs strong_pf010), the τ-geometry scalar
  responds in the **direction GR would expect**:
  - More mass → stronger effective curvature signal.
  - Spatial structure of the mass core is reflected in τ-shell behavior.
- There is no regression to a pre-written Einstein equation; the code simply
  measures how τ and mass co-vary under STRICT PP rules.

Verdict (scalar):

- At 512×512 PPV1, τ-geometry behaves as a **strict-PP scalar EFE-00 proxy**:
  - The universe built from traces, partial order, and Markov kernels exhibits
    a scalar “gravity” that tracks mass in the expected way.
  - This supports an **emergent EFE-00-like relation** in the tested cases,
    without injecting GR structure by hand.

This is **not** a proof for all possible graphs / mass patterns; it is strong,
explicit evidence across the ms080 / strong_pf010 scenarios and their shells.

### 2.2 Shapiro Markov-τ at 512

Example report:

- `src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_512_strong_pf010_PPV1.json`
  (and the analogous ms080 file)

What these do:

- Use Markov commute times on flat vs curved PPV1 edges to define:
  - Through path: walker goes “through” the mass region.
  - Around path: walker goes “around” the mass region.
- Measure **extra Markov τ** for the through path relative to around:
  - An operational Shapiro delay in Markov time.

Verdict (scalar Shapiro):

- Shapiro tests at 512 show an **extra Markov time delay** through mass,
  consistent with the scalar τ-geometry picture.
- Again, no PDE or metric is injected; all times are walk counters.

### 2.3 Perihelion and deflection front

Perihelion:

- There are Markov-dynamic perihelion runs (primarily at smaller grids), with
  results consistent with 1PN-type behavior under STRICT PP.
- At 512, perihelion is not the central pillar; τ-geometry + Shapiro carry more weight.

Deflection front (strict PP):

- The **strict-PP deflection front** at 512 is deliberately conservative:
  - Uses finite-time Markov fronts,
  - Compares curved vs flat behavior,
  - Avoids any PDE smoothing.
- In many 512 runs, the current criterion yields **saturated or indeterminate**
  behavior (e.g. norms flat/curved ≈ 1, or zero Markov distance shift).
- This is recorded honestly as **non-PASS or inconclusive**, not massaged.

Scalar summary:

- τ-geometry + Shapiro provide strong, structurally consistent scalar evidence.
- Perihelion is supportive where run.
- Strict-PP deflection at 512 is **not yet a clean PASS**; it is documented as
  saturated/indeterminate rather than claimed as success.

---

## 3. Vector status at 512×512 (PPV1)

Vector-like observables are in `src/gr_strict_pp_vector/` and are documented in:

- `src/gr_strict_pp_vector/README_STRICT_PP_VECTOR_512_PPV1.md`

At 512×512 PPV1:

- We use the same edges, trace weights, and masks as the scalar side.
- We define three vector proxies:
  1. **Azimuthal circulation bias**,  
  2. **Frame-dragging proxy**,  
  3. **Feeder-shear**.

The pack and suite JSONs:

- `src/gr_strict_pp_vector/PP_vector_pack_512_ms080_bt001_PPV1.json`
- `src/gr_strict_pp_vector/PP_vector_pack_512_strong_pf010_bt001_PPV1.json`
- `src/gr_strict_pp_vector/PP_vector_suite_512_ms080_pf010_bt001_PPV1_v2.json`

Key summary:

1. **Azimuthal circulation (bt001): PASS**  
   - Nonzero Markov circulation bias around the mass core at 512×512 PPV1.
   - Bias survives strict PPV1 orbit/mass selection with a modest threshold
     (`bias_threshold = 0.01`).
   - Evidence of **vector-like structure** in flow around mass, derived only
     from Markov transitions and trace-defined masks.

2. **Frame-dragging proxy: NA**  
   - Under current PPV1 orbits and admissibility gates at 512:
     - `FRAME_DRAGGING_APPLICABLE = False`
     - `PASS_frame_dragging_PP = null`
   - There are no robust opposite-band pairs that survive gating, so the proxy
     cannot be evaluated.
   - We do **not** claim frame-dragging detection at 512.

3. **Feeder-shear: saturated, non-gating**  
   - In the admissible feeder band at 512, inflow efficiency to mass saturates:
     nearly every node has an “inward” edge.
   - This collapses the shear proxy to 0; it becomes “saturated” rather than
     informative.
   - In the 512-aware aggregation (`overall_PASS_vector_strict_PP_v2`), feeder-shear
     is recorded but **not used as a gate**.

Vector verdict:

- The **vector suite v2** at 512 yields:
  - Per-pack: `overall_PASS_vector_strict_PP_v2 = True`
  - Suite: `ALL_PASS_vector_strict_PP_v2 = True`
- This means:
  - We have **robust azimuthal circulation** evidence at 512 under STRICT PP.
  - Frame-dragging is explicitly NA (not claimed).
  - Feeder-shear is saturated and treated as non-gating.

We do **not** yet extract a full `G_{0i}` or `G_{ij}` tensor from these proxies.

---

## 4. What is honestly supported about EFE at 512×512?

### 4.1 Positive result: scalar EFE-00–like behavior

Under STRICT PP at 512×512 PPV1:

- τ-geometry + Shapiro (and supporting perihelion runs) demonstrate that:
  - A **scalar gravitational behavior** emerges from CA/MM traces and
    Markov kernels.
  - This scalar behavior tracks trace-defined mass in the way GR’s `G_{00}`
    term would lead you to expect:
    - More mass → stronger curvature proxy.
    - Mass localization → structure in τ shells.
    - Extra τ delay through mass (Shapiro-like effect).

It is therefore reasonable to say, **for the tested cases**:

> We have a strong, production-grade **scalar EFE-00–like result** emerging
> from traces, partial order, and Markov kernels, without injecting a GR ansatz
> or PDE.

This is evidence, not a closed-form theorem covering all possible graphs.

### 4.2 Partial vector result: circulation, but no full tensor

On the vector side, we can honestly say:

- There is **nonzero Markov circulation** around mass at 512×512 PPV1
  (azimuthal bias PASS under STRICT PP).
- The inflow structure to mass is **saturated** in a way consistent with a
  strong central attractor.
- Frame-dragging, as currently defined, is NA at this scale and edge policy.

We **cannot** yet say:

- That we have computed a full **Lorentzian metric** `g_{μν}` from STRICT-PP data.
- That we have recovered all **tensor components** `G_{μν}` and `T_{μν}` in an
  Einstein-equation sense.
- That vector proxies at 512 match GR’s quantitative predictions for
  frame-dragging (Lense–Thirring) or other off-diagonal effects.

So the EFE statement at 512×512 PPV1 is currently:

- **Yes**: strong scalar EFE-00–like behavior from STRICT-PP CA/MM.
- **Yes**: first vector-like evidence (circulation, inflow saturation).
- **No**: full spatial-tensor EFE `G_{μν} = κ T_{μν}` has **not yet** been
  demonstrated from STRICT-PP data alone at this resolution.

---

## 5. Where a skeptic could push (and how to answer)

A serious GR/quantum gravity skeptic might reasonably ask:

1. “Isn’t this just curve-fitting GR?”  
   - Answer:
     - No GR PDE or metric is injected.
     - No `G_{00} = κρ` is fit; instead we measure τ and mass and report their
       relationship directly from Markov data.
     - The code paths are explicit and reproducible.

2. “Is scalar EFE-00 enough to claim ‘GR emerges’?”  
   - Answer:
     - No: scalar EFE-00 is **necessary but not sufficient** for full GR.
     - We present it as **partial emergence**: scalar gravity and Shapiro-like
       behavior from CA/MM, not a full EFE derivation.

3. “What about vector/tensor structure?”  
   - Answer:
     - We show azimuthal circulation and inflow saturation, which are
       vector-like.
     - We explicitly do **not** claim full tensor recovery yet.
     - This is an open target for future STRICT-PP work.

---

## 6. Practical summary

At 512×512 PPV1, with STRICT PP enforced:

- **Proven in code / reports:**
  - Scalar τ-geometry tracking trace-defined mass (scalar EFE-00 proxy).
  - Shapiro-like extra Markov time through mass.
  - Vector-like azimuthal circulation bias around mass.
  - Inflow saturation into mass cores.

- **Honestly not yet achieved:**
  - Full recovery of `G_{μν} = κ T_{μν}` (all components) fr

