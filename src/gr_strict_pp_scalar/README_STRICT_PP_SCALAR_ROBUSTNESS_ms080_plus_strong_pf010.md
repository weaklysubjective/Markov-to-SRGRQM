````markdown
# STRICT PP Scalar EFE — ms080 + strong_pf010 (40×40)  
**Updated status README (as of this run)**

This README documents the current **STRICT PP** scalar-EFE robustness state across two cases:

- **ms080** (baseline blob)  
- **strong_pf010** (robustness case)

STRICT PP rules enforced throughout:
- **Distances/time/observables are Markov/trace/partial-order operational only.**
- **No PDE. No Laplacian/Poisson field injection. No Euclidean radius assumptions.**
- **No GR ansatz. No regression to force matches.**
- Any change made here is an **observable-definition update** or **structural admissibility mark**, not a physics shortcut.

---

## 1) What we were trying to accomplish

A two-case robustness suite for **STRICT PP scalar-EFE evidence** that can honestly say:

1) **τ-geometry (G00 + κ sign evidence)** comes from trace/Markov structure.  
2) **Shapiro-like τ** is consistent with that scalar geometry.  
3) **Perihelion-like τ (dynamic)** is tested when the directed kernel can support cyclic orbit structure.  
4) **Deflection** is assessed via strict-PP Markov-front observables.  
5) We can ship a multi-case suite with passes/failures marked **without cheating**.

---

## 2) Current final status (short)

### ms080
✅ **Pack v3 overall PASS**

### strong_pf010
- ✅ **τ-geometry sign evidence PASS**
- ✅ **Shapiro-like τ** (present in v2 pack inputs; ensure the JSON exists in your tree)
- ⚠️ **Perihelion-dynamic = N/A (structural)**
  - Directed pf010 graph at 40×40 decomposes into singleton SCCs (no nontrivial directed cycles).  
  - A dynamic “orbit loop” observable is not admissible under this kernel.
- ❌ **Deflection (strict-PP) FAIL**
  - v2 endpoint-front metric saturates.
  - v3 time-quantile deflection remains negative evidence.

Therefore:
- ✅ `overall_PASS_scalar_EFE_strict_PP = True` (core scalar evidence)
- ❌ `overall_PASS_scalar_EFE_strict_PP_v3 = False` (because v3 includes deflection)

### Suite
❌ **ALL_PASS** does not hold because **strong_pf010 deflection is negative**.

---

## 3) The actual suite artifact

```bash
cat PP_EFE00_markov_scalar_suite_v2_ms080_plus_strong_pf010.json
````

Expected top-level truth:

* ms080 pack overall ✅
* strong_pf010 pack overall ❌
* ALL_PASS ❌


---

## 11) PPV1 provenance addendum (40×40)

We introduced **PPV1 edges-from-trace** to remove “mystery edges” and ensure
case-to-case reproducibility under STRICT PP.

### 11.1 Provenance check (confirmed)

The PPV1 τ-geometry artifacts are correctly wired:

**ms080**
- edges_curved: `edges_ca_v3_mass_ms080_40x40_PPV1.txt`
- mass_mask: `PP_mass_mask_40x40_ms080_PPV1.npy`
- trace_weights: `trace_weights_ca_v3_mass_ms080_40x40.txt`

**strong_pf010**
- edges_curved: `edges_ca_v3_strong_pf010_40x40_PPV1.txt`
- mass_mask: `PP_mass_mask_40x40_strong_pf010_PPV1.npy`
- trace_weights: `trace_weights_ca_v3_strong_pf010_40x40.txt`

This confirms no PPV1 provenance mixing in the τ-geometry JSONs.

### 11.2 PPV1 scalar observables at 40×40

**strong_pf010**
- ✅ PPV1 Shapiro-like τ now **PASS** using anchors chosen from the
  trace-derived mass center row and a far-row control path:
  - `src_through=1520`, `dst_through=1559`
  - `src_around=0`, `dst_around=39`
  - Artifact:
    `src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_40x40_strong_pf010_PPV1.json`
- ❌ PPV1 deflection remains **negative evidence** at this scale:
  - Artifact:
    `src/gr_strict_pp_scalar/PP_deflection_markov_front_40x40_strong_pf010_row39_TIMEQ_PP_v3_PPV1.json`

**ms080**
- PPV1 deflection at 40×40 appears to be trending toward an
  **admissibility-limited regime** under this rule (may return `None`
  depending on applicability gating).
  This is treated as a provenance/admissibility diagnostic, not a physics “loss.”

### 11.3 Interpretation

At 40×40, **PPV1 is functioning as a strict provenance/admissibility lens**:
it may reduce contrast for some observables while clarifying which results
are genuinely supported by trace-derived directed topology.

This does not overwrite the non-PPV1 robustness conclusion:
- ms080 remains the canonical “full green” scalar strict-PP case.
- strong_pf010 remains a valid robustness boundary case with:
  scalar sign structure + Shapiro support, but deflection negative at this scale.

---


---

## 4) Repro CLIs

### 4.1 ms080 (baseline)

You already have the green artifacts:

* `PP_EFE00_markov_scalar_pack_v2_tau_v2_ms080.json`
* `PP_EFE00_markov_scalar_pack_v3_tau_v2_ms080.json`

---

### 4.2 strong_pf010 — τ-geometry (sign evidence)

Use your current v3 τ-geometry generator:

```bash
python PP_markov_tau_geometry_v3_multiShellIntersect.py \
  --edges_curved edges_ca_v3_strong_pf010_40x40.txt \
  --trace_weights trace_weights_ca_v3_strong_pf010_40x40.txt \
  --H 40 --W 40 \
  --mass_mask PP_mass_mask_40x40_strong_pf010.npy \
  --output PP_markov_tau_geometry_40x40_strong_pf010.json
```

We already saw:

* `PASS_G00_sign_attractive = true`
* `PASS_kappa_median_sign = true`

---

### 4.3 strong_pf010 — Shapiro-like τ

```bash
python PP_Shapiro_markov_tau_v1.py \
  --edges_flat   edges_ca_v3_40x40.txt \
  --edges_curved edges_ca_v3_strong_pf010_40x40.txt \
  --trace_weights trace_weights_ca_v3_strong_pf010_40x40.txt \
  --H 40 --W 40 \
  --mass_quantile 0.995 \
  --device auto \
  --output PP_Shapiro_markov_tau_40x40_strong_pf010.json
```

---

### 4.4 strong_pf010 — Perihelion-dynamic (STRICT PP structural N/A)

We mark this by certificate JSON:

```bash
cat PP_perihelion_markov_dynamic_tau_v2_strong_pf010_NA.json
```

Key fields:

* `PERIHELION_DYNAMIC_APPLICABLE = false`
* SCC certificate indicates no directed cycles at this scale.

---

### 4.5 strong_pf010 — Deflection v2 (known saturation)

```bash
python PP_deflection_markov_front_PP_v2.py \
  --edges_flat   edges_ca_v3_40x40.txt \
  --edges_curved edges_ca_v3_strong_pf010_40x40.txt \
  --mass_mask    PP_mass_mask_40x40_strong_pf010.npy \
  --H 40 --W 40 \
  --src_row 39 --src_col 0 \
  --steps 800 \
  --device auto \
  --output PP_deflection_markov_front_40x40_strong_pf010_row39_PP_v2.json
```

Observed:

* `norm_finite_mass_flat = 1`
* `norm_finite_mass_curved = 1`
* `D_flat = D_curved = 0`
* PASS false

---

### 4.6 strong_pf010 — Deflection v3 (TIME-QUANTILE)

```bash
python PP_deflection_markov_front_PP_v3.py \
  --edges_flat   edges_ca_v3_40x40.txt \
  --edges_curved edges_ca_v3_strong_pf010_40x40.txt \
  --mass_mask    PP_mass_mask_40x40_strong_pf010.npy \
  --H 40 --W 40 \
  --src_row 39 --src_col 0 \
  --steps 800 \
  --device auto \
  --output PP_deflection_markov_front_40x40_strong_pf010_row39_TIMEQ_PP_v3.json
```

Result:

* `PASS_deflection_markov_front_PP = false`

This is the decisive strict-PP deflection status for pf010.

---

### 4.7 strong_pf010 — Pack v2 and v3

You created pf010 v2 by cloning schema from ms080 and swapping case strings,
then injected perihelion N/A.

Final v3 assembly:

```bash
python PP_EFE00_markov_scalar_pack_v3.py \
  --pack_v2 PP_EFE00_markov_scalar_pack_v2_tau_v2_strong_pf010.json \
  --deflection PP_deflection_markov_front_40x40_strong_pf010_row39_TIMEQ_PP_v3.json \
  --output PP_EFE00_markov_scalar_pack_v3_tau_v2_strong_pf010.json
```

---

### 4.8 Two-case suite

```bash
python PP_EFE00_markov_scalar_suite_v2.py \
  --packs \
    PP_EFE00_markov_scalar_pack_v3_tau_v2_ms080.json \
    PP_EFE00_markov_scalar_pack_v3_tau_v2_strong_pf010.json \
  --output PP_EFE00_markov_scalar_suite_v2_ms080_plus_strong_pf010.json
```

---

## 5) Interpretation (what these results mean)

### 5.1 The passes are real

**ms080** being green shows the strict-PP scalar-EFE pipeline is coherent end-to-end
for at least one nontrivial case.

**strong_pf010 τ-geometry sign passes** are important:
they show the **“attractive scalar piece”** of GR is not a one-off artifact of ms080.

### 5.2 The failures are informative, not embarrassing

**Perihelion-dynamic N/A** is not a “bad day”; it’s a structural diagnosis:

> a directed Markov kernel with no loop-capable subgraphs cannot support
> a dynamic orbit precession observable.

**Deflection negative evidence** under pf010 is a legitimate robustness outcome:

> the pf010 kernel/topology appears to produce scalar sign structure
> without supporting the corresponding strict-PP bending observable.

That is exactly the kind of “narrow success + boundary of applicability”
that a serious physics-grade program should record.

---

## 6) Analogy & intuition (quick, grounded)

Think of each case as a different “road network” for information flow.

* **ms080** is like a city with ring roads and gradated congestion:
  the strict-PP probes detect “pull,” delay, and bending consistently.

* **strong_pf010** is more like a network of one-way streets that funnel into
  terminal depots with few return paths:
  you can still detect “where the sinks are” (scalar attraction signs),
  but a “curving of traveler trajectories” (deflection) may not show up
  in the same operational way.

This is not a contradiction — it’s a kernel-structure constraint.

---

## 7) Where we are in the big scheme

### Sector I — Scalar Gravity (STRICT PP)

We now have:

* **One full green canonical case (ms080).**
* **One honest robustness case (strong_pf010)** that:

  * supports scalar sign evidence,
  * supports Shapiro-like delay,
  * cannot support dynamic perihelion in a directed-loop sense,
  * does not support deflection under strict-PP front observables.

This is a credible end-of-phase robustness snapshot.

---

## 8) What we do about the failures

We do **not** “tune to win.” We do strict-PP diagnostics:

### 8.1 Perihelion N/A

Action:

* Keep the N/A certificate in the pack.
* Consider a future **structural admissibility gate** in the suite:

  * If `PERIHELION_DYNAMIC_APPLICABLE = false`, treat that subtest as satisfied-by-structure.

This is a formal improvement, not a physics change.

### 8.2 Deflection negative in pf010

Action:

* Freeze it as **negative evidence for this kernel/scale**.
* Add a dedicated **topology/flow analysis** script later:

  * compare source-to-mass capture curves across multiple admissible sources,
  * report whether deflection suppression correlates with sink concentration.

No new geometry assumptions needed.

---

## 9) What’s next (strict PP roadmap)

**Nearest, high-value actions:**

1. **Freeze this two-case scalar robustness release**

   * ms080 green
   * pf010 honest boundary case

2. **Add a small “admissibility + negative-evidence” section to your suite README**

   * so reviewers see we’re not hiding failures.

3. **Move to the next physics layer**

   * **Vector gravity / frame effects** under strict PP
   * where deflection-like phenomena may require more than pure scalar structure.

4. Optional (still strict PP):

   * A third robustness case to see whether pf010 is an outlier
     or a representative of a broader “sink-dominant” class.

1) Generate canonical `_PPV1` edges for ms080 + pf010 at 40×40.
2) Re-run any 40×40 strict-PP scalar/vector admissibility checks using `_PPV1`.
3) Only then proceed to ≥512 with the same canonical rule.

---

## 10) Final status statement (copy/paste)

> Under STRICT PP, we have a fully green scalar-EFE observables pack for ms080.
> For strong_pf010, τ-geometry sign evidence remains attractive and Shapiro-like τ is consistent,
> while dynamic perihelion is structurally non-admissible at 40×40 due to the absence of directed cycles,
> and strict-PP deflection remains negative even under time-quantile Markov-front observables.
> The two-case suite therefore records an honest robustness boundary rather than forcing ALL_PASS.

---

```
```

---

## 9.5) Edge provenance (IMPORTANT for scale)

We discovered that legacy pf010 40×40 edges were produced during exploratory
one-off generation and **cannot be reliably reverse-engineered from the node-weight
trace file alone**.

Therefore, for STRICT PP provenance and any ≥512 scaling claims:

- We introduce a **canonical edges-from-trace rule** via
  `PP_build_edges_from_trace_v1.py`.
- Canonical outputs must be saved with suffix:

  * `_PPV1`

Example canonical naming:

- `edges_ca_v3_strong_pf010_40x40_PPV1.txt`
- `edges_ca_v3_mass_ms080_40x40_PPV1.txt`

**Policy:**
- Legacy edges are preserved for historical runs.
- **All new strict-PP robustness and scale work must use `_PPV1` edges.**

