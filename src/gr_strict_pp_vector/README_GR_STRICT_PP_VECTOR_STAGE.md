# GR — STRICT PP Vector Stage (WIP)

This module begins **Sector II: Vector Gravity** under STRICT PP rules.

Rules:
- Distances, time, and observables are **Markov/trace/partial-order operational only**.
- **No PDE. No Laplacian/Poisson. No Euclidean radius injection.**
- **No GR ansatz. No regression to force matches.**
- Grid coordinates/atan2 may be used **only as labels** for ordering ring nodes,
  never as a metric in PASS conditions.

Rationale:
Scalar robustness is now frozen:
- ms080: PASS
- strong_pf010: scalar signs + Shapiro PASS, perihelion dynamic N/A (no directed cycles), deflection negative

Vector stage goal:
Define STRICT-PP vector observables that can succeed across kernel classes
where scalar deflection can fail.

Planned deliverables:
1) **Vector-circulation observable** (Markov-only)
2) **Frame-dragging proxy** (Markov-only)
3) Vector pack + suite integrated with scalar

Expected outcomes:
- Some kernels may show scalar attraction without vector circulation.
- We record PASS/FAIL/N/A honestly per case.
