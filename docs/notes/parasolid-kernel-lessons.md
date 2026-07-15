# Production-solver lessons from the Parasolid RE

*Added 2026-07-14. Provenance: reverse-engineering of Parasolid `pskernel.dll`
V37.01.243 (a shipping industrial geometric kernel), in the sibling repo
`../../../parasolid-re/`. Primary sources: `parasolid-re/docs/NUMERIC_CORE_ALGORITHMS.md`
(the decompiled solver cores), `NUMERIC_MODEL.md` (tolerance model), `CPP_ENGINE.md`
(the C++ optimizer stack). Evidence grade: `static-observed` (decompilation). These are
**independently-derivable engineering lessons**, not code to copy — Parasolid is evidence
of what a production kernel actually does under load, nothing more.*

## TL;DR

Parasolid runs **two numeric lanes**, picked by problem type: a **damped Newton on the
residual system** (`REL_to_two_equations`: Cramer's-rule step, ≤20 iters, ≤10 step-halvings,
singular-Jacobian guard) for small well-determined clusters, and a **C++ optimizer stack**
(`GradientDescent` = steepest descent + delegated line search; `BrentsMethod` = 1-D
parabolic/golden minimizer) driven by `ImplicitFunctionRootSolver` for the implicit/voxel
lane. solverang is **already ahead** on solver breadth (BFGS/L-BFGS-B/LM/ALM/trust-region +
Strong-Wolfe line search + symbolic Jacobians + JIT + decomposition). The value here is the
**production-hardening tricks** a from-first-principles solver usually lacks, and the
**solver↔kernel contract** for sitting on top of cadabra2.

## Where solverang already matches or beats Parasolid (don't re-do these)

| capability | Parasolid | solverang |
|---|---|---|
| direction methods | steepest descent + damped Newton only | + LM, BFGS, L-BFGS-B, ALM, trust-region |
| line search | Brent (1-D) / pluggable step-size object | Strong-Wolfe/Armijo with safeguarded interpolation |
| small-system direct solve | Cramer's rule (2–3 eqns) | `solve/closed_form.rs` (pattern-matched) |
| singular/rectangular | determinant guard | SVD pseudoinverse |
| decomposition | per-case dispatch (`REL__gen_two_su`) | graph clustering + DOF/redundancy analysis |
| refuse plausible-wrong | output sentinel-poison + status codes | final residual certification |

So the additions below are **surgical**, not architectural.

## Lessons worth acting on (ordered by value)

### 1. Per-component residual noise-gating (dead-zone) — GAP
Parasolid floors every residual component below an absolute floor (`DAT_1845d31c0`) to zero
*before* forming `‖r‖²`, in **both** lanes (`REL_to_two_equations`, `REL_step_from_svec`).
This kills sub-tolerance drift so it can't accumulate across iterations or pollute the
convergence norm — the single most-repeated robustness trick in the binary.
**Action:** add an optional per-component dead-zone to the residual/gradient norm in
`solver/config.rs` (`residual_floor: Option<f64>`), applied in `newton_raphson.rs` and the
gradient norm used by BFGS. Especially valuable for mixed-unit residual vectors (see #2).

### 2. Role-aware linear/angular tolerance model — GAP
Parasolid never uses one scalar tolerance. It carries a **session precision** (default
`1e-8` linear / `1e-11` angular) plus role-specific **ε-multiples** (`~468·ε ≈ 1.04e-13` and
`~161·ε` *relative* comparison tolerances), recomputed per operation/scale, never a global
magic number (`NUMERIC_MODEL.md §1`). A CAD constraint system mixes **distance residuals (mm)
and angle residuals (rad)** in one vector; normalizing them against a single tolerance is
dimensionally wrong and biases the solve toward whichever unit is numerically larger.
**Action:** give constraints a *residual kind* (linear / angular / dimensionless) and scale
each residual by its kind's tolerance before the solver sees it (a diagonal weighting), so
`‖r‖` is unit-consistent. Default `1e-8` linear / `1e-11` angular matches Parasolid and OCCT.

### 3. Damped-Newton-Cramer fast path for micro-clusters — PARTIAL
After decomposition, most CAD sketch sub-problems are **2–3 equations in 2–3 unknowns**.
Parasolid solves exactly these with **Cramer's rule + monotone-residual-decrease acceptance +
≤10 step-halvings toward the previous point (factor 0.5) + a determinant-ratio singular
guard**, capped at 20 outer iterations — no matrix factorization, no line-search machinery.
This is leaner and more robust than routing a 2×2 system through the general NR/SVD path.
**Action:** add a `solve/closed_form.rs` (or `solver/`) branch that, for clusters with
`n ≤ 3` and a square Jacobian, runs direct-Cramer damped Newton with the exact accept-else-
halve cadence. Route decomposed micro-clusters there before the general solver. (You already
route *patterns* like `HorizontalVertical`; this handles the *generic* tiny square system.)

### 4. "Refuse to return a plausible-but-wrong answer" — HAVE, reinforce
Parasolid layers three guards: (a) **poison the output** to a sentinel (`-3.14e13`) on hard
divergence so a stale buffer can't read as a valid root; (b) **status codes**, not a bool —
converged / step-too-small / singular / diverged are distinct; (c) an **in-range post-check**
rejecting out-of-domain solutions. solverang's residual certification already covers the
spirit. **Action:** ensure every solver returns a *typed* failure (you have
`OptimizationStatus::LineSearchFailed` — extend to `SingularJacobian`, `Diverged`,
`MaxItersUnconverged`) and that a diverged solve leaves params in a *documented* state
(unchanged or NaN-poisoned), never a half-applied step.

### 5. Determinism / reproducibility contract — GAP / RISK
Parasolid deliberately does its geometry math in **scalar double precision, not SIMD**
(~178:1 scalar:packed), specifically so the same model gives **bit-identical results on every
OS/CPU** (`NUMERIC_MODEL.md §2`) — a hard CAD requirement (a sketch must solve the same way on
every machine). solverang uses `rayon` (parallel) and `faer` (sparse); **parallel reductions
reorder floating-point ops and break bit-reproducibility.**
**Action:** document a reproducibility contract; for the deterministic path, use a fixed
reduction order (or Kahan/pairwise summation with a defined order) and gate parallel solving
behind a flag that the contract marks as "may differ in the last ULPs." At minimum, add a
regression test asserting identical results single- vs multi-threaded on a reference sketch.

### 6. Multi-criterion convergence with an additive floor — PARTIAL
`GradientDescent` stops on **any** of: gradient stationarity `‖g‖ ≤ (|f| + c)·tol` (with
`c = 1.0`, a confirmed absolute floor), step length, or objective change. The `+c` floor is
what keeps the relative test valid **when the objective passes through zero** — exactly the
root-finding regime, where a pure relative tolerance collapses. **Action:** audit solverang's
convergence tests for a zero-objective trap; use `(|f| + c)·tol` form for any relative test
that fires near a root.

### 7. Trial-and-rollback around risky steps — PARTIAL
Before a tentative step, Parasolid **snapshots the full state** and rolls back if the residual
doesn't improve. solverang has a warm-start `SolutionCache`; the missing piece is a cheap
**checkpoint/restore around a single speculative step** in the damped path (#3), so a rejected
step costs nothing. **Action:** reuse `ParamStore` snapshotting inside the micro-cluster
damped-Newton loop.

## The solver ↔ kernel contract (solverang on cadabra2)

The most important architectural point once solverang sits on the geometric kernel: **the
kernel's own relaxation correctors ARE the evaluators for geometric constraints.** In
Parasolid, a "point on surface" / "tangent" relationship is enforced by the same
`REL_*`/`prepare_su_ests` machinery we reversed. So for solverang:

- A geometric constraint like `PointOnSurface(p, nurbs)` or `TangentToSurface(line, surf)`
  should get its **residual and Jacobian rows from cadabra2's surface evaluation** (position,
  first/second derivatives), not re-implement surface math in the solver. The kernel already
  computes `∂S/∂u, ∂S/∂v`; those are the constraint Jacobian.
- This makes the constraint solver and the kernel **share one tolerance model** (#2): the
  linear/angular session precision must be the *same* object the kernel uses, or a sketch can
  "solve" to a tolerance the kernel then rejects (Parasolid emits exactly this diagnostic:
  "edge/face tolerance must be ≥ Parasolid tolerance").
- **Two levels of Newton nest**: the constraint solver Newton-iterates the sketch DOFs while
  each geometric constraint residual may itself require the kernel to project a point onto a
  surface (an inner relaxation). Budget for it — cache the kernel's projection (warm start)
  across outer iterations; do not re-project from scratch each constraint eval.

## Suggested next steps (prioritized)

1. **#2 role-aware tolerances** — highest leverage; fixes a latent unit-mixing bug and is the
   shared contract with the kernel. Small, mechanical.
2. **#1 residual noise-gating** — a few lines, broadly robustifying.
3. **#3 damped-Newton-Cramer micro-cluster path** — measurable speedup on decomposed sketches;
   pairs with #7 rollback.
4. **#5 determinism regression test** — cheap to add, catches a class of "works on my machine"
   bugs that are fatal for CAD.
5. **Solver↔kernel constraint-evaluation seam** — design work; do it before wiring cadabra2 in,
   so geometric constraints delegate to kernel derivatives from day one.

## Cross-links
- `../../../parasolid-re/docs/NUMERIC_CORE_ALGORITHMS.md` — decompiled solver cores (the loops).
- `../../../parasolid-re/docs/NUMERIC_MODEL.md` §1 (tolerances), §2 (determinism), §3.5 (corrector layering).
- `../../../parasolid-re/docs/CPP_ENGINE.md` — the C++ optimizer/implicit lane.
- `../../../cadabra2/docs/re/` — the kernel RE→implementation guide this solver will sit on.
