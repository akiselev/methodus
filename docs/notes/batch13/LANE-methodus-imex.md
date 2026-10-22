# Lane `methodus-imex` (repository: methodus, branch `claude/b13-methodus-imex`, base `8bd8333`)

Work package: **SV2-C4b**: the three items the W12 Methodus lane recorded open in `STATUS.md`
(`ImexOperator` and an IMEX integrator; low-storage execution of SSP-RK(10,4); embedded
error-estimate step control). The first consumer is SV9 E5 stage 1 (advection explicit, stiff
diffusion implicit on one FV mesh) and the Euler cases of SV9 F1 (error-controlled explicit
steps). Methodus is physics-neutral: no scientific types, nothing from Scientia or Finitum.

## Packages

1. **`ImexOperator`.** A trait with the explicit part `F_E(t, y)`, the implicit part `F_I(t, y)`,
   and the implicit stage solve hook (either a JVP for Newton–Krylov through the existing
   `newton_krylov`, or a caller-provided linear solve for a linear implicit part; both paths
   exercised). Additive Runge–Kutta pairs stored as tableau data with declared order and
   stiff-accuracy flag: ARS(2,2,2) and ARS(4,4,3) (Ascher–Ruuth–Spiteri 1997) at least; one
   Kennedy–Carpenter ARK3(2)4L[2]SA if time allows. Shu–Osher-style documented summation order,
   no FMA, as `explicit.rs` does.
2. **`imex_step` / `integrate_imex`** with the existing `StepPolicy::{Fixed, Cfl}` applied to the
   explicit part's `stable_step`, admissibility checks per stage, the same typed rejections and
   `ExplicitStepReport`-like telemetry (conservation drift per invariant when the implicit part
   declares its boundary rates). New `SolveError` variants are fine (not serialized; say so).
3. **Embedded error-estimate step control.** `StepPolicy::ErrorEstimate { atol, rtol, safety,
   min_step, max_step, growth limits }` for the explicit schemes that have an embedded pair
   (SSP-RK3 with the SSP-RK2 embedded estimate; RK4 with a declared estimate, or Bogacki–Shampine
   3(2) added as a scheme) and for the ARK pairs; PI controller; reject/retry through the existing
   rejection path; deterministic.
4. **Low-storage SSP-RK(10,4).** Ketcheson's two-register form; result equal to the ten-register
   reference to a predeclared bound (summation order differs; bitwise equality is not expected,
   state the bound from the operation count), memory measured by the number of live vectors.
5. **Tests** (`tests/imex_integration.rs`, extensions of `tests/explicit_integration.rs`):
   - order fits within 0.1 of nominal for each ARK pair on a stiff linear system `y' = -λ y + f(t)`
     with `λ` large (implicit) plus a nonstiff explicit part, and on a nonlinear split (van der
     Pol in the stiff limit as explicit/implicit halves);
   - stiff stability: the explicit CFL bound is set by the explicit part only; the implicit part
     with `λ Δt = 1e4` stays bounded;
   - error control reaches a requested tolerance on a problem with a known solution, with step
     counts recorded, and rejects when the tolerance is tightened by 100× (more steps, not NaN);
   - low-storage vs reference within the bound; both TVD on upwind advection at the SSP CFL;
   - conservation telemetry for IMEX with a conservative pair of operators.

## Files you own

New `src/imex.rs`, `src/explicit.rs` (step policy, low-storage path, embedded pairs),
`src/error.rs` (variants), `src/lib.rs` exports, the tests above, your STATUS section.

## Done means

Packages 1–5 landed, `cargo test --locked -p methodus` green (all targets, the crate is small),
clippy and rustdoc clean, report §4 with the trait signatures and config shapes Krasis and Sinbad
will consume (Krasis's `ExplicitExecution` is the model for an `ImexExecution` next phase).

## Refuse rather than guess

Multirate, exponential integrators, and implicit-part solver tuning beyond the existing
Newton–Krylov and linear solvers are out of scope.
