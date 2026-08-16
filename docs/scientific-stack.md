# Solverang in the scientific refinement stack

Solverang remains a batteries-included numerical and geometric/general constraint solver.
Resolvent does not replace its public product API.

## Two roles are intentional

Solverang is both:

1. an independent high-level constraint product (`Sketch2D`, `Sketch3D`, rigid assemblies,
   constraint diagnosis, optimization, drag/continuation behavior); and
2. a numerical algorithm provider used by Sinbad for operator systems emitted by Resolvent.

Those roles reinforce rather than conflict with one another.

## Optional symbolic capability

`Constraint::symbolic_residuals` is an additive capability with a default of `None`.
Solverang defines only the object-safe `constraint::symbolic::SymbolicSink` vocabulary. It
does not own a second CAS AST and does not take a mandatory dependency on Resolvent.

A Resolvent adapter can implement the sink to obtain the exact residual expression graph.
That enables, progressively:

- generic Jacobian rank over finite fields;
- dependency certificates for `RedundantConstraint.implied_by`;
- polynomial recognition and small-cluster algebraic solving;
- exact event/root analysis where appropriate;
- checkable inconsistency certificates for tractable algebraic clusters.

A constraint with no symbolic representation, or a consumer that never installs such an
adapter, follows the existing numerical path unchanged.

## Numeric algorithms stay here

Solverang continues to own solver policy and algorithms such as nonlinear globalization,
least squares, sparse/direct/iterative orchestration, DAE/ODE integration, continuation,
branch tracking and optimization. Resolvent's `OperatorProgram` describes the mathematical
problem; it does not decide how to solve it.

Malleus remains responsible for compiled finite-precision residual/JVP/VJP execution. Solverang
does not reacquire its old JIT implementation.

## Diagnostics are layered

Native Solverang diagnostics remain available on every build. Optional symbolic/exact
backends can return `DiagnosticSupplement` information rather than replacing the native
analysis. In particular, exact/generic redundancy analysis should eventually populate the
already-existing `implied_by` concept with a certificate, while numeric rank and conflict
analysis remain useful interactive fallbacks.

## Migration safety

This integration must not require rewriting all constraint types at once. The default method
means the existing constraint corpus compiles unchanged. Symbolic coverage can be added one
constraint family at a time and differential-tested against `residuals` and `jacobian`.

For every symbolic constraint implementation, test at randomized finite points that:

1. evaluating the emitted expression equals `Constraint::residuals` within the declared
   floating-point interpretation;
2. differentiating the emitted expression agrees with `Constraint::jacobian`;
3. unsupported/transcendental cases fail closed to the numerical path rather than silently
   changing the constraint.
