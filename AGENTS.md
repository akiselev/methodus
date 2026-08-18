# Agent instructions

Read `STATUS.md` before substantial work and update it before every handoff or pull request.

## STATUS.md policy

`STATUS.md` is a compact current-state ledger, not an append-only journal.

- Keep it under 300 lines; target under 200 and compact stale sections aggressively.
- Record current milestone, implemented numerical contracts/algorithms, exact validation results, blockers, compatibility constraints, and next concrete work.
- Never say a solver or contract is verified unless the corresponding tests actually ran.
- Move historical detail to Git history, PRs, ADRs, or dedicated design documents.
- Keep Solverang physics-neutral: no RSL, materials, function-space, field-name, or physics-specific branching belongs here.
