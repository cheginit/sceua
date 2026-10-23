# Reviewing sceua

For every reviewer of a change here, a person or an agent. Read `AGENTS.md` first, and
the `docs/design.md` entries the change touches.

## What failure looks like here

1. **The result lies.** `Result.x` and `Result.fun` disagree, `Result.fun` is not the
    smallest value in `Result.funv`, a point lies outside the bounds, or `nfev` differs
    from the length of the history (D-02, L-01, L-02, L-04, L-07).
1. **The search degrades unmeasured.** A change to complex evolution, recovery, or the
    defaults lowers solution quality or spends more evaluations, and no comparison on
    fixed problems and seeds shows it (Q-01, L-05).
1. **Bad input runs.** An input `minimize()` should reject runs, or the error comes
    after the objective was already called (D-07, D-08).
1. **Units decide the search.** A heuristic computed in raw parameter units behaves
    differently when a parameter changes units (D-05, L-03).
1. **The public surface moves silently.** A `minimize()` argument, default, or `Result`
    field changes without a breaking-change commit that the release notes pick up.
1. **A claim outruns its test.** A docstring, example, or design-record sentence states
    behavior that no test pins.

## Out of scope

- `pixi.lock`, notebook outputs in `docs/examples/*.ipynb`, and the pages `pixi r docs`
    generates.
- Anything the gate already enforces: ruff with `select = ["ALL"]`, pyright strict on
    `src/`, mdformat, codespell, vulture.
- Accepted entries in `docs/design.md`. Challenge one only with evidence or a
    consequence the entry does not account for.

## How to report

- Each finding names the file and line, the input or state that triggers it, and what
    goes wrong. A guess from a name is no finding.
- Mark each finding verified (you read the code or ran a probe) or inferred.
- Tag each one important, nit, or pre-existing (present before this change). Report at
    most three nits.
- Report the class of a bug: when one site has it, list every site that does, so one fix
    round covers them.
- Open with the count, "N findings (I important, P pre-existing)", and say "No findings"
    when there are none.
- On a re-review, give a fixed or still-open verdict on each earlier finding first,
    review only the new diff, and repeat no finding already answered.

## Depth

Match the depth to what a missed bug would cost:

| Change touches                                                                                                                                              | Review                                                                                                                              |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Only docs, `pixi.lock`, pre-commit revisions, or tests                                                                                                      | CI only                                                                                                                             |
| Other code in `src/` or `scripts/`                                                                                                                          | A review of the diff                                                                                                                |
| `_evolve_complexes()`, `_pca_recovery()`, `_update_population()`, `_improves()`, the loop in `minimize()`, `Result`, `__init__.py`, or `.github/workflows/` | A review of the whole project at the change, and for a search change, solution quality on fixed problems and seeds before and after |

After three review rounds on one change, stop and reconsider the approach before a
fourth.
