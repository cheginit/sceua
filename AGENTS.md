# Contributing

Conventions for human contributors and AI coding agents working in this repository. The
docs site renders this file as its contributing guide.

## Project

`sceua` is a Python implementation of the Shuffled Complex Evolution (SCE-UA) algorithm
for bound-constrained, single-objective global optimization, aimed at calibrating models
such as HYMOD. The algorithm lives in `src/sceua/sceua.py`, and `sceua.minimize` and
`sceua.Result` are the public API.

## Design record

`docs/design.md` holds every design decision (`D-`), the lessons from past mistakes
(`L-`), and the open questions (`Q-`), each with its rationale and rejected
alternatives.

- Before designing a component or proposing an alternative to a rule below, search it.
    Do not reopen an accepted decision without evidence it does not already account for.
- When a change makes or reverses a decision, answers a question, or exposes a mistake
    worth not repeating, update `docs/design.md` in the same change, following the rules
    at its top.

## Design rules

One-line summaries of the decisions any code change can break. The IDs point into
`docs/design.md`.

- `minimize()` and `Result` are the public API, and numpy and scipy are the only runtime
    dependencies (D-01).
- Every evaluated point lands in `Result.xv` with its value in `Result.funv`. `Result.x`
    is one of those points, `Result.fun` is its value and equals `nanmin(Result.funv)`,
    and every point lies within the bounds (D-02).
- The best point joins every complex, listed once (D-03).
- Any step that moves a point evaluates it before it competes, and the best point is
    never moved (D-05, L-01).
- Scale-dependent heuristics work on parameters scaled to [0, 1] by the bounds (D-05,
    L-03).
- Comparisons of objective values go through `_improves()`, which ranks NaN worse than
    any number (D-06, L-07).
- Caller input errors raise `TypeError` or `ValueError` before the first evaluation
    (D-07).
- `max_evals` is honored as given and must be at least the population size (D-08).
- Populations and objective values are stored as float64 (D-09).
- A run stops when neither the best value nor the population median beats its record by
    `tolerance` times the population's objective IQR for `max_tolerant_iter` iterations,
    or when every parameter's range is below `x_tolerance` of its bounds (D-14).
- Python packages come from PyPI. `[tool.pixi.dependencies]` holds only system libraries
    (D-11).

## Commands

Install [pixi](https://pixi.sh) and run everything through it; each environment is
created on first use. Never call bare `pytest`, `python`, or `pyright`.

| Task                       | Command                             |
| -------------------------- | ----------------------------------- |
| Lint (pre-commit)          | `pixi r lint`                       |
| Dependency audit           | `pixi r audit`                      |
| Type check                 | `pixi r typecheck`                  |
| Test (3.13)                | `pixi r -e test313 test`            |
| Test (3.15)                | `pixi r -e test315 test`            |
| Coverage report / HTML     | `pixi r -e test315 report` / `html` |
| Benchmarks (CodSpeed)      | `pixi r benchmark`                  |
| Run Python                 | `pixi r -e dev python ...`          |
| Spell check (writes fixes) | `pixi r spell`                      |
| Changelog preview          | `pixi r changelog`                  |
| Docs, strict build         | `pixi r docs`                       |
| Docs, live preview         | `pixi r docs-serve`                 |
| Notebooks, sync and run    | `pixi r nb-run`                     |
| Notebooks, pair a new one  | `pixi r nb-pair`                    |

Environments: `dev`, `test313`, `test315`, `typecheck`, `benchmark`, `lint`, `docs`.

A change is done when lint, typecheck, and the tests on both test313 and test315 pass,
plus `pixi r docs` when it touches `docs/`, `README.md`, or `zensical.toml`.
`.github/workflows/ci.yml` runs the first three, and `.github/workflows/docs.yml` builds
the site on every PR and deploys `main` to GitHub Pages.

A change to the search itself (`_evolve_complexes()`, `_pca_recovery()`,
`_update_population()`, or the defaults of `minimize()`) also reports solution quality
before and after on fixed problems and seeds, with the prediction written down first
(REVIEW.md).

## Code style

- Python >= 3.13, with `from __future__ import annotations` in every file outside
    `docs/examples/`.
- ruff with 100-character lines and `select = ["ALL"]`; pyright in strict mode. Both are
    configured in `pyproject.toml`.
- NumPy-convention docstrings.
- Source in `src/sceua/`, tests in `tests/`. Tests make no network requests.

## Gotchas

- `addopts` sets `--cov-append` and coverage runs in parallel mode, so coverage
    accumulates across runs. Delete `.coverage` and any `.coverage.*` files before
    trusting a number.
- pyright checks only `src/`, so a green `typecheck` says nothing about `tests/`.
- `pixi r lint` runs vulture at 80% confidence, which catches only unused imports,
    unused function arguments, and unreachable code. Unused functions, classes, and
    local variables score 60% and pass.
- ruff auto-inserts `from __future__ import annotations`.
- mdformat rewraps markdown to 88 columns during `pixi r lint`. pre-commit's
    `--all-files` sees only git-tracked files, so `pixi r lint` runs
    `scripts/lint_untracked.py` after it to lint untracked, non-ignored files by name.
- `docs/index.md` only includes `README.md`, and `docs/contributing.md` only includes
    this file. Edit the source files. Links in both must be absolute URLs: a relative
    link resolves against the site, where its target does not exist, and the strict
    build does not catch it.
- Each example in `docs/examples/` is a notebook paired with a jupytext `py:percent`
    script; edit either and commit both, the notebook with its outputs. `pixi r nb-run`
    syncs and executes them in the `dev` environment. `calibration.ipynb` downloads its
    data from the USGS NLDI and NWIS services and from GridMET when run.
- `pixi r docs` converts each notebook into a Markdown page beside it without running
    it. The generated `.md` files and `*_files/` folders are git-ignored, apart from
    `index.md`. The site also publishes the `.ipynb` and `.py` files, so a link to
    `other.ipynb` downloads the notebook; link `other.md` to reach its page.
- `tests/test_examples.py` runs the code cells of the notebooks it lists, skipping cells
    tagged `docs-only`, so an example's `assert` statements are tests. Plotting cells
    carry that tag because the test environments have no matplotlib.
- Each notebook gets a card in `docs/examples/index.md` with a thumbnail in
    `docs/examples/images/`, following the existing card. The strict build fails if a
    card links to a notebook that does not exist.
- `pixi r nb-sync` and `pixi r nb-pair` copy whichever of a notebook and its script has
    the newer modification time over the other. A checkout or rebase resets those times,
    so an edit can be overwritten by the stale side. Regenerate the stale side
    explicitly, for example
    `pixi r -e dev jupytext --to py:percent docs/examples/<name>.ipynb`.
- Skills live in `.agents/skills/`. `.claude/skills` is a symlink to it for Claude Code,
    which does not read `.agents/`.
- A ruleset on `main` names its required checks by job: `Test (test313)`,
    `Test (test315)`, `typecheck`, and `lint` from `ci.yml`, and `build` from
    `docs.yml`. Renaming a job or a matrix environment leaves the old name unreported,
    so the renaming PR and every PR after it wait forever. Swap the name in the ruleset
    as that PR merges.

## Git

- Conventional Commits: `<type>[optional scope]: <description>`, with type one of
    `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.
- Subject under 72 characters, imperative mood.
- Every change reaches `main` through a PR from its own branch, named
    `<type>/<short-description>` (`fix/nan-best-point`).
- PRs fill in `.github/pull_request_template.md`, and name the issue they finish with
    `Closes #N`.
- The GitHub releases page is the changelog. git-cliff writes each release's notes from
    the commit messages (D-13), so there is no `CHANGELOG.md` to edit.
- `pixi r release v<version> [notes-file]` tags and pushes that tag, which publishes to
    PyPI (D-13). Run it only when a maintainer asks.

## Writing style

These rules cover code comments, docstrings, commit messages, PR descriptions, issues,
and docs.

- American spelling. Fix British spellings on sight.
- Active voice, present tense, concise.
- Use an em or en dash only when no comma, colon, semicolon, parentheses, or new
    sentence works.
- Be concrete. A number, name, or mechanism beats an abstraction: "improves convergence"
    becomes "reaches NSE 0.5315 on all five seeds".
- No contrastive framing. Say what a thing is and stop, without a negated counterpart
    added for rhythm or emphasis: "the launch is checked rather than trusted" reads "the
    launch is checked". The ban covers "rather than", "not X but Y", "instead of", and
    "X, not Y" in any position. A real either/or, where both options are live and the
    reader needs to know which was picked, stays.
- Keep terms consistent. Once something has a name, reuse it, because a reader takes a
    new word for a new thing. If the text calls it a complex, later sentences do not
    switch to "group" or "cluster".
- Cut filler ("it's worth noting", "in order to", "when it comes to"), puffery ("plays a
    vital role", "stands as a testament"), and closing summaries that restate what came
    before.
- Banned words: delve, foster, empower, streamline, facilitate, cutting-edge, seamless,
    multifaceted, nuanced, intricate, meticulous, paramount, transformative, elevate,
    embark, supercharge.

Comments and commits:

- Comment only what the code does not say: a decision a reader would not guess, or
    behavior a reader would not see from reading it.
- Keep a comment or commit body to a line or two. Cut the rationale to the sentence that
    stops the next reader from undoing the change.
- Comments describe the code as it stands. History ("used to", "the old code", the bug
    that prompted the change) goes in the commit message.

Reviews and reports:

- A code review follows `REVIEW.md`: what failure looks like here, what to skip, how to
    report, and how deep to go.
- Say which claims you checked against the source and which you infer.
- Reserve prose for what changes the conclusion. Minor issues go in a numbered list at
    the end.
