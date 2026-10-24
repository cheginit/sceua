# Design decisions

This page records the design decisions behind `sceua`, the mistakes that shaped them,
and the questions still open. Contributors and coding agents check it before proposing a
design change and update it in the same change that makes or reverses a decision.

## Maintaining this page

- Search here before deciding. If a decision (`D-`) or lesson (`L-`) covers the case,
    follow it, or argue against it with evidence it does not already account for.
- A new decision gets a `D-` entry with its date. A mistake worth not repeating gets an
    `L-` entry that names the check catching it.
- To reverse a decision, set the old entry's status to `superseded by D-NN` and keep its
    text. The record of what was tried is the point of this page.
- An answered question (`Q-`) becomes a `D-` entry, and the question's status becomes
    `answered by D-NN`.
- Some decisions have a one-line summary under "Design rules" in `AGENTS.md`. Update
    that line in the same change.
- Never renumber. New entries take the next free number.
- The page started on 2026-10-23. Entries marked "predates this page" describe the code
    as it stood then and carry no individual dates.

Each decision lists its status, the decision, why, what was rejected, and its limits.

## Scope

`sceua` minimizes a scalar objective over a box of bounded parameters with the Shuffled
Complex Evolution algorithm (Duan et al., 1992, 1994), extended with ideas from Muttil
and Jayawardena (2008), Chu et al. (2010), and the SAMBO package. Its users calibrate
models, HYMOD being the reference case, where one objective evaluation runs a model.
Derivatives, constraints beyond bounds, and multiple objectives are out of scope.

## Decisions

### D-01: `minimize()` and `Result` are the public API

- **Status:** accepted; predates this page.
- **Decision:** The package exports `minimize()` and `Result`. `Result` carries `x`,
    `fun`, `nit`, `nfev`, `message`, and `success`, named as in SciPy, plus `xv` and
    `funv`, every evaluated point and its value. numpy and scipy are the only runtime
    dependencies.
- **Why:** Calibration users know `scipy.optimize.minimize`, and the history arrays feed
    convergence plots and diagnostics.
- **Limits:** The history grows with every evaluation: 50,000 evaluations in 1,000
    dimensions hold 400 MB of coordinates.

### D-02: The result agrees with the history

- **Status:** accepted.
- **Date:** 2026-10-23.
- **Decision:** Every evaluated point lands in `Result.xv` with its value at the same
    index of `Result.funv`. `Result.x` is one of those points, `Result.fun` is its value
    and equals `nanmin(Result.funv)`, every point lies within the bounds, and
    `Result.nfev` equals the length of the history.
- **Why:** A calibration reports `Result.x` as the parameter set and `Result.fun` as its
    score. Before L-01 was fixed, HYMOD with default settings reported NSE 0.517 for a
    parameter set that scores 0.355.
- **Limits:** With `max_workers > 1`, the history of the initial population follows
    completion order, so two runs with the same seed can list it in different orders.
- **Check:** `test_result_invariants` over HYMOD, a mixed-scale sphere, and a 1-D
    function, each with five seeds.

### D-03: The best point joins every complex, once

- **Status:** accepted; predates this page. Listing it once dates from 2026-10-23
    (L-02).
- **Decision:** After each shuffle, complex `k` holds the points ranked `k`, `k + p`,
    `k + 2p`, and so on, plus the best point (index 0), each index listed once.
- **Why:** SAMBO's SCE-UA does this and labels it an improvement.
- **Rejected:** Duan's independent complexes, for now (Q-01).
- **Limits:** Every complex reads and may replace index 0, so complexes evolve in
    sequence and cannot run in parallel (Q-04).

### D-04: New points shift toward the best point by a θ set from the bounds

- **Status:** accepted; predates this page.
- **Decision:** Reflection and contraction points move toward the complex's best point,
    `x_new = (1 - θ) x + θ x_best` (Muttil and Jayawardena, 2008). `θ` is interpolated
    from the base-10 logarithm of the widest bound range, between 0.2 at a range of 100
    and 0.5 at 100,000, as SAMBO does.
- **Why:** The shift speeds convergence on the test functions Muttil and Jayawardena
    report.
- **Limits:** The paper treats `θ` as a fixed, problem-dependent value, best between 0.1
    and 0.5, and found `θ = 0` best for 2-D Rastrigin. Deriving it from the widest range
    ties the search to parameter units (Q-02).

### D-05: Lost dimensions are found and searched in bound-scaled units

- **Status:** accepted.
- **Date:** 2026-10-23.
- **Decision:** Every `pca_freq` iterations, `_pca_recovery()` scales the population to
    [0, 1] by the bounds and computes its principal components. When the smallest
    eigenvalue is below `pca_tol` times the largest, it moves a tenth of the population,
    never the best point, by up to 5% of the bound range along that component, and
    evaluates each moved point before it competes.
- **Why:** Chu et al. (2010) show populations collapsing onto a subspace. In raw units a
    parameter with range 0.13 next to one with range 1,499 always looked collapsed, so
    on HYMOD recovery ran on every iteration (1,425 of 1,425 over 20 seeds) and moved
    points without evaluating them (L-01, L-03). In scaled units it ran on none of
    4,424, and all 20 seeds reached the same NSE.
- **Rejected:** SP-UCI's full module (Chu et al., 2011), which works per complex on
    standardized coordinates, flags a component below 10% of the mean variance, and
    samples along it from the complex centroid, for now (Q-01).
- **Limits:** A population collapsed in every direction has a largest eigenvalue of 0,
    so the check never fires.

### D-06: NaN ranks worse than any number

- **Status:** accepted.
- **Date:** 2026-10-23.
- **Decision:** Every comparison of objective values goes through `_improves()`, which
    treats NaN as worse than any number, so a finite value always replaces a NaN one.
    When every evaluation returns NaN, `Result.success` is `False` and the message says
    so.
- **Why:** Models fail for some parameter sets, and `x < nan` is false for every `x`, so
    plain comparisons discarded finite candidates whenever a complex's worst value was
    NaN (L-07).

### D-07: Caller input errors raise before the first evaluation

- **Status:** accepted. The `x0` bounds check dates from 2026-10-23.
- **Decision:** `minimize()` raises `TypeError` for an objective that is not callable,
    and `ValueError` for bounds that are not `(min, max)` pairs or have a lower bound at
    or above its upper bound, for an `x0` of the wrong shape or outside the bounds, and
    for a `max_evals` below the population size (D-08). Each check runs before the
    objective is called.
- **Why:** A model run can take minutes, and an `x0` outside the bounds came back as the
    answer.
- **Rejected:** Clipping `x0` into the bounds, which changes what the caller asked to
    evaluate. SciPy's `differential_evolution` also raises.
- **Limits:** `n_complexes` below 2 still falls back to the default without an error.

### D-08: `max_evals` is honored as given

- **Status:** accepted.
- **Date:** 2026-10-23.
- **Decision:** The loop stops at the first iteration boundary at or past `max_evals`. A
    `max_evals` below the population size, `n_complexes * n_points_complex`, raises
    `ValueError`. With `p` complexes, a run ends at most `3p` plus a tenth of the
    population above the budget.
- **Why:** `max_evals` used to be raised silently to twice the population size, so a
    budget of 50 ran 72 evaluations.
- **Rejected:** SciPy's approach of evaluating the whole initial population and then
    stopping, chosen against by the maintainer.

### D-09: Populations and objective values are float64

- **Status:** accepted.
- **Date:** 2026-10-23.
- **Decision:** `x0` and the initial objective values are converted to float64 on entry.
- **Why:** An integer `x0` or integer objective values made the population arrays
    integer, which truncated every later float written into them (L-04).

### D-10: Python 3.13 and later

- **Status:** accepted.
- **Date:** 2026-10-23.
- **Decision:** `requires-python` is `>=3.13`, and the tests run on 3.13 and 3.15.
- **Why:** Chosen by the maintainer, matching water-ogcapi.
- **Rejected:** Keeping 3.10 through 3.12, which released versions support.

### D-11: Python packages come from PyPI

- **Status:** accepted.
- **Date:** 2026-10-23.
- **Decision:** Every pixi environment installs its Python packages from PyPI: `sceua`
    through the editable entry in `[tool.pixi.pypi-dependencies]` and the tooling
    through `[dependency-groups]`. conda-forge supplies Python.
    `[tool.pixi.dependencies]` holds only system libraries, and the project needs none.
- **Why:** The test environments then run the wheels a `pip install` gets, as in
    water-ogcapi's D-27.
- **Rejected:** Tooling as `optional-dependencies`, which publishes the tool sets as
    installable extras of the package.

### D-12: Documentation on Zensical, examples from committed notebooks

- **Status:** accepted.
- **Date:** 2026-10-23.
- **Decision:** The site builds with Zensical from `zensical.toml`, and
    `.github/workflows/docs.yml` builds it on every PR and deploys `main` to GitHub
    Pages. The home page includes `README.md`, the contributing page `AGENTS.md`, and
    the license page `LICENSE`. Each example in `docs/examples/` is a notebook paired
    with a jupytext `py:percent` script and committed with its outputs.
    `scripts/convert_notebooks.py` converts the notebooks to Markdown pages before each
    build without running them. `tests/test_examples.py` runs the code cells of the
    notebooks it lists, except cells tagged `docs-only`.
- **Why:** Chosen by the maintainer, matching water-ogcapi's D-16 and D-17. Running the
    examples' code in the tests keeps their claims true. `calibration.ipynb` downloads
    its data, so it is converted from its saved outputs and left out of the tests.
- **Rejected:** MkDocs on Read the Docs, whose strict build broke when mkdocstrings
    renamed its `import` option.
- **Limits:** The docs URL moved from `sceua.readthedocs.io`. Pages deploys only after
    the repository's Pages source is set to GitHub Actions.

### D-13: Releases publish from a `v` tag through trusted publishing

- **Status:** accepted.
- **Date:** 2026-10-23.
- **Decision:** `pixi r release vX.Y.Z [notes-file]` tags a commit already on
    `origin/main` and pushes the tag, which runs `.github/workflows/release.yml`.
    git-cliff writes the GitHub release notes from the commits since the previous final
    release, headed by the tag message. `uv build` builds the distributions, each must
    install alone and report the tag's version, and `uv publish` uploads them through
    PyPI trusted publishing. The GitHub releases page is the changelog.
- **Why:** The same pipeline as water-ogcapi's D-32, with no PyPI token stored.
- **Rejected:** A tracked `CHANGELOG.md`, which needs a commit on `main` before every
    tag.

### D-14: A run stops when neither the best value nor the population median progresses

- **Status:** accepted; answers Q-03. Its `tolerance` margin is superseded by D-15.
- **Date:** 2026-10-23.
- **Decision:** An iteration makes progress when the best objective value or the
    population's median value beats its record by more than `tolerance` times the
    interquartile range of the population's finite values. NaN counts as worse than any
    number in both statistics, and a record moves only when its statistic makes
    progress. A run stops after `max_tolerant_iter` iterations without progress, or once
    every parameter's range across the population is below `x_tolerance` (default 1e-4)
    of its bound range. `_Progress` holds the records.
- **Why:** Under the previous rule, an absolute improvement of the best value, every
    tuning run ended by stalling, after 355 to 4,746 evaluations of a 50,000 budget. On
    HYMOD, two of the three early stops kept the best point of the starting sample for
    30 iterations while the population median rose from NSE 0.30 to 0.46. Benchmark with
    default settings, 20 seeds per problem, 5-D, solved meaning within 1e-2 of the known
    optimum (1e-3 of the best value found for HYMOD). Tuning set, seeds 0 to 19, solved
    before and after: HYMOD NSE 17 and 20, Rosenbrock 6 and 14, Rastrigin 0 and 8,
    Griewank 0 and 0, Styblinski-Tang 8 and 13. Holdout set, seeds 100 to 119, run once:
    HYMOD log-flow NSE 17 and 20, Ackley 1 and 1, Zakharov 15 and 20, Dixon-Price 5 and
    17, a mixed-scale sphere 20 and 20. On the seeds the old rule solved, every one
    stays solved, at 1.00 to 1.38 times the old median evaluations per problem and up to
    2.61 times on a single seed. HYMOD with its objective scaled by 1e-8, shifted by
    1e6, or with `c_max` in meters solves on 20 of 20 seeds, against 0, 17, and 15
    before.
- **Rejected:** Measured on the tuning set. Median progress with an absolute threshold
    still stopped the scaled objective after about 360 evaluations. A threshold from the
    initial population's spread was about 0.05 on Rosenbrock and solved none of it. A
    threshold relative to the current value stopped the shifted objective after 359
    evaluations, and its zero-optimum runs went to `max_iter`. Five times the patience
    kept the absolute threshold and cost 56% more evaluations on HYMOD. On both sets,
    the closest published rule, the Stagnation criterion of CMA-ES (Hansen, 2016),
    compares the median of the newest and the oldest 30% of the best and median
    histories over at least 139 iterations here. It solved Rosenbrock on 20 of 20 tuning
    seeds and Ackley on 18 of 20 holdout seeds, against 14 and 1, and took 1.5 times the
    median evaluations on both HYMOD problems. Its evaluations per solved run, counted
    over all runs, were more than 10% lower on 1 of 9 problems and more than 10% higher
    on 4. The K-iterations rule, criterion C-IV of Fernández-Vargas et al. (2016),
    counts any strict improvement of the best value as progress and solved exactly the
    old rule's counts on both sets: HYMOD 17, Rosenbrock 6, Rastrigin 0, Dixon-Price 5.
    Published rules with absolute or offset-dependent thresholds, such as CMA-ES's
    TolFun (Tanabe, 2026), SciPy's differential evolution, and Diff (Zielinski and Laur,
    2008), stop the scaled or the shifted objective early by construction.
- **Limits:** Evaluations grow with the population: HYMOD with 15 complexes takes about
    9,800 and with 30 complexes reaches `max_iter` near 33,000, where the old rule
    stopped at 1,000 to 6,000 evaluations with NSE 0.50 to 0.53. Patience counts
    iterations, so the population turnovers it allows vary with the complex size, which
    the 5-D benchmark could not test. Every Ackley holdout run stops in the global
    basin, at f = 0.009 to 0.042, and 19 of 20 stop short of the 1e-2 threshold;
    continued, the same runs pass it on 17 of those 19 seeds (Q-07). `tolerance` and
    `x_tolerance` changed meaning, so callers passing absolute values must convert them.

### D-15: Progress is any strict gain of the best value or the median

- **Status:** accepted.
- **Date:** 2026-10-24.
- **Decision:** An iteration makes progress when the best objective value or the
    population's median value is lower than its record, compared through `_improves()`
    with no margin. `minimize()` has no `tolerance` argument; the rest of D-14 stands.
- **Why:** D-14's margin, `tolerance` times the population's current IQR, never decided
    a stop. A record moves only on progress while the IQR shrinks, so gains accumulated
    against the record soon exceed the margin. `tolerance` from 0 to 1e-2 left the stops
    of HYMOD, Rosenbrock, and Rastrigin unchanged. With the margin at 0, every run
    matched D-14 in value, evaluations, and iterations: 335 of 335 on the tuning pool
    (D-14's tuning and holdout sets, the diagnostics, and a cost set with 15 complexes,
    an insensitive parameter, 1-D, and 20-D) and 140 of 140 on a fresh holdout of 7
    problems never run before, seeds 200 to 219.
- **Rejected:** An objective-space stop at a population IQR below `f_tolerance` times
    the initial population's IQR. At 1e-4 to 1e-6 it ended problems whose initial values
    span orders of magnitude far from the optimum: Zakharov went from 20 solved seeds to
    0 at each of those values. At 1e-8, the only value that lost at most one seed on
    every tuning problem, it cut the fresh holdout's evaluations by 7.8% against a 10%
    bar set in advance, and stopped Qing at a median f of 15.0 against 2.4. A threshold
    scaled by the initial sample inherits that sample's range. A windowed record that
    would give `tolerance` an effect was not measured.
- **Limits:** While an objective keeps yielding strict gains, the progress rule cannot
    end the run, and the x-stop, `max_iter`, or `max_evals` does: a 3-D sphere with the
    x-stop off reaches `max_iter` at 7,290 evaluations. The progress rule ends a run
    once gains stop or round away, as in all 20 Rosenbrock tuning runs, constant
    regions, objectives computed in float32 such as HYMOD, and the same sphere shifted
    by 100, which stalls after 552 iterations. Where a parameter has no effect, the
    x-stop cannot fire and the run goes on until gains stop: a 5-D sphere that ignores
    one parameter takes 3,783 evaluations on average, against 2,472 under the rule
    before D-14, which reached 1e-6 on 9 of 20 seeds against 20. The cost set shows the
    other large increases: a 1-D sphere takes 968 evaluations against 276 (within 1e-6
    on 20 of 20 seeds against 2), and 20-D Rastrigin 10,380 against 1,098 (median f 7.0
    against 126). Callers passing `tolerance` get a `TypeError`.

## Lessons

### L-01: Moved points kept the values of where they had been

- **What happened:** Lost-dimension recovery moved points without evaluating them, so
    they competed under their old values, and when it moved the best point, `Result.x`
    stopped matching `Result.fun`.
- **Rule:** A step that moves a point evaluates it before the point competes (D-05).
- **Check:** `test_pca_recovery_evaluates_moved_points` and `test_result_invariants`.

### L-02: A repeated index lost a new best point

- **What happened:** The first complex listed index 0 twice. numpy's fancy assignment
    keeps the last write to a repeated index, so writing a sorted complex back put the
    old best over a new one: 45 of 6,435 complex evolutions lost a point better than the
    whole population on 20-D Rosenbrock.
- **Rule:** Index lists used for write-back hold unique indices (D-03).
- **Check:** `test_new_best_is_kept`.

### L-03: Raw units made a narrow parameter look collapsed

- **What happened:** The covariance of HYMOD's population, in raw units, put the
    variance of `k_s` (range 0.13) near 1e-8 of `c_max`'s (range 1,499), below the 1e-3
    threshold on every iteration.
- **Rule:** Heuristics that compare parameters work in bound-scaled units (D-05).
- **Check:** `test_pca_recovery_ignores_parameter_units`.

### L-04: Integer inputs truncated float results

- **What happened:** An integer `x0` that filled the population, or an objective
    returning integers at the start, made the arrays integer, and later floats were
    truncated on assignment while their scores were kept.
- **Rule:** Convert arrays to float64 on entry (D-09).
- **Check:** `test_integer_x0` and `test_integer_objective_values`.

### L-05: Accuracy tests passed without any evolution

- **What happened:** `test_objective` ran 200 to 400 complexes on 2-D functions, so the
    initial Latin hypercube sample of up to 320,400 points met every tolerance, and all
    three cases passed with `max_iter=0`.
- **Rule:** A new accuracy test must fail when run with `max_iter=0`.
- **Check:** None automatic. Run the new test once with `max_iter=0` before adding it.

### L-06: A seed-pinned reproduction stopped reproducing

- **What happened:** A defect found with a fixed seed stopped showing at that seed after
    an earlier fix changed how many random numbers each run drew, while the defect
    remained.
- **Rule:** Regression tests pin the mechanism with a constructed input, and are checked
    by breaking the fix once and watching the test fail.
- **Check:** The tests that call `_pca_recovery()` and `_evolve_complexes()` directly.

### L-07: NaN comparisons discarded finite values

- **What happened:** `x < nan` is false for every `x`, so when a complex's worst value
    was NaN, a finite reflection or contraction lost the comparison and a random point
    took its place, and the run could report every value as NaN.
- **Rule:** Compare objective values through `_improves()` (D-06).
- **Check:** `test_finite_candidate_replaces_nan_worst` and
    `test_nan_worst_keeps_finite_values`.

### L-08: A single run's NSE differed by platform

- **What happened:** The reliability example asserted that seed 0's run with a NaN
    region reached the NSE of seed 0's plain run. HYMOD computes in float32, and on
    Windows the small arithmetic differences sent the same seed down another path, which
    stalled at NSE 0.504 after 365 evaluations (Q-03).
- **Rule:** Assertions on search results either state a guarantee that holds on every
    run (D-02, D-06, D-08) or compare the best of several seeds.
- **Check:** None automatic, since CI runs the tests on Ubuntu only.

## Open questions

### Q-01: Canonical complex evolution

**Status:** open.

Each shuffle evolves every complex by one reflection of its worst point through the
centroid of the whole complex. Duan et al. (1994) evolve a sub-complex of `d + 1`
points, chosen with triangular probability, `2d + 1` times per complex before shuffling,
with complexes of `2d + 1` points. On 10-D Rosenbrock the median over 10 seeds is 4.82,
and 4,800 more evaluations lower one run by 0.001, while SciPy's differential evolution
reaches 5.6e-5 in 50,000 evaluations. The candidates are Duan's step and SP-UCI's
modified step (Chu et al., 2011), measured on a fixed set of problems, budgets, and
seeds.

### Q-02: θ depends on parameter units

**Status:** open.

D-04 derives `θ` from the widest bound range, so expressing `c_max` in meters in place
of millimeters changes the search for every parameter, though the final NSE on HYMOD
stays the same. The options are a fixed `θ` as Muttil and Jayawardena use, or FASCE's
adaptive `θ` (Liu et al., 2024), which needs a guard for objectives near zero.

### Q-03: Stopping rules depend on scale

**Status:** answered by D-14.

`tolerance` is an absolute improvement per iteration and `x_tolerance` an absolute sum
of parameter ranges. A 2-D quadratic on `[0, 1e-12]²` stops after one iteration. Duan's
reference code uses the percent change over the last few shuffles and a range normalized
by the bounds.

The rule also ends runs before the search has moved. On HYMOD with default settings, 3
of 20 seeds stop after 355 to 513 evaluations at NSE 0.49 to 0.51, while the other 17
reach 0.5314 or better (L-08).

### Q-04: Parallel evolution

**Status:** open.

`max_workers` parallelizes only the initial population. Evolving each complex against a
snapshot of the best point would remove the coupling in D-03, and offspring could then
be evaluated through a thread pool or a user-supplied map-like `workers` callable, as
SciPy accepts.

### Q-05: `alpha` and `beta` collide with Duan's notation

**Status:** open.

In Duan et al. (1994), α counts offspring per sub-complex and β evolution steps per
complex. Here they are the reflection and contraction coefficients, so a value copied
from the papers, such as `β = 2d + 1`, gives a contraction coefficient of 41 at 20-D.

### Q-06: Dynamically Dimensioned Search as a second solver

**Status:** open.

DDS (Tolson and Shoemaker, 2007) has one parameter, schedules itself by the evaluation
budget, and does well on expensive watershed models (Arsenault et al., 2014). It fits
the package's users and API.

### Q-07: Patience is 30 iterations whatever the problem

**Status:** open.

D-14 ends a run after 30 iterations without progress, which cuts every Ackley holdout
run during a pause. The Stagnation criterion of CMA-ES, whose window is at least 139
iterations here, solves 18 of the 20 against 1 at 2.7 times the median evaluations, and
costs 1.5 times on HYMOD, where both rules solve every seed. A patience that grows with
the dimension or with the iterations already run is untested. The 2026-10-23 holdout has
informed this question, so an answer needs new holdout problems and seeds.

## References

- Arsenault, R., Poulin, A., Côté, P., and Brissette, F. (2014). Comparison of
    stochastic optimization algorithms in hydrological model calibration. *Journal of
    Hydrologic Engineering*, 19(7). <https://doi.org/10.1061/(ASCE)HE.1943-5584.0000938>
- Chu, W., Gao, X., and Sorooshian, S. (2010). Improving the shuffled complex evolution
    scheme for optimization of complex nonlinear hydrological systems. *Water Resources
    Research*, 46. <https://doi.org/10.1029/2010WR009224>
- Chu, W., Gao, X., and Sorooshian, S. (2011). A new evolutionary search strategy for
    global optimization of high-dimensional problems. *Information Sciences*, 181(22).
    <https://doi.org/10.1016/j.ins.2011.06.024>
- Duan, Q., Sorooshian, S., and Gupta, V. K. (1992). Effective and efficient global
    optimization for conceptual rainfall-runoff models. *Water Resources Research*,
    28(4). <https://doi.org/10.1029/91WR02985>
- Duan, Q., Sorooshian, S., and Gupta, V. K. (1994). Optimal use of the SCE-UA global
    optimization method for calibrating watershed models. *Journal of Hydrology*, 158.
    <https://doi.org/10.1016/0022-1694(94)90057-4>
- Fernández-Vargas, J. A., Bonilla-Petriciolet, A., Rangaiah, G. P., and Fateen, S. K.
    (2016). Performance analysis of stopping criteria of population-based metaheuristics
    for global optimization in phase equilibrium calculations and modeling. *Fluid Phase
    Equilibria*, 427. <https://doi.org/10.1016/j.fluid.2016.06.037>
- Hansen, N. (2016). The CMA evolution strategy: a tutorial. arXiv:1604.00772.
    <https://arxiv.org/abs/1604.00772>
- Liu, J., Hong, H., Fan, W., and Chen, Y. (2024). A novel fast and efficient adaptive
    shuffled complex evolution algorithm for model parameter calibration. *Frontiers in
    Environmental Science*, 11. <https://doi.org/10.3389/fenvs.2023.1341017>
- Muttil, N., and Jayawardena, A. W. (2008). Shuffled Complex Evolution model
    calibrating algorithm: enhancing its robustness and efficiency. *Hydrological
    Processes*, 22(23). <https://doi.org/10.1002/hyp.7082>
- Tanabe, R. (2026). Quantitative performance analysis of stopping criteria for CMA-ES.
    *Parallel Problem Solving from Nature, PPSN XIX*, 114-129.
    <https://doi.org/10.1007/978-3-032-36226-1_8>
- Tolson, B. A., and Shoemaker, C. A. (2007). Dynamically dimensioned search algorithm
    for computationally efficient watershed model calibration. *Water Resources
    Research*, 43. <https://doi.org/10.1029/2005WR004723>
- Zielinski, K., and Laur, R. (2008). Stopping criteria for differential evolution in
    constrained single-objective optimization. In *Advances in Differential Evolution*,
    Studies in Computational Intelligence 143, 111-138.
    <https://doi.org/10.1007/978-3-540-68830-3_4>
