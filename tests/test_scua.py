# pyright: reportMissingParameterType=false
from __future__ import annotations

import numpy as np
import pytest
from landscapes import single_objective

import sceua
from sceua.sceua import _evolve_complexes, _parameters_converged, _pca_recovery, _Progress

MIXED_SCALE = np.array([1e3, 1, 1e-2, 1, 1])


def mixed_scale_sphere(x):
    return float(np.sum(((x - 0.3 * MIXED_SCALE) / MIXED_SCALE) ** 2))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("case", ["hymod", "mixed_scale", "one_dimensional"])
def test_result_invariants(case, seed, hymod):
    func, bounds = {
        "hymod": (hymod.fit, hymod.bounds),
        "mixed_scale": (mixed_scale_sphere, [(-5 * s, 5 * s) for s in MIXED_SCALE]),
        "one_dimensional": (lambda x: float(np.cos(5 * x[0]) + x[0] ** 2), [(-2, 2)]),
    }[case]
    lo, up = np.array(bounds, dtype=float).T
    result = sceua.minimize(func, bounds, seed=seed)
    assert func(result.x) == pytest.approx(result.fun)
    assert result.fun == result.funv.min()
    assert np.all(result.xv == result.x, axis=1).any()
    assert np.all((result.xv >= lo) & (result.xv <= up))
    assert len(result.xv) == len(result.funv) == result.nfev


@pytest.mark.parametrize("max_evals", [50, 200])
def test_max_evals_overshoot_is_bounded(max_evals):
    n_complexes, n_points_complex = 4, 9
    result = sceua.minimize(
        single_objective.rosenbrock,
        [(-5, 5)] * 10,
        n_complexes=n_complexes,
        n_points_complex=n_points_complex,
        max_evals=max_evals,
        max_tolerant_iter=10**6,
        seed=0,
    )
    # An iteration starts below the budget and adds up to three evaluations per complex
    # plus the points moved by PCA recovery.
    n_moved = max(1, int(0.1 * n_complexes * n_points_complex))
    assert max_evals <= result.nfev <= max_evals - 1 + 3 * n_complexes + n_moved


def test_max_evals_below_population_raises():
    calls = []
    with pytest.raises(ValueError, match="max_evals"):
        sceua.minimize(
            lambda x: calls.append(x) or 0.0,
            [(0, 1)] * 2,
            n_complexes=4,
            n_points_complex=9,
            max_evals=20,
        )
    assert not calls


def test_x0():
    b1 = (-10, 10)
    b2 = (-10, 10)
    expected = 0
    result = sceua.minimize(
        single_objective.levi_n13,
        [b1, b2],
        x0=[(1.01, 1.01)],
        seed=42,
        n_complexes=3 * 2,
        max_workers=2,
    )
    assert abs(result.fun - expected) < 1e-2


def test_integer_x0():
    def func(x):
        return float(np.sum((x - 2.4) ** 2))

    x0 = np.random.default_rng(6).integers(0, 10, (6, 2))
    result = sceua.minimize(
        func, [(0, 10)] * 2, x0=x0, n_complexes=2, n_points_complex=3, pca_tol=0, max_iter=1, seed=0
    )
    assert func(result.x) == pytest.approx(result.fun)


def test_integer_objective_values():
    def func(x):
        return 1 if x[0] == x[1] else float(x @ x)

    t = np.linspace(0.2, 0.8, 6)
    result = sceua.minimize(
        func, [(0, 1)] * 2, x0=np.c_[t, t], n_complexes=2, n_points_complex=3, max_iter=2, seed=1
    )
    assert func(result.x) == pytest.approx(result.fun)


def test_finite_value_replaces_nan_best():
    def func(x):
        return np.nan if np.all(x > 0.8) else float(x @ x)

    result = sceua.minimize(
        func,
        [(0, 1)] * 2,
        x0=np.full((6, 2), 0.9),
        n_complexes=2,
        n_points_complex=3,
        max_iter=2,
        seed=0,
    )
    assert result.success
    assert func(result.x) == pytest.approx(result.fun)
    assert result.fun == np.nanmin(result.funv)


@pytest.mark.parametrize(
    ("func", "kept"),
    [
        (lambda x: x[0] if x[0] < 0.3 else np.nan, 0.18),
        (lambda x: x[0] if 0.6 < x[0] < 0.7 else np.nan, 0.66),
    ],
    ids=["reflection", "contraction"],
)
def test_finite_candidate_replaces_nan_worst(func, kept):
    population = np.array([[0.5], [0.9]])
    values = np.array([np.nan, np.nan])
    rng = np.random.default_rng(0)
    _evolve_complexes(
        func,
        (),
        population,
        values,
        np.arange(2),
        lo_bounds=np.zeros(1),
        up_bounds=np.ones(1),
        rng=rng,
        alpha=1.0,
        beta=0.5,
    )
    assert values[0] == pytest.approx(kept)


def test_nan_worst_keeps_finite_values():
    def func(x):
        return x[0] if 0.67 < x[0] < 0.69 else np.nan

    x0 = [[0.8], [0.9], [0.95], [0.99]]
    result = sceua.minimize(
        func, [(0, 1)], x0=x0, n_complexes=2, n_points_complex=2, max_iter=1, seed=0
    )
    assert result.success
    assert result.fun == np.nanmin(result.funv)


def test_all_nan_objective_is_not_success():
    result = sceua.minimize(lambda _: np.nan, [(0, 1)] * 2, seed=0)
    assert not result.success
    assert "NaN" in result.message


def test_single_point_complexes_get_finite_points():
    result = sceua.minimize(
        lambda x: float(x @ x), [(0, 1)] * 2, n_complexes=2, n_points_complex=1, max_iter=2, seed=0
    )
    assert np.all(np.isfinite(result.xv))


def test_x0_out_of_bounds():
    with pytest.raises(ValueError, match="x0"):
        sceua.minimize(lambda x: float(x @ x), [(1, 2)] * 2, x0=[(0, 0)])


def test_new_best_is_kept():
    def func(x):
        return float(np.sin(10 * x[0]) + np.sin(10 * x[1]))

    result = sceua.minimize(
        func, [(0, 1)] * 2, n_complexes=2, n_points_complex=3, pca_tol=0, max_iter=1, seed=1
    )
    assert result.fun == pytest.approx(result.funv.min())


def test_one_dimensional():
    result = sceua.minimize(lambda x: float((x[0] - 0.3) ** 2), [(-1, 1)], seed=42)
    # The x-stop ends these runs below 1e-9 across seeds.
    assert result.fun < 1e-8


def test_hymod_without_rain_makes_no_runoff(hymod):
    hymod.prcp[:] = 0
    hymod.pet[:] = 0
    x = np.array([lo + 0.5 * (up - lo) for lo, up in hymod.bounds])
    assert np.all(hymod(x) == 0)


def test_model(hymod):
    result = sceua.minimize(hymod.fit, hymod.bounds, seed=42, n_complexes=3 * len(hymod.bounds))
    assert result.success


def test_pca_recovery_evaluates_moved_points():
    def func(x):
        return float(x.sum())

    rng = np.random.default_rng(0)
    t = np.sort(rng.random(20))
    population = np.c_[t, 0.5 * t]  # collinear, so one direction is lost
    values = np.array([func(x) for x in population])
    best = population[0].copy()
    moved = _pca_recovery(
        func, (), population, values, lo_bounds=np.zeros(2), up_bounds=np.ones(2), tol=1e-3, rng=rng
    )
    assert moved
    assert np.array_equal(population[0], best)
    assert np.allclose(values, [func(x) for x in population])
    assert not np.allclose(population[:, 1], 0.5 * population[:, 0])


def test_pca_recovery_ignores_parameter_units():
    lo, up = np.array([(1, 1500), (0.0, 1.99), (0.01, 0.99), (0.01, 0.14), (0.14, 0.99)]).T
    rng = np.random.default_rng(0)
    population = lo + rng.random((100, 5)) * (up - lo)  # spread evenly in every parameter
    values = np.zeros(100)
    assert not _pca_recovery(
        lambda _: 0.0, (), population, values, lo_bounds=lo, up_bounds=up, tol=1e-3, rng=rng
    )


def _sphere(x):
    return float(np.sum((x - 0.3) ** 2))


def test_stopping_ignores_objective_scale_and_shift():
    bounds = [(-5, 5)] * 3
    plain = sceua.minimize(_sphere, bounds, seed=0)
    scaled = sceua.minimize(lambda x: 1e-8 * _sphere(x), bounds, seed=0)
    shifted = sceua.minimize(lambda x: _sphere(x) + 100.0, bounds, seed=0)
    assert plain.nfev == scaled.nfev == shifted.nfev
    assert plain.fun < 1e-4


def _quantized_sphere(x):
    return float(np.floor(1e3 * _sphere(x)) / 1e3)


def test_stall_rule_ignores_objective_scale_and_shift():
    # With the x-stop off, the quantized objective ends every run by the stall rule.
    bounds = [(-5, 5)] * 3
    runs = [
        sceua.minimize(f, bounds, seed=0, x_tolerance=0)
        for f in (
            _quantized_sphere,
            lambda x: 1e-8 * _quantized_sphere(x),
            lambda x: _quantized_sphere(x) + 100.0,
        )
    ]
    assert all(r.message.startswith("No progress") for r in runs)
    assert runs[0].nfev == runs[1].nfev == runs[2].nfev
    # The best value and the median reach the quantized floor of 0, after which no
    # iteration can progress, so the stop comes exactly max_tolerant_iter later.
    longer = sceua.minimize(_quantized_sphere, bounds, seed=0, x_tolerance=0, max_tolerant_iter=40)
    assert runs[0].nit > 30
    assert longer.nit == runs[0].nit + 10


def test_flat_objective_stops_after_patience():
    result = sceua.minimize(lambda _: 1.0, [(0, 1)] * 2, max_tolerant_iter=5, seed=0)
    assert result.message.startswith("No progress")
    assert result.nit == 5


def test_parameter_convergence_needs_every_range():
    lo, up = np.zeros(5), np.array([1.0, 10, 100, 1e3, 1e4])
    pop = lo + np.outer([0, 0.6e-4], up - lo)
    assert _parameters_converged(pop, lo_bounds=lo, up_bounds=up, x_tolerance=1e-4)
    pop[1, 2] = 2e-4 * up[2]  # one parameter stays wide
    assert not _parameters_converged(pop, lo_bounds=lo, up_bounds=up, x_tolerance=1e-4)


def test_parameter_convergence_uses_bound_scaled_ranges():
    width = 1e-12
    result = sceua.minimize(
        lambda x: float(np.sum((x / width - 0.3) ** 2)), [(0, width)] * 2, seed=0
    )
    assert result.nit > 1
    assert result.fun < 1e-4


def test_progress_counts_a_better_median():
    pop = np.linspace(1.0, 2.0, 20)
    progress = _Progress.start(pop)
    for _ in range(5):
        pop = np.r_[pop[0], pop[1:] - 0.01]  # best unchanged, median improving
        progress.update(pop)
    assert progress.stalled == 0


def test_progress_counts_a_better_median_among_nan_points():
    pop = np.linspace(1.0, 2.0, 20)
    pop[-3:] = np.nan
    progress = _Progress.start(pop)
    progress.update(np.r_[pop[0], pop[1:-3] - 0.01, pop[-3:]])  # best and NaN points stay
    assert progress.stalled == 0


def test_progress_ignores_a_rebounding_median():
    low, high = np.linspace(1.0, 2.0, 20), np.linspace(1.0, 2.0, 20) + 0.1
    high[0] = low[0]
    progress = _Progress.start(low)
    for pop in (high, low, high, low):
        progress.update(pop)
    assert progress.stalled == 4


def test_progress_does_not_count_finite_values_turning_nan():
    pop = np.linspace(1.0, 2.0, 20)
    progress = _Progress.start(pop)
    worse = pop.copy()
    worse[10:] = np.nan  # the worst half becomes NaN
    progress.update(worse)
    assert progress.stalled == 1


def test_progress_survives_values_near_the_float_limit():
    # Averaging two central values of -1e308 would overflow the median record to -inf.
    central = np.array([-1.5e308, -1.2e308, -1e308, -1e308, 1e308, 1.2e308])
    progress = _Progress.start(central)
    progress.update(np.r_[central[:2], -1.05e308, -1.05e308, central[4:]])
    assert progress.stalled == 0


def test_rosenbrock():
    n_dim = 20
    result = sceua.minimize(
        single_objective.rosenbrock,
        [(-5, 5)] * n_dim,
        seed=42,
    )
    assert result.success


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("func", "bound", "expected"),
    [
        (single_objective.booth, (-10, 10), 0),
        (single_objective.goldstein_price, (-2, 2), 3),
        (single_objective.levi_n13, (-10, 10), 0),
    ],
)
def test_objective(func, bound, expected):
    # The initial sample alone misses each tolerance, so a pass needs the evolution (L-05).
    result = sceua.minimize(func, [bound] * 2, seed=42)
    assert abs(result.fun - expected) < 1e-2
