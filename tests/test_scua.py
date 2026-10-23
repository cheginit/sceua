# pyright: reportMissingParameterType=false
from __future__ import annotations

import numpy as np
import pytest
from landscapes import single_objective

import sceua
from sceua.sceua import _pca_recovery

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


def test_max_evals_overshoot_is_bounded():
    n_complexes, n_points_complex, max_evals = 4, 9, 200
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
    # The absolute stall rule stops runs near 1e-5 here, across seeds.
    assert result.fun < 1e-3


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
    moved = _pca_recovery(func, (), population, values, np.zeros(2), np.ones(2), 1e-3, rng)
    assert moved
    assert np.array_equal(population[0], best)
    assert np.allclose(values, [func(x) for x in population])
    assert not np.allclose(population[:, 1], 0.5 * population[:, 0])


def test_pca_recovery_ignores_parameter_units():
    lo, up = np.array([(1, 1500), (0.0, 1.99), (0.01, 0.99), (0.01, 0.14), (0.14, 0.99)]).T
    rng = np.random.default_rng(0)
    population = lo + rng.random((100, 5)) * (up - lo)  # spread evenly in every parameter
    values = np.zeros(100)
    assert not _pca_recovery(lambda _: 0.0, (), population, values, lo, up, 1e-3, rng)


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
    ("func", "b1", "b2", "n_complexes", "expected"),
    [
        (single_objective.booth, (-10, 10), (-10, 10), 200, 0),
        (single_objective.goldstein_price, (-2, 2), (-2, 2), 400, 3),
        (single_objective.levi_n13, (-10, 10), (-10, 10), 400, 0),
    ],
)
def test_objective(func, b1, b2, n_complexes, expected):
    result = sceua.minimize(func, [b1, b2], seed=42, n_complexes=n_complexes)
    assert abs(result.fun - expected) < 1e-2
