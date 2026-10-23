# pyright: reportMissingParameterType=false
from __future__ import annotations

import numpy as np
import pytest
from landscapes import single_objective

import sceua


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


def test_x0_out_of_bounds():
    with pytest.raises(ValueError, match="x0"):
        sceua.minimize(lambda x: float(x @ x), [(1, 2)] * 2, x0=[(0, 0)])


def test_one_dimensional():
    result = sceua.minimize(lambda x: float((x[0] - 0.3) ** 2), [(-1, 1)], seed=42)
    assert result.fun < 1e-6


def test_model(hymod):
    result = sceua.minimize(hymod.fit, hymod.bounds, seed=42, n_complexes=3 * len(hymod.bounds))
    assert result.success


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
