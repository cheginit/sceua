# pyright: reportMissingParameterType=false
from __future__ import annotations

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
