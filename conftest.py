"""Pytest configuration for the HYMOD model tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from numba import config as numba_config
from numba import njit

numba_config.THREADING_LAYER = "workqueue"

FloatArray = npt.NDArray[np.float32]


@njit(nogil=True)
def _linear_reservoir(
    x_slow: np.float32, inflow: np.float32, k_s: np.float32
) -> tuple[np.float32, np.float32]:
    """Run the linear reservoir model."""
    xn_slow = (1 - k_s) * x_slow + (1 - k_s) * inflow
    outflow = k_s / (1 - k_s) * xn_slow
    return xn_slow, outflow


@njit(nogil=True)
def _excess(
    prcp_t: np.float32,
    pet_t: np.float32,
    x_loss: np.float32,
    c_max: np.float32,
    b_exp: np.float32,
) -> tuple[np.float32, np.float32]:
    """Calculate excess precipitation and evaporation."""
    ct_prev = c_max * (1 - np.power(np.abs(1 - ((b_exp + 1) * (x_loss) / c_max)), 1 / (b_exp + 1)))
    er_1 = np.maximum(prcp_t - c_max + ct_prev, 0)
    s_1 = prcp_t - er_1
    dummy = np.minimum((ct_prev + s_1) / c_max, 1)
    xn = c_max / (b_exp + 1) * (1 - np.power(np.abs(1 - dummy), b_exp + 1))

    er_2 = np.maximum(s_1 - (xn - x_loss), 0.2)

    evap = (1 - (((c_max / (b_exp + 1)) - xn) / (c_max / (b_exp + 1)))) * pet_t
    xn = np.maximum(xn - evap, 0)

    return er_1 + er_2, xn


@njit(nogil=True)
def _simulate(
    prcp: FloatArray,
    pet: FloatArray,
    c_max: np.float32,
    b_exp: np.float32,
    alpha: np.float32,
    k_s: np.float32,
    k_q: np.float32,
) -> FloatArray:
    """Run HYMOD model."""
    x_loss = np.float32(0.0)
    x_slow = np.float32(0.0)
    outflow = np.float32(0.0)
    x_quick = np.zeros(3, dtype="f4")
    n_steps = prcp.shape[0]
    q_out = np.zeros(n_steps, dtype="f4")

    for t in range(n_steps):
        et, x_loss = _excess(prcp[t], pet[t], x_loss, c_max, b_exp)
        u_q = alpha * et
        u_s = (1 - alpha) * et
        x_slow, q_s = _linear_reservoir(x_slow, u_s, k_s)
        inflow = u_q
        for i in range(3):
            x_quick[i], outflow = _linear_reservoir(x_quick[i], inflow, k_q)
            inflow = outflow
        q_out[t] = q_s + outflow
    return q_out


@njit(nogil=True)
def compute_nse(sim: FloatArray, obs: FloatArray) -> float:
    """Compute Nash-Sutcliffe Efficiency."""
    nse = 1 - np.sum(np.square(obs - sim)) / np.sum(np.square(obs - np.mean(obs)))
    return nse


class HYMOD:
    """Simulate a watershed using HYMOD model."""

    def __init__(self, clm: pd.DataFrame, qobs: pd.Series[float], warmup_years: int) -> None:
        if not all(clm.columns.intersection(["pr", "pet"]) == ["pr", "pet"]):
            raise ValueError("clm must contain 'pr' and 'pet' columns")
        if len(clm) != len(qobs):
            raise ValueError("clm and qobs must have the same length")
        if len(qobs) < 365 * warmup_years:
            raise ValueError("Not enough data for the warmup period")
        self.prcp = clm["pr"].to_numpy("f4")
        self.pet = clm["pet"].to_numpy("f4")
        self.qobs = qobs.to_numpy("f4")
        self.cal_idx = np.s_[warmup_years * 365 :]
        self.bounds = [(1, 1500), (0.0, 1.99), (0.01, 0.99), (0.01, 0.14), (0.14, 0.99)]

    def fit(self, x: FloatArray) -> float:
        """Compute objective functions (NSE)."""
        qsim = self(x)
        nse = compute_nse(qsim[self.cal_idx], self.qobs[self.cal_idx])
        return -nse

    def __call__(self, x: FloatArray) -> FloatArray:
        """Simulate the watershed."""
        c_max, b_exp, alpha, k_s, k_q = x.astype("f4")
        return _simulate(self.prcp, self.pet, c_max, b_exp, alpha, k_s, k_q)


@pytest.fixture
def hymod() -> HYMOD:
    """Set up the HYMOD model with sample data."""
    data = pd.read_csv("tests/01666500_hymod.csv")
    return HYMOD(data[["pr", "pet"]], data["qobs"], 1)
