# SCE-UA: Parameter Calibration with Shuffle Complex Evolution Algorithm

[![PyPi](https://img.shields.io/pypi/v/sceua.svg)](https://pypi.python.org/pypi/sceua)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/sceua.svg)](https://anaconda.org/conda-forge/sceua)
[![CodeCov](https://codecov.io/gh/cheginit/sceua/branch/main/graph/badge.svg)](https://codecov.io/gh/cheginit/sceua)
[![Python Versions](https://img.shields.io/pypi/pyversions/sceua.svg)](https://pypi.python.org/pypi/sceua)
[![Downloads](https://static.pepy.tech/badge/sceua)](https://pepy.tech/project/sceua)

[![CodeFactor](https://www.codefactor.io/repository/github/cheginit/sceua/badge)](https://www.codefactor.io/repository/github/cheginit/sceua)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cheginit/sceua/HEAD?labpath=docs%2Fexamples)

SCE-UA is a lightweight Python package that implements the Shuffled Complex Evolution
(SCE-UA) algorithm for parameter calibration of hydrological models. Built on NumPy and
SciPy, it's designed for seamless integration into existing Python workflows while
providing robust global optimization capabilities.

Full documentation is available at [sceua.readthedocs.io](https://sceua.readthedocs.io).

## 🌟 Features

The major improvements over the original SCE-UA algorithm include:

- **Adaptive smoothing parameter (theta)** based on problem scale, which helps with
    numerical stability and convergence
- **Latin Hypercube Sampling** for initial population generation instead of uniform
    random sampling, providing better coverage of the parameter space
- **PCA-based recovery mechanism** to identify and recover lost dimensions in the
    population (from [Chu et al. 2010](https://doi.org/10.1029/2010wr009224))
- **Best solution inclusion** in every complex to accelerate convergence
- **Optimized complex evolution strategy** with improved reflection, contraction, and
    mutation operations
- **Automatic parameter determination** with sensible defaults based on problem
    dimensionality
- **Multithreading support** for parallel evaluation of the objective function
- **Comprehensive convergence criteria** including function value tolerance, parameter
    tolerance, and maximum iterations
- **Detailed results object** that provides comprehensive information about the
    optimization process
- **Type hinting and modern Python implementation** following best practices for better
    code maintainability

## 📦 Installation

Choose your preferred installation method:

### Using `pip`

```console
pip install sceua
```

### Using `micromamba` (recommended)

```console
micromamba install -c conda-forge sceua
```

Alternatively, you can use `conda` or `mamba`.

## 🚀 Quick Start Guide

The SCE-UA package has one main function called `minimize`, which is used to optimize a
given objective function. The API follows a similar pattern to `scipy.optimize.minimize`
for easy adoption.

### Key Parameters

The most important tuning parameters are `n_complexes` and `n_points_complex` for all
cases, and `pca_freq` and `pca_tol` for cases with highly correlated parameters and/or
high-dimensional problems. The default values are set to reasonable defaults, but you
can adjust them based on your specific problem, if needed. Please refer to the
[API reference](https://sceua.readthedocs.io/en/latest/reference/) for more details.

### Example Usage

The signature for the objective function can be either:

```python
def objective(params: FloatArray) -> float:
    """Objective function to minimize."""
    # Your implementation here
```

or

```python
def objective(params: FloatArray, *args: Any) -> float:
    """Objective function to minimize."""
    # Your implementation here
```

where `params` is a NumPy array of parameters to optimize, and `args` is a tuple of
additional arguments to pass to the objective function. The function should return a
single floating-point value representing the objective function value.

Here's a simple example of how to use the package:

```python
from typing import Any
import numpy as np
from numpy.typing import NDArray

import sceua

FloatArray = NDArray[np.floating[Any]]


def objective(params: FloatArray, *args: Any) -> float:
    """Objective function to minimize."""
    sim_arg1, sim_arg2, obs = args
    sim = simulation_func(params, sim_arg1, sim_arg2)
    return metric_func(sim, obs)


# Define parameter bounds as a sequence of (min, max) pairs
bounds = [(lower1, upper1), (lower2, upper2), ..., (lowerN, upperN)]

# Run optimization
result = sceua.minimize(objective, bounds, args=(sim_arg1, sim_arg2, obs), seed=42, max_workers=8)

# Access the optimization results
best_params = result.x
best_function_value = result.fun
num_iterations = result.nit
num_function_evaluations = result.nfev
```

### Result Object

The `result` object contains the following attributes:

- `x` (`numpy.ndarray`): Best parameters found.
- `fun` (`float`): Best function value corresponding to the best parameters.
- `nit` (`int`): Number of iterations.
- `nfev` (`int`): Number of function evaluations.
- `message` (`str`): Message describing the termination reason.
- `success` (`bool`): Whether the optimization was successful.
- `xv` (`numpy.ndarray`): All evaluated parameter sets.
- `funv` (`numpy.ndarray`): Function values for all evaluated parameter sets.

### Visualization

The result attributes can be used to create convergence plots and analyze optimization
performance:

```python
import matplotlib.pyplot as plt


_, ax1 = plt.subplots()
ax1.plot(np.minimum.accumulate(results.funv))
ax2 = ax1.twinx()
ax2.set_yscale("log")
ax2.plot(np.cumsum(results.funv - true_min), color="r")
```

![Convergence](https://raw.githubusercontent.com/cheginit/sceua/main/docs/examples/images/convergence.png)

Check the [docs](https://sceua.readthedocs.io) for more examples and API details.

## 📚 References

This package is based on the following references:

- Duan, Q., Sorooshian, S., & Gupta, V. K. (1992). Effective and efficient global
    optimization for conceptual rainfall-runoff models. Water Resources Research, 28(4),
    1015-1031. [doi:10.1029/91WR02985](https://doi.org/10.1029/91WR02985)
- Duan, Q., Gupta, V. K., & Sorooshian, S. (1994). Optimal use of the SCE-UA global
    optimization method for calibrating watershed models. Journal of Hydrology,
    158(3-4), 265-284.
    [doi:10.1016/0022-1694(94)90057-4](<https://doi.org/10.1016/0022-1694(94)90057-4>)
- Duan, Q., Sorooshian, S., & Gupta, V. K. (1994). A shuffled complex evolution approach
    for effective and efficient global minimization. Journal of optimization theory and
    applications, 76(3), 501-521.
    [doi:10.1007/BF00939380](https://doi.org/10.1007/BF00939380)
- Muttil, N., & Jayawardena, A. W. (2008). Shuffled Complex Evolution model calibrating
    algorithm: enhancing its robustness and efficiency. Hydrological Processes, 22(23),
    4628-4638. Portico. [doi:10.1002/hyp.7082](https://doi.org/10.1002/hyp.7082)
- Chu, W., Gao, X., & Sorooshian, S. (2010). Improving the shuffled complex evolution
    scheme for optimization of complex nonlinear hydrological systems: Application to
    the calibration of the Sacramento soil-moisture accounting model. Water Resources
    Research, 46(9). Portico.
    [doi:10.1029/2010wr009224](https://doi.org/10.1029/2010wr009224)

Additionally, some ideas were inspired by [UQPyL](https://github.com/smasky/UQPyL) and
[SAMBO](https://github.com/sambo-optimization/sambo) Python packages.

## 🤝 Contributing

We welcome contributions! Please see the
[contributing](https://sceua.readthedocs.io/en/latest/CONTRIBUTING/) section for
guidelines and instructions.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
