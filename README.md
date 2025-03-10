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

SCE-UA is a lightweight Python package 

📚 Full documentation is available [here](https://sceua.readthedocs.io).

## Installation

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

## Quick Start Guide


```python
slope_files = s3dep.get_map("Slope Degrees", bbox, data_dir)
dem = s3dep.tiffs_to_da(slope_files, bbox)
```

![Slope Example](https://raw.githubusercontent.com/cheginit/sceua/main/docs/examples/images/slope_dynamic.png)

## Contributing

We welcome contributions! Please see the
[contributing](https://sceua.readthedocs.io/en/latest/CONTRIBUTING/) section for
guidelines and instructions.
