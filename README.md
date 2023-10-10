# DISP-S1
[![Pytest and build docker image](https://github.com/opera-adt/disp-s1/actions/workflows/test-build-push.yml/badge.svg?branch=main)](https://github.com/opera-adt/disp-s1/actions/workflows/test-build-push.yml)

Surface Displacement workflows for OPERA DISP-S1 products.

Creates the science application software (SAS) using the [dolphin](https://github.com/opera-adt/dolphin) library.


## Development setup


### Prerequisite installs
1. Download source code:
```bash
git clone https://github.com/isce-framework/dolphin.git
git clone https://github.com/isce-framework/tophu.git
git clone https://github.com/opera-adt/disp-s1.git
```
2. Install dependencies, either to a new environment:
```bash
mamba env create --name my-disp-env --file disp-s1/conda-env.yml
conda activate my-disp-env
```
or install within your existing env with mamba.

3. Install `tophu, dolphin` and `disp-s1` via pip in editable mode
```bash
python -m pip install --no-deps -e dolphin/ tophu/ disp-s1/
```

### Setup for contributing


We use [pre-commit](https://pre-commit.com/) to automatically run linting, formatting, and [mypy type checking](https://www.mypy-lang.org/).
Additionally, we follow [`numpydoc` conventions for docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
To install pre-commit locally, run:

```bash
pre-commit install
```
This adds a pre-commit hooks so that linting/formatting is done automatically. If code does not pass the checks, you will be prompted to fix it before committing.
Remember to re-add any files you want to commit which have been altered by `pre-commit`. You can do this by re-running `git add` on the files.

Since we use [black](https://black.readthedocs.io/en/stable/) for formatting and [flake8](https://flake8.pycqa.org/en/latest/) for linting, it can be helpful to install these plugins into your editor so that code gets formatted and linted as you save.

### Running the unit tests

After making functional changes and/or have added new tests, you should run pytest to check that everything is working as expected.

First, install the extra test dependencies:
```bash
python -m pip install --no-deps -e .[test]
```

Then run the tests:

```bash
pytest
```

### Optional GPU setup

To enable GPU support (on aurora with CUDA 11.6 installed), install the following extra packages:
```bash
mamba install -c conda-forge "cudatoolkit=11.6" cupy "pynvml>=11.0"
```
