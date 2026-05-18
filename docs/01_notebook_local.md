# Local Notebook Guide

This guide is only for running and understanding `notebooks/infra-mlpipeline.ipynb`.
It is an exploratory local workflow (separate from the production-oriented `training/train.py` and CI pipelines).

## 1) Create and activate a Python virtual environment

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
```

Verify:

```bash
python --version
which python
```

## 2) Install notebook dependencies

Install the packages used by the notebook imports:

```bash
pip install \
  jupyter \
  numpy \
  pandas \
  scikit-learn \
  xgboost \
  joblib \
  pandas-gbq \
  google-cloud-bigquery \
  google-auth
```

## 3) Start Jupyter

```bash
jupyter notebook
```

This prints a local URL (usually `http://localhost:8888/...`) in the terminal.

## 4) Open the notebook

In Jupyter, navigate to:

`notebooks/infra-mlpipeline.ipynb`

Then run cells top-to-bottom.

## Notes

- The notebook is retained as an exploratory local workflow for traceability.
- Current productionized training/inference flows are documented in:
  - `docs/02_training.md`
  - `docs/03_inference.md`
