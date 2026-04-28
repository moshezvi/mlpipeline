# Local Notebook Guide

This guide is only for running and understanding `notebooks/infra-takehome.ipynb`.

## 1) Create and activate a Python virtual environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify:

```bash
python --version
which python
```

## 2) Install only notebook dependencies

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

`notebooks/infra-takehome.ipynb`

Then run cells top-to-bottom.
