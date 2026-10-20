# Homework 6 — Tree Ensembles

ML Zoomcamp 2026 | Module 6

## Setup

```powershell
cd hw6
pip install -r requirements.txt
```

## Run

```powershell
python run_hw6.py
jupyter nbconvert --to notebook --execute homework6.ipynb --inplace
```

## Files

- `homework6.ipynb` — executed notebook
- `answers.md` — verified quiz answers
- `data/car_fuel_efficiency_2026.csv` — 2026 cohort dataset

## Methods

Regression on `fuel_efficiency_mpg` with `DictVectorizer(sparse=True)` fit on training only. Models: shallow decision tree, random forests (tuning `n_estimators` and `max_depth`), and XGBoost with fixed 100 boosting rounds.

## LinkedIn draft

Wrapped Module 6 of #mlzoomcamp 2026 — decision trees, random forests, and XGBoost for fuel efficiency prediction. `model_year` drove the depth-1 split; slower learning (`eta=0.1`) beat `0.3` on validation RMSE. Thanks Alexey Grigorev and DataTalksClub! Repo: `<YOUR_GITHUB_URL>/tree/master/hw6`
