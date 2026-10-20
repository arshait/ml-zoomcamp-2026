# Homework 4 — Evaluation Metrics

ML Zoomcamp 2026 | Module 4

## Setup

```powershell
cd hw4
# System Python 3.11+ with pandas, scikit-learn, matplotlib, jupyter
pip install -r requirements.txt
```

## Run

```powershell
python run_hw4.py
jupyter nbconvert --to notebook --execute homework4.ipynb --output homework4_executed.ipynb
```

## Files

- `homework4.ipynb` / `homework4_executed.ipynb` — full workflow
- `answers.md` — verified quiz answers
- `data/course_lead_scoring_2026.csv` — 2026 cohort dataset
- `plots/precision_recall_vs_threshold.png` — Q3 plot

## Methods

Lead conversion classification with train/validation/test splits (`random_state=1`). Missing categoricals → `"NA"`, numerics → `0.0`. One-hot encoding via `DictVectorizer`. Metrics: ROC AUC, precision/recall curves, F1, and 5-fold cross-validation for regularization tuning.

## LinkedIn draft

Just finished Homework 4 of #mlzoomcamp 2026 — evaluation metrics for lead scoring: ROC AUC, precision/recall trade-offs, F1, and cross-validated regularization. Key takeaway: `C=0.001` beat the default `C=1` on mean CV AUC (0.749 vs 0.732). Thanks Alexey Grigorev and DataTalksClub! Repo: `<YOUR_GITHUB_URL>/tree/master/hw4`
