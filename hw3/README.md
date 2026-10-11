# ML Zoomcamp 2026 — Homework 3 (Classification)

Lead-scoring classification homework using the pinned 2026 dataset:

`https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv`

## Files

| File | Description |
|---|---|
| `course_lead_scoring_2026.csv` | Dataset (5000 rows) |
| `hw3_solution.py` | Executable solution (prints all answers) |
| `hw3_classification_2026.ipynb` | Beginner-friendly notebook |
| `homework_answers.md` | Verified answers for submission |
| `requirements.txt` | Python dependencies |
| `run_solution.ps1` | Docker runner (see below) |
| `outputs/correlation_heatmap.png` | Q2 heatmap (created on run) |

## Setup (local virtual environment)

```powershell
cd "D:\Zoomcamp\Machine Learning Zoomcamp\hw3"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Python solution

```powershell
python hw3_solution.py
```

## Run the notebook

```powershell
jupyter notebook hw3_classification_2026.ipynb
```

Or execute headless:

```powershell
jupyter nbconvert --to notebook --execute hw3_classification_2026.ipynb --output hw3_classification_2026_executed.ipynb
```

## Windows App Control / DLL note

On some Windows machines, `numpy` / `scikit-learn` native DLLs are blocked by Application Control policies (`ImportError: DLL load failed ... Application Control policy has blocked this file`).

If that happens, use **Docker Desktop** (included in this repo):

```powershell
.\run_solution.ps1
```

This runs the same script inside `python:3.12-slim` with dependencies installed in the container.

## Regenerate the notebook

```powershell
python build_notebook.py
```

## Official links

- Homework: https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/03-classification/homework.md
- Submit: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw03
