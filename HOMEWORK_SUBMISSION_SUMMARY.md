# Homework Submission Summary — ML Zoomcamp 2026

Generated after local execution on Windows (PowerShell). **Not committed or pushed** unless you approve.

## Homework 4 — Evaluation Metrics

| Q | Answer | Evidence |
|---|--------|----------|
| Q1 | **lead_score** | Train AUCs: lead_score 0.788, interaction_count 0.765, number_of_courses_viewed 0.723, annual_income 0.608 |
| Q2 | **0.732** | Validation ROC AUC (3 dp) |
| Q3 | **0.63** | Min \|precision − recall\| among valid thresholds |
| Q4 | **0.41** | Max F1 ≈ 0.758 |
| Q5 | **0.007** | 5-fold CV std (mean AUC 0.732) |
| Q6 | **0.001** (C) | Mean AUC 0.749, std 0.010 (beats C=1 mean 0.732) |

**Test status:** **PASS** — `run_hw4.py` and executed `hw4/homework4.ipynb`

**Submission:** https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw04

---

## Homework 5 — Deployment

| Q | Answer | Evidence |
|---|--------|----------|
| Q1 | **uv 0.12.17** | Ungraded setup check |
| Q2 | **1.7.2** | `pyproject.toml` / `uv.lock` |
| Q3 | **0.533** | `predict_proba` on specified lead |
| Q4 | **0.770** | POST `/predict` → 0.769799, conversion true |
| Q5 | **python:3.11.15-slim-bookworm** | `Dockerfile` |
| Q6 | **0.770** | Docker health OK; `q6_test.py` → 0.769799 (matches Q4) |

**Test status:** **PASS**

- Smoke test (`uv run python smoke_test.py`)
- Local API: `GET /health` and `POST /predict` → HTTP 200; Q4 prob **0.769799**
- Docker: build/run + `q6_test.py` → same prediction
- **BLOCKED (launcher only):** `uv run uvicorn ...` (Application Control). Workaround: `.\.venv\Scripts\python.exe -m uvicorn predict:app --host 0.0.0.0 --port 9696`

**SHA256 `pipeline.bin`:** `1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`

**Submission:** https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw05

---

## Homework 6 — Tree Ensembles

| Q | Answer | Evidence |
|---|--------|----------|
| Q1 | **model_year** | Depth-1 tree root feature |
| Q2 | **1.837** | RF n=10 val RMSE 1.8332 |
| Q3 | **150** | Lowest RMSE at 3 dp (1.769) |
| Q4 | **10** | Lowest mean RMSE across forest sizes (1.7544) |
| Q5 | **vehicle_weight** | Importance 0.203 vs others under 0.08 |
| Q6 | **0.1** | Val RMSE 1.7248 vs 1.8265 for eta=0.3 |

**Test status:** **PASS** — `run_hw6.py` and executed `hw6/homework6.ipynb`

**Submission:** https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw06

---

## Outstanding setup notes

1. **Windows Application Control** may block `.venv\Scripts\pip.exe` and **`uv run uvicorn`** (process spawn only). HW5 API and Docker tests **passed** via `python -m uvicorn` workaround. HW4/HW6: system Python when venv DLLs are blocked.
2. **HW4/HW6 `.venv`** under `hw4` is optional and gitignored; prefer system Python 3.12 if local venv DLLs are blocked.
3. **HW5 `_mlz_sparse/`** is a temporary sparse clone — gitignored; canonical files copied to `hw5/` root.

## Git files ready to commit (when approved)

```
.gitignore
README.md
HOMEWORK_SUBMISSION_SUMMARY.md
hw4/
hw5/   (excluding .venv, _mlz_sparse)
hw6/
```

## Reproduce (root)

```powershell
cd "D:\Zoomcamp\Machine Learning Zoomcamp\hw4"
python run_hw4.py

cd ..\hw5
uv sync --locked
uv run python smoke_test.py
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
python q6_test.py

cd ..\hw6
python run_hw6.py
```

## Suggested commit (after your approval)

```powershell
cd "D:\Zoomcamp\Machine Learning Zoomcamp"
git add .gitignore README.md HOMEWORK_SUBMISSION_SUMMARY.md hw4 hw5 hw6
git status
git commit -m "Add homework 4-6: metrics, deployment, and tree ensembles"
```

Do **not** push until you review.
