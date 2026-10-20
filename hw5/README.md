# Homework 5 — Model Deployment

ML Zoomcamp 2026 | Module 5

Official reference files from [05-deployment](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/05-deployment) (frozen `pipeline.bin`, `uv.lock`).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python **3.11.15** pinned via `.python-version`)
- Docker (for Q6)

## Setup

```powershell
cd hw5
uv --version
uv sync --locked
uv run python smoke_test.py
```

Verify model checksum:

```powershell
Get-FileHash pipeline.bin -Algorithm SHA256
# Expected: 1646BBDCD38D4F044DA6B630C5B332C93A314245A8C21929011C42DE51F629F1
```

## Local API (Q4)

### Course command (may be blocked on Windows)

```powershell
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

On some Windows hosts, **Application Control** blocks `uv run uvicorn` (spawn failure). That is an OS policy issue, not a broken app.

### Supported PowerShell workaround

Run from the `hw5` directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn predict:app --host 0.0.0.0 --port 9696
```

Test (expect HTTP 200):

```powershell
Invoke-RestMethod http://127.0.0.1:9696/health
Invoke-RestMethod http://127.0.0.1:9696/predict -Method Post -ContentType "application/json" -Body '{"lead_source":"organic_search","industry":"technology","employment_status":"employed","location":"europe","number_of_courses_viewed":4,"annual_income":80304.0,"interaction_count":7,"lead_score":0.74}'
```

Verified `conversion_probability`: **0.769799** (quiz: **0.770**).

Stop the local server before Docker (Q6) to free port 9696.

## Docker (Q6)

```powershell
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
python q6_test.py
```

Verified: health OK, same prediction as local API (**0.769799**).

## LinkedIn draft

Deployed a lead-scoring model for #mlzoomcamp 2026 with FastAPI, uv lockfiles, and Docker — including SHA256 verification of the frozen sklearn pipeline. Thanks Alexey Grigorev and DataTalksClub! Repo: `<YOUR_GITHUB_URL>/tree/master/hw5`
