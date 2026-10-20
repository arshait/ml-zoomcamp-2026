# Homework 5 — Answers (ML Zoomcamp 2026)

## Verification status

| Check | Result |
|-------|--------|
| `uv sync --locked` + `uv run python smoke_test.py` | **PASS** |
| Local FastAPI (`GET /health`, `POST /predict`) | **PASS** (HTTP 200) |
| Docker build/run + `python q6_test.py` | **PASS** |
| `uv run uvicorn predict:app ...` | **BLOCKED** (Application Control; not an app failure) |

---

## Question 1 — uv version (setup, ungraded)

**uv 0.12.17** (635500036 2026-09-18 x86_64-pc-windows-msvc)

---

## Question 2 — scikit-learn version

From `pyproject.toml` and `uv.lock`: **scikit-learn==1.7.2**

**Answer: 1.7.2**

---

## Question 3 — Pickle predict_proba (paid ads lead)

Verified with `uv run python` after SHA256 check on `pipeline.bin`:

```python
# lead: paid_ads, technology, employed, north_america;
# number_of_courses_viewed=2, annual_income=79276.0, interaction_count=4, lead_score=0.41
```

**Conversion probability: 0.533**

**Answer: 0.533**

---

## Question 4 — Local FastAPI (port 9696)

### Blocked command (course default)

```powershell
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

**Failed** on this machine: Windows Application Control blocked process spawning (`uvicorn` exit code 2). This does **not** indicate a broken FastAPI app.

### Supported Windows workaround

From the `hw5` directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn predict:app --host 0.0.0.0 --port 9696
```

**Verified:**

- `GET /health` → HTTP **200**, `status=ok`, expected `model_sha256`
- `POST /predict` (organic_search / europe client) → HTTP **200**
- `conversion_probability`: **0.769799** → report **0.770** (3 decimals)
- `conversion`: `true` (boolean); probability in (0, 1)

The Uvicorn process was **stopped intentionally** before Docker testing; exit code `4294967295` is from forced termination, not an application crash.

**Answer: 0.770**

---

## Question 5 — Docker base image

From `Dockerfile` line 1:

**Answer: `python:3.11.15-slim-bookworm`**

---

## Question 6 — Containerized API

Stop any local service on port 9696, then:

```powershell
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
python q6_test.py
```

**Verified:**

- `GET /health` → `status=ok`, `model_sha256=1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`
- `q6_test.py` → `conversion_probability=0.769799`, matches Q4 within tolerance

**Answer: 0.770**

---

## Artifact provenance

**SHA256 `pipeline.bin` (verified before unpickling):**

`1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`
