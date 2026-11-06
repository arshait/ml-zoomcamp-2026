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

## Homework 8 — Neural networks

Session start for this work: 2026-11-06T17:20:26+05:30. One CPU run of `reference_train.py`, 5 minutes 50.6 seconds.

| Q | Answer | Evidence |
|---|--------|----------|
| Q1 | **BCEWithLogitsLoss** | One logit; official criterion |
| Q2 | **20073473** | `sum(p.numel() for p in model.parameters())` |
| Q3 | **0.81** | Median baseline train accuracy 0.808125 |
| Q4 | **0.139** | `np.std(train_loss, ddof=0)` = 0.139341 |
| Q5 | **0.576** | Mean augmented evaluation loss 0.575828 |
| Q6 | **0.72** | Mean of last five augmented evaluation accuracies 0.720398 |

**Test status:** **PASS**

Archive SHA-256 `9e53453e3017502f22860cb08558f0cdb343e102fb61feab75960616185e7d67`. ImageFolder 800/201, classes curly=0 and straight=1. Histories are length 10 and finite.

**Submission:** https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw08

**GitHub path:** `hw8/`

**Learning-in-public draft:** `hw8/README.md` and `LEARNING_IN_PUBLIC.md`

---

## Homework 9 — Serverless

| Q | Answer | Evidence |
|---|--------|----------|
| Q1 | **output** | `session.get_outputs()[0].name` |
| Q2 | **200x200** | `prepare_image` bilinear resize; tensor `(1, 3, 200, 200)` |
| Q3 | **-1.056** | `tensor[0, 0, 0, 0]` = -1.056169 |
| Q4 | **0.728** | Local ONNX output 0.727697 |
| Q5 | **public.ecr.aws/lambda/python:3.13** | Dockerfile `FROM` line |
| Q6 | **0.728** | Container body `straight_probability` 0.727697 |

**Test status:** **PASS**

All three asset SHA-256 values matched `asset_manifest.json`. Smoke test, both handler event shapes, Docker build, and container invocation passed. Difference between Q4 and Q6 is 0. AWS was not used.

**Submission:** https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw09

**GitHub path:** `hw9/` (ONNX weights gitignored; manifest has the download URLs)

**Learning-in-public draft:** `hw9/README.md` and `LEARNING_IN_PUBLIC.md`

---

## Homework 10 — Kubernetes

| Q | Answer | Evidence |
|---|--------|----------|
| Q1 | **0.770** | `q6_test.py` on `zoomcamp-model:2026-hw10` returned 0.769799 |
| Q2 | **ungraded** | kind v0.33.0; kubectl client v1.36.1 |
| Q3 | **Pod** | Smallest deployable unit |
| Q4 | **ClusterIP** | Live `kubernetes` service on `kind-mlzoomcamp-2026` |
| Q5 | **kind load docker-image** | Image loaded onto `mlzoomcamp-2026-control-plane` |
| Q6 | **9696** | Manifest and live deployment; pod 1/1 Running |
| Q7 | **app: subscription** | Service selector; port-forward `q6_test.py` returned 0.769799 |
| Q8 | **3** | `hpa.yaml` `maxReplicas`; `kubectl get hpa` showed MAXPODS 3 |

**Test status:** **PASS** for image, cluster, deployment, service, and forwarded inference.

HPA scaling is **not verified**. `kubectl get hpa` showed `cpu: <unknown>/20%` because metrics-server is absent. The cluster `mlzoomcamp-2026` was left running. It was not deleted. Context was returned to `kind-agent-relay`. The pre-existing cluster `agent-relay` was not changed.

**Submission:** https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw10

**GitHub path:** `hw10/`

**Learning-in-public draft:** `hw10/README.md` and `LEARNING_IN_PUBLIC.md`

---

## Outstanding setup notes

1. **Windows Application Control** may block `.venv\Scripts\pip.exe` and **`uv run uvicorn`** (process spawn only). HW5 API and Docker tests **passed** via `python -m uvicorn` workaround. HW4/HW6: system Python when venv DLLs are blocked.
2. **HW4/HW6 `.venv`** under `hw4` is optional and gitignored; prefer system Python 3.12 if local venv DLLs are blocked.
3. **HW5 `_mlz_sparse/`** is a temporary sparse clone — gitignored; canonical files copied to `hw5/` root.
4. **HW8 torch install:** `uv pip install -r requirements-cpu.txt` did not see `torch==2.9.0+cpu` under uv's default index policy. The same pins were installed from `https://download.pytorch.org/whl/cpu`.
5. **HW10 HPA:** metrics-server is missing, so CPU is `<unknown>`. Scaling was not demonstrated.
6. **HW10 cluster:** `mlzoomcamp-2026` is still up. Delete only with `kind delete cluster --name mlzoomcamp-2026` after approval. Do not delete `agent-relay`.

## Git files ready to commit (when approved)

HW4–HW6 are already on `master`. New work:

```
.gitignore
README.md
HOMEWORK_SUBMISSION_SUMMARY.md
LEARNING_IN_PUBLIC.md
hw8/   (excluding .venv, data/, data.zip, train_run.txt)
hw9/   (excluding .venv and the ONNX weight files)
hw10/
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

HW8–HW10 commands are in each folder README. Do not rerun the 20-epoch HW8 training if `history_baseline.json` and `history_augmented.json` are already present.

## Suggested commit (after your approval)

```powershell
cd "D:\Zoomcamp\Machine Learning Zoomcamp"
git status --short
git diff -- hw1 hw2 hw3 hw4 hw5 hw6
git add .gitignore README.md HOMEWORK_SUBMISSION_SUMMARY.md LEARNING_IN_PUBLIC.md hw8 hw9 hw10
git commit -m "Add homework 8-10: neural nets, serverless inference, and Kubernetes"
```

Do **not** push until you review.
