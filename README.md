# Machine Learning Zoomcamp 2026

Personal homework repository for [DataTalksClub ML Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Homework

| Module | Folder | Topic |
|--------|--------|--------|
| 1 | [hw1](hw1/) | Introduction / setup |
| 2 | [hw2](hw2/) | Linear regression (car fuel efficiency) |
| 3 | [hw3](hw3/) | Classification (lead scoring) |
| 4 | [hw4](hw4/) | Evaluation metrics (ROC AUC, PR curves, CV) |
| 5 | [hw5](hw5/) | Model deployment (FastAPI, uv, Docker) |
| 6 | [hw6](hw6/) | Tree ensembles (RF, XGBoost) |
| 8 | [hw8](hw8/) | Neural networks (curly vs straight hair) |
| 9 | [hw9](hw9/) | Serverless deep learning (ONNX, Lambda) |
| 10 | [hw10](hw10/) | Kubernetes and model serving |

## Modules 4–6 (summary)

- **Module 4** — Lead conversion classification: single-feature AUC, logistic regression, precision/recall/F1 thresholds, and k-fold regularization tuning.
- **Module 5** — Deploy a frozen scikit-learn pipeline with FastAPI, reproducible `uv` environments, and a containerized service with health checks.
- **Module 6** — Regression with decision trees, random forests, and XGBoost on the 2026 car fuel efficiency dataset.
- **Module 8** — A fixed CPU PyTorch hair classifier, then the same model trained for 10 more epochs with the official augmentation.
- **Module 9** — A checksummed ONNX hair classifier behind the official Lambda handler, including a local container invocation.
- **Module 10** — The Homework 5 lead-scoring image on a named kind cluster, with a ClusterIP service and an HPA.

## Reproduce results

### Homework 4

```powershell
cd hw4
pip install -r requirements.txt
python run_hw4.py
jupyter nbconvert --to notebook --execute homework4.ipynb --inplace
```

Dataset: [course_lead_scoring_2026.csv](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv)

### Homework 5

```powershell
cd hw5
uv sync --locked
uv run python smoke_test.py
.\.venv\Scripts\python.exe -m uvicorn predict:app --host 0.0.0.0 --port 9696
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
python q6_test.py
```

Reference: [05-deployment](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/05-deployment)

### Homework 6

```powershell
cd hw6
pip install -r requirements.txt
python run_hw6.py
jupyter nbconvert --to notebook --execute homework6.ipynb --inplace
```

Dataset: [car_fuel_efficiency_2026.csv](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv)

### Homework 8

```powershell
cd hw8
# see README.md for the CPU wheel install
.\.venv\Scripts\python.exe reference_train.py --data-dir data --output-dir .
```

Reference: [08-deep-learning](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/08-deep-learning)

### Homework 9

```powershell
cd hw9
.\.venv\Scripts\python.exe smoke_test.py
docker build -t mlzoomcamp-2026-serverless .
docker run --rm -p 9000:8080 mlzoomcamp-2026-serverless
```

Reference: [09-serverless](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/09-serverless)

### Homework 10

```powershell
cd hw5
docker build -t zoomcamp-model:2026-hw10 .
kind create cluster --name mlzoomcamp-2026
kind load docker-image zoomcamp-model:2026-hw10 --name mlzoomcamp-2026
kubectl apply -f ..\hw10\deployment.yaml --context kind-mlzoomcamp-2026
kubectl apply -f ..\hw10\service.yaml --context kind-mlzoomcamp-2026
kubectl apply -f ..\hw10\hpa.yaml --context kind-mlzoomcamp-2026
```

Reference: [10-kubernetes](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/10-kubernetes)

## Notes

- Python 3.12 for HW4/HW6 notebooks; HW5 uses **Python 3.11.15** via `uv`.
- Virtual environments (`.venv`) are gitignored; run installs locally per folder.
- Quiz answers and metrics: see each `hw*/answers.md` and [HOMEWORK_SUBMISSION_SUMMARY.md](HOMEWORK_SUBMISSION_SUMMARY.md).
