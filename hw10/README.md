# Homework 10 — Kubernetes and Model Serving

ML Zoomcamp 2026. Manifests are the official `deployment.yaml`, `service.yaml`, and `hpa.yaml`. The image is built from the existing Homework 5 project. Do not retrain the model.

## Build and check the image

```powershell
cd hw5
Get-FileHash pipeline.bin -Algorithm SHA256
# 1646BBDCD38D4F044DA6B630C5B332C93A314245A8C21929011C42DE51F629F1
docker build -t zoomcamp-model:2026-hw10 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw10
```

In another shell:

```powershell
cd hw5
python q6_test.py
```

Verified probability: `0.769799` (`0.770` to 3 decimals). Stop that container before using port 9696 for Kubernetes.

## Kind cluster

This machine already had a kind cluster named `agent-relay`. It was not modified.

```powershell
kind --version
kubectl version --client
kind get clusters
kind create cluster --name mlzoomcamp-2026
```

Use the context explicitly:

```powershell
kubectl get services --context kind-mlzoomcamp-2026
kind load docker-image zoomcamp-model:2026-hw10 --name mlzoomcamp-2026
kubectl apply -f deployment.yaml --context kind-mlzoomcamp-2026
kubectl rollout status deployment/subscription --context kind-mlzoomcamp-2026
kubectl apply -f service.yaml --context kind-mlzoomcamp-2026
kubectl apply -f hpa.yaml --context kind-mlzoomcamp-2026
kubectl get hpa subscription-hpa --context kind-mlzoomcamp-2026
kubectl port-forward service/subscription 9696:80 --context kind-mlzoomcamp-2026
```

Then run `hw5\q6_test.py` again. The forwarded call returned the same `0.769799`.

The HPA CPU target stayed `<unknown>/20%` because metrics-server is not installed. That does not change `maxReplicas: 3`.

## Leave the cluster until you approve deletion

```powershell
kind delete cluster --name mlzoomcamp-2026
```

That command was not run. The kubectl current context was returned to `kind-agent-relay` after the checks. The `mlzoomcamp-2026` cluster is still present.

## Answers

See `answers.md`.

Q1 `0.770`. Q2 ungraded (`kind v0.33.0`, kubectl client `v1.36.1`). Q3 `Pod`. Q4 `ClusterIP`. Q5 `kind load docker-image`. Q6 `9696`. Q7 `app: subscription`. Q8 `3`.

## Learning in public

Completed Module 10 of Machine Learning Zoomcamp 2026.

I loaded the Homework 5 lead-scoring image into a kind cluster named mlzoomcamp-2026, exposed the subscription service with port-forwarding, and applied an HPA with a maximum of 3 replicas. The conversion probability through Kubernetes was 0.770, the same as the standalone container.

My repo: <YOUR_GITHUB_URL>/tree/master/hw10

#mlzoomcamp #DataTalksClub Alexey Grigorev
