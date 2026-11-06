# Homework 10 — Answers (ML Zoomcamp 2026)

HW5 `pipeline.bin` SHA-256 rechecked before use:

`1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`

The image was built from the existing `hw5` Dockerfile. HW5 files were not modified. Manifests in this folder are the official 2026 copies.

An existing kind cluster named `agent-relay` was left untouched. All kubectl commands used `--context kind-mlzoomcamp-2026`. After the checks, the current kubectl context was switched back to `kind-agent-relay`. The new cluster was not deleted.

## Question 1 — local container probability

```text
docker build -t zoomcamp-model:2026-hw10 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw10
python q6_test.py
```

Health: `status=ok`, model SHA-256 matches the expected artifact.

`q6_test.py`: `{'conversion_probability': 0.769799, 'conversion': True}`

**Answer: `0.770`**

This matches the previously verified Homework 5 value. The standalone container was stopped before port-forwarding.

## Question 2 — environment check (ungraded)

```text
kind version
kind v0.33.0 go1.26.7 windows/amd64

kubectl version --client
Client Version: v1.36.1
Kustomize Version: v5.8.1
```

The created node image was `kindest/node:v1.37.0`. These versions are diagnostics, not a graded choice.

## Question 3 — smallest deployable unit

**Answer: `Pod`**

## Question 4 — default service type

```text
kubectl get services --context kind-mlzoomcamp-2026
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.96.0.1    <none>        443/TCP   14s
```

**Answer: `ClusterIP`**

## Question 5 — load the local image

The command that placed `zoomcamp-model:2026-hw10` on node `mlzoomcamp-2026-control-plane` was:

```text
kind load docker-image zoomcamp-model:2026-hw10 --name mlzoomcamp-2026
```

**Answer: `kind load docker-image`**

## Question 6 — container port

`deployment.yaml` sets `containerPort: 9696`. After apply:

```text
kubectl rollout status deployment/subscription --context kind-mlzoomcamp-2026
deployment "subscription" successfully rolled out

subscription-7c9c5d48f4-bwkf2   1/1   Running
```

Live container port from the deployment object: `9696`.

**Answer: `9696`**

## Question 7 — service selector and forwarded inference

Service selector from the live object: `{"app":"subscription"}`. That matches the deployment pod label.

```text
kubectl port-forward service/subscription 9696:80 --context kind-mlzoomcamp-2026
python ..\hw5\q6_test.py
{'conversion_probability': 0.769799, 'conversion': True}
```

Difference from Q1 is 0, inside 0.005.

**Answer: `app: subscription`**

Forwarded probability rounded to 3 decimals: `0.770`.

## Question 8 — HPA maxReplicas

`hpa.yaml` declares `minReplicas: 1`, `maxReplicas: 3`, CPU target `averageUtilization: 20`.

```text
kubectl get hpa subscription-hpa --context kind-mlzoomcamp-2026
subscription-hpa   Deployment/subscription   cpu: <unknown>/20%   1   3   1
```

Deployment requests are `cpu: 100m` and `memory: 64Mi`.

CPU metrics are `<unknown>` because this kind cluster has no metrics-server (`pods.metrics.k8s.io` was not found). The HPA object was applied. Replica scaling was not observed and is not claimed.

**Answer: `3`**

## Test status

| Check | Status |
|-------|--------|
| Image build, health, Q1 `q6_test.py` | PASS |
| Named cluster, default service, image load, rollout | PASS |
| Service selector and port-forward inference | PASS |
| HPA manifest applied; maxReplicas 3 | PASS |
| HPA scale decision | not verified (metrics-server missing) |
| Cluster deletion | not run |

Cleanup, only when you approve it:

```powershell
kind delete cluster --name mlzoomcamp-2026
```
