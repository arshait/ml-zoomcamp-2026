# Homework 9 — Serverless Deep Learning

ML Zoomcamp 2026. Official files from `cohorts/2026/homework/09-serverless`. The ONNX pair is the frozen course model, not an export from Homework 8.

## Download and verify

Large weights are gitignored. Download them beside `lambda_function.py`:

```powershell
cd hw9
$prefix = "https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
Invoke-WebRequest "$prefix/hair_classifier_v1.onnx" -OutFile hair_classifier_v1.onnx
Invoke-WebRequest "$prefix/hair_classifier_v1.onnx.data" -OutFile hair_classifier_v1.onnx.data
Invoke-WebRequest "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg" -OutFile sample.jpeg
Get-FileHash hair_classifier_v1.onnx, hair_classifier_v1.onnx.data, sample.jpeg -Algorithm SHA256
```

Expected hashes are in `asset_manifest.json`.

## Local inference

```powershell
uv venv .venv --python 3.11
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe smoke_test.py
```

## Lambda container

```powershell
docker build -t mlzoomcamp-2026-serverless .
docker run --rm -p 9000:8080 mlzoomcamp-2026-serverless
```

From another shell, post the body in `invoke_body.json`:

```powershell
curl.exe -sS -X POST "http://127.0.0.1:9000/2015-03-31/functions/function/invocations" -H "Content-Type: application/json" --data-binary "@invoke_body.json"
```

Verified container body: `straight_probability` 0.727697 (`0.728` to 3 decimals), `straight` true. This matches local ONNX inference.

## Answers

See `answers.md`.

## Learning in public

Completed Module 9 of Machine Learning Zoomcamp 2026.

I served a checksummed ONNX hair classifier with the official Lambda handler, checked bilinear 200x200 preprocessing, and invoked the same image through a local Lambda container. Local and container probabilities both rounded to 0.728.

My repo: <YOUR_GITHUB_URL>/tree/master/hw9

#mlzoomcamp #DataTalksClub Alexey Grigorev
