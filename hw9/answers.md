# Homework 9 — Answers (ML Zoomcamp 2026)

Start of this session's asset checks: 2026-11-06 (local). All three SHA-256 values were checked with `Get-FileHash` before inference.

| File | SHA-256 | Match |
|------|---------|-------|
| `hair_classifier_v1.onnx` | `2b1adcb51745b73609ac7efebf3b6199483a931ed6dfe35ab925978f4357c2d8` | yes |
| `hair_classifier_v1.onnx.data` | `6ba88582d6098a535f918d9a56ad23800f919ad7bd5aad67e5b2bb3e289e5a7e` | yes |
| `sample.jpeg` | `6b3ae4003b5b6d54b51ee4d5fc5f183126bb1be0bf7066e5a846785fdcbfd2f5` | yes |

Local environment: Python 3.11 venv, `onnxruntime==1.23.2`, `numpy==2.3.3`, `Pillow==11.3.0`, provider `CPUExecutionProvider`.

## Question 1 — output node name

`session.get_inputs()[0].name` = `input`, shape `['s77', 3, 200, 200]`.

`session.get_outputs()[0].name` = `output`, shape `['s77', 1]`.

**Answer: `output`**

## Question 2 — target image size

`prepare_image` resizes with `Image.Resampling.BILINEAR` to `(200, 200)`. Sample file is 1024x1024 RGB before resize. Prepared tensor shape is `(1, 3, 200, 200)`.

**Answer: `200x200`**

## Question 3 — first normalized pixel

`tensor[0, 0, 0, 0]` = `-1.0561692714691162`.

**Answer: `-1.056`**

## Question 4 — local ONNX probability

```text
session.run(["output"], {"input": prepared_tensor})[0] -> 0.7276966571807861
```

`python smoke_test.py` printed `{'straight_probability': 0.727697, 'straight': True}`.

Direct event and API Gateway-style event both returned `statusCode` 200 with `straight_probability` and boolean `straight`.

**Answer: `0.728`**

## Question 5 — Dockerfile base image

`Dockerfile` line 1:

`FROM public.ecr.aws/lambda/python:3.13@sha256:0fc65c4c9bace18ce7cb5b670119b847140a9c1c45f2b24d06ce4e35bef0a9be`

**Answer: `public.ecr.aws/lambda/python:3.13`**

## Question 6 — container probability

Image `mlzoomcamp-2026-serverless` was built and run with `-p 9000:8080`.

`POST http://127.0.0.1:9000/2015-03-31/functions/function/invocations`

Response:

```json
{"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": "{\"straight_probability\": 0.727697, \"straight\": true}"}
```

Rounded to 3 decimals: **0.728**. Absolute difference from Q4 is 0. Container test **PASS**.

**Answer: `0.728`**

## Test status

| Check | Status |
|-------|--------|
| Checksums | PASS |
| Local ONNX + smoke test + both event shapes | PASS |
| Docker build and invocation | PASS |
| AWS deploy | not run (optional, not graded) |
