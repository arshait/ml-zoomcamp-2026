# Homework 8 — Neural Networks

ML Zoomcamp 2026. Official files: `reference_train.py`, `requirements-cpu.txt`, `dataset_manifest.json`.

## Data

The image archive is gitignored. Download and check it before unzipping:

```powershell
cd hw8
Invoke-WebRequest "https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip" -OutFile data.zip
Get-FileHash data.zip -Algorithm SHA256
# 9E53453E3017502F22860CB08558F0CDB343E102FB61FEAB75960616185E7D67
Expand-Archive data.zip -DestinationPath .
```

`ImageFolder` reads `data/train` (800 images) and `data/test` (201 images). One GIF under `data/train/curly` is ignored.

## Environment

Python 3.11 and the CPU wheels in `requirements-cpu.txt`.

On this machine, uv's default index policy rejected `torch==2.9.0+cpu` when the requirements file listed both the PyTorch CPU index and PyPI. The pins were installed explicitly:

```powershell
uv venv .venv --python 3.11
uv pip install --python .\.venv\Scripts\python.exe "torch==2.9.0+cpu" "torchvision==0.24.0+cpu" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .\.venv\Scripts\python.exe "numpy==2.3.3" "Pillow==11.3.0"
```

## Train

```powershell
.\.venv\Scripts\python.exe reference_train.py --data-dir data --output-dir .
```

This writes `history_baseline.json` and `history_augmented.json`. Do not start a second 20-epoch run if those files are already present. The recorded CPU run took 5 minutes 50.6 seconds.

Plots (separate from the pinned training environment):

```powershell
python plot_history.py
```

## Answers

See `answers.md`.

Q1 `BCEWithLogitsLoss`. Q2 `20073473`. Q3 `0.81`. Q4 `0.139`. Q5 `0.576`. Q6 `0.72`.

## Learning in public

Completed Module 8 of Machine Learning Zoomcamp 2026.

I trained the fixed CPU hair classifier for 10 baseline epochs and 10 more epochs with the official augmentation, without resetting the model or optimizer. Median baseline train accuracy was 0.81. The mean augmented evaluation loss was 0.576.

My repo: <YOUR_GITHUB_URL>/tree/master/hw8

#mlzoomcamp #DataTalksClub Alexey Grigorev
