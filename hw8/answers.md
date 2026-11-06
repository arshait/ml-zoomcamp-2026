# Homework 8 — Answers (ML Zoomcamp 2026)

Dataset archive SHA-256 (verified before extraction):

`9e53453e3017502f22860cb08558f0cdb343e102fb61feab75960616185e7d67`

ImageFolder after ignoring `data/train/curly/images700.gif`:

| Split | curly | straight | total |
|-------|------:|---------:|------:|
| train | 410 | 390 | 800 |
| held-out | 103 | 98 | 201 |

Class map: `curly=0`, `straight=1`. No new split. Evaluation images were not augmented.

Environment: Python 3.11.16 venv, `torch==2.9.0+cpu`, `torchvision==0.24.0+cpu`, `numpy==2.3.3`, `Pillow==11.3.0`, `torch.cuda.is_available() == False`.

`uv pip install -r requirements-cpu.txt` failed because uv only consults the first index that publishes `torch`. The same pins were installed from the official CPU index `https://download.pytorch.org/whl/cpu`, then numpy and Pillow were pinned back to the required versions.

Training used unmodified `reference_train.py` on CPU. One run, wall clock **5 minutes 50.6 seconds** (`EXIT 0`). Histories have 10 finite values in each series.

## Question 1 — loss

The model returns one logit. The reference criterion is `nn.BCEWithLogitsLoss()`.

**Answer: `BCEWithLogitsLoss`**

## Question 2 — parameter count

`sum(p.numel() for p in HairModel().parameters())` = **20073473** (biases included).

**Answer: `20073473`**

## Question 3 — median baseline train accuracy

`train_accuracy`:

`0.645, 0.7175, 0.755, 0.79125, 0.80875, 0.8075, 0.83625, 0.885, 0.91, 0.93875`

Median = `0.808125`.

**Answer: `0.81`**

## Question 4 — population std of baseline train loss

`np.std(train_loss, ddof=0)` = `0.13934069829701942`.

**Answer: `0.139`**

## Question 5 — mean augmented evaluation loss

Mean of the 10 augmented `evaluation_loss` values = `0.575828219008683`.

**Answer: `0.576`**

## Question 6 — mean of last five augmented evaluation accuracies

`0.7014925373134329, 0.7412935323383084, 0.7064676616915423, 0.7213930348258707, 0.7313432835820896`

Mean = `0.7203980099502487`.

**Answer: `0.72`**

## Test status

**PASS** — checksum, class counts, single canonical CPU run, both history files.
