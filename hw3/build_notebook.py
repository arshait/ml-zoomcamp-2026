"""Generate hw3_classification_2026.ipynb from hw3_solution logic."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "hw3_classification_2026.ipynb"

cells = []

def md(source: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})

def code(source: str):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": source.splitlines(keepends=True)})

md("""# ML Zoomcamp 2026 — Homework 3: Classification

**Objective:** Lead scoring classification using the pinned 2026 dataset (`course_lead_scoring_2026.csv`).

Target variable: `converted`.
""")

code("""import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split

DATA_PATH = "course_lead_scoring_2026.csv"
""")

md("## Load and explore the dataset")
code("""df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
display(df.head())
print("Columns:", list(df.columns))
print(df.dtypes)
print("Missing values:")
print(df.isna().sum())
print("Target distribution:")
print(df["converted"].value_counts())
""")

md("## Missing-value treatment")
code("""target = "converted"
feature_cols = [c for c in df.columns if c != target]

cat_cols = df[feature_cols].select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

for col in cat_cols:
    df[col] = df[col].fillna("NA")
for col in num_cols:
    df[col] = df[col].fillna(0.0)

print("Categorical:", cat_cols)
print("Numerical:", num_cols)
""")

md("## Question 1 — Industry mode")
code("""industry_freq = df["industry"].value_counts()
print(industry_freq)
q1_answer = df["industry"].mode().iloc[0]
print("Q1 answer:", q1_answer)
""")

md("## Question 2 — Correlation matrix and heatmap")
code("""corr = df[num_cols].corr(method="pearson")
display(corr.round(4))

pairs = [
    ("interaction_count", "lead_score"),
    ("number_of_courses_viewed", "lead_score"),
    ("number_of_courses_viewed", "interaction_count"),
    ("annual_income", "interaction_count"),
]
labels = [
    "interaction_count and lead_score",
    "number_of_courses_viewed and lead_score",
    "number_of_courses_viewed and interaction_count",
    "annual_income and interaction_count",
]

best_label, best_val = None, -np.inf
for (a, b), label in zip(pairs, labels):
    val = corr.loc[a, b]
    print(f"{label}: {val:.6f}")
    if val > best_val:
        best_val, best_label = val, label

print("Q2 answer:", best_label, best_val)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Pearson Correlation Matrix")
plt.tight_layout()
plt.show()
""")

md("## Train / validation / test split")
code("""df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

y_train = df_train["converted"]
y_val = df_val["converted"]
y_test = df_test["converted"]

X_train = df_train.drop(columns=["converted"])
X_val = df_val.drop(columns=["converted"])
X_test = df_test.drop(columns=["converted"])

print(len(df_train), len(df_val), len(df_test))
assert "converted" not in X_train.columns
""")

md("## Question 3 — Mutual information (training set only)")
code("""mi_rows = []
for col in cat_cols:
    score = mutual_info_score(df_train["converted"], df_train[col])
    mi_rows.append({"feature": col, "mi": score, "mi_rounded": round(score, 2)})

mi_df = pd.DataFrame(mi_rows).sort_values("mi", ascending=False)
display(mi_df)

compare = ["industry", "location", "lead_source", "employment_status"]
q3_answer = max(compare, key=lambda c: mutual_info_score(df_train["converted"], df_train[c]))
print("Q3 answer:", q3_answer)
""")

md("## Question 4 — Logistic regression")
code("""def train_lr(X_tr, X_va, y_tr, y_va, C=1.0):
    dv = DictVectorizer(sparse=False)
    X_tr_enc = dv.fit_transform(X_tr.to_dict(orient="records"))
    X_va_enc = dv.transform(X_va.to_dict(orient="records"))
    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000, random_state=42)
    model.fit(X_tr_enc, y_tr)
    y_pred = model.predict(X_va_enc)
    return accuracy_score(y_va, y_pred)

baseline_acc = train_lr(X_train, X_val, y_train, y_val, C=1.0)
q4_rounded = round(baseline_acc, 2)
print("Raw:", baseline_acc, "Rounded:", q4_rounded)
""")

md("## Question 5 — Feature elimination")
code("""candidates = ["lead_source", "number_of_courses_viewed", "interaction_count"]
rows = []
for feat in candidates:
    acc = train_lr(X_train.drop(columns=[feat]), X_val.drop(columns=[feat]), y_train, y_val)
    rows.append({
        "removed": feat,
        "baseline": baseline_acc,
        "without": acc,
        "difference": baseline_acc - acc,
    })

fe_df = pd.DataFrame(rows)
display(fe_df)
q5_answer = fe_df.loc[fe_df["difference"].idxmin(), "removed"]
print("Q5 answer:", q5_answer)
""")

md("## Question 6 — Regularization")
code("""C_values = [0.000001, 0.00001, 0.0001, 0.001]
reg_rows = []
for C in C_values:
    acc = train_lr(X_train, X_val, y_train, y_val, C=C)
    reg_rows.append({"C": C, "accuracy": acc, "rounded_3": round(acc, 3)})

reg_df = pd.DataFrame(reg_rows)
display(reg_df)

best_rounded = reg_df["rounded_3"].max()
q6_answer = reg_df[reg_df["rounded_3"] == best_rounded]["C"].min()
print("Q6 answer:", q6_answer)
""")

md("""## Final answers

| Q | Answer |
|---|--------|
| 1 | technology |
| 2 | interaction_count and lead_score |
| 3 | lead_source |
| 4 | 0.65 |
| 5 | number_of_courses_viewed |
| 6 | 0.001 |
""")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": cells,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
