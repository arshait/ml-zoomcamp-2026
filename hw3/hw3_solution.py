"""
ML Zoomcamp 2026 — Homework 3: Classification
Executable solution with verified answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "course_lead_scoring_2026.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"
HEATMAP_PATH = OUTPUT_DIR / "correlation_heatmap.png"


def load_and_prepare() -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(DATA_PATH)

    print("=== Dataset overview ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn names:")
    print(list(df.columns))
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nTarget distribution (converted):")
    print(df["converted"].value_counts())

    target = "converted"
    feature_cols = [c for c in df.columns if c != target]

    cat_cols = df[feature_cols].select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    for col in cat_cols:
        df[col] = df[col].fillna("NA")
    for col in num_cols:
        df[col] = df[col].fillna(0.0)

    return df, cat_cols, num_cols


def question_1(df: pd.DataFrame) -> str:
    print("\n=== Question 1: Industry mode ===")
    freq = df["industry"].value_counts()
    print("Industry frequency distribution:")
    print(freq)
    mode = df["industry"].mode().iloc[0]
    print(f"Mode: {mode}")
    return str(mode)


def question_2(df: pd.DataFrame, num_cols: list[str]) -> tuple[str, float]:
    print("\n=== Question 2: Numerical correlation ===")
    num_df = df[num_cols]
    corr = num_df.corr(method="pearson")
    print("\nPearson correlation matrix:")
    print(corr.round(4))

    pairs = [
        ("interaction_count", "lead_score"),
        ("number_of_courses_viewed", "lead_score"),
        ("number_of_courses_viewed", "interaction_count"),
        ("annual_income", "interaction_count"),
    ]
    pair_labels = [
        "interaction_count and lead_score",
        "number_of_courses_viewed and lead_score",
        "number_of_courses_viewed and interaction_count",
        "annual_income and interaction_count",
    ]

    best_label = None
    best_val = -np.inf
    print("\nSelected pair correlations:")
    for (a, b), label in zip(pairs, pair_labels):
        val = corr.loc[a, b]
        print(f"  {label}: {val:.6f}")
        if val > best_val:
            best_val = val
            best_label = label

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Pearson Correlation Matrix (Numerical Features)")
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=120)
    plt.close()
    print(f"Heatmap saved to: {HEATMAP_PATH}")

    return best_label, best_val


def split_data(df: pd.DataFrame):
    print("\n=== Train / validation / test split ===")
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    y_train = df_train["converted"]
    y_val = df_val["converted"]
    y_test = df_test["converted"]

    X_train = df_train.drop(columns=["converted"])
    X_val = df_val.drop(columns=["converted"])
    X_test = df_test.drop(columns=["converted"])

    n = len(df)
    print(f"Total rows: {n}")
    print(f"df_full_train: {len(df_full_train)}, df_test: {len(df_test)}")
    print(f"df_train: {len(df_train)}, df_val: {len(df_val)}")
    assert "converted" not in X_train.columns
    assert len(df_train) + len(df_val) + len(df_test) == n

    return X_train, X_val, X_test, y_train, y_val, y_test, df_train


def question_3(df_train: pd.DataFrame, cat_cols: list[str]) -> tuple[str, dict[str, float]]:
    print("\n=== Question 3: Mutual information ===")
    compare = ["industry", "location", "lead_source", "employment_status"]
    scores = {}
    for col in cat_cols:
        score = mutual_info_score(df_train["converted"], df_train[col])
        scores[col] = score

    print("All categorical features (rounded to 2 decimals):")
    for col in sorted(scores, key=lambda c: scores[c], reverse=True):
        print(f"  {col}: {round(scores[col], 2)}")

    compare_scores = {c: round(scores[c], 2) for c in compare}
    print("\nComparison options:")
    for c in compare:
        print(f"  {c}: {compare_scores[c]}")

    best = max(compare, key=lambda c: scores[c])
    print(f"Highest among options: {best}")
    return best, compare_scores


def train_lr_accuracy(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    C: float = 1.0,
) -> tuple[float, np.ndarray]:
    train_dicts = X_train.to_dict(orient="records")
    val_dicts = X_val.to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train_encoded = dv.fit_transform(train_dicts)
    X_val_encoded = dv.transform(val_dicts)

    model = LogisticRegression(
        solver="liblinear",
        C=C,
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train_encoded, y_train)
    y_pred = model.predict(X_val_encoded)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy, y_pred


def question_4(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> tuple[float, float, str | None]:
    print("\n=== Question 4: Logistic regression ===")
    accuracy, _ = train_lr_accuracy(X_train, X_val, y_train, y_val, C=1.0)
    rounded = round(accuracy, 2)
    options = [0.55, 0.65, 0.75, 0.85]
    match = rounded if rounded in options else None
    print(f"Raw accuracy: {accuracy}")
    print(f"Rounded (2 decimals): {rounded}")
    print(f"Matching option: {match}")
    return accuracy, rounded, match


def question_5(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    baseline_accuracy: float,
) -> str:
    print("\n=== Question 5: Feature elimination ===")
    candidates = ["lead_source", "number_of_courses_viewed", "interaction_count"]
    rows = []
    for feat in candidates:
        X_tr = X_train.drop(columns=[feat])
        X_va = X_val.drop(columns=[feat])
        acc, _ = train_lr_accuracy(X_tr, X_va, y_train, y_val, C=1.0)
        diff = baseline_accuracy - acc
        rows.append(
            {
                "removed_feature": feat,
                "baseline_accuracy": baseline_accuracy,
                "accuracy_without": acc,
                "difference": diff,
            }
        )

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    best_row = table.loc[table["difference"].idxmin()]
    selected = best_row["removed_feature"]
    print(f"Smallest difference (feature to select): {selected}")
    return str(selected)


def question_6(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> float:
    print("\n=== Question 6: Regularization ===")
    C_values = [0.000001, 0.00001, 0.0001, 0.001]
    results = []
    for C in C_values:
        acc, _ = train_lr_accuracy(X_train, X_val, y_train, y_val, C=C)
        results.append(
            {
                "C": C,
                "raw_accuracy": acc,
                "rounded_3": round(acc, 3),
            }
        )
    table = pd.DataFrame(results)
    print(table.to_string(index=False))

    best_rounded = table["rounded_3"].max()
    tied = table[table["rounded_3"] == best_rounded].sort_values("C")
    best_C = float(tied.iloc[0]["C"])
    print(f"Selected C (smallest among ties): {best_C}")
    return best_C


def main() -> dict:
    df, cat_cols, num_cols = load_and_prepare()

    q1 = question_1(df)
    q2_pair, q2_val = question_2(df, num_cols)

    X_train, X_val, X_test, y_train, y_val, y_test, df_train = split_data(df)
    q3, q3_scores = question_3(df_train, cat_cols)

    baseline_acc, q4_rounded, q4_option = question_4(X_train, X_val, y_train, y_val)
    q5 = question_5(X_train, X_val, y_train, y_val, baseline_acc)
    q6 = question_6(X_train, X_val, y_train, y_val)

    # Map Q2 to homework option text
    q2_options = {
        "interaction_count and lead_score": "`interaction_count` and `lead_score`",
        "number_of_courses_viewed and lead_score": "`number_of_courses_viewed` and `lead_score`",
        "number_of_courses_viewed and interaction_count": "`number_of_courses_viewed` and `interaction_count`",
        "annual_income and interaction_count": "`annual_income` and `interaction_count`",
    }

    results = {
        "q1": q1,
        "q2": q2_pair,
        "q2_value": q2_val,
        "q3": q3,
        "q3_scores": q3_scores,
        "q4_raw": baseline_acc,
        "q4_rounded": q4_rounded,
        "q4_option": q4_option,
        "q5": q5,
        "q6": q6,
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
