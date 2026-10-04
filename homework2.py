"""ML Zoomcamp 2026 - Homework 2. Reproduces all numerical answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "car_fuel_efficiency_2026.csv"
FEATURE_COLUMNS = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
]
TARGET = "fuel_efficiency_mpg"

Q1_OPTIONS = FEATURE_COLUMNS
Q2_OPTIONS = [204, 254, 304, 354]
Q3_OPTIONS = ["With 0", "With mean", "Both are equally good"]
Q4_OPTIONS = [0, 0.01, 0.1, 1, 5, 10, 100]
Q5_OPTIONS = [0.006, 0.016, 0.029, 0.036]
Q6_OPTIONS = [0.236, 2.236, 22.10, 221.0]


def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X_b = np.column_stack([ones, X])
    XTX = X_b.T.dot(X_b)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X_b.T).dot(y)
    return w_full[0], w_full[1:]


def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X_b = np.column_stack([ones, X])
    XTX = X_b.T.dot(X_b)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X_b.T).dot(y)
    return w_full[0], w_full[1:]


def split_df(df: pd.DataFrame, seed: int):
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].copy()
    df_val = df.iloc[idx[n_train : n_train + n_val]].copy()
    df_test = df.iloc[idx[n_train + n_val :]].copy()
    return df_train, df_val, df_test


def prepare_X(df: pd.DataFrame, fill_value=None, train_means: dict | None = None):
    X = df[FEATURE_COLUMNS].copy()
    if fill_value is not None:
        X = X.fillna(fill_value)
    elif train_means is not None:
        for col in FEATURE_COLUMNS:
            X[col] = X[col].fillna(train_means[col])
    else:
        X = X.fillna(0)
    return X.values


def predict(w0, w, X):
    return w0 + X.dot(w)


def closest_option(value: float, options: list[float], tol: float = 1e-9) -> float:
    for opt in options:
        if abs(value - opt) < tol:
            return opt
    return min(options, key=lambda o: abs(o - value))


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[FEATURE_COLUMNS + [TARGET]]

    # --- Q1 ---
    missing = df[FEATURE_COLUMNS].isna().sum()
    q1_col = missing.idxmax() if missing.any() else None
    for col in FEATURE_COLUMNS:
        if missing[col] > 0:
            q1_col = col
            break

    # --- Q2 ---
    hp_median = df["horsepower"].median()

    # --- Q3 ---
    df_train, df_val, df_test = split_df(df, seed=42)
    y_train = df_train[TARGET].values
    y_val = df_val[TARGET].values

    X_train_0 = prepare_X(df_train, fill_value=0)
    X_val_0 = prepare_X(df_val, fill_value=0)
    w0, w = train_linear_regression(X_train_0, y_train)
    rmse_zero = rmse(y_val, predict(w0, w, X_val_0))

    train_means = df_train[FEATURE_COLUMNS].mean(numeric_only=True).to_dict()
    X_train_mean = prepare_X(df_train, train_means=train_means)
    X_val_mean = prepare_X(df_val, train_means=train_means)
    w0_m, w_m = train_linear_regression(X_train_mean, y_train)
    rmse_mean = rmse(y_val, predict(w0_m, w_m, X_val_mean))

    rmse_zero_r = round(rmse_zero, 3)
    rmse_mean_r = round(rmse_mean, 3)
    print(f"RMSE with 0 = {rmse_zero_r}")
    print(f"RMSE with mean = {rmse_mean_r}")

    if rmse_zero_r < rmse_mean_r:
        q3_answer = "With 0"
    elif rmse_mean_r < rmse_zero_r:
        q3_answer = "With mean"
    else:
        q3_answer = "Both are equally good"

    # --- Q4 ---
    r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
    q4_results = {}
    best_r = None
    best_rmse_q4 = None
    print("r | validation_rmse")
    for r in r_values:
        w0_r, w_r = train_linear_regression_reg(X_train_0, y_train, r=r)
        val_rmse = rmse(y_val, predict(w0_r, w_r, X_val_0))
        q4_results[r] = val_rmse
        print(f"{r} | {round(val_rmse, 4)}")
        if best_rmse_q4 is None or val_rmse < best_rmse_q4 - 1e-15:
            best_rmse_q4 = val_rmse
            best_r = r

    # --- Q5 ---
    seeds = list(range(10))
    q5_scores = []
    for seed in seeds:
        tr, va, _ = split_df(df, seed=seed)
        y_tr = tr[TARGET].values
        y_va = va[TARGET].values
        X_tr = prepare_X(tr, fill_value=0)
        X_va = prepare_X(va, fill_value=0)
        w0_s, w_s = train_linear_regression(X_tr, y_tr)
        q5_scores.append(rmse(y_va, predict(w0_s, w_s, X_va)))

    q5_std = np.std(q5_scores)
    q5_std_r = round(q5_std, 3)

    # --- Q6 ---
    tr9, va9, te9 = split_df(df, seed=9)
    df_train_val = pd.concat([tr9, va9], ignore_index=True)
    y_train_val = df_train_val[TARGET].values
    y_test = te9[TARGET].values
    X_train_val = prepare_X(df_train_val, fill_value=0)
    X_test = prepare_X(te9, fill_value=0)
    w0_t, w_t = train_linear_regression_reg(X_train_val, y_train_val, r=0.001)
    test_rmse = rmse(y_test, predict(w0_t, w_t, X_test))

    q2_answer = closest_option(hp_median, Q2_OPTIONS)
    q5_answer = closest_option(q5_std_r, Q5_OPTIONS)
    q6_answer = closest_option(round(test_rmse, 3), Q6_OPTIONS)

    print()
    print("========================")
    print("ML ZOOMCAMP 2026 - HW2")
    print("========================")
    print()
    print("Q1:")
    print(f"Missing column: {q1_col}")
    print()
    print("Q2:")
    print(f"Horsepower median: {hp_median}")
    print(f"Answer: {q2_answer}")
    print()
    print("Q3:")
    print(f"RMSE with 0: {rmse_zero_r}")
    print(f"RMSE with mean: {rmse_mean_r}")
    print(f"Answer: {q3_answer}")
    print()
    print("Q4:")
    print(f"Best r: {best_r}")
    print(f"Best RMSE: {round(best_rmse_q4, 4)}")
    print(f"Answer: {best_r}")
    print()
    print("Q5:")
    print("RMSE scores:")
    for seed, score in zip(seeds, q5_scores):
        print(f"  seed {seed}: {score}")
    print(f"Standard deviation: {q5_std_r} (raw std: {q5_std})")
    print(f"Answer: {q5_answer}")
    print()
    print("Q6:")
    print(f"Test RMSE: {test_rmse}")
    print(f"Rounded test RMSE: {round(test_rmse, 3)}")
    print(f"Answer: {q6_answer}")


if __name__ == "__main__":
    main()
