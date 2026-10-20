"""HW4 computation script - generates answers and plots."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

DATA = Path(__file__).parent / "data" / "course_lead_scoring_2026.csv"
PLOTS = Path(__file__).parent / "plots"
PLOTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
target = "converted"
feature_cols = [c for c in df.columns if c != target]

cat_cols = df[feature_cols].select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

for col in cat_cols:
    df[col] = df[col].fillna("NA")
for col in num_cols:
    df[col] = df[col].fillna(0.0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

print("Shapes:", df_train.shape, df_val.shape, df_test.shape)

# Q1
q1_features = [
    "lead_score",
    "number_of_courses_viewed",
    "interaction_count",
    "annual_income",
]
y_train = df_train[target].values
best_feat = None
best_auc = -1.0
q1_results = {}
for feat in q1_features:
    x = df_train[feat].values.astype(float)
    auc = roc_auc_score(y_train, x)
    used_neg = False
    if auc < 0.5:
        x = -x
        auc = roc_auc_score(y_train, x)
        used_neg = True
    q1_results[feat] = (auc, used_neg)
    print(f"Q1 {feat}: AUC={auc:.6f} neg={used_neg}")
    if auc > best_auc:
        best_auc = auc
        best_feat = feat

print("Q1 answer:", best_feat)

# Q2
train_dicts = df_train[feature_cols].to_dict(orient="records")
val_dicts = df_val[feature_cols].to_dict(orient="records")
y_val = df_val[target].values

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
model.fit(X_train, y_train)
y_val_proba = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)
print(f"Q2 val AUC: {val_auc:.3f}")

# Q3 & Q4
thresholds = np.arange(0.0, 1.01, 0.01)
precisions = []
recalls = []
f1s = []
valid_thresholds = []

for t in thresholds:
    y_pred = (y_val_proba >= t).astype(int)
    tp = ((y_pred == 1) & (y_val == 1)).sum()
    fp = ((y_pred == 1) & (y_val == 0)).sum()
    fn = ((y_pred == 0) & (y_val == 1)).sum()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if prec == 0.0 and rec == 0.0:
        continue

    valid_thresholds.append(t)
    precisions.append(prec)
    recalls.append(rec)
    if prec + rec > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0
    f1s.append(f1)

precisions = np.array(precisions)
recalls = np.array(recalls)
f1s = np.array(f1s)
valid_thresholds = np.array(valid_thresholds)

diff = np.abs(precisions - recalls)
q3_idx = np.argmin(diff)
q3_thresh = valid_thresholds[q3_idx]
print(f"Q3 threshold: {q3_thresh:.2f} prec={precisions[q3_idx]:.4f} rec={recalls[q3_idx]:.4f}")

q4_idx = np.argmax(f1s)
q4_thresh = valid_thresholds[q4_idx]
print(f"Q4 threshold: {q4_thresh:.2f} max F1={f1s[q4_idx]:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(valid_thresholds, precisions, label="Precision")
plt.plot(valid_thresholds, recalls, label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(PLOTS / "precision_recall_vs_threshold.png", dpi=150, bbox_inches="tight")
plt.close()

# Q5
kf = KFold(n_splits=5, shuffle=True, random_state=1)
full_train_dicts = df_full_train[feature_cols].to_dict(orient="records")
y_full = df_full_train[target].values
fold_aucs = []

for train_idx, val_idx in kf.split(df_full_train):
    tr_dicts = [full_train_dicts[i] for i in train_idx]
    va_dicts = [full_train_dicts[i] for i in val_idx]
    y_tr = y_full[train_idx]
    y_va = y_full[val_idx]

    dv_f = DictVectorizer(sparse=False)
    X_tr = dv_f.fit_transform(tr_dicts)
    X_va = dv_f.transform(va_dicts)

    m = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    m.fit(X_tr, y_tr)
    proba = m.predict_proba(X_va)[:, 1]
    fold_aucs.append(roc_auc_score(y_va, proba))

fold_aucs = np.array(fold_aucs)
print(f"Q5 fold AUCs: {fold_aucs}")
print(f"Q5 mean={fold_aucs.mean():.6f} std={fold_aucs.std():.6f}")

# Q6
c_values = [0.000001, 0.001, 1]
q6_results = []
for c in c_values:
    fold_aucs_c = []
    for train_idx, val_idx in kf.split(df_full_train):
        tr_dicts = [full_train_dicts[i] for i in train_idx]
        va_dicts = [full_train_dicts[i] for i in val_idx]
        y_tr = y_full[train_idx]
        y_va = y_full[val_idx]

        dv_f = DictVectorizer(sparse=False)
        X_tr = dv_f.fit_transform(tr_dicts)
        X_va = dv_f.transform(va_dicts)

        m = LogisticRegression(solver="liblinear", C=c, max_iter=1000)
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_va)[:, 1]
        fold_aucs_c.append(roc_auc_score(y_va, proba))

    mean_r = round(float(np.mean(fold_aucs_c)), 3)
    std_r = round(float(np.std(fold_aucs_c)), 3)
    q6_results.append((c, mean_r, std_r, fold_aucs_c))
    print(f"Q6 C={c}: mean={mean_r} std={std_r} folds={fold_aucs_c}")

# pick best C
q6_results.sort(key=lambda x: (-x[1], x[2], x[0]))
print("Q6 best C:", q6_results[0][0])
