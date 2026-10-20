"""Build homework6.ipynb."""
import json
from pathlib import Path

cells = []

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})

def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "source": s.splitlines(keepends=True), "outputs": [], "execution_count": None})

md("# Homework 6 — Tree Ensembles\n\nML Zoomcamp 2026 | Car fuel efficiency regression")

code("""from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = Path('data/car_fuel_efficiency_2026.csv')

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def original_column(name):
    return name.split('=', 1)[0] if '=' in name else name""")

code("""df = pd.read_csv(DATA_PATH)
target = 'fuel_efficiency_mpg'
feature_cols = [c for c in df.columns if c != target]
cat_cols = df[feature_cols].select_dtypes(include=['object', 'string']).columns.tolist()
for col in feature_cols:
    if col in cat_cols:
        df[col] = df[col].fillna('NA')  # no missing categoricals in this dataset
    else:
        df[col] = df[col].fillna(0.0)
print('Missing after fill:', df.isna().sum().sum())
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
train_dicts = df_train[feature_cols].to_dict(orient='records')
val_dicts = df_val[feature_cols].to_dict(orient='records')
y_train = df_train[target].values
y_val = df_val[target].values
dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
feature_names = dv.get_feature_names_out()""")

code("""dt1 = DecisionTreeRegressor(max_depth=1)
dt1.fit(X_train, y_train)
root = feature_names[dt1.tree_.feature[0]]
print('Q1 root:', original_column(root))""")

code("""rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
print(f'Q2 RMSE: {rmse(y_val, rf.predict(X_val)):.4f}')""")

code("""for n in [10, 50, 100, 150]:
    m = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    m.fit(X_train, y_train)
    print(n, round(rmse(y_val, m.predict(X_val)), 3))""")

code("""for d in [10, 15, 20, 25]:
    scores = []
    for n in [10, 50, 100, 150]:
        m = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=1, n_jobs=-1)
        m.fit(X_train, y_train)
        scores.append(rmse(y_val, m.predict(X_val)))
    print('depth', d, 'mean', round(float(np.mean(scores)), 4))""")

code("""m5 = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
m5.fit(X_train, y_train)
cols = ['vehicle_weight', 'horsepower', 'acceleration', 'engine_displacement']
agg = {c: 0.0 for c in cols}
for name, val in zip(feature_names, m5.feature_importances_):
    base = original_column(name)
    if base in agg:
        agg[base] += val
print(agg)
print('Q5:', max(agg, key=agg.get))""")

code("""X_tr = X_train.toarray(); X_va = X_val.toarray()
dtrain = xgb.DMatrix(X_tr, label=y_train, feature_names=list(feature_names))
dval = xgb.DMatrix(X_va, label=y_val, feature_names=list(feature_names))
watch = [(dtrain, 'train'), (dval, 'val')]
base = {'max_depth': 6, 'min_child_weight': 1, 'objective': 'reg:squarederror', 'nthread': 8, 'seed': 1, 'verbosity': 0}
for eta in [0.3, 0.1]:
    m = xgb.train({**base, 'eta': eta}, dtrain, num_boost_round=100, evals=watch, verbose_eval=False)
    print(f'eta={eta} RMSE={rmse(y_val, m.predict(dval)):.4f}')""")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "cells": cells}
Path("homework6.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
