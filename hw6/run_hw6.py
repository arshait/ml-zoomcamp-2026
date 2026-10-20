"""HW6 computation script."""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

DATA = Path(__file__).parent / "data" / "car_fuel_efficiency_2026.csv"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def original_column(feature_name: str) -> str:
    if "=" in feature_name:
        return feature_name.split("=", 1)[0]
    return feature_name


df = pd.read_csv(DATA)
target = "fuel_efficiency_mpg"
feature_cols = [c for c in df.columns if c != target]

cat_cols = df[feature_cols].select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

for col in feature_cols:
    if col in cat_cols:
        df[col] = df[col].fillna("NA")
    else:
        df[col] = df[col].fillna(0.0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

train_dicts = df_train[feature_cols].to_dict(orient="records")
val_dicts = df_val[feature_cols].to_dict(orient="records")
test_dicts = df_test[feature_cols].to_dict(orient="records")

y_train = df_train[target].values
y_val = df_val[target].values

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
feature_names = dv.get_feature_names_out()

# Q1
dt1 = DecisionTreeRegressor(max_depth=1, random_state=1)
dt1.fit(X_train, y_train)
root_idx = dt1.tree_.feature[0]
root_feat = feature_names[root_idx]
root_col = original_column(root_feat)
print(f"Q1 root feature: {root_feat} -> {root_col}")

# Q2
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
pred_val = rf.predict(X_val)
q2_rmse = rmse(y_val, pred_val)
print(f"Q2 RMSE: {q2_rmse:.4f}")

# Q3
for n in [10, 50, 100, 150]:
    m = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    m.fit(X_train, y_train)
    r = rmse(y_val, m.predict(X_val))
    print(f"Q3 n={n}: RMSE={r:.4f} (3dp {round(r,3)})")

# Q4
depths = [10, 15, 20, 25]
nest = [10, 50, 100, 150]
for d in depths:
    rmses = []
    for n in nest:
        m = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=1, n_jobs=-1)
        m.fit(X_train, y_train)
        rmses.append(rmse(y_val, m.predict(X_val)))
    print(f"Q4 depth={d}: mean RMSE={np.mean(rmses):.4f} individual={[round(x,4) for x in rmses]}")

# Q5
m5 = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
m5.fit(X_train, y_train)
imp = m5.feature_importances_
check_cols = ["vehicle_weight", "horsepower", "acceleration", "engine_displacement"]
col_imp = {c: 0.0 for c in check_cols}
for fname, score in zip(feature_names, imp):
    col = original_column(fname)
    if col in col_imp:
        col_imp[col] += score
print("Q5 importances:", col_imp)
print("Q5 max:", max(col_imp, key=col_imp.get))

# Q6 XGBoost
# Build dense matrices with same encoding - XGBoost uses DMatrix
# For missing values: we filled with 0, use same feature matrix
X_train_dense = X_train.toarray()
X_val_dense = X_val.toarray()

dtrain = xgb.DMatrix(X_train_dense, label=y_train, feature_names=list(feature_names))
dval = xgb.DMatrix(X_val_dense, label=y_val, feature_names=list(feature_names))
watchlist = [(dtrain, "train"), (dval, "val")]

params_base = {
    "max_depth": 6,
    "min_child_weight": 1,
    "objective": "reg:squarederror",
    "nthread": 8,
    "seed": 1,
    "verbosity": 1,
}

for eta in [0.3, 0.1]:
    params = {**params_base, "eta": eta}
    model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=False)
    pred = model.predict(dval)
    r = rmse(y_val, pred)
    print(f"Q6 eta={eta}: RMSE={r:.4f}")
