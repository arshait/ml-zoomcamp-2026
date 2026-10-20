# Homework 6 â€” Answers (ML Zoomcamp 2026)

Dataset: `data/car_fuel_efficiency_2026.csv`
Target: `fuel_efficiency_mpg`
Missing values: numerical features â†’ `0.0` (horsepower, acceleration); categoricals have no missing rows (policy documented in notebook).

## Question 1 â€” Decision tree depth 1 root split

Root one-hot feature: `model_year` â†’ original column **`model_year`**

## Question 2 â€” Random forest (10 trees) validation RMSE

**RMSE â‰ˆ 1.8332** â†’ quiz option **1.837**

## Question 3 â€” Best `n_estimators` (10, 50, 100, 150)

| n_estimators | Val RMSE (3 dp) |
|--------------|-----------------|
| 10 | 1.833 |
| 50 | 1.774 |
| 100 | 1.770 |
| **150** | **1.769** |

**Answer: 150**

## Question 4 â€” Best `max_depth` (mean RMSE over four forest sizes)

| max_depth | Mean RMSE |
|-----------|-----------|
| **10** | **1.7544** |
| 15 | 1.7842 |
| 20 | 1.7868 |
| 25 | 1.7863 |

**Answer: 10**

## Question 5 â€” Highest feature importance (10 trees, depth 20)

Aggregated importances among listed columns:

- vehicle_weight: **0.203**
- engine_displacement: 0.061
- horsepower: 0.078
- acceleration: 0.055

**Answer: vehicle_weight**

## Question 6 â€” XGBoost learning rate (100 rounds)

Same data, `max_depth=6`, `seed=1`:

- `eta=0.3` â†’ val RMSE **1.8265**
- `eta=0.1` â†’ val RMSE **1.7248** (lower)

**Answer: 0.1**
