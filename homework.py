import pandas as pd
import numpy as np

df = pd.read_csv("../car_fuel_efficiency_2026.csv")

# Q1
print("Pandas version:", pd.__version__)

# Q2
print("Records:", len(df))

# Q3
print("Fuel types:", df["fuel_type"].nunique())

# Q4
print("Columns with missing values:", df.isna().any().sum())

# Q5
asia = df[df["origin"] == "Asia"]
print("Max fuel efficiency in Asia:", asia["fuel_efficiency_mpg"].max())

# Q6
before = asia["horsepower"].median()
mode = asia["horsepower"].mode()[0]
asia = asia.copy()
asia["horsepower"] = asia["horsepower"].fillna(mode)
after = asia["horsepower"].median()

print("Horsepower median before:", before)
print("Horsepower median after:", after)

# Q7
X = asia[["vehicle_weight", "model_year"]].iloc[:7].to_numpy()
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = np.linalg.inv(X.T @ X) @ X.T @ y

print("Weights:", w)
print("Sum of weights:", w.sum())