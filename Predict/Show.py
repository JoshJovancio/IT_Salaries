import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# === Load Data ===
data = pd.read_csv("../Data.csv")

# Group by year: average salary
yearly_avg = data.groupby("work_year")["salary_in_usd"].mean().reset_index()

# Prepare regression
X = yearly_avg["work_year"].values.reshape(-1,1)  # feature = year
y = yearly_avg["salary_in_usd"].values

model = LinearRegression()
model.fit(X, y)

# Predict for 2026–2028
future_years = np.array([2026, 2027, 2028]).reshape(-1,1)
future_preds = model.predict(future_years)

# === Visualization ===
plt.figure(figsize=(10,6))

# Actual average salaries
plt.plot(yearly_avg["work_year"], yearly_avg["salary_in_usd"], "o-", label="Observed Avg Salary")

# Regression line (historical + future)
all_years = np.concatenate([X, future_years])
all_preds = model.predict(all_years)
plt.plot(all_years, all_preds, "r--", label="Regression Trend")

# Future predictions
plt.plot(future_years, future_preds, "ro", markersize=8, label="Predicted (2026–2028)")

# Labels & formatting
plt.title("Average Software Engineer Salaries Over Time")
plt.xlabel("Year")
plt.ylabel("Average Salary (USD)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
