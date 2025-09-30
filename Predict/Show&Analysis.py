import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# === Load Data ===
data = pd.read_csv("../Data.csv")

# === Group by year & experience level (average salary) ===
yearly_avg = data.groupby(["work_year", "experience_level"])["salary_in_usd"].mean().reset_index()

# === Unique experience levels ===
experience_levels = yearly_avg["experience_level"].unique()

# === Store regression results ===
results = []

plt.figure(figsize=(12, 7))

for exp in experience_levels:
    df_exp = yearly_avg[yearly_avg["experience_level"] == exp]

    # Prepare regression on yearly averages
    X = df_exp["work_year"].values.reshape(-1, 1)
    y = df_exp["salary_in_usd"].values

    model = LinearRegression().fit(X, y)

    # Predict for all historical + 2026–2028
    all_years = np.arange(df_exp["work_year"].min(), 2028 + 1).reshape(-1, 1)
    all_preds = model.predict(all_years)

    # Save results
    for year, pred in zip(all_years.flatten(), all_preds):
        results.append([year, exp, round(pred, 2)])

    # === Plotting ===
    plt.plot(df_exp["work_year"], df_exp["salary_in_usd"], "o-", label=f"{exp} Avg Salary")
    plt.plot(all_years, all_preds, "--", linewidth=2, label=f"{exp} Regression")

# === Show regression results as table ===
results_df = pd.DataFrame(results, columns=["Year", "Experience Level", "Predicted Avg Salary (USD)"])
print(results_df.sort_values(by=["Year", "Experience Level"]))

# === Labels & formatting ===
plt.title("Predicted Software Engineer Salaries by Experience Level (Yearly Averages + Regression)")
plt.xlabel("Year")
plt.ylabel("Average Salary (USD)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
