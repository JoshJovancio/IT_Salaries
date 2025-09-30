import pandas as pd
import matplotlib.pyplot as plt

# === Load Data ===
data = pd.read_csv("../Data.csv")

# Group by year and experience level
yearly_exp = data.groupby(["work_year", "experience_level"])["salary_in_usd"].mean().reset_index()

# === Visualization ===
plt.figure(figsize=(10,6))

# Plot each experience level separately
for level in yearly_exp["experience_level"].unique():
    subset = yearly_exp[yearly_exp["experience_level"] == level]
    plt.plot(subset["work_year"], subset["salary_in_usd"], marker="o", linestyle="-", label=level)

plt.title("Average Salaries by Experience Level (Year to Year)")
plt.xlabel("Year")
plt.ylabel("Average Salary (USD)")
plt.legend(title="Experience Level")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
