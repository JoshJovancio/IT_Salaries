import pandas as pd
import matplotlib.pyplot as plt

# === Load Data ===
data = pd.read_csv("../Data.csv")

# Group by year: average salary
yearly_avg = data.groupby("work_year")["salary_in_usd"].mean().reset_index()

# === Visualization ===
plt.figure(figsize=(10,6))
plt.plot(yearly_avg["work_year"], yearly_avg["salary_in_usd"], marker="o", linestyle="-", color="blue")

plt.title("Average Software Engineer Salaries (Year to Year)")
plt.xlabel("Year")
plt.ylabel("Average Salary (USD)")
plt.grid(True, linestyle="--", alpha=0.7)

# Show exact values on points
for i, row in yearly_avg.iterrows():
    plt.text(row["work_year"], row["salary_in_usd"]+2000, f"{row['salary_in_usd']:.0f}", 
             ha="center", fontsize=9, color="black")

plt.show()
