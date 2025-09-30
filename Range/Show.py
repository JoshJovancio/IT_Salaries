import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# === Load Data (already filtered for SE Software Engineers) ===
data = pd.read_csv("Range2025.csv")  
salaries = data["salary_in_usd"]

# Compute stats
mean_salary = salaries.mean()
std_salary = salaries.std(ddof=1)
n = len(salaries)
se = std_salary / (n**0.5)

# 95% Confidence Interval for the mean
ci_low, ci_high = stats.norm.interval(0.95, loc=mean_salary, scale=se)

# === Visualization ===
plt.figure(figsize=(10,6))
plt.hist(salaries, bins=30, color="skyblue", edgecolor="black", alpha=0.7)

# Bounds (test range)
plt.axvline(156000, color="red", linestyle="--", label="Lower Bound (156k)")
plt.axvline(245000, color="red", linestyle="--", label="Upper Bound (245k)")

# Mean salary
plt.axvline(mean_salary, color="green", linestyle="-", linewidth=2, label=f"Mean = {mean_salary:,.0f}")

# Confidence Interval
plt.axvspan(ci_low, ci_high, color="orange", alpha=0.2, label=f"95% CI [{ci_low:,.0f}, {ci_high:,.0f}]")

plt.title("Senior SWE (Software Engineer) Salary Distribution")
plt.xlabel("Salary (USD)")
plt.ylabel("Count")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
