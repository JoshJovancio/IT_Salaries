import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load data ===
remote_data = pd.read_csv("Remote.csv")   # salaries for remote workers
onsite_data = pd.read_csv("Onsite.csv")   # salaries for onsite workers

# Add a label column to each dataset
remote_data["Work_Type"] = "Remote"
onsite_data["Work_Type"] = "Onsite"

# Combine into one DataFrame
data = pd.concat([remote_data, onsite_data], ignore_index=True)

# === Visualization 1: Boxplot ===
plt.figure(figsize=(8,6))
sns.boxplot(x="Work_Type", y="salary_in_usd", data=data, palette="Set2")
plt.title("Salary Distribution: Remote vs Onsite Workers")
plt.xlabel("Work Type")
plt.ylabel("Salary (USD)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# === Visualization 2: Barplot with CI ===
plt.figure(figsize=(8,6))
sns.barplot(x="Work_Type", y="salary_in_usd", data=data, palette="Set2", ci=95, capsize=0.1)
plt.title("Average Salary with 95% CI: Remote vs Onsite")
plt.xlabel("Work Type")
plt.ylabel("Mean Salary (USD)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
