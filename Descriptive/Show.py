import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../Data.csv")

# Function to compute custom stats
def custom_stats(x):
    mean = x.mean()
    median = x.median()
    mode = x.mode().iloc[0] if not x.mode().empty else np.nan
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    std = x.std()
    data_range = x.max() - x.min()
    skewness = x.skew()
    kurtosis = x.kurt()
    
    return pd.Series({
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "IQR": iqr,
        "Std Dev": std,
        "Range": data_range,
        "Skewness": skewness,
        "Kurtosis": kurtosis
    })

# Compute stats per year
yearly_stats = df.groupby("work_year")["salary_in_usd"].apply(custom_stats).unstack()

# Reset index so "work_year" is a column
yearly_stats = yearly_stats.reset_index()

# Now yearly_stats should have columns:
# ['work_year', 'Mean', 'Median', 'Mode', 'IQR', 'Std Dev', 'Range', 'Skewness', 'Kurtosis']

print(yearly_stats.head())

# Plot each variable separately
variables = ["Mean", "Median", "Mode", "IQR", "Std Dev", "Range", "Skewness", "Kurtosis"]

for var in variables:
    plt.figure(figsize=(8,5))
    plt.plot(yearly_stats["work_year"], yearly_stats[var], marker="o", linestyle="-", linewidth=2)
    plt.title(f"{var} of Salaries", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel(var)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(yearly_stats["work_year"])
    plt.tight_layout()
    plt.show()
