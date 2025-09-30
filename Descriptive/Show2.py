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
yearly_stats = df.groupby("work_year")["salary_in_usd"].apply(custom_stats).unstack().reset_index()

years = yearly_stats["work_year"]

# === 1. Mean, Median, Mode ===
plt.figure(figsize=(8,5))
plt.plot(years, yearly_stats["Mean"], marker="o", label="Mean")
plt.plot(years, yearly_stats["Median"], marker="s", label="Median")
plt.plot(years, yearly_stats["Mode"], marker="^", label="Mode")
plt.title("Measure of location", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Value (USD)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.xticks(years)
plt.tight_layout()
plt.show()


# === 3. Skewness, Kurtosis ===
plt.figure(figsize=(8,5))
plt.plot(years, yearly_stats["Skewness"], marker="o", label="Skewness")
plt.plot(years, yearly_stats["Kurtosis"], marker="s", label="Kurtosis")
plt.title("Shape of Distribution (Skewness & Kurtosis)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Value")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.xticks(years)
plt.tight_layout()
plt.show()
