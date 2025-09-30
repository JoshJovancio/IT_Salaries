import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Data.csv")

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

# Apply stats per year (assuming 'work_year' is the year column)
yearly_stats = df.groupby("work_year")["salary_in_usd"].apply(custom_stats)

# Show results
print(yearly_stats)

# Save to CSV
yearly_stats.to_csv("yearly_custom_statistics.csv")
