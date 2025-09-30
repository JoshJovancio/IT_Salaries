import pandas as pd
from scipy import stats

# Load datasets
remote = pd.read_csv("Remote.csv")   
onsite = pd.read_csv("Onsite.csv")   

remote_salaries = remote["salary_in_usd"]
onsite_salaries = onsite["salary_in_usd"]

# --- Descriptive statistics ---
def describe_data(data, label):
    print(f"\n{label} Statistics:")
    print(f"Sample size (n): {len(data)}")
    print(f"Mean: {data.mean():.2f}")
    print(f"Variance: {data.var(ddof=1):.2f}")
    print(f"Standard Deviation: {data.std(ddof=1):.2f}")

describe_data(remote_salaries, "Remote Workers (100% Remote)")
describe_data(onsite_salaries, "Onsite Workers (0% Remote)")

# --- Welch’s two-sample t-test ---
t_stat, p_value = stats.ttest_ind(remote_salaries, onsite_salaries, equal_var=False)

print("\nWelch’s Two-Sample t-test Results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("✅ Reject H0: The mean salaries of remote and onsite workers are significantly different.")
else:
    print("❌ Fail to reject H0: No significant difference in mean salaries.")
