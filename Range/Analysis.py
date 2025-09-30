import pandas as pd
import numpy as np
from scipy import stats

# === Step 1: Load data ===
df = pd.read_csv("Range2025.csv")  
salaries = df["salary_in_usd"].dropna()

# === Step 2: Hypotheses setup ===
lower_bound = 156000
upper_bound = 245000
alpha = 0.05   

# === Step 3: Sample stats ===
n = len(salaries)
sample_mean = np.mean(salaries)
sample_std = np.std(salaries, ddof=1)  
se = sample_std / np.sqrt(n)

print(f"Sample size: {n}")
print(f"Sample mean: {sample_mean:.2f}")
print(f"Std: {sample_std:.3f}")
print(f"Standard Error: {se:.2f}\n")

# === Step 4: Test A - Lower bound ===
z_lower = (sample_mean - lower_bound) / se
p_lower = 1 - stats.norm.cdf(z_lower)   # one-sided p-value (right-tail)

print("=== Test A: Mean > 156,000 ===")
print(f"H0: μ ≤ 156,000")
print(f"H1: μ > 156,000")
print(f"z = {z_lower:.4f}, p = {p_lower:.4f}")
if p_lower < alpha:
    print("Reject H0 → mean is significantly greater than 156,000.\n")
else:
    print("Fail to reject H0 → cannot conclude mean > 156,000.\n")

# === Step 5: Test B - Upper bound ===
z_upper = (sample_mean - upper_bound) / se
p_upper = stats.norm.cdf(z_upper)       # one-sided p-value (left-tail)

print("=== Test B: Mean < 245,000 ===")
print(f"H0: μ ≥ 245,000")
print(f"H1: μ < 245,000")
print(f"z = {z_upper:.4f}, p = {p_upper:.4f}")
if p_upper < alpha:
    print("Reject H0 → mean is significantly less than 245,000.\n")
else:
    print("Fail to reject H0 → cannot conclude mean < 245,000.\n")
