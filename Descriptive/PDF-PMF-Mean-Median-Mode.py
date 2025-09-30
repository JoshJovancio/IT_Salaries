import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the CSV file
df = pd.read_csv('../Data.csv')

# Display basic info
print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 3 rows:")
print(df[['work_year', 'salary_in_usd']].head(3))

# Check year range and data availability
print(f"\nYear range: {df['work_year'].min()} - {df['work_year'].max()}")
print("\nData distribution by year:")
year_counts = df['work_year'].value_counts().sort_index()
print(year_counts)

# Clean data - THIS CREATES df_clean
df_clean = df[['work_year', 'salary_in_usd']].dropna()
print(f"\nClean data shape: {df_clean.shape}")

# PMF (Probability Mass Function) - Year Distribution
print("\n" + "=" * 50)
print("PROBABILITY MASS FUNCTION (PMF) - YEAR DISTRIBUTION")
print("=" * 50)

# Calculate PMF for years
year_pmf = df_clean['work_year'].value_counts(normalize=True).sort_index()
print("\nPMF for Years:")
for year, prob in year_pmf.items():
    count = year_counts[year]
    print(f"{year}: {prob:.4f} ({prob * 100:.2f}%) - {count} records")

# Basic statistics by year
print("\n" + "=" * 50)
print("SALARY STATISTICS BY YEAR")
print("=" * 50)

# Calculate mode for each year
print("\nMode by Year:")
mode_salaries = []
for year in sorted(df_clean['work_year'].unique()):
    year_data = df_clean[df_clean['work_year'] == year]['salary_in_usd']
    mode_result = year_data.mode()
    if not mode_result.empty:
        mode_val = mode_result.iloc[0]
        print(f"{year}: ${mode_val:.2f}")
        mode_salaries.append(mode_val)
    else:
        print(f"{year}: No unique mode")
        mode_salaries.append(np.nan)

# Create the main figure with 3 subplots
plt.figure(figsize=(15, 10))

# Plot 1: PMF - Year Distribution
plt.subplot(2, 2, 1)
year_pmf.plot(kind='bar', color='lightblue', edgecolor='black')
plt.title('PMF of Data Distribution by Year')
plt.xlabel('Year')
plt.ylabel('Probability')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(year_pmf):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Plot 2: Mean, Median and Mode Salary Trends
plt.subplot(2, 2, 2)
years = sorted(df_clean['work_year'].unique())
mean_salaries = [df_clean[df_clean['work_year'] == year]['salary_in_usd'].mean() for year in years]
median_salaries = [df_clean[df_clean['work_year'] == year]['salary_in_usd'].median() for year in years]

plt.plot(years, mean_salaries, marker='o', linewidth=2, label='Mean Salary', color='blue')
plt.plot(years, median_salaries, marker='s', linewidth=2, label='Median Salary', color='red')
plt.plot(years, mode_salaries, marker='^', linewidth=2, label='Mode Salary', color='green')

plt.title('Salary Trends by Year (Mean, Median & Mode)')
plt.xlabel('Year')
plt.ylabel('Salary in USD')
plt.legend()
plt.grid(alpha=0.3)

# Add value annotations
for i, (year, mean_val, median_val, mode_val) in enumerate(zip(years, mean_salaries, median_salaries, mode_salaries)):
    plt.annotate(f'${mean_val:,.0f}', (year, mean_val), textcoords="offset points", xytext=(0, 10), ha='center',
                 fontsize=8)
    plt.annotate(f'${median_val:,.0f}', (year, median_val), textcoords="offset points", xytext=(0, -15), ha='center',
                 fontsize=8)
    if not np.isnan(mode_val):
        plt.annotate(f'${mode_val:,.0f}', (year, mode_val), textcoords="offset points", xytext=(0, 10), ha='center',
                     fontsize=8)

# Plot 3: Individual PDFs for each year
plt.subplot(2, 2, 3)
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
color_idx = 0

for year in sorted(df_clean['work_year'].unique()):
    year_salaries = df_clean[df_clean['work_year'] == year]['salary_in_usd']
    # Plot KDE for each year
    sns.kdeplot(year_salaries, label=f'{year}', linewidth=2.5, color=colors[color_idx])
    color_idx = (color_idx + 1) % len(colors)

plt.title('PDF - Salary Distribution by Year')
plt.xlabel('Salary in USD')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)

# Remove the 4th subplot since we don't need CDF
plt.delaxes(plt.subplot(2, 2, 4))

plt.tight_layout()
plt.show()

# PDF Analysis - Salary Distribution by Year
print("\n" + "=" * 50)
print("PDF ANALYSIS - SALARY DISTRIBUTION BY YEAR")
print("=" * 50)

# Create individual PDF plots for each year in a separate figure
plt.figure(figsize=(15, 12))
years = sorted(df_clean['work_year'].unique())
n_years = len(years)

# Calculate optimal grid size
n_cols = 2
n_rows = (n_years + 1) // n_cols

for i, year in enumerate(years):
    plt.subplot(n_rows, n_cols, i + 1)

    year_salaries = df_clean[df_clean['work_year'] == year]['salary_in_usd']

    # Plot histogram and KDE
    n, bins, patches = plt.hist(year_salaries, bins=20, density=True, alpha=0.7,
                                color=colors[i % len(colors)], edgecolor='black')
    sns.kdeplot(year_salaries, color='darkred', linewidth=2)

    # Add statistical information
    mean_salary = year_salaries.mean()
    median_salary = year_salaries.median()
    mode_result = year_salaries.mode()

    plt.axvline(mean_salary, color='blue', linestyle='--', linewidth=2, label=f'Mean: ${mean_salary:,.0f}')
    plt.axvline(median_salary, color='green', linestyle='--', linewidth=2, label=f'Median: ${median_salary:,.0f}')

    if not mode_result.empty:
        mode_salary = mode_result.iloc[0]
        plt.axvline(mode_salary, color='orange', linestyle='--', linewidth=2, label=f'Mode: ${mode_salary:,.0f}')

    plt.title(f'PDF - {year} (n={len(year_salaries)})')
    plt.xlabel('Salary in USD')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Final summary
print("\n" + "=" * 50)
print("OVERALL SUMMARY")
print("=" * 50)
years_sorted = sorted(df_clean['work_year'].unique())
print(f"Dataset spans {len(years_sorted)} years: {min(years_sorted)} to {max(years_sorted)}")
print(f"Total records: {len(df_clean)}")
print(f"Overall mean salary: ${df_clean['salary_in_usd'].mean():.2f}")
print(f"Overall median salary: ${df_clean['salary_in_usd'].median():.2f}")
print(
    f"Overall mode salary: ${df_clean['salary_in_usd'].mode().iloc[0] if not df_clean['salary_in_usd'].mode().empty else 'No mode'}")
print(f"Overall salary standard deviation: ${df_clean['salary_in_usd'].std():.2f}")

print("\nPDF Interpretation Guide:")
print("- Higher peaks indicate more concentrated salary distributions")
print("- Wider spreads indicate more salary variability")
print("- Right-skewed distributions show few high earners")

print(f"Total records in dataset: {len(df)}")
