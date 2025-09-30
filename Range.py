import pandas as pd

# Load your dataset
data = pd.read_csv("Data2025.csv")

# Example: group by remote_ratio (0 = onsite, 100 = remote)
range = data[(data["experience_level"] == "SE") & (data["job_title"] == "Software Engineer")].sample(1000, random_state=42)

# Save to CSV
range.to_csv("Range/Range2025.csv", index=False)