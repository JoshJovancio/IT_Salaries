import pandas as pd

# Load your dataset
data = pd.read_csv("Data2025.csv")

# Example: group by remote_ratio (0 = onsite, 100 = remote)
onsite = data[data["remote_ratio"] == 0].sample(500, random_state=42)
remote = data[data["remote_ratio"] == 100].sample(500, random_state=42)

# Combine into one DataFrame
sampled_data = pd.concat([onsite, remote])

# Save to CSV
sampled_data.to_csv("sampled_remote_onsite.csv", index=False)