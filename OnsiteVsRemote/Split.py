import pandas as pd

# Load your dataset
data = pd.read_csv("sampled_remote_onsite.csv")

# Split dataset
onsite = data[data["remote_ratio"] == 0]     # onsite employees
remote = data[data["remote_ratio"] == 100]   # remote employees

# Save to CSV
onsite.to_csv("Onsite.csv", index=False)
remote.to_csv("Remote.csv", index=False)
