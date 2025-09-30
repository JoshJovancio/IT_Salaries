import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- Step 1: Load Data ---
data = pd.read_csv("../Data.csv")

# Keep only useful columns 
df = data[["work_year", "experience_level", "salary_in_usd"]]

# Encode categorical experience level
le = LabelEncoder()
df["experience_encoded"] = le.fit_transform(df["experience_level"])

# Features (year + experience level) and target (salary)
X = df[["work_year", "experience_encoded"]]
y = df["salary_in_usd"]

# --- Step 2: Train Regression Model ---
model = LinearRegression()
model.fit(X, y)

# --- Step 3: Predict Future Salaries ---

future_years = [2026, 2027, 2028]
experience_levels = le.classes_  

predictions = []

for year in future_years:
    for exp in experience_levels:
        exp_code = le.transform([exp])[0]
        salary_pred = model.predict([[year, exp_code]])[0]
        predictions.append([year, exp, round(salary_pred, 2)])

pred_df = pd.DataFrame(predictions, columns=["Year", "Experience_Level", "Predicted_Salary"])
print(pred_df)
