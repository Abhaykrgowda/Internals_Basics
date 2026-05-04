import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

combined_df = pd.concat([train_df, new_df])

X = combined_df.drop("engagement_score", axis=1)
y = combined_df["engagement_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load champion
champion = joblib.load("models/best_model.pkl")
champ_pred = champion.predict(X_test)
champ_rmse = np.sqrt(mean_squared_error(y_test, champ_pred))

# Retrain (assume Ridge or LinearRegression both okay)
model = type(champion)()
model.fit(X_train, y_train)

re_pred = model.predict(X_test)
re_rmse = np.sqrt(mean_squared_error(y_test, re_pred))

improvement = champ_rmse - re_rmse

if improvement >= 0.5:
    action = "promoted"
    joblib.dump(model, "models/best_model.pkl")
else:
    action = "kept_champion"

output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_rmse": champ_rmse,
    "retrained_rmse": re_rmse,
    "improvement": improvement,
    "min_improvement_threshold": 0.5,
    "action": action,
    "comparison_metric": "rmse"
}

os.makedirs("results", exist_ok=True)
with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Retraining complete:", action)