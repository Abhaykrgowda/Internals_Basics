import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Create folders if not exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load data
df = pd.read_csv("data/training_data.csv")
X = df.drop("engagement_score", axis=1)
y = df["engagement_score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Metrics function
def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, r2, mape

mlflow.set_experiment("buzzmetric-engagement-score")

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0)
}

results = []
best_rmse = float("inf")
best_model_name = None
best_model = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae, rmse, r2, mape = get_metrics(y_test, y_pred)

        mlflow.log_param("model", name)
        if name == "Ridge":
            mlflow.log_param("alpha", 1.0)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)

        mlflow.set_tag("domain", "social_media")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model

# Save best model
joblib.dump(best_model, "models/best_model.pkl")

# Save JSON
output = {
    "experiment_name": "buzzmetric-engagement-score",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": best_rmse
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Training complete. Best model:", best_model_name)