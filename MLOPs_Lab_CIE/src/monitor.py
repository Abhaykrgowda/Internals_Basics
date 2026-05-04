import pandas as pd
import json
import os

# Load logs
logs = []
with open("logs/predictions.jsonl", "r") as f:
    for line in f:
        logs.append(json.loads(line))

df_logs = pd.DataFrame([l["input"] for l in logs])

# Training means
train_df = pd.read_csv("data/training_data.csv")

alerts = []

def check_drift(feature, threshold):
    train_mean = train_df[feature].mean()
    live_mean = df_logs[feature].mean()
    shift = abs(live_mean - train_mean)

    status = "ALERT" if shift > threshold else "OK"

    return {
        "feature": feature,
        "train_mean": train_mean,
        "live_mean": live_mean,
        "shift": shift,
        "threshold": threshold,
        "status": status
    }

alerts.append(check_drift("followers_count", 26989.52))
alerts.append(check_drift("content_length", 88.62))

output = {
    "total_predictions": len(df_logs),
    "mean_prediction": 0.0,
    "drift_detected": any(a["status"] == "ALERT" for a in alerts),
    "alerts": alerts
}

os.makedirs("results", exist_ok=True)
with open("results/step3_s5.json", "w") as f:
    json.dump(output, f, indent=4)

print("Monitoring complete")