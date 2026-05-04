import requests
import pandas as pd
import time

url = "http://127.0.0.1:8000/predict"

train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

success = 0

# Send 35 normal
for i in range(35):
    row = train_df.sample(1).iloc[0]
    payload = {
        "followers_count": int(row["followers_count"]),
        "post_hour": int(row["post_hour"]),
        "has_media": int(row["has_media"]),
        "content_length": int(row["content_length"])
    }

    r = requests.post(url, json=payload)
    print("Normal:", r.status_code)

    if r.status_code == 200:
        success += 1

    time.sleep(0.2)   # instead of 0.05

# Send 15 drifted
for i in range(15):
    row = new_df.sample(1).iloc[0]
    payload = {
        "followers_count": int(row["followers_count"]),
        "post_hour": int(row["post_hour"]),
        "has_media": int(row["has_media"]),
        "content_length": int(row["content_length"])
    }

    r = requests.post(url, json=payload)
    print("Drift:", r.status_code)

    if r.status_code == 200:
        success += 1

    time.sleep(0.05)

print("Total successful requests:", success)