# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
from collections import Counter

# File paths
json_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\aave_projectnewml\\scored_wallets.json"
output_img_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\aave_projectnewml\\credit_score_disturbution.png"

# Load scored data
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract and bucket predicted credit scores
scores = [entry['predicted_credit_score'] for entry in data]
bucketed = [((score // 100) * 100) for score in scores]  # Group into 0–99, 100–199, etc.
score_counter = Counter(bucketed)

# Sort buckets
sorted_ranges = sorted(score_counter.items())

# Prepare labels and values
labels = [f"{rng}-{rng+99}" for rng, _ in sorted_ranges]
values = [count for _, count in sorted_ranges]

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(labels, values, color="skyblue", edgecolor="black")
plt.title("Credit Score Distribution by Range")
plt.xlabel("Credit Score Range")
plt.ylabel("Number of Wallets")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the chart
plt.savefig(output_img_path)
print(f"Distribution graph saved to: {output_img_path}")
