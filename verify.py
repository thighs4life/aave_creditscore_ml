import json
from collections import Counter

# Path to the predicted output file
file_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\aave_projectnewml\\scored_wallets.json"

# Load scored wallets
with open(file_path, "r") as f:
    data = json.load(f)

# Extract credit scores
scores = [entry['predicted_credit_score'] for entry in data]

# Count occurrences of each score
score_counts = Counter(scores)

# Display demographic
print("Credit Score Distribution:\n")
for score in sorted(score_counts):
    print(f"Score {score}: {score_counts[score]} wallets")

# Optional: Total wallets
print(f"\nTotal wallets scored: {len(scores)}")
