import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

# File paths
input_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\aave_projectnewml\\user_wallet_transactions.json"
output_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\aave_projectnewml\\labeleed_transaction.json"

print("Loading data...")
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize(data)

# Convert amount to USD
def normalize_amount(row):
    try:
        amt = float(row['actionData.amount'])
        symbol = row['actionData.assetSymbol'].upper()
        amt = amt / 1e6 if symbol == "USDC" else amt / 1e18
        return amt * float(row['actionData.assetPriceUSD'])
    except:
        return 0

df['amount_usd'] = df.apply(normalize_amount, axis=1)

# Group transactions by wallet
grouped = df.groupby("userWallet")

print("Engineering features...")
features_df = grouped.agg(
    total_tx_count=('action', 'count'),
    unique_actions=('action', lambda x: x.nunique()),
    total_deposit_usd=('amount_usd', lambda x: x[df['action'] == 'deposit'].sum()),
    total_borrow_usd=('amount_usd', lambda x: x[df['action'] == 'borrow'].sum()),
    total_repay_usd=('amount_usd', lambda x: x[df['action'] == 'repay'].sum()),
    total_liquidations=('action', lambda x: (x == 'liquidationcall').sum()),
    total_redeem_usd=('amount_usd', lambda x: x[df['action'] == 'redeemunderlying'].sum()),
    first_tx=('timestamp', 'min'),
    last_tx=('timestamp', 'max')
).fillna(0)

# Derived features
features_df['repay_borrow_ratio'] = features_df['total_repay_usd'] / (features_df['total_borrow_usd'] + 1e-6)
features_df['active_days'] = (features_df['last_tx'] - features_df['first_tx']) / (60 * 60 * 24)
features_df['liquidation_rate'] = features_df['total_liquidations'] / (features_df['total_tx_count'] + 1e-6)

# Select features
features = [
    'total_tx_count', 'unique_actions', 'total_deposit_usd', 'total_borrow_usd',
    'total_repay_usd', 'total_redeem_usd', 'repay_borrow_ratio',
    'active_days', 'liquidation_rate'
]

X = features_df[features].replace([np.inf, -np.inf], 0).fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Running KMeans clustering (n=5)...")
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Map clusters to credit scores based on size (small clusters = higher scores)
cluster_counts = Counter(clusters)
sorted_clusters = sorted(cluster_counts, key=cluster_counts.get, reverse=True)
score_values = [300, 600, 900]

cluster_to_score = {cluster_id: score_values[i] for i, cluster_id in enumerate(sorted_clusters)}

features_df['credit_score'] = [cluster_to_score[c] for c in clusters]

# Show distribution
print("Cluster distribution (wallets per score):")
print(Counter(features_df['credit_score']))

# Assign credit scores to each transaction
wallet_to_score = features_df['credit_score'].to_dict()
for tx in data:
    tx['credit_score'] = wallet_to_score.get(tx['userWallet'], 300)

# Save labeled data
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Labeled data saved to:", output_path)
