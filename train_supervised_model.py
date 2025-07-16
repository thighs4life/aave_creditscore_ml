import json
import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load labeled transaction data
print("Loading labeled data...")
input_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\aave_projectnewml\\labeleed_transaction.json"
with open(input_path, "r") as f:
    transactions = json.load(f)

df = pd.json_normalize(transactions)

# Print original score distribution
print("Original credit score distribution:")
print(Counter(df['credit_score']))

# Balance the dataset
min_class_size = min(df['credit_score'].value_counts().values)
min_class_size = min(min_class_size, 1000)  # Limit to 1000 per class to avoid overfitting

df_balanced = pd.concat([
    resample(df[df['credit_score'] == score], replace=False, n_samples=min_class_size, random_state=42)
    for score in df['credit_score'].unique()
])

print("\nBalanced credit score distribution:")
print(Counter(df_balanced['credit_score']))

# Replace df with balanced data
df = df_balanced.copy()

# Normalize amount to USD
def normalize_amount(row):
    try:
        amt = float(row['actionData.amount'])
        symbol = row['actionData.assetSymbol'].upper()
        amt = amt / 1e6 if symbol == "USDC" else amt / 1e18
        return amt * float(row['actionData.assetPriceUSD'])
    except:
        return 0

df['amount_usd'] = df.apply(normalize_amount, axis=1)

# Subsets by action
deposits = df[df['action'] == 'deposit']
borrows = df[df['action'] == 'borrow']
repays = df[df['action'] == 'repay']
redeems = df[df['action'] == 'redeemunderlying']
liquidations = df[df['action'] == 'liquidationcall']

# Aggregate wallet features
agg = df.groupby("userWallet").agg(
    tx_count=('action', 'count'),
    first_time=('timestamp', 'min'),
    last_time=('timestamp', 'max'),
    credit_score=('credit_score', 'mean')  # label
).reset_index()

agg = agg.merge(
    deposits.groupby("userWallet")['amount_usd'].sum().rename("total_deposit_usd"),
    on="userWallet", how="left"
).merge(
    borrows.groupby("userWallet")['amount_usd'].sum().rename("total_borrow_usd"),
    on="userWallet", how="left"
).merge(
    repays.groupby("userWallet")['amount_usd'].sum().rename("total_repay_usd"),
    on="userWallet", how="left"
).merge(
    redeems.groupby("userWallet")['amount_usd'].sum().rename("total_redeem_usd"),
    on="userWallet", how="left"
).merge(
    deposits.groupby("userWallet")['action'].count().rename("deposit_count"),
    on="userWallet", how="left"
).merge(
    borrows.groupby("userWallet")['action'].count().rename("borrow_count"),
    on="userWallet", how="left"
).merge(
    repays.groupby("userWallet")['action'].count().rename("repay_count"),
    on="userWallet", how="left"
).merge(
    redeems.groupby("userWallet")['action'].count().rename("redeem_count"),
    on="userWallet", how="left"
).merge(
    liquidations.groupby("userWallet")['action'].count().rename("liquidation_count"),
    on="userWallet", how="left"
)

# Fill missing
agg.fillna(0, inplace=True)

# Derived metrics
agg['repay_borrow_ratio'] = agg['total_repay_usd'] / (agg['total_borrow_usd'] + 1e-6)
agg['liquidation_ratio'] = agg['liquidation_count'] / (agg['borrow_count'] + 1e-6)
agg['active_days'] = (agg['last_time'] - agg['first_time']) / (60 * 60 * 24)

# Final features
features = [
    'total_deposit_usd', 'total_borrow_usd', 'total_repay_usd',
    'repay_borrow_ratio', 'liquidation_ratio', 'tx_count', 'active_days'
]

# Training
print("\nSplitting data and training model...")
X = agg[features]
y = agg['credit_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\nModel training complete.")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("R^2 Score:", round(r2_score(y_test, y_pred), 2))

# Save model
model_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\aave_projectnewml\\model.pkl"
joblib.dump(model, model_path)
print("Model saved to:", model_path)
