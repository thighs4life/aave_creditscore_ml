import json
import pandas as pd
import numpy as np
import joblib

# Paths
input_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\aave_projectnewml\\user_wallet_transactions.json"
model_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\aave_projectnewml\\model.pkl"
output_path = "C:\\Users\\Asus\\OneDrive\\Desktop\\aave_projectnewml\\scored_wallets.json"

print("Loading model...")
model = joblib.load(model_path)

print("Loading input data...")
with open(input_path, "r") as f:
    data = json.load(f)

df = pd.json_normalize(data)

# Normalize amount to USD
def normalize_amount(row):
    try:
        amt = float(row['actionData.amount'])
        symbol = row['actionData.assetSymbol'].upper()
        if symbol == "USDC":
            amt = amt / 1e6
        else:
            amt = amt / 1e18
        return amt * float(row['actionData.assetPriceUSD'])
    except:
        return 0

df['amount_usd'] = df.apply(normalize_amount, axis=1)

# Pre-filtered views
deposits = df[df['action'] == 'deposit']
borrows = df[df['action'] == 'borrow']
repays = df[df['action'] == 'repay']
redeems = df[df['action'] == 'redeemunderlying']
liquidations = df[df['action'] == 'liquidationcall']

# Aggregate features
agg = df.groupby("userWallet").agg(
    tx_count=('action', 'count'),
    first_time=('timestamp', 'min'),
    last_time=('timestamp', 'max')
).reset_index()

# Merge behavior features
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

# Fill missing values
agg.fillna(0, inplace=True)

# Derived ratios
agg['repay_borrow_ratio'] = agg['total_repay_usd'] / (agg['total_borrow_usd'] + 1e-6)
agg['liquidation_ratio'] = agg['liquidation_count'] / (agg['borrow_count'] + 1e-6)
agg['active_days'] = (agg['last_time'] - agg['first_time']) / (60 * 60 * 24)

# Final feature set
features = [
    'total_deposit_usd', 'total_borrow_usd', 'total_repay_usd',
    'repay_borrow_ratio', 'liquidation_ratio', 'tx_count', 'active_days'
]

print("Predicting credit scores...")
X = agg[features]
agg['predicted_credit_score'] = model.predict(X).round().astype(int)

# Save results
output = agg[['userWallet', 'predicted_credit_score']].to_dict(orient='records')

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Predictions saved to: {output_path}")
