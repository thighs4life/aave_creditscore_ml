DeFi Credit Scoring — Aave V2 Wallet Analysis
Problem Statement
Traditional credit scoring systems help evaluate risk in centralized finance. But in DeFi, credit infrastructure is nearly nonexistent. Users are anonymous, and lending often relies on over-collateralization. This makes borrowing expensive and limits broader adoption.

The goal here was to use raw transaction data from Aave V2 to build a credit scoring system that could assign scores between 300 and 900 to wallets, based purely on their on-chain behavior. Higher scores represent responsible users; lower scores reflect risky or bot-like behavior.

Dataset Overview
The dataset provided contains over 100,000 wallet-level transactions from the Aave V2 protocol on the Polygon network. Each transaction includes:

The wallet address

The action performed (e.g., deposit, borrow, repay, redeemunderlying, liquidationcall)

Token involved and its amount

Price of the asset in USD

Timestamps and protocol metadata

The dataset was entirely unlabeled — no predefined credit scores — so the first step was to create those labels from scratch.

Project Structure
Feature Engineering:
I began by grouping transactions per wallet and extracting behavioral indicators such as:

Number of transactions

Types of actions performed

Total value deposited, borrowed, repaid, and redeemed (in USD)

Repayment-to-borrow ratio

Number of times the wallet was liquidated

Time duration between first and last transaction

These features provided a numeric summary of wallet behavior.

Label Generation via Clustering:
Since the data was unlabeled, I used unsupervised clustering (KMeans) to group wallets with similar behaviors. Based on those clusters, I manually mapped them to credit scores:

Lower score (300) for low activity or risky patterns

Mid-range score (600) for balanced usage

High score (900) for consistent, responsible usage

The labeled dataset was saved as labeleed_transaction.json.

Model Training:
I trained a regression model on the labeled data using scikit-learn. The model predicts credit scores for new wallets based on their transaction patterns. It was saved as model.pkl.

Prediction:
Using the trained model, I scored the original unlabeled dataset and stored the output in scored_wallets.json.

Results
Mean Absolute Error (MAE): 94.03

R² Score: 0.22

These are acceptable scores for a first version, especially since labels were generated via unsupervised clustering. The credit score predictions were not overly concentrated in a single range, which suggests the model captured variations in behavior.

Example score spread from scored_wallets.json:

yaml
Copy
Edit
Score 300: 659 wallets
Score 410: 276 wallets
Score 600: 64 wallets
Score 900: 1 wallet
...
Total: 3497 wallets
Timeline
Assigned: July 15, 2025

Completed: July 17, 2025

Turnaround Time: 2 days

All stages — from parsing the data, creating features, clustering, training the model, and generating predictions — were completed within the deadline.

Folder Structure
aave_projectnewml/
├── user_wallet_transactions.json         # Raw data
├── credit_score_clustering.py            # Clustering & labeling
├── labeleed_transaction.json             # Labeled output
├── train_supervised_model.py            # Trains regression model
├── model.pkl                             # Saved model
├── predict_score.py                      # Predicts credit scores
├── scored_wallets.json                   # Scored wallets
├── analyze_distribution.py               # Score distribution analysis
└── README.md                             # Project documentation