Credit Score Analysis
After processing and scoring 3,497 DeFi wallet addresses from the Aave V2 protocol on the Polygon network, we conducted a detailed analysis of the resulting credit score distribution and user behavior patterns. This document summarizes the insights we derived from the score data and outlines how wallet behaviors correlate with different credit score ranges.

Score Range Distribution
Each wallet was assigned a predicted credit score between 300 and 900, based on its transaction behavior using a machine learning model trained on features like total deposits, borrow amounts, repayment ratios, liquidation events, and wallet activity span.

We grouped these scores into buckets by ranges of 100 to better understand the concentration of users. The resulting distribution shows a strong skew toward lower ranges, especially between 300–400.

Key Findings
300–399: This range had the highest concentration of users. These wallets typically had very minimal activity, low diversity in transaction types, and higher exposure to liquidation events. Many of these users either deposited very small amounts or did not engage in repayment behavior.

400–599: This group displayed slightly healthier financial behavior. Wallets in this band tended to show more repayment activity and moderate borrow/deposit ratios. Some of these users may be newer to the platform but show signs of responsible usage.

600–799: Wallets scoring in this range generally had higher transaction diversity and lower liquidation rates. These users often interacted with multiple assets and maintained a consistent presence over time. Many of these wallets showed strong repayment patterns.

800–900: This was the rarest range and contained very few wallets. These users exhibited high-value transactions, diversified activity across asset types, and no record of liquidation. In short, they appeared highly reliable from a credit risk perspective.

Behavioral Trends
A few behavioral insights stood out clearly:

Low-score wallets (300–399) tended to either use the platform minimally or behave in risky ways — for instance, borrowing without significant repayments or encountering frequent liquidation calls.

Mid-score wallets (500–600) were more stable. They typically had fewer liquidation events and maintained decent repayment-to-borrow ratios, but lacked transaction depth or long-term usage patterns.

High-score wallets (700–900) behaved like model users: well-balanced activity, no risky debt positions, and responsible repayment behavior. These users also had longer active spans and more consistent use of the platform.

Conclusion
The score predictions reveal that a majority of users fall into lower credit score categories, which may reflect the open and experimental nature of DeFi, where many participants are testing the waters with small or short-lived activities. However, the data also shows that trustworthy behavior can be clearly detected and modeled using transactional history alone.

This credit scoring method, although basic in its current form, demonstrates that trust signals do exist on-chain. With better feature engineering and clustering, we can fine-tune predictions further. Future work could include time-series modeling, improved behavioral clustering, or incorporating cross-protocol wallet activity.

Score Distribution Chart:

![Credit Score Distribution](credit_score_disturbution.png)