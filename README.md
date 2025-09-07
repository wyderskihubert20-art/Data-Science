# Credit Card Customer Segmentation

This project explores customer segmentation for credit card users using a real-world dataset. The workflow combines Python for preprocessing, clustering, and model building with R/Shiny for creating an interactive dashboard to visualize business insights.

--------------

Features

Data cleaning and preprocessing (handling missing values, feature engineering).

Customer segmentation using clustering techniques.

Business insights derived from spending, payments, and risk behaviors.

Machine learning models to predict risk scores.

Deployment of an interactive Shiny dashboard for exploration.

Tech Stack

Python (Pandas, Scikit-learn, Matplotlib/Seaborn)

R (Shiny, tidyverse)

GitHub for version control and portfolio showcase

Goals

Provide meaningful segmentation of credit card customers for business strategy.

Demonstrate integration of Python analysis with R Shiny visualization.

Showcase end-to-end workflow from raw dataset to polished insights.

--------------

### Dashboard
![Low Risk Prediction](images/Screenshot%202025-09-07%20121112.png)

### Clusters
![Medium Risk Prediction](images/Screenshot%202025-09-07%20121132.png)

### High Risk
![High Risk Prediction](images/Screenshot%202025-09-07%20121147.png)

### Medium Risk
![Low Risk Prediction](images/Screenshot%202025-09-07%20121523.png)

### Low Risk
![Low Risk Prediction](images/Screenshot%202025-09-07%20121347.png)

### Dataset Distributution
![Low Risk Prediction](images/Screenshot%202025-09-07%20121603.png)

**Dataset:** Credit Card Customers Dataset from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

--------------

### Key Findings & Business Insights

### 1. Customer Segmentation

Using KMeans clustering, we identified 4 distinct customer groups based on financial behavior:

Cluster	Profile Summary	Business Insight
0	High balances, high spending, moderate credit utilization	VIP customers – most profitable; target with premium offers, loyalty programs, and personalized promotions.
1	Low balances, low spending, low payments	Low-value customers – may need engagement campaigns to increase usage and retention.
2	Moderate balances, low purchases, moderate cash advance	Conservative spenders – incentivize installment purchases or cross-selling services.
3	Low balances, minimal purchases, high cash advances	High-risk customers – monitor for potential default; offer risk-mitigating services.

Clusters help the bank identify which customer segments to focus on, tailor promotions, and manage risk.

### 2. Risk Prediction Insights

We built a Random Forest model to predict the Risk_Score for new customers:

Low Risk: Moderate-to-high payments, low cash advance reliance, controlled credit utilization.

High Risk: High balances with low payments, high cash advances, or low purchase activity.

Use Case:
The risk score can guide the bank in pre-approving credit limits, detecting potential defaulters, and adjusting interest rates.

### 3. Feature Importance

The Random Forest model highlighted the most influential features for predicting customer risk:

Credit Utilization – High usage of available credit increases risk.

Payment Ratio – Paying less than minimum payments indicates higher risk.

Total Spending – High spending without proportional payments can indicate default potential.

Balance – Larger balances relative to credit limit increase exposure.

Focus on these indicators to monitor and mitigate risk effectively.

### 4. Strategic Recommendations

VIP Programs: Reward high-value clusters with loyalty incentives, exclusive offers, and personalized services.

Risk Management: Monitor high-risk clusters closely; provide interventions like credit counseling or reminders.

Promotion Optimization: Track which promotions drive higher purchases in profitable clusters.

Customer Retention: Engage low-value clusters with targeted campaigns to increase usage without increasing risk.
