import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from math import sqrt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =============================
# 1. Load Dataset
# =============================
df = pd.read_csv("CC GENERAL.csv")   
print(df.shape)
df.head()

# =============================
# 2. Initial Checks
# =============================
print(df.info())
print(df.describe().T)
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# =============================
# 3. Data Cleaning
# =============================
df = df.drop_duplicates()

zero_fill = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "PAYMENTS"]
df[zero_fill] = df[zero_fill].fillna(0)

df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(0)
df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
df = df[df['BALANCE'] >= 0]

# =============================
# 4. Univariate Analysis
# =============================
num_cols = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(15,10))
for i, col in enumerate(num_cols[:6], 1):
    plt.subplot(2,3,i)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# =============================
# 5. Bivariate Analysis
# =============================
plt.figure(figsize=(8,6))
sns.scatterplot(x="BALANCE", y="PURCHASES", data=df, alpha=0.5)
plt.title("Balance vs Purchases")
plt.show()

numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# =============================
# 6. Feature Engineering
# =============================
df["Purchase_Installment_Ratio"] = df["PURCHASES_INSTALLMENTS_FREQUENCY"] / (df["PURCHASES_FREQUENCY"] + 1e-5)
df["Credit_Utilization"] = df["BALANCE"] / (df["CREDIT_LIMIT"] + 1e-5)
df["Payment_Ratio"] = df["PAYMENTS"] / (df["MINIMUM_PAYMENTS"] + 1e-5)
df["TOTAL_SPENDING"] = df["PURCHASES"] + df["CASH_ADVANCE"]
df["Risk_Score"] = df["CASH_ADVANCE"] / (df["CREDIT_LIMIT"] + 1e-5)

balance_thresh = df["BALANCE"].quantile(0.75)
purchase_thresh = df["PURCHASES"].quantile(0.75)
df["VIP"] = ((df["BALANCE"] > balance_thresh) & (df["PURCHASES"] > purchase_thresh)).astype(int)

# =============================
# 7. Customer Segmentation
# =============================
features = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS",
            "Purchase_Installment_Ratio", "Credit_Utilization", "Payment_Ratio",
            "TOTAL_SPENDING", "Risk_Score"]
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# =============================
# 8. Cluster Profiling
# =============================
print(df["Cluster"].value_counts())
plt.figure(figsize=(6,4))
sns.countplot(x="Cluster", data=df)
plt.title("Number of Customers per Cluster")
plt.show()

plot_features = features + ["VIP"]
num_plots = len(plot_features)
cols = 2
rows = (num_plots + 1) // cols

plt.figure(figsize=(12, rows*4))
gs = gridspec.GridSpec(rows, cols)

for i, feature in enumerate(plot_features):
    ax = plt.subplot(gs[i])
    sns.boxplot(x="Cluster", y=feature, data=df, ax=ax)
    ax.set_title(f"{feature} by Cluster")

plt.tight_layout()
plt.show()

cluster_summary = df.groupby("Cluster")[features + ["VIP"]].mean()
print(cluster_summary)

plt.figure(figsize=(12,6))
sns.heatmap(cluster_summary, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Average Feature Values per Cluster")
plt.show()

# =============================
# 9. Risk Score Prediction (Regression)
# =============================
features_reg = ["BALANCE", "PURCHASES", "PAYMENTS",
                "Purchase_Installment_Ratio", "Credit_Utilization",
                "Payment_Ratio", "TOTAL_SPENDING"]

X_reg = df[features_reg]
y_reg = df["Risk_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled = scaler_reg.transform(X_test)

# --- Linear Regression ---
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

rmse_lr = sqrt(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression Performance:")
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("RMSE:", rmse_lr)

# --- Random Forest with GridSearchCV ---
rf = RandomForestRegressor(random_state=42)
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(rf, rf_params, cv=5, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

rmse_rf = sqrt(mean_squared_error(y_test, y_pred_rf))
print("\nRandom Forest Performance (after GridSearchCV):")
print("Best Params:", grid_rf.best_params_)
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("RMSE:", rmse_rf)

# --- Feature Importance ---
importances = best_rf.feature_importances_
feat_importance = pd.Series(importances, index=features_reg).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_importance, y=feat_importance.index)
plt.title("Feature Importance for Risk Prediction")
plt.show()

# --- Actual vs Predicted Plot ---
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.xlabel("Actual Risk Score")
plt.ylabel("Predicted Risk Score")
plt.title("Actual vs Predicted Risk Score (Random Forest)")
plt.show()

# =============================
# 10. Future Prediction Workflow
# =============================
joblib.dump(best_rf, "risk_rf_model.pkl")
joblib.dump(scaler_reg, "risk_scaler.pkl")

new_customer = pd.DataFrame({
    "BALANCE": [5000],
    "PURCHASES": [1500],
    "PAYMENTS": [400],
    "Purchase_Installment_Ratio": [0.3],
    "Credit_Utilization": [0.25],
    "Payment_Ratio": [1.2],
    "TOTAL_SPENDING": [2000]
})

rf_model = joblib.load("risk_rf_model.pkl")
scaler_model = joblib.load("risk_scaler.pkl")
new_customer_scaled = scaler_model.transform(new_customer)
predicted_risk = rf_model.predict(new_customer_scaled)
print("\nPredicted Risk Score for new customer:", predicted_risk[0])

# =============================
# 11. Save Cleaned & Clustered Data for Shiny
# =============================
df_to_save = df.copy()
cols_order = ["CUST_ID", "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", 
              "PAYMENTS", "MINIMUM_PAYMENTS", "Purchase_Installment_Ratio", 
              "Credit_Utilization", "Payment_Ratio", "TOTAL_SPENDING", 
              "Risk_Score", "VIP", "Cluster"]
df_to_save = df_to_save[cols_order]
df_to_save.to_csv("CC_CUSTOMERS_CLEANED.csv", index=False)
print("Cleaned and clustered dataset saved as 'CC_CUSTOMERS_CLEANED.csv'")
