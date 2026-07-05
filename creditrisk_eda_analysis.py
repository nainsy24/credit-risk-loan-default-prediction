import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

sns.set_style("whitegrid")

import os
from urllib.parse import quote_plus

db_password = os.environ.get("DB_PASSWORD")
encoded_password = quote_plus(db_password)
engine = create_engine(f"postgresql://postgres:{encoded_password}@localhost:5432/lending_club")

print("Loading data...")
df = pd.read_sql("SELECT * FROM loans WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default');", engine)
print(f"Shape: {df.shape}")

df["is_default"] = df["loan_status"].apply(lambda x: 1 if x in ["Charged Off", "Default"] else 0)

# Summary statistics

print("\n--- Numeric summary ---")
print(df.describe().T)

# Distribution of key numeric features

num_features = ["loan_amnt", "int_rate", "annual_inc", "dti", "installment"]
num_features = [c for c in num_features if c in df.columns]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(num_features):
    sns.histplot(df[col], bins=40, ax=axes[i], kde=True)
    axes[i].set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150)
plt.show()

# Outlier check via boxplots

fig, axes = plt.subplots(1, len(num_features), figsize=(18, 5))
for i, col in enumerate(num_features):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()
plt.savefig("eda_boxplots_outliers.png", dpi=150)
plt.show()

# Correlation heatmap (numeric features vs target)

corr_cols = num_features + ["is_default"]
corr = df[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", dpi=150)
plt.show()

# Default rate by key categorical features (annual_inc binned, dti binned)

df["income_bracket"] = pd.cut(df["annual_inc"], bins=[0, 30000, 60000, 90000, 120000, 1e7],
                               labels=["<30k", "30-60k", "60-90k", "90-120k", "120k+"])

income_default = df.groupby("income_bracket")["is_default"].mean().reset_index()
income_default["is_default"] = income_default["is_default"] * 100

plt.figure(figsize=(8, 5))
sns.barplot(data=income_default, x="income_bracket", y="is_default", color="purple")
plt.title("Default Rate by Income Bracket")
plt.ylabel("Default Rate (%)")
plt.xlabel("Annual Income Bracket")
plt.tight_layout()
plt.savefig("eda_default_by_income.png", dpi=150)
plt.show()

print("\nEDA complete. Charts saved as PNG files in the project folder.")