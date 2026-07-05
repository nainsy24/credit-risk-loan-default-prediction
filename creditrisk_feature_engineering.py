import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
from urllib.parse import quote_plus

db_password = os.environ.get("DB_PASSWORD")
encoded_password = quote_plus(db_password)
engine = create_engine(f"postgresql://postgres:{encoded_password}@localhost:5432/lending_club")

print("Loading data from PostgreSQL...")
df = pd.read_sql("SELECT * FROM loans;", engine)
print(f"Loaded shape: {df.shape}")

# Keep only completed loans (known outcome)

df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
print(f"After filtering to completed loans: {df.shape}")

# Create target variable

df["is_default"] = df["loan_status"].apply(lambda x: 1 if x in ["Charged Off", "Default"] else 0)
print("\nTarget distribution:")
print(df["is_default"].value_counts(normalize=True).round(3))

# Remove leakage columns
# These describe outcomes that happen AFTER the loan is funded -
# including them would let the model "cheat"

leakage_cols = [
    "loan_status", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "out_prncp", "out_prncp_inv", "debt_settlement_flag", "hardship_flag",
    "settlement_status", "funded_amnt", "funded_amnt_inv",
    "last_fico_range_low", "last_fico_range_high"
]
leakage_cols_present = [c for c in leakage_cols if c in df.columns]
df = df.drop(columns=leakage_cols_present)
print(f"\nDropped {len(leakage_cols_present)} leakage columns: {leakage_cols_present}")

# Drop columns that are pure identifiers / free text (not useful features)

id_cols = ["id", "member_id", "url", "desc", "title", "emp_title", "zip_code", "policy_code"]
id_cols_present = [c for c in id_cols if c in df.columns]
df = df.drop(columns=id_cols_present)
print(f"Dropped {len(id_cols_present)} identifier/text columns: {id_cols_present}")

# Drop any remaining date columns (not directly usable by basic models)
# We already used issue_d for vintage analysis in SQL; for modeling we drop raw dates

date_cols = [c for c in df.columns if df[c].dtype == "object" and "_d" in c]
date_cols += [c for c in ["issue_d", "earliest_cr_line"] if c in df.columns]
date_cols = list(set([c for c in date_cols if c in df.columns]))
df = df.drop(columns=date_cols, errors="ignore")
print(f"Dropped date columns: {date_cols}")

# Encode remaining categorical columns

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"\nEncoding categorical columns: {cat_cols}")

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Final check - drop any remaining missing values

df = df.dropna()
print(f"\nFinal shape after dropping any remaining NaNs: {df.shape}")

# Train/test split

X = df.drop(columns=["is_default"])
y = df["is_default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train default rate: {y_train.mean():.3f}, Test default rate: {y_test.mean():.3f}")

# Save everything for the next step (model training)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nSaved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("DONE - ready for model training")