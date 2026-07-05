import pandas as pd
import numpy as np
import os
import sys

# LOAD DATA

file_path = "accepted_2007_to_2018Q4.csv"

print("Current Folder:", os.getcwd())
print("File Exists:", os.path.exists(file_path))

if not os.path.exists(file_path):
    print(f"ERROR: {file_path} not found!")
    sys.exit()

print("\nLoading dataset...")

# Load only first 100,000 rows
df = pd.read_csv(
    file_path,
    nrows=100000,
    low_memory=False
)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# BASIC INFO

print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes.value_counts())

# MISSING VALUES

missing_pct = (
    df.isnull().sum() / len(df) * 100
).sort_values(ascending=False)

print("\nTop 10 columns with missing values:")
print(missing_pct.head(10))

# DROP HIGHLY MISSING COLUMNS

threshold = 50

cols_to_drop = missing_pct[
    missing_pct > threshold
].index

df_clean = df.drop(columns=cols_to_drop)

print(
    f"\nDropped {len(cols_to_drop)} columns "
    f"having more than {threshold}% missing values."
)

# REMOVE DUPLICATES

before = len(df_clean)

df_clean = df_clean.drop_duplicates()

after = len(df_clean)

print(
    f"Duplicate rows removed: {before - after}"
)

# CLEAN INTEREST RATE

for col in ["int_rate", "revol_util"]:

    if col in df_clean.columns:

        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.replace("%", "", regex=False)
        )

        df_clean[col] = pd.to_numeric(
            df_clean[col],
            errors="coerce"
        )

# CLEAN TERM

if "term" in df_clean.columns:

    df_clean["term"] = (
        df_clean["term"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )

# CLEAN EMPLOYMENT LENGTH

if "emp_length" in df_clean.columns:

    def clean_emp(val):

        if pd.isna(val):
            return np.nan

        val = str(val)

        if "10+" in val:
            return 10

        if "<" in val:
            return 0

        digits = "".join(
            filter(str.isdigit, val)
        )

        return (
            float(digits)
            if digits
            else np.nan
        )

    df_clean["emp_length"] = (
        df_clean["emp_length"]
        .apply(clean_emp)
    )

# DATE COLUMNS

date_cols = [
    "issue_d",
    "earliest_cr_line",
    "last_pymnt_d",
    "last_credit_pull_d"
]

for col in date_cols:

    if col in df_clean.columns:

        df_clean[col] = pd.to_datetime(
            df_clean[col],
            format="%b-%Y",
            errors="coerce"
        )


# FILL MISSING VALUES


num_cols = df_clean.select_dtypes(
    include=[np.number]
).columns

df_clean[num_cols] = (
    df_clean[num_cols]
    .fillna(df_clean[num_cols].median())
)

cat_cols = df_clean.select_dtypes(
    include=["object"]
).columns

df_clean[cat_cols] = (
    df_clean[cat_cols]
    .fillna("Unknown")
)

# FINAL CHECK


print("\nFinal Shape:")
print(df_clean.shape)

print(
    "\nRemaining Missing Values:",
    df_clean.isnull().sum().sum()
)

# SAVE CLEANED FILE

output_file = "loan_cleaned.csv"

df_clean.to_csv(
    output_file,
    index=False
)

print(
    f"\nCleaned dataset saved as: "
    f"{output_file}"
)

print("\nPROCESS COMPLETED SUCCESSFULLY!")
