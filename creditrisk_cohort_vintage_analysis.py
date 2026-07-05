import pandas as pd
from sqlalchemy import create_engine

import os
from urllib.parse import quote_plus

db_password = os.environ.get("DB_PASSWORD")
encoded_password = quote_plus(db_password)
engine = create_engine(f"postgresql://postgres:{encoded_password}@localhost:5432/lending_club")

# Check issue_d data type

q_check = """
SELECT issue_d, pg_typeof(issue_d) AS data_type
FROM loans
LIMIT 5;
"""
print("--- Checking issue_d type ---")
type_check = pd.read_sql(q_check, engine)
print(type_check)

issue_d_type = str(type_check["data_type"].iloc[0])
print(f"\nDetected issue_d type: {issue_d_type}")

# Build the date expression depending on type

if "date" in issue_d_type or "timestamp" in issue_d_type:
    # Already a proper date/timestamp column
    year_expr = "EXTRACT(YEAR FROM issue_d)"
else:
    # Stored as text -> try casting (handles 'YYYY-MM-DD' style strings)
    year_expr = "EXTRACT(YEAR FROM issue_d::date)"

print(f"\nUsing year expression: {year_expr}")

# Vintage analysis - default rate by issue year

q_vintage = f"""
SELECT
    {year_expr}::int AS issue_year,
    COUNT(*) AS completed_loans,
    SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
    ROUND(
        100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS default_rate_pct,
    ROUND(AVG(int_rate)::numeric, 2) AS avg_int_rate,
    ROUND(AVG(loan_amnt)::numeric, 2) AS avg_loan_amt
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
GROUP BY issue_year
ORDER BY issue_year;
"""

print("\n--- Vintage Analysis: Default Rate by Issue Year ---")
try:
    df_vintage = pd.read_sql(q_vintage, engine)
    print(df_vintage)
except Exception as e:
    print("ERROR running vintage query:")
    print(e)
    df_vintage = None

# Vintage analysis - default rate by issue year AND grade
# (window function: rank years within each grade by default rate)

q_vintage_grade = f"""
SELECT
    {year_expr}::int AS issue_year,
    grade,
    COUNT(*) AS completed_loans,
    SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
    ROUND(
        100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS default_rate_pct,
    RANK() OVER (
        PARTITION BY grade
        ORDER BY ROUND(
            100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
            2
        ) DESC
    ) AS risk_rank_within_grade
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
GROUP BY issue_year, grade
ORDER BY grade, issue_year;
"""

print("\n--- Vintage Analysis: Default Rate by Year x Grade (with risk rank) ---")
try:
    df_vintage_grade = pd.read_sql(q_vintage_grade, engine)
    print(df_vintage_grade)
except Exception as e:
    print("ERROR running vintage x grade query:")
    print(e)
    df_vintage_grade = None

# Save results to CSV for later use (visualization / report)

if df_vintage is not None:
    df_vintage.to_csv("vintage_analysis_by_year.csv", index=False)
    print("\nSaved: vintage_analysis_by_year.csv")

if df_vintage_grade is not None:
    df_vintage_grade.to_csv("vintage_analysis_by_year_grade.csv", index=False)
    print("Saved: vintage_analysis_by_year_grade.csv")

print("\nDONE")