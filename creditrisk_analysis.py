import pandas as pd
from sqlalchemy import create_engine

# CONNECT TO POSTGRESQL

import os
from urllib.parse import quote_plus

db_password = os.environ.get("DB_PASSWORD")
encoded_password = quote_plus(db_password)
engine = create_engine(f"postgresql://postgres:{encoded_password}@localhost:5432/lending_club")

try:
   
    # Loan status distribution (all loans, raw counts)

    q1 = """
    SELECT loan_status, COUNT(*) AS count
    FROM loans
    GROUP BY loan_status
    ORDER BY count DESC;
    """
    print("\n--- Q1: Loan Status Distribution ---")
    print(pd.read_sql(q1, engine))

    # Average interest rate by grade
    
    q2 = """
    SELECT grade, ROUND(AVG(int_rate)::numeric, 2) AS avg_int_rate, COUNT(*) AS num_loans
    FROM loans
    GROUP BY grade
    ORDER BY grade;
    """
    print("\n--- Q2: Avg Interest Rate by Grade ---")
    print(pd.read_sql(q2, engine))

    # Default rate by grade (completed loans only)
    
    q3 = """
    SELECT
        grade,
        COUNT(*) AS completed_loans,
        SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
        ROUND(
            100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
            2
        ) AS default_rate_pct
    FROM loans
    WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
    GROUP BY grade
    ORDER BY grade;
    """
    print("\n--- Q3: Default Rate by Grade (completed loans only) ---")
    print(pd.read_sql(q3, engine))

    # Loan amount and interest rate by purpose

    q4 = """
    SELECT
        purpose,
        COUNT(*) AS num_loans,
        ROUND(AVG(loan_amnt)::numeric, 2) AS avg_loan_amt,
        ROUND(AVG(int_rate)::numeric, 2) AS avg_int_rate
    FROM loans
    GROUP BY purpose
    ORDER BY num_loans DESC;
    """
    print("\n--- Q4: Loan Amount & Interest Rate by Purpose ---")
    print(pd.read_sql(q4, engine))

    # Top 10 states by loan volume
  
    q5 = """
    SELECT
        addr_state,
        COUNT(*) AS num_loans,
        ROUND(AVG(int_rate)::numeric, 2) AS avg_int_rate
    FROM loans
    GROUP BY addr_state
    ORDER BY num_loans DESC
    LIMIT 10;
    """
    print("\n--- Q5: Top 10 States by Loan Volume ---")
    print(pd.read_sql(q5, engine))

    # Default rate by loan purpose (completed loans only)

    q6 = """
    SELECT
        purpose,
        COUNT(*) AS completed_loans,
        SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) AS defaults,
        ROUND(
            100.0 * SUM(CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END) / COUNT(*),
            2
        ) AS default_rate_pct
    FROM loans
    WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default')
    GROUP BY purpose
    ORDER BY default_rate_pct DESC;
    """
    print("\n--- Q6: Default Rate by Purpose (completed loans only) ---")
    print(pd.read_sql(q6, engine))

    print("\nSUCCESS - all queries ran")

except Exception as e:
    print("ERROR OCCURRED:")
    print(e)