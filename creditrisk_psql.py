import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv("loan_cleaned.csv")

engine = create_engine("postgresql://postgres:Nainsy%232005@localhost:5432/lending_club")

df.to_sql("loans", engine, if_exists="replace", index=False)

print("Data loaded into PostgreSQL -> table 'loans'")
print("Rows:", len(df))