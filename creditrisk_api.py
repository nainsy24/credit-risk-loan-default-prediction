from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Credit Risk Prediction API")

model = joblib.load("final_credit_risk_model.pkl")
feature_cols = joblib.load("model_feature_columns.pkl")
feature_defaults = joblib.load("model_feature_defaults.pkl")

class LoanInput(BaseModel):
    loan_amnt: float
    int_rate: float
    dti: float
    annual_inc: float
    term: int
    grade: str

@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is running"}

@app.post("/predict")
def predict(loan: LoanInput):
    input_data = pd.DataFrame([feature_defaults[feature_cols].values], columns=feature_cols)

    if "loan_amnt" in input_data.columns:
        input_data["loan_amnt"] = loan.loan_amnt
    if "int_rate" in input_data.columns:
        input_data["int_rate"] = loan.int_rate
    if "dti" in input_data.columns:
        input_data["dti"] = loan.dti
    if "annual_inc" in input_data.columns:
        input_data["annual_inc"] = loan.annual_inc
    if "term" in input_data.columns:
        input_data["term"] = loan.term
    if "grade" in input_data.columns:
        grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
        input_data["grade"] = grade_map[loan.grade]

    proba = model.predict_proba(input_data[feature_cols])[0][1]
    risk_label = "High Risk" if proba > 0.5 else "Low Risk"

    return {
        "default_probability": round(float(proba), 4),
        "risk_label": risk_label
    }