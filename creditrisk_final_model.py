import pandas as pd
import joblib
from xgboost import XGBClassifier

# LOAD TRAIN/TEST DATA

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

print(f"Training final model on {X_train.shape[0]} rows...")

# TRAIN FINAL XGBOOST MODEL

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

final_model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss",
    random_state=42
)
final_model.fit(X_train, y_train)

# SAVE MODEL TO DISK

joblib.dump(final_model, "final_credit_risk_model.pkl")
print("Saved: final_credit_risk_model.pkl")

joblib.dump(list(X_train.columns), "model_feature_columns.pkl")
print("Saved: model_feature_columns.pkl")

print("\nDONE - model ready for reuse without retraining")