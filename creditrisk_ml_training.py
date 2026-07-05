import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("xgboost not installed - run: pip install xgboost")

# LOAD TRAIN/TEST DATA 

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train default rate: {y_train.mean():.3f}")

results = {}

def evaluate_model(name, model, X_te, y_te, y_pred, y_proba):
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_te, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred))

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
    return y_proba

# Logistic Regression
# (scale features first - LogReg is sensitive to feature scale)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
proba_lr = evaluate_model("Logistic Regression", log_reg, X_test_scaled, y_test, y_pred_lr, y_proba_lr)

# Random Forest

rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
proba_rf = evaluate_model("Random Forest", rf, X_test, y_test, y_pred_rf, y_proba_rf)

# Feature importance from Random Forest
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\n--- Top 10 Feature Importances (Random Forest) ---")
print(importances.head(10))

plt.figure(figsize=(8, 6))
importances.head(10).plot(kind="barh")
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_rf.png", dpi=150)
plt.show()

# XGBoost

proba_xgb = None
if XGB_AVAILABLE:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=42, use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    proba_xgb = evaluate_model("XGBoost", xgb, X_test, y_test, y_pred_xgb, y_proba_xgb)

# COMPARISON TABLE

results_df = pd.DataFrame(results).T
print("\n=== MODEL COMPARISON ===")
print(results_df.round(4))
results_df.to_csv("model_comparison_results.csv")

# ROC CURVE COMPARISON

plt.figure(figsize=(8, 6))
for name, proba in [("Logistic Regression", proba_lr), ("Random Forest", proba_rf), ("XGBoost", proba_xgb)]:
    if proba is not None:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=150)
plt.show()

print("\nDONE - results saved to model_comparison_results.csv")