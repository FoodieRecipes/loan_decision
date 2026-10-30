import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def evaluate_models():
    base_dir = r"c:\Users\USER\Desktop\Goody_Project"
    df = pd.read_csv(os.path.join(base_dir, "nigeria_credit_risk_dataset.csv"))
    models_dir = os.path.join(base_dir, "models")
    
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    numeric_features = [
        'loan_amount', 'loan_tenor_days', 'interest_rate_pct',
        'applicant_age', 'monthly_income', 'debt_to_income_ratio',
        'credit_score_num', 'loan_to_income_ratio', 'monthly_debt_repayment',
        'total_interest_amount', 'estimated_monthly_installment',
        'installment_to_income_ratio', 'total_debt_service_ratio',
        'disposable_income_after_loan'
    ]
    categorical_features = [
        'loan_product', 'employment_type', 'credit_score_band',
        'state', 'channel', 'age_group', 'income_segment', 'tenor_category'
    ]
    
    X = preprocessor.transform(df[numeric_features + categorical_features])
    y_risk = df['risk_level'].map({'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}).values
    y_bin = df['is_default'].values
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42)
    xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, eval_metric='logloss')
    
    rf_bin_scores = cross_val_score(rf, X, y_bin, cv=cv, scoring='roc_auc')
    xgb_bin_scores = cross_val_score(xgb, X, y_bin, cv=cv, scoring='roc_auc')
    
    rf_risk_acc = cross_val_score(rf, X, y_risk, cv=cv, scoring='accuracy')
    xgb_risk_acc = cross_val_score(xgb, X, y_risk, cv=cv, scoring='accuracy')
    
    print(f"5-Fold Binary ROC-AUC -> RF: {rf_bin_scores.mean():.4f} (+/- {rf_bin_scores.std():.4f}), XGB: {xgb_bin_scores.mean():.4f} (+/- {xgb_bin_scores.std():.4f})")
    print(f"5-Fold Risk Level Acc -> RF: {rf_risk_acc.mean():.4f} (+/- {rf_risk_acc.std():.4f}), XGB: {xgb_risk_acc.mean():.4f} (+/- {xgb_risk_acc.std():.4f})")

if __name__ == "__main__":
    evaluate_models()

