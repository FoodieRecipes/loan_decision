import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def get_dataset_path():
    base_dir = r"c:\Users\USER\Desktop\Goody_Project"
    candidates = [
        os.path.join(base_dir, "nigeria_credit_risk_dataset_2500.csv"),
        os.path.join(base_dir, "nigeria_credit_risk_dataset_1500.csv"),
        os.path.join(base_dir, "nigeria_credit_risk_dataset_augmented.csv"),
        os.path.join(base_dir, "nigeria-digital-lending-credit-risk", "nigeria_credit_risk_dataset.csv"),
        os.path.join(base_dir, "nigeria_credit_risk_dataset.csv")
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if len(df) >= 2000:
                return p
    return candidates[0]

def build_and_train_models_with_tuning():
    base_dir = r"c:\Users\USER\Desktop\Goody_Project"
    dataset_path = get_dataset_path()
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset Shape: {df.shape} (Total Applicants: {len(df)})")
    
    numeric_features = [
        'loan_amount', 'loan_tenor_days', 'interest_rate_pct',
        'applicant_age', 'monthly_income', 'debt_to_income_ratio',
        'credit_score_num', 'loan_to_income_ratio', 'monthly_debt_repayment',
        'total_interest_amount', 'estimated_monthly_installment',
        'installment_to_income_ratio', 'total_debt_service_ratio',
        'disposable_income_after_loan', 'affordability_index',
        'credit_vulnerability_index'
    ]
    
    categorical_features = [
        'loan_product', 'employment_type', 'credit_score_band',
        'state', 'channel', 'age_group', 'income_segment', 'tenor_category'
    ]
    
    X = df[numeric_features + categorical_features].copy()
    
    risk_classes = ['Low Risk', 'Medium Risk', 'High Risk']
    risk_mapping = {'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}
    y_risk = df['risk_level'].map(risk_mapping).values
    y_binary = df['is_default'].values
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + encoded_cat_names
    
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(all_feature_names, os.path.join(models_dir, "feature_names.joblib"))
    
    # Stratified 80/20 train-test split
    X_train, X_test, y_train_risk, y_test_risk, y_train_bin, y_test_bin = train_test_split(
        X_processed, y_risk, y_binary, test_size=0.20, random_state=42, stratify=y_risk
    )
    
    print("\n=======================================================")
    print("HYPERPARAMETER TUNING & OPTIMIZATION (STRATIFIED CV)")
    print("=======================================================")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # 1. Random Forest Tuning
    print("Tuning Random Forest...")
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    
    rf_base = RandomForestClassifier(random_state=42)
    grid_rf = GridSearchCV(
        rf_base,
        param_grid_rf,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    grid_rf.fit(X_train, y_train_risk)
    best_rf_risk = grid_rf.best_estimator_
    print(f"Optimal Random Forest Parameters: {grid_rf.best_params_}")
    print(f"Random Forest Best CV Accuracy: {grid_rf.best_score_*100:.2f}%")
    
    # 2. XGBoost Tuning
    print("Tuning XGBoost...")
    param_grid_xgb = {
        'n_estimators': [100, 150],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.08],
        'subsample': [0.85],
        'colsample_bytree': [0.85],
        'reg_lambda': [1.0]
    }
    
    xgb_base = XGBClassifier(random_state=42, eval_metric='mlogloss')
    grid_xgb = GridSearchCV(
        xgb_base,
        param_grid_xgb,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    grid_xgb.fit(X_train, y_train_risk)
    best_xgb_risk = grid_xgb.best_estimator_
    print(f"Optimal XGBoost Parameters: {grid_xgb.best_params_}")
    print(f"XGBoost Best CV Accuracy: {grid_xgb.best_score_*100:.2f}%")
    
    # 3. Binary Default Models
    print("Fitting Binary Default Predictors...")
    best_rf_bin = RandomForestClassifier(
        n_estimators=grid_rf.best_params_['n_estimators'],
        max_depth=grid_rf.best_params_['max_depth'],
        min_samples_leaf=grid_rf.best_params_['min_samples_leaf'],
        class_weight=grid_rf.best_params_['class_weight'],
        random_state=42
    )
    best_rf_bin.fit(X_train, y_train_bin)
    
    best_xgb_bin = XGBClassifier(
        n_estimators=grid_xgb.best_params_['n_estimators'],
        max_depth=grid_xgb.best_params_['max_depth'],
        learning_rate=grid_xgb.best_params_['learning_rate'],
        subsample=grid_xgb.best_params_['subsample'],
        colsample_bytree=grid_xgb.best_params_['colsample_bytree'],
        reg_lambda=grid_xgb.best_params_['reg_lambda'],
        random_state=42,
        eval_metric='logloss'
    )
    best_xgb_bin.fit(X_train, y_train_bin)
    
    # 4. Comprehensive Evaluation on Holdout Test Set (300 Loans)
    print("\n=======================================================")
    print("TEST SET EVALUATION RESULTS (HOLDOUT 20% = 300 LOANS)")
    print("=======================================================")
    
    rf_pred_risk = best_rf_risk.predict(X_test)
    rf_proba_risk = best_rf_risk.predict_proba(X_test)
    rf_pred_bin = best_rf_bin.predict(X_test)
    rf_proba_bin = best_rf_bin.predict_proba(X_test)[:, 1]
    
    rf_acc = accuracy_score(y_test_risk, rf_pred_risk)
    rf_prec = precision_score(y_test_risk, rf_pred_risk, average='weighted', zero_division=0)
    rf_rec = recall_score(y_test_risk, rf_pred_risk, average='weighted', zero_division=0)
    rf_f1 = f1_score(y_test_risk, rf_pred_risk, average='weighted', zero_division=0)
    rf_auc = roc_auc_score(y_test_risk, rf_proba_risk, multi_class='ovr', average='weighted')
    rf_bin_auc = roc_auc_score(y_test_bin, rf_proba_bin)
    rf_cm = confusion_matrix(y_test_risk, rf_pred_risk).tolist()
    
    xgb_pred_risk = best_xgb_risk.predict(X_test)
    xgb_proba_risk = best_xgb_risk.predict_proba(X_test)
    xgb_pred_bin = best_xgb_bin.predict(X_test)
    xgb_proba_bin = best_xgb_bin.predict_proba(X_test)[:, 1]
    
    xgb_acc = accuracy_score(y_test_risk, xgb_pred_risk)
    xgb_prec = precision_score(y_test_risk, xgb_pred_risk, average='weighted', zero_division=0)
    xgb_rec = recall_score(y_test_risk, xgb_pred_risk, average='weighted', zero_division=0)
    xgb_f1 = f1_score(y_test_risk, xgb_pred_risk, average='weighted', zero_division=0)
    xgb_auc = roc_auc_score(y_test_risk, xgb_proba_risk, multi_class='ovr', average='weighted')
    xgb_bin_auc = roc_auc_score(y_test_bin, xgb_proba_bin)
    xgb_cm = confusion_matrix(y_test_risk, xgb_pred_risk).tolist()
    
    print(f"[Tuned Random Forest] -> Accuracy: {rf_acc*100:.2f}% | Precision: {rf_prec*100:.2f}% | Recall: {rf_rec*100:.2f}% | F1: {rf_f1*100:.2f}% | Multiclass AUC: {rf_auc:.4f} | Binary AUC: {rf_bin_auc:.4f}")
    print(f"[Tuned XGBoost]       -> Accuracy: {xgb_acc*100:.2f}% | Precision: {xgb_prec*100:.2f}% | Recall: {xgb_rec*100:.2f}% | F1: {xgb_f1*100:.2f}% | Multiclass AUC: {xgb_auc:.4f} | Binary AUC: {xgb_bin_auc:.4f}")
    
    if rf_acc > xgb_acc:
        best_model_name = "Random Forest"
        best_risk_model = best_rf_risk
    elif xgb_acc > rf_acc:
        best_model_name = "XGBoost"
        best_risk_model = best_xgb_risk
    else:
        best_model_name = "XGBoost" if xgb_f1 >= rf_f1 else "Random Forest"
        best_risk_model = best_xgb_risk if xgb_f1 >= rf_f1 else best_rf_risk
        
    print(f"\n>>> BEST MODEL SELECTED: {best_model_name} <<<")
    
    # Save models
    joblib.dump(best_rf_risk, os.path.join(models_dir, "rf_risk_model.joblib"))
    joblib.dump(best_xgb_risk, os.path.join(models_dir, "xgb_risk_model.joblib"))
    joblib.dump(best_rf_bin, os.path.join(models_dir, "rf_binary_model.joblib"))
    joblib.dump(best_xgb_bin, os.path.join(models_dir, "xgb_binary_model.joblib"))
    joblib.dump(best_risk_model, os.path.join(models_dir, "best_risk_model.joblib"))
    
    rf_importance = dict(sorted(
        zip(all_feature_names, [float(x) for x in best_rf_risk.feature_importances_]),
        key=lambda x: x[1], reverse=True
    ))
    
    xgb_importance = dict(sorted(
        zip(all_feature_names, [float(x) for x in best_xgb_risk.feature_importances_]),
        key=lambda x: x[1], reverse=True
    ))
    
    metrics = {
        "best_model": best_model_name,
        "classes": risk_classes,
        "dataset_rows": len(df),
        "test_samples": len(y_test_risk),
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "all_encoded": all_feature_names
        },
        "hyperparameter_tuning": {
            "technique": "GridSearchCV with 3-Fold Stratified Cross-Validation",
            "scoring": "Stratified Classification Accuracy",
            "rf_best_params": grid_rf.best_params_,
            "rf_cv_score": round(float(grid_rf.best_score_), 4),
            "xgb_best_params": grid_xgb.best_params_,
            "xgb_cv_score": round(float(grid_xgb.best_score_), 4)
        },
        "random_forest": {
            "accuracy": round(float(rf_acc), 4),
            "precision": round(float(rf_prec), 4),
            "recall": round(float(rf_rec), 4),
            "f1_score": round(float(rf_f1), 4),
            "auc_roc": round(float(rf_auc), 4),
            "binary_auc": round(float(rf_bin_auc), 4),
            "confusion_matrix": rf_cm,
            "classification_report": classification_report(y_test_risk, rf_pred_risk, target_names=risk_classes, output_dict=True)
        },
        "xgboost": {
            "accuracy": round(float(xgb_acc), 4),
            "precision": round(float(xgb_prec), 4),
            "recall": round(float(xgb_rec), 4),
            "f1_score": round(float(xgb_f1), 4),
            "auc_roc": round(float(xgb_auc), 4),
            "binary_auc": round(float(xgb_bin_auc), 4),
            "confusion_matrix": xgb_cm,
            "classification_report": classification_report(y_test_risk, xgb_pred_risk, target_names=risk_classes, output_dict=True)
        },
        "feature_importance": {
            "random_forest": rf_importance,
            "xgboost": xgb_importance
        }
    }
    
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nMetrics and optimal hyperparameters successfully saved to: {metrics_path}")

if __name__ == "__main__":
    build_and_train_models_with_tuning()
