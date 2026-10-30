import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, average_precision_score
)

BASE_DIR = r"c:\Users\USER\Desktop\Goody_Project"
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "nigeria_credit_risk_dataset.csv")

RISK_CLASSES = ['Low Risk', 'Medium Risk', 'High Risk']
RISK_MAP = {'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}
INV_RISK_MAP = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

NUMERIC_FEATURES = [
    'loan_amount', 'loan_tenor_days', 'interest_rate_pct',
    'applicant_age', 'monthly_income', 'debt_to_income_ratio',
    'credit_score_num', 'loan_to_income_ratio', 'monthly_debt_repayment',
    'total_interest_amount', 'estimated_monthly_installment',
    'installment_to_income_ratio', 'total_debt_service_ratio',
    'disposable_income_after_loan', 'affordability_index',
    'credit_vulnerability_index'
]

CATEGORICAL_FEATURES = [
    'loan_product', 'employment_type', 'credit_score_band',
    'state', 'channel', 'age_group', 'income_segment', 'tenor_category'
]

CREDIT_MAP = {'A': 800, 'B': 700, 'C': 600, 'D': 500, 'E': 400}

def load_artifacts():
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    rf_risk = joblib.load(os.path.join(MODELS_DIR, "rf_risk_model.joblib"))
    xgb_risk = joblib.load(os.path.join(MODELS_DIR, "xgb_risk_model.joblib"))
    rf_bin = joblib.load(os.path.join(MODELS_DIR, "rf_binary_model.joblib"))
    xgb_bin = joblib.load(os.path.join(MODELS_DIR, "xgb_binary_model.joblib"))
    best_risk = joblib.load(os.path.join(MODELS_DIR, "best_risk_model.joblib"))
    
    with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
        
    return {
        "preprocessor": preprocessor,
        "rf_risk": rf_risk,
        "xgb_risk": xgb_risk,
        "rf_bin": rf_bin,
        "xgb_bin": xgb_bin,
        "best_risk": best_risk,
        "metrics": metrics
    }

def compute_derived_features(input_dict):
    """
    Computes all engineered financial and demographic features for a single applicant.
    """
    loan_amount = float(input_dict.get('loan_amount', 0.0))
    monthly_income = float(input_dict.get('monthly_income', 1.0))
    if monthly_income <= 0:
        monthly_income = 1.0
        
    loan_tenor_days = int(input_dict.get('loan_tenor_days', 30))
    interest_rate_pct = float(input_dict.get('interest_rate_pct', 15.0))
    applicant_age = int(input_dict.get('applicant_age', 30))
    debt_to_income_ratio = float(input_dict.get('debt_to_income_ratio', 20.0))
    credit_score_band = str(input_dict.get('credit_score_band', 'C')).upper().strip()
    
    # Financial computations
    credit_score_num = CREDIT_MAP.get(credit_score_band, 600)
    loan_to_income_ratio = round(loan_amount / monthly_income, 4)
    monthly_debt_repayment = round(monthly_income * (debt_to_income_ratio / 100.0), 2)
    total_interest_amount = round(loan_amount * (interest_rate_pct / 100.0), 2)
    expected_total_due = loan_amount + total_interest_amount
    
    months = max(loan_tenor_days / 30.0, 0.5)
    estimated_monthly_installment = round(expected_total_due / months, 2)
    installment_to_income_ratio = round((estimated_monthly_installment / monthly_income) * 100.0, 2)
    total_debt_service_ratio = round(debt_to_income_ratio + installment_to_income_ratio, 2)
    disposable_income_after_loan = round(monthly_income - monthly_debt_repayment - estimated_monthly_installment, 2)
    
    # Advanced interaction ratios
    affordability_index = round(max(0.0, disposable_income_after_loan) / max(estimated_monthly_installment, 1.0), 2)
    credit_vulnerability_index = round((debt_to_income_ratio * max(0, 850 - credit_score_num)) / 100.0, 2)
    
    # Demographic binnings
    if applicant_age <= 25:
        age_group = '18-25'
    elif applicant_age <= 35:
        age_group = '26-35'
    elif applicant_age <= 45:
        age_group = '36-45'
    elif applicant_age <= 55:
        age_group = '46-55'
    else:
        age_group = '56+'
        
    if monthly_income < 100000:
        income_segment = 'Low (<₦100k)'
    elif monthly_income <= 250000:
        income_segment = 'Lower-Middle (₦100k-₦250k)'
    elif monthly_income <= 500000:
        income_segment = 'Middle (₦250k-₦500k)'
    else:
        income_segment = 'High (>₦500k)'
        
    if loan_tenor_days <= 30:
        tenor_category = 'Ultra-Short (<=30d)'
    elif loan_tenor_days <= 90:
        tenor_category = 'Short (31-90d)'
    elif loan_tenor_days <= 180:
        tenor_category = 'Medium (91-180d)'
    else:
        tenor_category = 'Long (>180d)'
        
    full_dict = {
        'loan_amount': loan_amount,
        'loan_tenor_days': loan_tenor_days,
        'interest_rate_pct': interest_rate_pct,
        'applicant_age': applicant_age,
        'monthly_income': monthly_income,
        'debt_to_income_ratio': debt_to_income_ratio,
        'credit_score_num': credit_score_num,
        'loan_to_income_ratio': loan_to_income_ratio,
        'monthly_debt_repayment': monthly_debt_repayment,
        'total_interest_amount': total_interest_amount,
        'estimated_monthly_installment': estimated_monthly_installment,
        'installment_to_income_ratio': installment_to_income_ratio,
        'total_debt_service_ratio': total_debt_service_ratio,
        'disposable_income_after_loan': disposable_income_after_loan,
        'affordability_index': affordability_index,
        'credit_vulnerability_index': credit_vulnerability_index,
        'loan_product': input_dict.get('loan_product', 'Personal Loan'),
        'employment_type': input_dict.get('employment_type', 'Employed'),
        'credit_score_band': credit_score_band,
        'state': input_dict.get('state', 'Lagos'),
        'channel': input_dict.get('channel', 'Mobile App'),
        'age_group': age_group,
        'income_segment': income_segment,
        'tenor_category': tenor_category
    }
    return full_dict

def predict_single_applicant(input_dict, artifacts=None):
    if artifacts is None:
        artifacts = load_artifacts()
        
    preprocessor = artifacts['preprocessor']
    rf_risk = artifacts['rf_risk']
    xgb_risk = artifacts['xgb_risk']
    rf_bin = artifacts['rf_bin']
    xgb_bin = artifacts['xgb_bin']
    best_model_name = artifacts['metrics']['best_model']
    
    full_dict = compute_derived_features(input_dict)
    df_single = pd.DataFrame([full_dict])
    
    # Preprocess
    X_single = preprocessor.transform(df_single[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    
    # RF predictions
    rf_risk_class_idx = rf_risk.predict(X_single)[0]
    rf_risk_proba = rf_risk.predict_proba(X_single)[0]
    rf_risk_class = INV_RISK_MAP[rf_risk_class_idx]
    rf_default_prob = rf_bin.predict_proba(X_single)[0][1]
    
    # XGB predictions
    xgb_risk_class_idx = xgb_risk.predict(X_single)[0]
    xgb_risk_proba = xgb_risk.predict_proba(X_single)[0]
    xgb_risk_class = INV_RISK_MAP[xgb_risk_class_idx]
    xgb_default_prob = xgb_bin.predict_proba(X_single)[0][1]
    
    # Primary model decision based on best selected model
    if best_model_name == "Random Forest":
        primary_risk = rf_risk_class
        primary_risk_idx = rf_risk_class_idx
        primary_proba = rf_risk_proba
        primary_default_prob = rf_default_prob
    else:
        primary_risk = xgb_risk_class
        primary_risk_idx = xgb_risk_class_idx
        primary_proba = xgb_risk_proba
        primary_default_prob = xgb_default_prob
        
    # Decision mapping according to Step 10 of Thesis Architecture:
    # IF Risk_Level == "Low Risk" THEN Recommendation = "Approve Loan" (Accept)
    # ELSE IF Risk_Level == "Medium Risk" THEN Recommendation = "Review Application" (Review)
    # ELSE Recommendation = "Reject Loan" (Reject)
    if primary_risk == "Low Risk":
        recommendation = "Approve Loan"
        decision = "Accept"
        approval_status = "Approved"
        decision_color = "#28a745" # Green
    elif primary_risk == "Medium Risk":
        recommendation = "Review Application"
        decision = "Review"
        approval_status = "Under Review"
        decision_color = "#ffc107" # Yellow
    else:
        recommendation = "Reject Loan"
        decision = "Reject"
        approval_status = "Declined"
        decision_color = "#dc3545" # Red
        
    # Risk Drivers and Explanations
    risk_drivers = []
    recommendations_list = []
    
    dti = full_dict['debt_to_income_ratio']
    tdsr = full_dict['total_debt_service_ratio']
    cs_band = full_dict['credit_score_band']
    disp_income = full_dict['disposable_income_after_loan']
    lti = full_dict['loan_to_income_ratio']
    
    if cs_band in ['A', 'B']:
        risk_drivers.append(f"✅ Strong Credit Bureau Standing: Band {cs_band} indicates established repayment history and high creditworthiness.")
    elif cs_band == 'C':
        risk_drivers.append(f"⚠️ Moderate Credit Standing: Band C reflects average credit history; standard monitoring warranted.")
    else:
        risk_drivers.append(f"🚨 Adverse Credit Standing: Band {cs_band} flags severe historical delinquency or thin file.")
        
    if dti >= 50.0:
        risk_drivers.append(f"🚨 Critical Existing Debt: Current DTI is {dti:.1f}% (exceeds 50% prudential danger threshold).")
    elif dti >= 40.0:
        risk_drivers.append(f"⚠️ Elevated Existing Debt: Current DTI is {dti:.1f}% (above 40% caution threshold).")
    else:
        risk_drivers.append(f"✅ Healthy Existing Debt: DTI is {dti:.1f}% (comfortably below 40%).")
        
    if tdsr > 60.0:
        risk_drivers.append(f"🚨 Unsustainable Debt Service: Total debt obligations including this loan consume {tdsr:.1f}% of monthly income.")
    elif tdsr > 45.0:
        risk_drivers.append(f"⚠️ Moderate Total Burden: Total debt service requires {tdsr:.1f}% of monthly cash flow.")
    else:
        risk_drivers.append(f"✅ Manageable Total Installment: Total debt service ratio is {tdsr:.1f}%.")
        
    if disp_income <= 0:
        risk_drivers.append(f"🚨 Negative Cash Cushion: Projected monthly disposable income is -₦{abs(disp_income):,.2f}.")
    else:
        risk_drivers.append(f"✅ Positive Cash Cushion: Net remaining disposable income after all debt is ₦{disp_income:,.2f}.")
        
    # Specific policy recommendations
    if decision == "Accept":
        recommendations_list.append("Loan qualifies for automated instant disbursement via connected channel.")
        recommendations_list.append("Standard direct debit mandate (NIBSS e-BillsPay / Paystack) should be scheduled against salary account.")
    elif decision == "Review":
        recommendations_list.append("Request 3-6 months verified bank statement or proof of payroll direct deposit.")
        if lti > 1.5:
            suggested_amount = full_dict['monthly_income'] * 0.8
            recommendations_list.append(f"Consider counter-offering a lower principal of ₦{suggested_amount:,.0f} to improve debt sustainability.")
        recommendations_list.append("Subject to loan committee approval or secondary underwriter sign-off.")
    else:
        recommendations_list.append("Decline application due to elevated probability of default.")
        recommendations_list.append("Send adverse action notification detailing credit score and DTI policy thresholds.")
        recommendations_list.append("Borrower eligible for re-evaluation in 90 days after reducing existing debt burdens.")
        
    return {
        "primary_risk": primary_risk,
        "recommendation": recommendation,
        "decision": decision,
        "approval_status": approval_status,
        "decision_color": decision_color,
        "best_model_name": best_model_name,
        "default_probability": float(primary_default_prob),
        "primary_proba": {k: float(v) for k, v in zip(RISK_CLASSES, primary_proba)},
        "rf": {
            "risk_class": rf_risk_class,
            "risk_proba": {k: float(v) for k, v in zip(RISK_CLASSES, rf_risk_proba)},
            "default_prob": float(rf_default_prob)
        },
        "xgb": {
            "risk_class": xgb_risk_class,
            "risk_proba": {k: float(v) for k, v in zip(RISK_CLASSES, xgb_risk_proba)},
            "default_prob": float(xgb_default_prob)
        },
        "derived_features": full_dict,
        "risk_drivers": risk_drivers,
        "policy_recommendations": recommendations_list
    }

def predict_batch(df_input, artifacts=None):
    """
    Takes a DataFrame of new loan applicants, computes all derived interaction
    ratios and demographic features, runs Random Forest and XGBoost inferences,
    and returns an augmented DataFrame with classifications and recommendations.
    """
    if artifacts is None:
        artifacts = load_artifacts()
        
    preprocessor = artifacts['preprocessor']
    rf_risk = artifacts['rf_risk']
    xgb_risk = artifacts['xgb_risk']
    rf_bin = artifacts['rf_bin']
    xgb_bin = artifacts['xgb_bin']
    best_model_name = artifacts['metrics']['best_model']
    
    df_clean = df_input.copy()
    
    # Process row by row through compute_derived_features
    processed_rows = []
    for _, row in df_clean.iterrows():
        row_dict = row.to_dict()
        full_dict = compute_derived_features(row_dict)
        processed_rows.append(full_dict)
        
    df_derived = pd.DataFrame(processed_rows)
    
    # Preprocess with trained ColumnTransformer
    X_proc = preprocessor.transform(df_derived[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    
    # Model predictions
    rf_preds = rf_risk.predict(X_proc)
    rf_default_probs = rf_bin.predict_proba(X_proc)[:, 1]
    
    xgb_preds = xgb_risk.predict(X_proc)
    xgb_default_probs = xgb_bin.predict_proba(X_proc)[:, 1]
    
    results = df_clean.copy()
    
    results['rf_predicted_risk'] = [INV_RISK_MAP[p] for p in rf_preds]
    results['rf_default_prob_pct'] = np.round(rf_default_probs * 100.0, 1)
    
    results['xgb_predicted_risk'] = [INV_RISK_MAP[p] for p in xgb_preds]
    results['xgb_default_prob_pct'] = np.round(xgb_default_probs * 100.0, 1)
    
    # Best Model Primary Classification
    if best_model_name == "Random Forest":
        results['predicted_risk_level'] = results['rf_predicted_risk']
        results['default_probability_pct'] = results['rf_default_prob_pct']
    else:
        results['predicted_risk_level'] = results['xgb_predicted_risk']
        results['default_probability_pct'] = results['xgb_default_prob_pct']
        
    # Map recommendations according to Architecture Step 10
    decision_map = {'Low Risk': 'Accept', 'Medium Risk': 'Review', 'High Risk': 'Reject'}
    rec_map = {'Low Risk': 'Approve Loan', 'Medium Risk': 'Review Application', 'High Risk': 'Reject Loan'}
    status_map = {'Low Risk': 'Approved', 'Medium Risk': 'Under Review', 'High Risk': 'Declined'}
    
    results['system_decision'] = results['predicted_risk_level'].map(decision_map)
    results['system_recommendation'] = results['predicted_risk_level'].map(rec_map)
    results['approval_status'] = results['predicted_risk_level'].map(status_map)
    
    # Add key calculated prudential ratios
    results['monthly_installment'] = df_derived['estimated_monthly_installment']
    results['total_debt_service_ratio'] = df_derived['total_debt_service_ratio']
    results['disposable_income_after_loan'] = df_derived['disposable_income_after_loan']
    results['affordability_index'] = df_derived['affordability_index']
    results['credit_vulnerability_index'] = df_derived['credit_vulnerability_index']
    
    return results

_DETAILED_EVAL_CACHE = None

def get_detailed_evaluation_data():
    """
    Computes complete comparative evaluation metrics on holdout test set (500 samples):
    - Binary ROC curves & AUC
    - Binary Precision-Recall curves & Average Precision
    - Multiclass One-vs-Rest (OvR) ROC curves for each risk category
    - Random Forest vs XGBoost agreement matrix and concordance %
    - Distribution of predicted default probabilities
    """
    global _DETAILED_EVAL_CACHE
    if _DETAILED_EVAL_CACHE is not None:
        return _DETAILED_EVAL_CACHE

    artifacts = load_artifacts()
    preprocessor = artifacts['preprocessor']
    rf_risk = artifacts['rf_risk']
    xgb_risk = artifacts['xgb_risk']
    rf_bin = artifacts['rf_bin']
    xgb_bin = artifacts['xgb_bin']
    
    candidates = [
        os.path.join(BASE_DIR, "nigeria_credit_risk_dataset_2500.csv"),
        os.path.join(BASE_DIR, "nigeria_credit_risk_dataset_1500.csv"),
        os.path.join(BASE_DIR, "nigeria_credit_risk_dataset.csv")
    ]
    data_file = DATASET_PATH
    for c in candidates:
        if os.path.exists(c):
            data_file = c
            break

    df = pd.read_csv(data_file)
    risk_mapping = {'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}
    y_risk = df['risk_level'].map(risk_mapping).values
    y_bin = df['is_default'].values
    
    X = preprocessor.transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    
    X_train, X_test, y_train_risk, y_test_risk, y_train_bin, y_test_bin = train_test_split(
        X, y_risk, y_bin, test_size=0.20, random_state=42, stratify=y_risk
    )
    
    # Multiclass probabilities
    rf_risk_probs = rf_risk.predict_proba(X_test)
    xgb_risk_probs = xgb_risk.predict_proba(X_test)
    
    # Binary default probabilities
    rf_bin_probs = rf_bin.predict_proba(X_test)[:, 1]
    xgb_bin_probs = xgb_bin.predict_proba(X_test)[:, 1]
    
    # 1. Binary ROC Curves
    rf_fpr, rf_tpr, _ = roc_curve(y_test_bin, rf_bin_probs)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test_bin, xgb_bin_probs)
    
    # 2. Precision-Recall Curves
    rf_prec_curve, rf_rec_curve, _ = precision_recall_curve(y_test_bin, rf_bin_probs)
    xgb_prec_curve, xgb_rec_curve, _ = precision_recall_curve(y_test_bin, xgb_bin_probs)
    rf_ap = average_precision_score(y_test_bin, rf_bin_probs)
    xgb_ap = average_precision_score(y_test_bin, xgb_bin_probs)
    
    # 3. Multiclass OvR ROC Curves
    ovr_roc = {}
    for i, cls_name in enumerate(RISK_CLASSES):
        y_c = (y_test_risk == i).astype(int)
        rf_c_fpr, rf_c_tpr, _ = roc_curve(y_c, rf_risk_probs[:, i])
        xgb_c_fpr, xgb_c_tpr, _ = roc_curve(y_c, xgb_risk_probs[:, i])
        rf_c_auc = roc_auc_score(y_c, rf_risk_probs[:, i])
        xgb_c_auc = roc_auc_score(y_c, xgb_risk_probs[:, i])
        ovr_roc[cls_name] = {
            "rf": {"fpr": rf_c_fpr.tolist(), "tpr": rf_c_tpr.tolist(), "auc": round(float(rf_c_auc), 4)},
            "xgb": {"fpr": xgb_c_fpr.tolist(), "tpr": xgb_c_tpr.tolist(), "auc": round(float(xgb_c_auc), 4)}
        }
        
    # 4. Model Predictions & Agreement
    rf_preds = rf_risk.predict(X_test)
    xgb_preds = xgb_risk.predict(X_test)
    agreement_cm = confusion_matrix(rf_preds, xgb_preds).tolist()
    agreement_pct = round(float(np.mean(rf_preds == xgb_preds) * 100), 2)
    
    _DETAILED_EVAL_CACHE = {
        "binary_roc": {
            "rf": {"fpr": rf_fpr.tolist(), "tpr": rf_tpr.tolist(), "auc": artifacts['metrics']['random_forest']['binary_auc']},
            "xgb": {"fpr": xgb_fpr.tolist(), "tpr": xgb_tpr.tolist(), "auc": artifacts['metrics']['xgboost']['binary_auc']}
        },
        "binary_pr": {
            "rf": {"precision": rf_prec_curve.tolist(), "recall": rf_rec_curve.tolist(), "ap": round(float(rf_ap), 4)},
            "xgb": {"precision": xgb_prec_curve.tolist(), "recall": xgb_rec_curve.tolist(), "ap": round(float(xgb_ap), 4)}
        },
        "ovr_roc": ovr_roc,
        "agreement": {
            "matrix": agreement_cm,
            "percentage": agreement_pct
        },
        "test_counts": {
            "total": len(y_test_risk),
            "low_risk": int(np.sum(y_test_risk == 0)),
            "medium_risk": int(np.sum(y_test_risk == 1)),
            "high_risk": int(np.sum(y_test_risk == 2))
        },
        "rf_bin_probs": rf_bin_probs.tolist(),
        "xgb_bin_probs": xgb_bin_probs.tolist()
    }
    return _DETAILED_EVAL_CACHE

def get_model_roc_data():
    """
    Computes ROC curve coordinates for Random Forest and XGBoost for plotting.
    """
    data = get_detailed_evaluation_data()
    return data['binary_roc']

def generate_random_test_batch(n=30):
    """
    Generates a realistic batch of unseen loan applications for testing.
    Prominently features Akwa Ibom, Lagos, Rivers, FCT, etc.
    """
    states = ['Akwa Ibom', 'Lagos', 'FCT', 'Rivers', 'Oyo', 'Ogun', 'Kaduna', 'Kano', 'Enugu']
    state_weights = [0.30, 0.20, 0.14, 0.11, 0.09, 0.06, 0.04, 0.03, 0.03]
    products = ['Salary Advance', 'Personal Loan', 'SME Loan', 'Buy Now Pay Later']
    bands = ['A', 'B', 'C', 'D', 'E']
    channels = ['Mobile App', 'USSD', 'Web', 'Agent']
    emp_types = ['Salaried', 'Self-Employed', 'Informal Trader', 'Student']
    
    names_pool = [
        "Aniefiok Bassey", "Udeme Okon", "Ime Archibong", "Mfonobong Ekpo", "Ekaette Akpan",
        "Chukwudi Okafor", "Ibrahim Danjuma", "Ngozi Eze", "Emeka Okonkwo", "Folake Adebayo",
        "Babatunde Fashina", "Fatima Abdullahi", "Chiamaka Nnamdi", "Yakubu Garba", "Segun Alabi",
        "Zainab Bello", "Kelechi Iheanacho", "Tunde Bakare", "Aisha Mohammed", "Victor Osimhen",
        "Edidiong Umoh", "Iniobong Peters", "Itoro Effiong", "Nsikan Akpabio", "Grace Udo"
    ]
    
    rows = []
    for i in range(n):
        name = np.random.choice(names_pool)
        state = np.random.choice(states, p=state_weights)
        prod = np.random.choice(products)
        band = np.random.choice(bands, p=[0.25, 0.25, 0.25, 0.15, 0.10])
        emp = np.random.choice(emp_types)
        chan = np.random.choice(channels)
        age = int(np.random.randint(21, 58))
        
        # State multiplier
        state_multiplier = 1.30 if state in ['Akwa Ibom', 'Rivers'] else (1.35 if state in ['Lagos', 'FCT'] else 1.0)
        base_inc = float(np.random.choice([150000, 220000, 350000, 480000, 650000, 85000, 110000])) * state_multiplier
        income = int(np.clip(base_inc, 30000, 2000000))
        
        if band in ['A', 'B']:
            dti = round(float(np.random.uniform(10.0, 30.0)), 1)
        elif band == 'C':
            dti = round(float(np.random.uniform(22.0, 45.0)), 1)
        else:
            dti = round(float(np.random.uniform(42.0, 75.0)), 1)
            
        if prod == 'Salary Advance':
            amt = int(np.clip(income * np.random.uniform(0.2, 0.7), 20000, 400000))
            tenor = int(np.random.choice([15, 30, 45]))
            rate = round(float(np.random.uniform(6.0, 12.0)), 1)
        elif prod == 'Buy Now Pay Later':
            amt = int(np.clip(income * np.random.uniform(0.15, 0.6), 15000, 300000))
            tenor = int(np.random.choice([14, 30, 60]))
            rate = round(float(np.random.uniform(8.0, 15.0)), 1)
        elif prod == 'Personal Loan':
            amt = int(np.clip(income * np.random.uniform(0.5, 2.0), 50000, 1200000))
            tenor = int(np.random.choice([60, 90, 180, 270]))
            rate = round(float(np.random.uniform(14.0, 24.0)), 1)
        else: # SME Loan
            amt = int(np.clip(income * np.random.uniform(0.8, 3.0), 100000, 2500000))
            tenor = int(np.random.choice([90, 180, 270, 365]))
            rate = round(float(np.random.uniform(16.0, 26.0)), 1)
            
        rows.append({
            'applicant_id': f"NEW-{i+1:03d}",
            'applicant_name': name,
            'loan_product': prod,
            'loan_amount': amt,
            'loan_tenor_days': tenor,
            'interest_rate_pct': rate,
            'applicant_age': age,
            'employment_type': emp,
            'monthly_income': income,
            'debt_to_income_ratio': dti,
            'credit_score_band': band,
            'state': state,
            'channel': chan
        })
    return pd.DataFrame(rows)

