import pandas as pd
import numpy as np
import os

def create_integrated_dataset():
    base_dir = r"c:\Users\USER\Desktop\Goody_Project"
    data_dir = os.path.join(base_dir, "nigeria-digital-lending-credit-risk")
    
    app_path = os.path.join(data_dir, "Loan_Applications.csv")
    perf_path = os.path.join(data_dir, "Loan_Performance.csv")
    
    print(f"Loading Loan Applications from {app_path}...")
    app_df = pd.read_csv(app_path)
    print(f"Loaded {len(app_df)} applications.")
    
    print(f"Loading Loan Performance from {perf_path}...")
    perf_df = pd.read_csv(perf_path)
    print(f"Loaded {len(perf_df)} performance records.")
    
    # Clean string columns
    for col in app_df.select_dtypes(include='object').columns:
        app_df[col] = app_df[col].astype(str).str.strip()
        
    for col in perf_df.select_dtypes(include='object').columns:
        perf_df[col] = perf_df[col].astype(str).str.strip()
        
    # Standardize missing decline_reason
    app_df['decline_reason'] = app_df['decline_reason'].replace({'nan': 'None', '': 'None'}).fillna('None')
    
    # Merge datasets on loan_id
    # We do a left merge from applications to preserve all 500 applicants
    merged = pd.merge(app_df, perf_df, on='loan_id', how='left', suffixes=('', '_perf'))
    
    # If loan_amount_perf exists, drop redundant column
    if 'loan_amount_perf' in merged.columns:
        merged.drop(columns=['loan_amount_perf'], inplace=True)
        
    # Impute missing values for performance columns for non-disbursed loans
    merged['repayment_status'] = merged['repayment_status'].fillna('Not Disbursed')
    merged['repayment_status'] = merged['repayment_status'].replace({'nan': 'Not Disbursed'})
    
    merged['days_past_due'] = merged['days_past_due'].fillna(0).astype(int)
    merged['number_of_late_payments'] = merged['number_of_late_payments'].fillna(0).astype(int)
    merged['early_repayment'] = merged['early_repayment'].fillna('N').replace({'nan': 'N'})
    
    merged['disbursement_date'] = merged['disbursement_date'].fillna('Not Applicable').replace({'nan': 'Not Applicable'})
    merged['due_date'] = merged['due_date'].fillna('Not Applicable').replace({'nan': 'Not Applicable'})
    
    # Calculate financial and interest metrics
    # Total repayment due if missing
    expected_interest = merged['loan_amount'] * (merged['interest_rate_pct'] / 100.0)
    expected_total_due = merged['loan_amount'] + expected_interest
    
    merged['total_repayment_due'] = merged['total_repayment_due'].fillna(expected_total_due)
    merged['amount_repaid'] = merged['amount_repaid'].fillna(0.0)
    
    # --- Feature Engineering ---
    # 1. Credit Score Numerical Mapping (Credit Bureau)
    # A: Excellent (800), B: Good (700), C: Fair (600), D: Poor (500), E: Very Poor (400)
    credit_score_map = {'A': 800, 'B': 700, 'C': 600, 'D': 500, 'E': 400}
    bureau_category_map = {
        'A': 'Prime / Very Low Risk',
        'B': 'Near Prime / Low Risk',
        'C': 'Standard / Moderate Risk',
        'D': 'Sub-Prime / High Risk',
        'E': 'Deep Sub-Prime / Very High Risk'
    }
    merged['credit_score_num'] = merged['credit_score_band'].map(credit_score_map).fillna(550).astype(int)
    merged['bureau_risk_category'] = merged['credit_score_band'].map(bureau_category_map).fillna('Standard')
    
    # 2. Financial ratios
    merged['loan_to_income_ratio'] = np.round(merged['loan_amount'] / merged['monthly_income'], 4)
    merged['monthly_debt_repayment'] = np.round(merged['monthly_income'] * (merged['debt_to_income_ratio'] / 100.0), 2)
    merged['total_interest_amount'] = np.round(expected_interest, 2)
    
    # Approximate monthly installment (tenor normalized to 30-day months)
    months = np.maximum(merged['loan_tenor_days'] / 30.0, 0.5)
    merged['estimated_monthly_installment'] = np.round(expected_total_due / months, 2)
    
    # Installment to income ratio
    merged['installment_to_income_ratio'] = np.round((merged['estimated_monthly_installment'] / merged['monthly_income']) * 100.0, 2)
    
    # Total debt burden (Existing DTI + New Installment DTI)
    merged['total_debt_service_ratio'] = np.round(merged['debt_to_income_ratio'] + merged['installment_to_income_ratio'], 2)
    
    # Disposable income after living debt obligations and loan installment
    merged['disposable_income_after_loan'] = np.round(
        merged['monthly_income'] - merged['monthly_debt_repayment'] - merged['estimated_monthly_installment'], 2
    )
    
    # 3. Categorical demographic binnings
    merged['age_group'] = pd.cut(
        merged['applicant_age'],
        bins=[18, 25, 35, 45, 55, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56+'],
        right=True
    ).astype(str)
    
    merged['income_segment'] = pd.cut(
        merged['monthly_income'],
        bins=[-1, 100000, 250000, 500000, 10000000],
        labels=['Low (<₦100k)', 'Lower-Middle (₦100k-₦250k)', 'Middle (₦250k-₦500k)', 'High (>₦500k)'],
        right=True
    ).astype(str)
    
    merged['tenor_category'] = pd.cut(
        merged['loan_tenor_days'],
        bins=[-1, 30, 90, 180, 1000],
        labels=['Ultra-Short (<=30d)', 'Short (31-90d)', 'Medium (91-180d)', 'Long (>180d)'],
        right=True
    ).astype(str)
    
    # --- Ground Truth Target Classification & Decision Support Assignment ---
    # Based on the Thesis Methodology, CBN guidelines, and Credit Bureau assessment:
    def classify_credit_record(row):
        app_status = row['approval_status']
        rep_status = row['repayment_status']
        dpd = row['days_past_due']
        late = row['number_of_late_payments']
        score = row['credit_score_band']
        dti = row['debt_to_income_ratio']
        reason = row['decline_reason']
        
        # 1. Historical Disbursed Loans
        if app_status == 'Approved':
            if rep_status in ['Written Off', '90+ DPD'] or dpd >= 90:
                # Stage 3 Default / Severe NPL
                return 'High Risk', 'Reject', 1, 'Stage 3 (Default/NPL)'
            elif rep_status == '60 DPD' or dpd >= 60 or late >= 2:
                # Stage 2 High Stress
                return 'High Risk', 'Reject', 1, 'Stage 2 (Severe Stress)'
            elif rep_status == '30 DPD' or dpd >= 30 or late == 1:
                # Stage 2 Early Delinquency / Watchlist
                return 'Medium Risk', 'Review', 0, 'Stage 2 (Early Delinquency)'
            elif rep_status in ['Fully Repaid', 'On Track']:
                # Stage 1 Performing
                return 'Low Risk', 'Accept', 0, 'Stage 1 (Performing)'
            else:
                return 'Medium Risk', 'Review', 0, 'Stage 2 (Watchlist)'
                
        # 2. Historical Declined Applications
        elif app_status == 'Declined':
            # Check for Query 8 "Missed Opportunities" (Incomplete Docs with high creditworthiness)
            if reason == 'Incomplete Docs' and score in ['A', 'B'] and dti < 40:
                return 'Medium Risk', 'Review', 0, 'Stage 2 (Documentation Review)'
            else:
                # High DTI, Low Credit Score, Insufficient Income
                return 'High Risk', 'Reject', 1, 'Stage 3 (Underwriting Decline)'
                
        # 3. Pending Underwriting Applications
        else:
            if score in ['A', 'B'] and dti < 35:
                return 'Low Risk', 'Accept', 0, 'Stage 1 (Eligible/Low Risk)'
            elif score in ['D', 'E'] or dti >= 50:
                return 'High Risk', 'Reject', 1, 'Stage 3 (Ineligible/High Risk)'
            else:
                return 'Medium Risk', 'Review', 0, 'Stage 2 (Manual Underwriting)'

    classifications = merged.apply(classify_credit_record, axis=1)
    merged['risk_level'] = [c[0] for c in classifications]
    merged['loan_decision'] = [c[1] for c in classifications]
    merged['is_default'] = [c[2] for c in classifications]
    merged['ifrs9_stage'] = [c[3] for c in classifications]
    merged['is_approved'] = (merged['approval_status'] == 'Approved').astype(int)
    
    # Output file paths
    out_csv1 = os.path.join(base_dir, "nigeria_credit_risk_dataset.csv")
    out_csv2 = os.path.join(data_dir, "nigeria_credit_risk_dataset.csv")
    
    merged.to_csv(out_csv1, index=False)
    merged.to_csv(out_csv2, index=False)
    
    print("\nIntegrated dataset created successfully!")
    print(f"Total Rows: {len(merged)}")
    print(f"Total Columns: {len(merged.columns)}")
    print(f"Saved to:\n  - {out_csv1}\n  - {out_csv2}")
    
    print("\n--- Summary of Key Target Labels ---")
    print("\nRisk Level Distribution:")
    print(merged['risk_level'].value_counts())
    print("\nLoan Decision Distribution:")
    print(merged['loan_decision'].value_counts())
    print("\nDefault Flag Distribution (0: Performing/Creditworthy, 1: Default/High Risk):")
    print(merged['is_default'].value_counts())
    print("\nIFRS 9 Staging Distribution:")
    print(merged['ifrs9_stage'].value_counts())
    
    print("\nColumns in integrated dataset:")
    print(merged.columns.tolist())

if __name__ == "__main__":
    create_integrated_dataset()

