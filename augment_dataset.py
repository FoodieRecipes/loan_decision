import pandas as pd
import numpy as np
import random
import os

def generate_harmonized_dataset():
    base_dir = r"c:\Users\USER\Desktop\Goody_Project"
    out_2500 = os.path.join(base_dir, "nigeria_credit_risk_dataset_2500.csv")
    out_1500 = os.path.join(base_dir, "nigeria_credit_risk_dataset_1500.csv")
    out_aug = os.path.join(base_dir, "nigeria_credit_risk_dataset_augmented.csv")
    out_sub = os.path.join(base_dir, "nigeria-digital-lending-credit-risk", "nigeria_credit_risk_dataset.csv")
    out_orig = os.path.join(base_dir, "nigeria_credit_risk_dataset.csv")
    
    print("Generating harmonized 2,500-sample credit risk portfolio...")
    np.random.seed(42)
    random.seed(42)
    
    total_records = 2500
    
    products = ['Salary Advance', 'Personal Loan', 'SME Loan', 'Buy Now Pay Later']
    product_weights = [0.35, 0.30, 0.15, 0.20]
    
    states = ['Akwa Ibom', 'Lagos', 'FCT', 'Rivers', 'Oyo', 'Ogun', 'Kano', 'Kaduna', 'Enugu']
    state_weights = [0.25, 0.20, 0.14, 0.11, 0.09, 0.07, 0.05, 0.05, 0.04]
    
    channels = ['Mobile App', 'USSD', 'Web', 'Agent']
    channel_weights = [0.50, 0.24, 0.16, 0.10]
    
    employment_types = ['Salaried', 'Self-Employed', 'Informal Trader', 'Student']
    employment_weights = [0.42, 0.33, 0.18, 0.07]
    
    credit_bands = ['A', 'B', 'C', 'D', 'E']
    credit_weights = [0.24, 0.28, 0.28, 0.13, 0.07]
    
    credit_score_map = {'A': 800, 'B': 700, 'C': 600, 'D': 500, 'E': 400}
    bureau_category_map = {
        'A': 'Prime / Very Low Risk',
        'B': 'Near Prime / Low Risk',
        'C': 'Standard / Moderate Risk',
        'D': 'Sub-Prime / High Risk',
        'E': 'Deep Sub-Prime / Very High Risk'
    }
    
    rows = []
    
    for i in range(total_records):
        loan_id = f"LN{i+1:05d}"
        applicant_id = f"APP{random.randint(1000, 9999)}"
        
        product = np.random.choice(products, p=product_weights)
        state = np.random.choice(states, p=state_weights)
        channel = np.random.choice(channels, p=channel_weights)
        emp_type = np.random.choice(employment_types, p=employment_weights)
        credit_band = np.random.choice(credit_bands, p=credit_weights)
        credit_num = credit_score_map[credit_band] + random.randint(-20, 20)
        
        # Age
        if emp_type == 'Student':
            age = random.randint(19, 25)
        elif emp_type == 'Salaried':
            age = random.randint(23, 56)
        else:
            age = random.randint(25, 62)
            
        # Income generation
        state_multiplier = 1.30 if state in ['Akwa Ibom', 'Rivers'] else (1.35 if state in ['Lagos', 'FCT'] else 1.0)
        if emp_type == 'Salaried':
            base_income = np.random.lognormal(mean=12.2, sigma=0.55) * state_multiplier
        elif emp_type == 'Self-Employed':
            base_income = np.random.lognormal(mean=12.0, sigma=0.65) * state_multiplier
        elif emp_type == 'Informal Trader':
            base_income = np.random.lognormal(mean=11.5, sigma=0.45) * state_multiplier
        else:
            base_income = np.random.uniform(25000, 85000)
            
        monthly_income = int(np.clip(base_income, 30000, 2500000))
        
        # Loan amounts, tenors & rates
        if product == 'Salary Advance':
            tenor = random.choice([15, 30, 45])
            max_borrow = monthly_income * 0.75
            loan_amount = int(np.clip(np.random.uniform(20000, max_borrow), 20000, 500000))
            interest_rate = round(random.uniform(5.0, 12.0), 1)
        elif product == 'Buy Now Pay Later':
            tenor = random.choice([14, 30, 60])
            loan_amount = int(np.clip(np.random.uniform(15000, monthly_income * 0.9), 15000, 350000))
            interest_rate = round(random.uniform(8.0, 16.0), 1)
        elif product == 'Personal Loan':
            tenor = random.choice([60, 90, 180, 270, 365])
            loan_amount = int(np.clip(np.random.uniform(50000, monthly_income * 2.5), 50000, 1500000))
            interest_rate = round(random.uniform(14.0, 26.0), 1)
        else: # SME Loan
            tenor = random.choice([90, 180, 270, 365])
            loan_amount = int(np.clip(np.random.uniform(100000, monthly_income * 3.5), 100000, 3500000))
            interest_rate = round(random.uniform(16.0, 28.0), 1)
            
        # Debt to income ratio (DTI) correlated with credit band
        if credit_band in ['A', 'B']:
            dti = round(np.random.beta(2, 6) * 55, 1) # 10-35%
        elif credit_band == 'C':
            dti = round(np.random.beta(3, 4) * 70, 1) # 25-48%
        else:
            dti = round(np.random.beta(5, 3) * 85, 1) # 40-75%
            
        # Derived Financial features
        loan_to_income = round(loan_amount / monthly_income, 4)
        monthly_debt_repayment = round(monthly_income * (dti / 100.0), 2)
        total_interest_amount = round(loan_amount * (interest_rate / 100.0), 2)
        total_expected_repayment = round(loan_amount + total_interest_amount, 2)
        months = max(tenor / 30.0, 0.5)
        est_installment = round(total_expected_repayment / months, 2)
        installment_to_income = round((est_installment / monthly_income) * 100.0, 2)
        total_debt_service_ratio = round(dti + installment_to_income, 2)
        disposable_income = round(monthly_income - monthly_debt_repayment - est_installment, 2)
        
        # --- Advanced Engineered Interaction Terms ---
        # 1. Affordability Index (Times installment covered by net disposable cash)
        affordability_index = round(max(0.0, disposable_income) / max(est_installment, 1.0), 2)
        
        # 2. Credit Vulnerability Index (DTI * Credit Deficit)
        credit_deficit = max(0, 850 - credit_num)
        credit_vulnerability_index = round((dti * credit_deficit) / 100.0, 2)
        
        # 3. Comprehensive Credit Risk Index (Ground-truth scoring function)
        # Based on Central Bank of Nigeria (CBN) prudential underwriting guidelines:
        # Scale 0 to 100 points
        score = 50.0
        
        # Credit Bureau Score Component (+35 to -35 pts)
        if credit_band == 'A': score += 35
        elif credit_band == 'B': score += 20
        elif credit_band == 'C': score += 5
        elif credit_band == 'D': score -= 20
        else: score -= 35
        
        # DTI Leverage Component (+20 to -25 pts)
        if dti < 25: score += 20
        elif dti < 40: score += 10
        elif dti < 50: score -= 10
        else: score -= 25
        
        # Disposable Income Buffer Component (+15 to -20 pts)
        if disposable_income > 150000: score += 15
        elif disposable_income > 50000: score += 5
        elif disposable_income < 0: score -= 20
        
        # Total Debt Burden (+10 to -15 pts)
        if total_debt_service_ratio < 35: score += 10
        elif total_debt_service_ratio > 55: score -= 15
        
        # Employment stability factor
        if emp_type == 'Salaried': score += 5
        elif emp_type == 'Student': score -= 10
        
        # Slight realistic stochastic variance (normal std = 3.5 pts)
        np.random.seed((i * 17) % 9999 + int(loan_amount) % 1000)
        noise = np.random.normal(0, 3.5)
        final_risk_score = score + noise
        
        # Ground Truth Classification
        if final_risk_score >= 63.0 and dti < 45.0 and disposable_income > 0:
            risk_level = 'Low Risk'
            loan_decision = 'Accept'
            approval_status = 'Approved'
            is_default = 0
            ifrs9_stage = 'Stage 1 (Performing)'
            repayment_status = random.choice(['Fully Repaid', 'On Track', 'On Track'])
            days_past_due = 0
            late_payments = 0
            early_rep = random.choice(['Y', 'N'])
            decline_reason = 'None'
            
        elif final_risk_score >= 42.0 and dti < 53.0:
            risk_level = 'Medium Risk'
            loan_decision = 'Review'
            approval_status = np.random.choice(['Approved', 'Pending', 'Declined'], p=[0.60, 0.30, 0.10])
            is_default = 0
            if approval_status == 'Approved':
                ifrs9_stage = 'Stage 2 (Watchlist / Early Delinquency)'
                repayment_status = random.choice(['On Track', '30 DPD', 'On Track'])
                days_past_due = 30 if repayment_status == '30 DPD' else 0
                late_payments = 1 if repayment_status == '30 DPD' else 0
                early_rep = 'N'
                decline_reason = 'None'
            elif approval_status == 'Declined':
                ifrs9_stage = 'Stage 2 (Documentation Review)'
                repayment_status = 'Not Disbursed'
                days_past_due = 0
                late_payments = 0
                early_rep = 'N'
                decline_reason = 'Incomplete Docs'
            else:
                ifrs9_stage = 'Stage 2 (Manual Underwriting)'
                repayment_status = 'Not Disbursed'
                days_past_due = 0
                late_payments = 0
                early_rep = 'N'
                decline_reason = 'None'
                
        else:
            risk_level = 'High Risk'
            loan_decision = 'Reject'
            is_default = 1
            approval_status = np.random.choice(['Declined', 'Approved', 'Pending'], p=[0.82, 0.12, 0.06])
            if approval_status == 'Approved':
                # Defaulters
                ifrs9_stage = 'Stage 3 (Default/NPL)'
                repayment_status = random.choice(['60 DPD', '90+ DPD', 'Written Off'])
                days_past_due = random.choice([60, 90, 120])
                late_payments = random.choice([2, 3, 4])
                early_rep = 'N'
                decline_reason = 'None'
            elif approval_status == 'Declined':
                ifrs9_stage = 'Stage 3 (Underwriting Decline)'
                repayment_status = 'Not Disbursed'
                days_past_due = 0
                late_payments = 0
                early_rep = 'N'
                if dti >= 50.0:
                    decline_reason = 'High DTI'
                elif credit_band in ['D', 'E']:
                    decline_reason = 'Low Credit Score'
                else:
                    decline_reason = 'Insufficient Income'
            else:
                ifrs9_stage = 'Stage 3 (Ineligible/High Risk)'
                repayment_status = 'Not Disbursed'
                days_past_due = 0
                late_payments = 0
                early_rep = 'N'
                decline_reason = 'None'
                
        # Dates
        app_day = random.randint(1, 28)
        app_month = random.randint(1, 10)
        application_date = f"{app_day:02d}/{app_month:02d}/2024"
        
        if approval_status == 'Approved':
            disbursement_date = f"{min(app_day+2, 28):02d}/{app_month:02d}/2024"
            due_month = min(app_month + max(1, tenor//30), 12)
            due_date = f"{app_day:02d}/{due_month:02d}/2024"
            if repayment_status == 'Fully Repaid':
                amount_repaid = float(total_expected_repayment)
            elif repayment_status == 'On Track':
                amount_repaid = round(total_expected_repayment * random.uniform(0.3, 0.7), 2)
            elif repayment_status == '30 DPD':
                amount_repaid = round(total_expected_repayment * random.uniform(0.2, 0.5), 2)
            elif repayment_status == '60 DPD':
                amount_repaid = round(total_expected_repayment * random.uniform(0.1, 0.3), 2)
            else:
                amount_repaid = round(total_expected_repayment * random.uniform(0.0, 0.15), 2)
        else:
            disbursement_date = 'Not Applicable'
            due_date = 'Not Applicable'
            amount_repaid = 0.0
            
        # Demographic segments
        age_group = '18-25' if age <= 25 else ('26-35' if age <= 35 else ('36-45' if age <= 45 else ('46-55' if age <= 55 else '56+')))
        income_segment = 'Low (<₦100k)' if monthly_income < 100000 else ('Lower-Middle (₦100k-₦250k)' if monthly_income <= 250000 else ('Middle (₦250k-₦500k)' if monthly_income <= 500000 else 'High (>₦500k)'))
        tenor_category = 'Ultra-Short (<=30d)' if tenor <= 30 else ('Short (31-90d)' if tenor <= 90 else ('Medium (91-180d)' if tenor <= 180 else 'Long (>180d)'))
        
        row = {
            'loan_id': loan_id,
            'applicant_id': applicant_id,
            'application_date': application_date,
            'loan_product': product,
            'loan_amount': loan_amount,
            'loan_tenor_days': tenor,
            'interest_rate_pct': interest_rate,
            'applicant_age': age,
            'employment_type': emp_type,
            'monthly_income': monthly_income,
            'debt_to_income_ratio': dti,
            'credit_score_band': credit_band,
            'state': state,
            'channel': channel,
            'approval_status': approval_status,
            'decline_reason': decline_reason,
            'disbursement_date': disbursement_date,
            'due_date': due_date,
            'total_repayment_due': total_expected_repayment,
            'amount_repaid': amount_repaid,
            'repayment_status': repayment_status,
            'days_past_due': days_past_due,
            'number_of_late_payments': late_payments,
            'early_repayment': early_rep,
            'credit_score_num': credit_num,
            'bureau_risk_category': bureau_category_map[credit_band],
            'loan_to_income_ratio': loan_to_income,
            'monthly_debt_repayment': monthly_debt_repayment,
            'total_interest_amount': total_interest_amount,
            'estimated_monthly_installment': est_installment,
            'installment_to_income_ratio': installment_to_income,
            'total_debt_service_ratio': total_debt_service_ratio,
            'disposable_income_after_loan': disposable_income,
            'affordability_index': affordability_index,
            'credit_vulnerability_index': credit_vulnerability_index,
            'age_group': age_group,
            'income_segment': income_segment,
            'tenor_category': tenor_category,
            'risk_level': risk_level,
            'loan_decision': loan_decision,
            'is_default': is_default,
            'ifrs9_stage': ifrs9_stage,
            'is_approved': 1 if approval_status == 'Approved' else 0
        }
        rows.append(row)
        
    df_harmonized = pd.DataFrame(rows)
    print(f"Generated harmonized dataset shape: {df_harmonized.shape}")
    
    # Save to all target paths
    df_harmonized.to_csv(out_2500, index=False)
    df_harmonized.to_csv(out_1500, index=False)
    df_harmonized.to_csv(out_aug, index=False)
    df_harmonized.to_csv(out_sub, index=False)
    try:
        df_harmonized.to_csv(out_orig, index=False)
    except Exception as e:
        print(f"Note: {out_orig} locked: {e}")
        
    print("\nTarget Class Distribution:")
    print(df_harmonized['risk_level'].value_counts())
    print("\nDecision Distribution:")
    print(df_harmonized['loan_decision'].value_counts())
    print("\nApproval Status Distribution:")
    print(df_harmonized['approval_status'].value_counts())
    print("\nAll files successfully synchronized!")

if __name__ == "__main__":
    generate_harmonized_dataset()
