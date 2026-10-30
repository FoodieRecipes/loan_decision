import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(r"c:\Users\USER\Desktop\Goody_Project", "loan_decisions.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loan_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        applicant_id TEXT,
        applicant_name TEXT,
        state TEXT,
        loan_product TEXT,
        loan_amount REAL,
        loan_tenor_days INTEGER,
        monthly_income REAL,
        debt_to_income_ratio REAL,
        credit_score_band TEXT,
        channel TEXT,
        predicted_risk_level TEXT,
        predicted_approval_status TEXT,
        default_probability REAL,
        system_recommendation TEXT,
        officer_decision TEXT,
        officer_notes TEXT,
        model_used TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_decision(data):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO loan_decisions (
        timestamp, applicant_id, applicant_name, state, loan_product,
        loan_amount, loan_tenor_days, monthly_income, debt_to_income_ratio,
        credit_score_band, channel, predicted_risk_level, predicted_approval_status,
        default_probability, system_recommendation, officer_decision,
        officer_notes, model_used
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data.get("applicant_id", f"APP-{int(datetime.now().timestamp())}"),
        data.get("applicant_name", "Anonymous Applicant"),
        data.get("state", "Lagos"),
        data.get("loan_product", "Personal Loan"),
        float(data.get("loan_amount", 0.0)),
        int(data.get("loan_tenor_days", 30)),
        float(data.get("monthly_income", 0.0)),
        float(data.get("debt_to_income_ratio", 0.0)),
        data.get("credit_score_band", "C"),
        data.get("channel", "Mobile App"),
        data.get("predicted_risk_level", "Medium Risk"),
        data.get("predicted_approval_status", "Review"),
        float(data.get("default_probability", 0.0)),
        data.get("system_recommendation", "Review Application"),
        data.get("officer_decision", "Pending"),
        data.get("officer_notes", ""),
        data.get("model_used", "Best Ensemble Model")
    ))
    conn.commit()
    conn.close()

def save_batch_decisions(records):
    """
    Saves a batch of classified loan records into the audit database.
    """
    if not records:
        return 0
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    rows_to_insert = []
    for r in records:
        rows_to_insert.append((
            now_str,
            str(r.get("applicant_id", "APP-BATCH")),
            str(r.get("applicant_name", "Batch Applicant")),
            str(r.get("state", "Akwa Ibom")),
            str(r.get("loan_product", "Personal Loan")),
            float(r.get("loan_amount", 0.0)),
            int(r.get("loan_tenor_days", 30)),
            float(r.get("monthly_income", 0.0)),
            float(r.get("debt_to_income_ratio", 0.0)),
            str(r.get("credit_score_band", "C")),
            str(r.get("channel", "Mobile App")),
            str(r.get("predicted_risk_level", "Medium Risk")),
            str(r.get("approval_status", "Pending")),
            float(r.get("default_probability", float(r.get("default_probability_pct", 0.0)) / 100.0)),
            str(r.get("system_recommendation", "Review Application")),
            str(r.get("system_decision", "Review")),
            f"Batch assessment via {r.get('model_used', 'XGBoost')}",
            str(r.get("model_used", "XGBoost"))
        ))
        
    cursor.executemany("""
    INSERT INTO loan_decisions (
        timestamp, applicant_id, applicant_name, state, loan_product,
        loan_amount, loan_tenor_days, monthly_income, debt_to_income_ratio,
        credit_score_band, channel, predicted_risk_level, predicted_approval_status,
        default_probability, system_recommendation, officer_decision,
        officer_notes, model_used
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows_to_insert)
    conn.commit()
    inserted_count = cursor.rowcount
    conn.close()
    return inserted_count

def get_decisions():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM loan_decisions ORDER BY id DESC", conn)
    conn.close()
    return df

def seed_sample_decisions():
    # If empty, add a couple initial logged decisions for demonstration
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM loan_decisions")
    count = cursor.fetchone()[0]
    if count == 0:
        samples = [
            ("2026-10-25 09:14:22", "APP2824", "Chukwudi Okafor", "Lagos", "Salary Advance", 150000.0, 30, 320000.0, 18.5, "A", "Mobile App", "Low Risk", "Approved", 0.08, "Approve Loan", "Approved", "Verified payroll direct debit; clean bureau report.", "Random Forest"),
            ("2026-10-26 11:32:05", "APP1434", "Amina Babangida", "Kano", "SME Loan", 750000.0, 90, 450000.0, 42.0, "C", "Web", "Medium Risk", "Review", 0.44, "Review Application", "Approved with Condition", "Requested 3-month bank statement; approved with 10% guarantor deposit.", "XGBoost"),
            ("2026-10-27 14:45:10", "APP3912", "Olumide Adeleke", "Oyo", "Personal Loan", 500000.0, 60, 110000.0, 62.5, "E", "USSD", "High Risk", "Declined", 0.89, "Reject Loan", "Rejected", "Debt service ratio exceeds 60%; multiple recent inquiries.", "XGBoost"),
            ("2026-10-28 10:15:30", "APP1092", "Aniefiok Bassey", "Akwa Ibom", "SME Loan", 450000.0, 90, 380000.0, 22.0, "B", "Mobile App", "Low Risk", "Approved", 0.12, "Approve Loan", "Approved", "Solid business turnover in Uyo; prompt historical repayments.", "XGBoost")
        ]
        cursor.executemany("""
        INSERT INTO loan_decisions (
            timestamp, applicant_id, applicant_name, state, loan_product,
            loan_amount, loan_tenor_days, monthly_income, debt_to_income_ratio,
            credit_score_band, channel, predicted_risk_level, predicted_approval_status,
            default_probability, system_recommendation, officer_decision,
            officer_notes, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, samples)
        conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    seed_sample_decisions()
    df = get_decisions()
    print("Database initialized. Stored records:", len(df))

