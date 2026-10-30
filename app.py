import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import joblib

# Import custom project modules
import model_engine

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Credit Risk Assessment & Loan Underwriting",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern FinTech Interface (Teal & Slate Theme)
st.markdown("""
<style>
    /* Global Canvas */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Typography & Headers */
    .main-header {
        font-size: 1.9rem;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 0.15rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.25rem;
    }
    
    /* Status Badges */
    .risk-badge-low {
        background-color: #ecfdf5;
        color: #065f46;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 20px;
        border: 1px solid #a7f3d0;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: #fffbeb;
        color: #92400e;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 20px;
        border: 1px solid #fde68a;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #fef2f2;
        color: #991b1b;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 20px;
        border: 1px solid #fecaca;
        display: inline-block;
    }
    
    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
        color: #475569;
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ecfdf5 !important;
        color: #0d9488 !important;
        border: 1px solid #a7f3d0 !important;
        border-bottom: 3px solid #0d9488 !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    [data-testid="stMetricValue"] {
        color: #0d9488 !important;
        font-weight: 750;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0d9488 0%, #059669 100%);
        color: #ffffff !important;
        font-weight: 650;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.4rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0f766e 0%, #047857 100%);
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.3);
        transform: translateY(-1px);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_dataset():
    base_dir = r"c:\Users\USER\Desktop\Goody_Project"
    candidates = [
        os.path.join(base_dir, "nigeria_credit_risk_dataset_2500.csv"),
        os.path.join(base_dir, "nigeria_credit_risk_dataset_1500.csv"),
        os.path.join(base_dir, "nigeria_credit_risk_dataset_augmented.csv"),
        os.path.join(base_dir, "nigeria_credit_risk_dataset.csv")
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if len(df) >= 2000:
                return df
    return pd.read_csv(candidates[0])

@st.cache_resource
def get_model_artifacts():
    return model_engine.load_artifacts()

df_dataset = load_dataset()
artifacts = get_model_artifacts()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=52)
    st.title("Credit Risk System")
    st.caption("Predictive Underwriting Engine")
    st.divider()
    
    menu = st.radio(
        "Navigation:",
        [
            "🎯 Credit Risk Assessment",
            "📊 Model Evaluation & Comparison",
            "🗄️ Portfolio Data & Insights"
        ],
        index=0
    )
    
    st.divider()
    best_model_name = artifacts['metrics']['best_model']
    best_acc = artifacts['metrics']['xgboost']['accuracy'] * 100 if best_model_name == 'XGBoost' else artifacts['metrics']['random_forest']['accuracy'] * 100
    st.success(f"🏆 **Primary Model:** {best_model_name} ({best_acc:.1f}% Acc)")

# -------------------------------------------------------------
# 1. CREDIT RISK ASSESSMENT (SINGLE & BATCH TEST DATA)
# -------------------------------------------------------------
if menu == "🎯 Credit Risk Assessment":
    st.markdown("<div class='main-header'>Credit Risk Assessment & Classification</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Real-Time Individual Applicant Underwriting and Batch Test File Ingestion</div>", unsafe_allow_html=True)
    
    tab_single, tab_batch = st.tabs([
        "👤 Single Applicant Assessment",
        "📁 Batch Test Data Assessment (Upload CSV)"
    ])
    
    # ---------------------------------------------------------
    # TAB 1: SINGLE APPLICANT ASSESSMENT
    # ---------------------------------------------------------
    with tab_single:
        # Top Action Bar with Clear Action
        c_head_left, c_head_right = st.columns([3, 1])
        with c_head_left:
            st.markdown("#### 📝 Applicant Profile & Loan Appraisal")
            st.caption("Select standard options from the dropdowns or type custom values manually into each field.")
        with c_head_right:
            if st.button("🧹 Clear All Inputs", use_container_width=True):
                # Clear all stored appraisal keys
                for k in ['inp_name', 'inp_id', 'inp_age', 'inp_income', 'inp_loan', 'inp_dti', 'inp_tenor', 'inp_rate', 'app_evaluated']:
                    if k in st.session_state:
                        del st.session_state[k]
                st.session_state['app_evaluated'] = False
                st.rerun()

        with st.form("appraisal_form"):
            st.markdown("##### 1. Applicant & Demographic Profile")
            c_app1, c_app2, c_app3, c_app4 = st.columns(4)
            with c_app1:
                app_name = st.text_input("Full Name", value=st.session_state.get('inp_name', ''), placeholder="e.g. Bassey Edet")
            with c_app2:
                app_id = st.text_input("Application ID", value=st.session_state.get('inp_id', ''), placeholder="e.g. APP-001")
            with c_app3:
                applicant_age = st.number_input("Age (Years)", min_value=18, max_value=85, value=st.session_state.get('inp_age', 30), step=1)
            with c_app4:
                state_choices = [
                    "Akwa Ibom", "Lagos", "Rivers", "FCT (Abuja)", "Oyo", "Ogun",
                    "Kaduna", "Kano", "Enugu", "Delta", "Anambra", "Edo", "Cross River",
                    "✍️ Enter State Manually"
                ]
                selected_state = st.selectbox("State of Residence", state_choices, index=0)
                if selected_state == "✍️ Enter State Manually":
                    custom_state = st.text_input("Type State Name", placeholder="e.g. Imo, Benue, Abia...")
                    state = custom_state.strip() if custom_state.strip() else "Akwa Ibom"
                else:
                    state = "FCT" if "Abuja" in selected_state else selected_state

            st.markdown("##### 2. Employment, Income & Credit Bureau Standing")
            c_fin1, c_fin2, c_fin3, c_fin4 = st.columns(4)
            with c_fin1:
                emp_choices = [
                    "Employed (Salaried)", "Civil Servant", "Self-Employed / Business",
                    "Trader / Merchant", "Student", "Unemployed", "✍️ Enter Employment Manually"
                ]
                selected_emp = st.selectbox("Employment Type", emp_choices, index=0)
                if selected_emp == "✍️ Enter Employment Manually":
                    custom_emp = st.text_input("Type Role/Job", placeholder="e.g. Architect, Consultant...")
                    employment_type = custom_emp.strip() if custom_emp.strip() else "Employed"
                else:
                    if "Civil Servant" in selected_emp:
                        employment_type = "Civil Servant"
                    elif "Self-Employed" in selected_emp or "Trader" in selected_emp:
                        employment_type = "Self-Employed"
                    elif "Student" in selected_emp:
                        employment_type = "Student"
                    elif "Unemployed" in selected_emp:
                        employment_type = "Unemployed"
                    else:
                        employment_type = "Employed"

            with c_fin2:
                quick_income = st.selectbox(
                    "Monthly Income (Select or Type):",
                    ["✍️ Enter Exact Amount Manually", "₦50,000", "₦100,000", "₦180,000", "₦250,000", "₦400,000", "₦650,000", "₦1,000,000+"]
                )
                if quick_income != "✍️ Enter Exact Amount Manually":
                    pre_inc = int(quick_income.split()[0].replace("₦", "").replace(",", "").replace("+", ""))
                else:
                    pre_inc = st.session_state.get('inp_income', 0)
                monthly_income = st.number_input(
                    "Monthly Income (₦)",
                    min_value=0,
                    max_value=20000000,
                    value=pre_inc,
                    step=10000,
                    placeholder="Enter monthly salary/revenue"
                )

            with c_fin3:
                quick_dti = st.selectbox(
                    "Existing Debt Burden (DTI %):",
                    ["✍️ Enter Exact DTI Manually", "Low Debt (10% DTI)", "Moderate Debt (25% DTI)", "High Debt (45% DTI)", "Severe Debt (65% DTI)"]
                )
                if "10%" in quick_dti:
                    pre_dti = 10.0
                elif "25%" in quick_dti:
                    pre_dti = 25.0
                elif "45%" in quick_dti:
                    pre_dti = 45.0
                elif "65%" in quick_dti:
                    pre_dti = 65.0
                else:
                    pre_dti = st.session_state.get('inp_dti', 0.0)
                dti = st.number_input(
                    "Existing Debt-to-Income (DTI %)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(pre_dti),
                    step=1.0,
                    placeholder="Enter DTI percentage"
                )

            with c_fin4:
                credit_mode = st.radio("Credit Bureau Score Entry:", ["Select Bureau Band", "Type Exact Score (300-850)"], horizontal=True)
                if credit_mode == "Select Bureau Band":
                    score_bands = ["A (Prime: 800+)", "B (Near Prime: 700-799)", "C (Standard: 600-699)", "D (Subprime: 500-599)", "E (Deep Subprime: <500)"]
                    band_sel = st.selectbox("Credit Bureau Band", score_bands, index=1)
                    credit_score_band = band_sel[0]
                else:
                    exact_score = st.number_input("Exact Score (300 to 850)", min_value=300, max_value=850, value=650, step=10)
                    if exact_score >= 750:
                        credit_score_band = "A"
                    elif exact_score >= 670:
                        credit_score_band = "B"
                    elif exact_score >= 580:
                        credit_score_band = "C"
                    elif exact_score >= 500:
                        credit_score_band = "D"
                    else:
                        credit_score_band = "E"
                    st.caption(f"Mapped Band: **{credit_score_band}** (Score {exact_score})")

            st.markdown("##### 3. Loan Application Details")
            c_loan1, c_loan2, c_loan3, c_loan4 = st.columns(4)
            with c_loan1:
                prod_choices = [
                    "Salary Advance", "Personal Loan", "SME Loan",
                    "Buy Now Pay Later", "Education Loan", "Agricultural Loan", "✍️ Enter Product Manually"
                ]
                selected_prod = st.selectbox("Loan Product", prod_choices, index=0)
                if selected_prod == "✍️ Enter Product Manually":
                    custom_prod = st.text_input("Type Loan Product", placeholder="e.g. Asset Financing...")
                    loan_product = custom_prod.strip() if custom_prod.strip() else "Personal Loan"
                else:
                    loan_product = selected_prod if selected_prod in ["Salary Advance", "Personal Loan", "SME Loan", "Buy Now Pay Later"] else "Personal Loan"

            with c_loan2:
                quick_amt = st.selectbox(
                    "Loan Amount (Select or Type):",
                    ["✍️ Enter Exact Amount Manually", "₦50,000", "₦100,000", "₦150,000", "₦250,000", "₦500,000", "₦1,000,000", "₦2,000,000"]
                )
                if quick_amt != "✍️ Enter Exact Amount Manually":
                    pre_amt = int(quick_amt.replace("₦", "").replace(",", ""))
                else:
                    pre_amt = st.session_state.get('inp_loan', 0)
                loan_amount = st.number_input(
                    "Loan Amount (₦)",
                    min_value=0,
                    max_value=10000000,
                    value=pre_amt,
                    step=10000,
                    placeholder="Enter amount in Naira"
                )

            with c_loan3:
                quick_tenor = st.selectbox(
                    "Loan Tenor (Select or Type):",
                    ["✍️ Enter Exact Days Manually", "14 Days (2 Weeks)", "30 Days (1 Month)", "60 Days (2 Months)", "90 Days (3 Months)", "180 Days (6 Months)", "365 Days (1 Year)"]
                )
                if "14 Days" in quick_tenor:
                    pre_tenor = 14
                elif "30 Days" in quick_tenor:
                    pre_tenor = 30
                elif "60 Days" in quick_tenor:
                    pre_tenor = 60
                elif "90 Days" in quick_tenor:
                    pre_tenor = 90
                elif "180 Days" in quick_tenor:
                    pre_tenor = 180
                elif "365 Days" in quick_tenor:
                    pre_tenor = 365
                else:
                    pre_tenor = st.session_state.get('inp_tenor', 0)
                loan_tenor_days = st.number_input(
                    "Tenor (Days)",
                    min_value=0,
                    max_value=365,
                    value=pre_tenor,
                    step=7,
                    placeholder="Enter duration in days"
                )

            with c_loan4:
                chan_choices = ["Mobile App", "Web", "USSD", "Agent", "✍️ Enter Channel Manually"]
                selected_chan = st.selectbox("Application Channel", chan_choices, index=0)
                if selected_chan == "✍️ Enter Channel Manually":
                    custom_chan = st.text_input("Type Channel", placeholder="e.g. Branch, WhatsApp...")
                    channel = custom_chan.strip() if custom_chan.strip() else "Mobile App"
                else:
                    channel = selected_chan

            interest_rate_pct = st.number_input("Interest Rate (%)", min_value=0.0, max_value=60.0, value=float(st.session_state.get('inp_rate', 12.0)), step=0.5)

            predict_btn = st.form_submit_button("⚡ Run Credit Risk Assessment", use_container_width=True)

        # Process Single Prediction
        if predict_btn:
            if loan_amount <= 0 or monthly_income <= 0:
                st.warning("⚠️ Please provide a valid **Loan Amount** and **Monthly Income** (both greater than ₦0) to calculate risk.")
            else:
                input_payload = {
                    'loan_product': loan_product,
                    'loan_amount': loan_amount,
                    'loan_tenor_days': max(loan_tenor_days, 7),
                    'interest_rate_pct': max(interest_rate_pct, 1.0),
                    'applicant_age': max(applicant_age, 18),
                    'employment_type': employment_type,
                    'monthly_income': monthly_income,
                    'debt_to_income_ratio': dti,
                    'credit_score_band': credit_score_band,
                    'state': state,
                    'channel': channel
                }

                assessment = model_engine.predict_single_applicant(input_payload, artifacts)

                st.divider()
                st.markdown("### 📋 Assessment Results & Underwriting Output")

                risk_label = assessment['primary_risk']
                rec_label = assessment['recommendation']
                dec_label = assessment['decision']
                
                # FinTech theme status colors
                if risk_label == "Low Risk":
                    dec_color = "#059669" # Emerald
                    bg_color = "#ecfdf5"
                elif risk_label == "Medium Risk":
                    dec_color = "#d97706" # Warm Amber
                    bg_color = "#fffbeb"
                else:
                    dec_color = "#dc2626" # Crimson
                    bg_color = "#fef2f2"

                col_res1, col_res2, col_res3 = st.columns([1.2, 1, 1.2])

                with col_res1:
                    st.markdown(f"""
                    <div style='background-color: {bg_color}; border: 2px solid {dec_color}; border-radius: 12px; padding: 22px; text-align:center; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);'>
                        <span style='font-size: 0.8rem; color: #475569; font-weight:750; text-transform: uppercase; letter-spacing: 0.5px;'>CLASSIFIED RISK LEVEL</span>
                        <h2 style='color: {dec_color}; margin: 8px 0; font-size: 1.85rem;'>{risk_label}</h2>
                        <span style='font-size: 0.95rem; font-weight:650; color: #0f172a;'>Decision: {dec_label} ({rec_label})</span>
                    </div>
                    """, unsafe_allow_html=True)

                with col_res2:
                    def_prob = assessment['default_probability'] * 100.0
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = def_prob,
                        title = {'text': "Default Risk Score (%)", 'font': {'size': 14, 'color': '#0f172a'}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickcolor': '#64748b'},
                            'bar': {'color': dec_color},
                            'steps': [
                                {'range': [0, 35], 'color': "#ecfdf5"},
                                {'range': [35, 60], 'color': "#fffbeb"},
                                {'range': [60, 100], 'color': "#fef2f2"}
                            ],
                            'threshold': {
                                'line': {'color': "#dc2626", 'width': 3},
                                'thickness': 0.75,
                                'value': 60
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col_res3:
                    derived = assessment['derived_features']
                    st.markdown(f"""
                    <div style='background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);'>
                        <div style='font-size: 0.85rem; font-weight: 700; color: #0d9488; text-transform: uppercase; margin-bottom: 6px;'>Financial Health Ratios</div>
                        <div style='margin: 4px 0; font-size: 0.9rem;'><b>Estimated Monthly Installment:</b> ₦{derived['estimated_monthly_installment']:,.2f}</div>
                        <div style='margin: 4px 0; font-size: 0.9rem;'><b>Total Debt Service Ratio:</b> {derived['total_debt_service_ratio']:.1f}%</div>
                        <div style='margin: 4px 0; font-size: 0.9rem;'><b>Net Monthly Cash Cushion:</b> ₦{derived['disposable_income_after_loan']:,.2f}</div>
                        <hr style='margin: 8px 0; border: none; border-top: 1px solid #e2e8f0;'/>
                        <div style='font-size: 0.85rem; color: #64748b;'><b>XGBoost Default Prob:</b> {assessment['xgb']['default_prob']*100:.1f}% ({assessment['xgb']['risk_class']})</div>
                        <div style='font-size: 0.85rem; color: #64748b;'><b>Random Forest Default Prob:</b> {assessment['rf']['default_prob']*100:.1f}% ({assessment['rf']['risk_class']})</div>
                    </div>
                    """, unsafe_allow_html=True)

                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.markdown("#### 🔍 Identified Risk Drivers")
                    for driver in assessment['risk_drivers']:
                        st.markdown(f"- {driver}")

                with col_exp2:
                    st.markdown("#### 💡 Policy Recommendations")
                    for rec in assessment['policy_recommendations']:
                        st.markdown(f"- {rec}")
        else:
            st.markdown("""
            <div style='background-color: #ffffff; border: 2px dashed #0d9488; border-radius: 12px; padding: 2.2rem; text-align: center; margin-top: 1.5rem; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);'>
                <div style='font-size: 2.2rem; margin-bottom: 0.4rem;'>💳</div>
                <h3 style='color: #0f172a; margin-bottom: 0.3rem;'>Applicant Appraisal Ready</h3>
                <p style='color: #64748b; font-size: 0.95rem; max-width: 520px; margin: 0 auto;'>
                    Select or manually enter applicant data in the form above and click <b>⚡ Run Credit Risk Assessment</b> to classify risk tier, default probability, and policy recommendations.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 2: BATCH TEST DATA ASSESSMENT (UPLOAD CSV)
    # ---------------------------------------------------------
    with tab_batch:
        st.markdown("### 📁 Batch Assessment on New Test Data")
        st.caption("Upload new test applications or load sample batches to classify risk levels and export decisions.")
        
        col_load1, col_load2, col_load3 = st.columns([1.5, 1, 1.2])
        
        with col_load1:
            uploaded_file = st.file_uploader(
                "📂 Upload New Test Applications (.csv or .xlsx)",
                type=["csv", "xlsx"]
            )
            
        with col_load2:
            st.markdown("**Quick Load Test Batches:**")
            if st.button("⚡ Load 15 Sample Test Applicants", use_container_width=True):
                sample_path = os.path.join(r"c:\Users\USER\Desktop\Goody_Project", "sample_new_test_applicants.csv")
                if os.path.exists(sample_path):
                    st.session_state['new_test_df'] = pd.read_csv(sample_path)
                    st.session_state['batch_results_df'] = None
                    st.success("Loaded 15 sample test applicants!")
            
            if st.button("🎲 Generate 50 Random Loans", use_container_width=True):
                st.session_state['new_test_df'] = model_engine.generate_random_test_batch(50)
                st.session_state['batch_results_df'] = None
                st.success("Generated 50 fresh random applicant profiles!")
                
        with col_load3:
            st.markdown("**Template Download:**")
            sample_template_path = os.path.join(r"c:\Users\USER\Desktop\Goody_Project", "sample_new_test_applicants.csv")
            if os.path.exists(sample_template_path):
                with open(sample_template_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Sample CSV Template",
                        data=f.read(),
                        file_name="new_applicants_test_template.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    st.session_state['new_test_df'] = pd.read_csv(uploaded_file)
                else:
                    st.session_state['new_test_df'] = pd.read_excel(uploaded_file)
                st.session_state['batch_results_df'] = None
                st.info(f"Loaded {len(st.session_state['new_test_df'])} applications from '{uploaded_file.name}'.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
        if 'new_test_df' in st.session_state and st.session_state['new_test_df'] is not None:
            raw_test_df = st.session_state['new_test_df']
            st.divider()
            st.markdown(f"#### 📄 Input Batch ({len(raw_test_df)} Applicants)")
            st.dataframe(raw_test_df.head(6), use_container_width=True)
            
            classify_col1, _ = st.columns([1.5, 2.5])
            with classify_col1:
                run_classify = st.button("🚀 Run Batch Credit Risk Classification", type="primary", use_container_width=True)
                
            if run_classify or ('batch_results_df' in st.session_state and st.session_state['batch_results_df'] is not None):
                if run_classify:
                    with st.spinner("Classifying test applications..."):
                        st.session_state['batch_results_df'] = model_engine.predict_batch(raw_test_df, artifacts)
                        
                classified_df = st.session_state['batch_results_df']
                st.success(f"✅ Classified {len(classified_df)} applications!")
                
                # Summary KPIs
                bk1, bk2, bk3, bk4, bk5 = st.columns(5)
                total_cnt = len(classified_df)
                low_cnt = sum(classified_df['predicted_risk_level'] == 'Low Risk')
                med_cnt = sum(classified_df['predicted_risk_level'] == 'Medium Risk')
                high_cnt = sum(classified_df['predicted_risk_level'] == 'High Risk')
                avg_prob = classified_df['default_probability_pct'].mean()
                
                bk1.metric("Total Evaluated", f"{total_cnt:,}")
                bk2.metric("🟢 Low Risk", f"{low_cnt} ({low_cnt/total_cnt*100:.1f}%)")
                bk3.metric("🟡 Medium Risk", f"{med_cnt} ({med_cnt/total_cnt*100:.1f}%)")
                bk4.metric("🔴 High Risk", f"{high_cnt} ({high_cnt/total_cnt*100:.1f}%)")
                bk5.metric("Avg Default Risk", f"{avg_prob:.1f}%")
                
                # Visual Charts for Batch
                b_c1, b_c2, b_c3 = st.columns([1, 1.2, 1.2])
                with b_c1:
                    risk_counts = classified_df['predicted_risk_level'].value_counts().reset_index()
                    risk_counts.columns = ['Risk Level', 'Count']
                    fig_b_pie = px.pie(
                        risk_counts, values='Count', names='Risk Level',
                        color='Risk Level',
                        color_discrete_map={'Low Risk': '#10b981', 'Medium Risk': '#f59e0b', 'High Risk': '#ef4444'},
                        title="Risk Proportions",
                        hole=0.45
                    )
                    fig_b_pie.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_b_pie, use_container_width=True)
                    
                with b_c2:
                    if 'state' in classified_df.columns:
                        state_risk = pd.crosstab(classified_df['state'], classified_df['predicted_risk_level']).reset_index()
                        for c in ['Low Risk', 'Medium Risk', 'High Risk']:
                            if c not in state_risk.columns:
                                state_risk[c] = 0
                        fig_b_state = px.bar(
                            state_risk, x='state', y=['Low Risk', 'Medium Risk', 'High Risk'],
                            color_discrete_map={'Low Risk': '#10b981', 'Medium Risk': '#f59e0b', 'High Risk': '#ef4444'},
                            title="Risk Distribution by State",
                            barmode='stack'
                        )
                        fig_b_state.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10), xaxis_title="State", yaxis_title="Applicants")
                        st.plotly_chart(fig_b_state, use_container_width=True)
                        
                with b_c3:
                    fig_b_hist = px.histogram(
                        classified_df, x='default_probability_pct',
                        nbins=20,
                        title="Default Probability Distribution (%)",
                        color_discrete_sequence=['#0d6efd']
                    )
                    fig_b_hist.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10), xaxis_title="Default Probability (%)", yaxis_title="Loans")
                    st.plotly_chart(fig_b_hist, use_container_width=True)
                    
                # Full Results Table
                st.markdown("#### 📋 Detailed Classification Output")
                filt_col1, filt_col2 = st.columns([1, 2])
                with filt_col1:
                    filter_risk = st.selectbox("Filter Risk Level", ["All", "Low Risk", "Medium Risk", "High Risk"])
                with filt_col2:
                    search_txt = st.text_input("🔍 Search Applicant ID or Name", "")
                    
                display_df = classified_df.copy()
                if filter_risk != "All":
                    display_df = display_df[display_df['predicted_risk_level'] == filter_risk]
                if search_txt:
                    name_mask = display_df['applicant_name'].str.contains(search_txt, case=False, na=False) if 'applicant_name' in display_df.columns else False
                    id_mask = display_df['applicant_id'].str.contains(search_txt, case=False, na=False) if 'applicant_id' in display_df.columns else False
                    display_df = display_df[name_mask | id_mask]
                    
                cols_to_format = ['loan_amount', 'monthly_income', 'monthly_installment', 'disposable_income_after_loan']
                preview_table = display_df.copy()
                for col in cols_to_format:
                    if col in preview_table.columns:
                        preview_table[col] = preview_table[col].apply(lambda x: f"₦{x:,.0f}" if pd.notnull(x) else "")
                        
                st.dataframe(preview_table, use_container_width=True)
                
                # Export actions
                st.divider()
                csv_export = classified_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Classified Results (CSV)",
                    data=csv_export,
                    file_name="classified_loan_applications_output.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

# -------------------------------------------------------------
# 2. MODEL EVALUATION & COMPARISON
# -------------------------------------------------------------
elif menu == "📊 Model Evaluation & Comparison":
    st.markdown("<div class='main-header'>Model Performance & Comparative Evaluation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Comparative Benchmark: Random Forest vs. XGBoost on 500-Loan Holdout Test Set</div>", unsafe_allow_html=True)
    
    metrics = artifacts['metrics']
    rf_m = metrics['random_forest']
    xgb_m = metrics['xgboost']
    best_m = metrics['best_model']
    
    # KPI cards
    col_metrics1, col_metrics2 = st.columns(2)
    with col_metrics1:
        st.markdown("### 🌲 Random Forest Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{rf_m['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{rf_m['precision']*100:.2f}%")
        m3.metric("Recall", f"{rf_m['recall']*100:.2f}%")
        
        m4, m5, m6 = st.columns(3)
        m4.metric("F1-Score", f"{rf_m['f1_score']*100:.2f}%")
        m5.metric("Multiclass AUC", f"{rf_m['auc_roc']:.4f}")
        m6.metric("Binary AUC-ROC", f"{rf_m['binary_auc']:.4f}")
        
    with col_metrics2:
        st.markdown(f"### 🚀 XGBoost Performance {'🏆 (Best Model)' if best_m == 'XGBoost' else ''}")
        x1, x2, x3 = st.columns(3)
        x1.metric("Accuracy", f"{xgb_m['accuracy']*100:.2f}%")
        x2.metric("Precision", f"{xgb_m['precision']*100:.2f}%")
        x3.metric("Recall", f"{xgb_m['recall']*100:.2f}%")
        
        x4, x5, x6 = st.columns(3)
        x4.metric("F1-Score", f"{xgb_m['f1_score']*100:.2f}%")
        x5.metric("Multiclass AUC", f"{xgb_m['auc_roc']:.4f}")
        x6.metric("Binary AUC-ROC", f"{xgb_m['binary_auc']:.4f}")
        
    st.divider()
    
    # 1. Overall Metrics Comparison Grouped Bar Chart
    st.markdown("### 1. Overall Performance Metrics Comparison")
    comp_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Multiclass AUC', 'Binary AUC']
    rf_vals = [rf_m['accuracy']*100, rf_m['precision']*100, rf_m['recall']*100, rf_m['f1_score']*100, rf_m['auc_roc']*100, rf_m['binary_auc']*100]
    xgb_vals = [xgb_m['accuracy']*100, xgb_m['precision']*100, xgb_m['recall']*100, xgb_m['f1_score']*100, xgb_m['auc_roc']*100, xgb_m['binary_auc']*100]
    
    col_comp_bar, col_cv_bar = st.columns([1.6, 1.0])
    with col_comp_bar:
        fig_bar_comp = go.Figure()
        fig_bar_comp.add_trace(go.Bar(
            name='Random Forest',
            x=comp_metrics,
            y=rf_vals,
            text=[f"{v:.1f}%" for v in rf_vals],
            textposition='auto',
            marker_color='#6366f1'
        ))
        fig_bar_comp.add_trace(go.Bar(
            name='XGBoost',
            x=comp_metrics,
            y=xgb_vals,
            text=[f"{v:.1f}%" for v in xgb_vals],
            textposition='auto',
            marker_color='#0d9488'
        ))
        fig_bar_comp.update_layout(
            barmode='group',
            yaxis=dict(title="Score (%)", range=[88, 102]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            height=380
        )
        st.plotly_chart(fig_bar_comp, use_container_width=True)
        
    with col_cv_bar:
        st.markdown("**Cross-Validation vs. Holdout Test Accuracy**")
        tuning_info = metrics.get('hyperparameter_tuning', {})
        rf_cv = tuning_info.get('rf_cv_score', 0.9575) * 100
        xgb_cv = tuning_info.get('xgb_cv_score', 0.9555) * 100
        
        cv_df = pd.DataFrame({
            'Model': ['Random Forest', 'Random Forest', 'XGBoost', 'XGBoost'],
            'Split': ['3-Fold CV', 'Holdout Test', '3-Fold CV', 'Holdout Test'],
            'Accuracy': [rf_cv, rf_m['accuracy']*100, xgb_cv, xgb_m['accuracy']*100]
        })
        fig_cv = px.bar(
            cv_df, x='Model', y='Accuracy', color='Split', barmode='group',
            text=cv_df['Accuracy'].apply(lambda x: f"{x:.2f}%"),
            color_discrete_map={'3-Fold CV': '#64748b', 'Holdout Test': '#0d9488'},
            range_y=[88, 100], height=380
        )
        fig_cv.update_traces(textposition='auto')
        fig_cv.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_cv, use_container_width=True)
        
    st.divider()
    
    # 2. Side-by-Side Confusion Matrices
    st.markdown("### 2. Side-by-Side Confusion Matrices (500 Test Loans)")
    col_cm1, col_cm2 = st.columns(2)
    cm_labels = ["Low Risk", "Medium Risk", "High Risk"]
    
    with col_cm1:
        st.markdown("#### Random Forest Confusion Matrix")
        fig_rf_cm = px.imshow(
            rf_m['confusion_matrix'],
            text_auto=True,
            x=cm_labels,
            y=cm_labels,
            labels=dict(x="Predicted", y="Actual", color="Loans"),
            color_continuous_scale="Greens",
            title=f"Random Forest (Accuracy: {rf_m['accuracy']*100:.2f}%)"
        )
        fig_rf_cm.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_rf_cm, use_container_width=True)
        
    with col_cm2:
        st.markdown("#### XGBoost Confusion Matrix")
        fig_xgb_cm = px.imshow(
            xgb_m['confusion_matrix'],
            text_auto=True,
            x=cm_labels,
            y=cm_labels,
            labels=dict(x="Predicted", y="Actual", color="Loans"),
            color_continuous_scale="Blues",
            title=f"XGBoost (Accuracy: {xgb_m['accuracy']*100:.2f}%)"
        )
        fig_xgb_cm.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_xgb_cm, use_container_width=True)

    st.divider()

    # 3. Granular Risk Tier Metrics Breakdown
    st.markdown("### 3. Per-Class Risk Tier Performance")
    rf_rep = rf_m['classification_report']
    xgb_rep = xgb_m['classification_report']
    
    per_class_data = []
    for cls in ["Low Risk", "Medium Risk", "High Risk"]:
        per_class_data.append({
            "Risk Tier": cls, "Metric": "Precision",
            "Random Forest": rf_rep[cls]['precision'] * 100,
            "XGBoost": xgb_rep[cls]['precision'] * 100
        })
        per_class_data.append({
            "Risk Tier": cls, "Metric": "Recall",
            "Random Forest": rf_rep[cls]['recall'] * 100,
            "XGBoost": xgb_rep[cls]['recall'] * 100
        })
        per_class_data.append({
            "Risk Tier": cls, "Metric": "F1-Score",
            "Random Forest": rf_rep[cls]['f1-score'] * 100,
            "XGBoost": xgb_rep[cls]['f1-score'] * 100
        })
    df_per_class = pd.DataFrame(per_class_data)
    
    col_pc1, col_pc2 = st.columns(2)
    with col_pc1:
        df_f1_rec = df_per_class[df_per_class['Metric'].isin(['F1-Score', 'Recall'])]
        fig_pc = px.bar(
            df_f1_rec, x='Risk Tier', y=['Random Forest', 'XGBoost'],
            barmode='group', facet_col='Metric',
            color_discrete_map={'Random Forest': '#6366f1', 'XGBoost': '#0d9488'},
            height=360
        )
        fig_pc.update_layout(yaxis=dict(range=[70, 102], title="Percentage (%)"))
        st.plotly_chart(fig_pc, use_container_width=True)
        
    with col_pc2:
        table_rows = []
        for cls in ["Low Risk", "Medium Risk", "High Risk"]:
            table_rows.append({
                "Risk Tier": cls,
                "RF Precision": f"{rf_rep[cls]['precision']*100:.1f}%",
                "XGB Precision": f"{xgb_rep[cls]['precision']*100:.1f}%",
                "RF Recall": f"{rf_rep[cls]['recall']*100:.1f}%",
                "XGB Recall": f"{xgb_rep[cls]['recall']*100:.1f}%",
                "RF F1": f"{rf_rep[cls]['f1-score']*100:.1f}%",
                "XGB F1": f"{xgb_rep[cls]['f1-score']*100:.1f}%",
                "Support": int(rf_rep[cls]['support'])
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.divider()

    # 4. ROC & Precision-Recall Curves
    st.markdown("### 4. Discrimination Power: ROC & Precision-Recall Curves")
    detailed_eval = model_engine.get_detailed_evaluation_data()
    
    col_roc, col_pr = st.columns(2)
    with col_roc:
        st.markdown("#### Binary Default ROC Curves")
        bin_roc = detailed_eval['binary_roc']
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=bin_roc['rf']['fpr'], y=bin_roc['rf']['tpr'],
            mode='lines', name=f"Random Forest (AUC = {bin_roc['rf']['auc']:.4f})",
            line=dict(color='#28a745', width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=bin_roc['xgb']['fpr'], y=bin_roc['xgb']['tpr'],
            mode='lines', name=f"XGBoost (AUC = {bin_roc['xgb']['auc']:.4f})",
            line=dict(color='#0d6efd', width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Baseline Chance (AUC = 0.500)',
            line=dict(dash='dash', color='gray')
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.45, y=0.15),
            margin=dict(l=20, r=20, t=30, b=20),
            height=380
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col_pr:
        st.markdown("#### Precision-Recall (PR) Curves")
        bin_pr = detailed_eval['binary_pr']
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=bin_pr['rf']['recall'], y=bin_pr['rf']['precision'],
            mode='lines', name=f"Random Forest (AP = {bin_pr['rf']['ap']:.4f})",
            line=dict(color='#28a745', width=2.5)
        ))
        fig_pr.add_trace(go.Scatter(
            x=bin_pr['xgb']['recall'], y=bin_pr['xgb']['precision'],
            mode='lines', name=f"XGBoost (AP = {bin_pr['xgb']['ap']:.4f})",
            line=dict(color='#0d6efd', width=2.5)
        ))
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            legend=dict(x=0.05, y=0.15),
            margin=dict(l=20, r=20, t=30, b=20),
            height=380
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # 5. Multiclass OvR ROC Curves
    st.markdown("#### Multi-Class One-vs-Rest (OvR) ROC Curves")
    ovr_data = detailed_eval['ovr_roc']
    col_ovr1, col_ovr2, col_ovr3 = st.columns(3)
    
    for col, cls_name in zip([col_ovr1, col_ovr2, col_ovr3], ["Low Risk", "Medium Risk", "High Risk"]):
        with col:
            cls_ovr = ovr_data[cls_name]
            fig_ovr_single = go.Figure()
            fig_ovr_single.add_trace(go.Scatter(
                x=cls_ovr['rf']['fpr'], y=cls_ovr['rf']['tpr'],
                mode='lines', name=f"RF (AUC = {cls_ovr['rf']['auc']:.3f})",
                line=dict(color='#28a745', width=2)
            ))
            fig_ovr_single.add_trace(go.Scatter(
                x=cls_ovr['xgb']['fpr'], y=cls_ovr['xgb']['tpr'],
                mode='lines', name=f"XGB (AUC = {cls_ovr['xgb']['auc']:.3f})",
                line=dict(color='#0d6efd', width=2)
            ))
            fig_ovr_single.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', name='Baseline',
                line=dict(dash='dash', color='gray')
            ))
            fig_ovr_single.update_layout(
                title=f"{cls_name} OvR",
                xaxis_title="FPR", yaxis_title="TPR",
                legend=dict(x=0.25, y=0.15, font=dict(size=9)),
                margin=dict(l=10, r=10, t=35, b=10),
                height=280
            )
            st.plotly_chart(fig_ovr_single, use_container_width=True)

    st.divider()

    # 6. Feature Importances Comparison
    st.markdown("### 5. Predictive Feature Importance Comparison")
    col_feat1, col_feat2 = st.columns(2)
    rf_feat = metrics['feature_importance']['random_forest']
    xgb_feat = metrics['feature_importance']['xgboost']
    
    with col_feat1:
        top_rf = list(rf_feat.items())[:12]
        df_rf_feat = pd.DataFrame(top_rf, columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
        fig_rf_feat = px.bar(
            df_rf_feat, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Greens',
            title="Random Forest Key Features (MDI)"
        )
        fig_rf_feat.update_layout(height=380, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_rf_feat, use_container_width=True)
        
    with col_feat2:
        top_xgb = list(xgb_feat.items())[:12]
        df_xgb_feat = pd.DataFrame(top_xgb, columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
        fig_xgb_feat = px.bar(
            df_xgb_feat, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Blues',
            title="XGBoost Key Features (Gain)"
        )
        fig_xgb_feat.update_layout(height=380, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_xgb_feat, use_container_width=True)

    st.divider()

    # 7. Model Concordance & Probability Calibration
    st.markdown("### 6. Model Concordance & Probability Calibration")
    col_agree, col_prob = st.columns(2)
    with col_agree:
        agree_cm = detailed_eval['agreement']['matrix']
        fig_agree = px.imshow(
            agree_cm,
            text_auto=True,
            x=cm_labels,
            y=cm_labels,
            labels=dict(x="XGBoost", y="Random Forest", color="Loans"),
            color_continuous_scale="Purples",
            title=f"Prediction Agreement: {detailed_eval['agreement']['percentage']}% Concordance"
        )
        fig_agree.update_layout(height=360, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_agree, use_container_width=True)
        
    with col_prob:
        prob_df = pd.DataFrame({
            'Default Probability': detailed_eval['rf_bin_probs'] + detailed_eval['xgb_bin_probs'],
            'Model': ['Random Forest'] * len(detailed_eval['rf_bin_probs']) + ['XGBoost'] * len(detailed_eval['xgb_bin_probs'])
        })
        fig_hist = px.histogram(
            prob_df, x='Default Probability', color='Model', barmode='overlay',
            nbins=30, opacity=0.7,
            color_discrete_map={'Random Forest': '#6366f1', 'XGBoost': '#0d9488'},
            title="Default Probability Distribution"
        )
        fig_hist.update_layout(
            xaxis_title="Default Probability",
            yaxis_title="Count",
            legend=dict(x=0.75, y=0.9),
            height=360, margin=dict(l=20, r=20, t=35, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------------------------------------
# 3. PORTFOLIO DATA & INSIGHTS
# -------------------------------------------------------------
elif menu == "🗄️ Portfolio Data & Insights":
    st.markdown("<div class='main-header'>Portfolio Dataset & Risk Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Consolidated Master Dataset (2,500 Loans) and Geographic Distribution</div>", unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    total_apps = len(df_dataset)
    total_disbursed = df_dataset[df_dataset['approval_status'] == 'Approved']['loan_amount'].sum()
    avg_income = df_dataset['monthly_income'].mean()
    avg_dti = df_dataset['debt_to_income_ratio'].mean()
    
    k1.metric("Total Loan Applications", f"{total_apps:,}")
    k2.metric("Portfolio Disbursed", f"₦{total_disbursed/1e6:.1f}M")
    k3.metric("Average Monthly Income", f"₦{avg_income:,.0f}")
    k4.metric("Average DTI", f"{avg_dti:.1f}%")
    
    st.markdown("#### Filter & Explore Master Dataset")
    with st.expander("🔍 Filter Options", expanded=False):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            sel_product = st.multiselect("Loan Product", options=df_dataset['loan_product'].unique(), default=df_dataset['loan_product'].unique())
        with fcol2:
            sel_state = st.multiselect("State", options=df_dataset['state'].unique(), default=df_dataset['state'].unique())
        with fcol3:
            sel_risk = st.multiselect("Risk Level", options=df_dataset['risk_level'].unique(), default=df_dataset['risk_level'].unique())
        with fcol4:
            sel_status = st.multiselect("Approval Status", options=df_dataset['approval_status'].unique(), default=df_dataset['approval_status'].unique())
            
    filtered_df = df_dataset[
        (df_dataset['loan_product'].isin(sel_product)) &
        (df_dataset['state'].isin(sel_state)) &
        (df_dataset['risk_level'].isin(sel_risk)) &
        (df_dataset['approval_status'].isin(sel_status))
    ]
    
    st.dataframe(
        filtered_df[[
            'loan_id', 'applicant_id', 'loan_product', 'loan_amount', 'loan_tenor_days',
            'monthly_income', 'debt_to_income_ratio', 'credit_score_band', 'state',
            'channel', 'approval_status', 'risk_level', 'loan_decision', 'repayment_status'
        ]],
        use_container_width=True,
        height=360
    )
    
    csv_data = df_dataset.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Master Dataset CSV (2,500 Rows)",
        data=csv_data,
        file_name="nigeria_credit_risk_dataset_2500.csv",
        mime="text/csv"
    )
    
    st.divider()
    st.markdown("#### Portfolio Risk Distributions")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        state_agg = df_dataset[df_dataset['approval_status'] == 'Approved'].groupby('state').agg(
            total_exposure=('loan_amount', 'sum'),
            default_count=('days_past_due', lambda x: (x >= 60).sum()),
            total_loans=('loan_id', 'count')
        ).reset_index()
        state_agg['default_rate_pct'] = (state_agg['default_count'] / state_agg['total_loans']) * 100.0
        
        fig_geo = px.bar(
            state_agg.sort_values('total_exposure', ascending=False),
            x='state', y='total_exposure', color='default_rate_pct',
            title="Exposure and Delinquency Rate by State",
            color_continuous_scale="Reds",
            labels={'total_exposure': 'Disbursed (₦)', 'state': 'State', 'default_rate_pct': 'Delinquency %'}
        )
        st.plotly_chart(fig_geo, use_container_width=True)
        
    with col_d2:
        prod_agg = df_dataset[df_dataset['approval_status'] == 'Approved'].groupby('loan_product').agg(
            par30=('days_past_due', lambda x: (x >= 30).sum()),
            npl=('days_past_due', lambda x: (x >= 90).sum()),
            total=('loan_id', 'count')
        ).reset_index()
        prod_agg['par30_pct'] = (prod_agg['par30'] / prod_agg['total']) * 100.0
        prod_agg['npl_pct'] = (prod_agg['npl'] / prod_agg['total']) * 100.0
        
        fig_prod = px.bar(
            prod_agg, x='loan_product', y=['par30_pct', 'npl_pct'],
            barmode='group', title="PAR30 vs NPL Rate by Product (%)",
            labels={'value': 'Percentage (%)', 'loan_product': 'Product', 'variable': 'Metric'},
            color_discrete_map={'par30_pct': '#f59e0b', 'npl_pct': '#ef4444'}
        )
        st.plotly_chart(fig_prod, use_container_width=True)
