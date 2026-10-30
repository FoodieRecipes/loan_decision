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
import database

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Credit Risk Assessment & Loan Underwriting",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Clean Interface
st.markdown("""
<style>
    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1a2a3a;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #556980;
        margin-bottom: 1rem;
    }
    .risk-badge-low {
        background-color: #d4edda;
        color: #155724;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #c3e6cb;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: #fff3cd;
        color: #856404;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #ffeeba;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #f5c6cb;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
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
            "🗄️ Portfolio Data & Insights",
            "📋 Decisions Audit Log"
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
        st.markdown("##### ⚡ Quick Load Sample Borrower Profiles:")
        col_preset1, col_preset2, col_preset3, col_preset4, col_preset5 = st.columns(5)
        
        preset_data = {}
        if col_preset1.button("🟢 Akwa Ibom Salaried (Uyo)", key="btn_p1"):
            preset_data = {
                'name': "Aniefiok Bassey", 'id': "APP-AKS-01", 'age': 33, 'state': "Akwa Ibom",
                'product': "Salary Advance", 'amount': 180000, 'tenor': 30, 'rate': 9.0,
                'income': 480000, 'dti': 16.5, 'credit_band': "A", 'employment': "Employed", 'channel': "Mobile App"
            }
        elif col_preset2.button("🟢 Prime Salaried (Lagos)", key="btn_p2"):
            preset_data = {
                'name': "Ibrahim Danjuma", 'id': "APP-PRIME-01", 'age': 34, 'state': "Lagos",
                'product': "Salary Advance", 'amount': 150000, 'tenor': 30, 'rate': 8.5,
                'income': 520000, 'dti': 14.0, 'credit_band': "A", 'employment': "Employed", 'channel': "Mobile App"
            }
        elif col_preset3.button("🟡 SME Expansion (Enugu)", key="btn_p3"):
            preset_data = {
                'name': "Ngozi Eze", 'id': "APP-SME-02", 'age': 41, 'state': "Enugu",
                'product': "SME Loan", 'amount': 600000, 'tenor': 90, 'rate': 16.0,
                'income': 380000, 'dti': 42.0, 'credit_band': "B", 'employment': "Self-Employed", 'channel': "Web"
            }
        elif col_preset4.button("🔴 Stressed Retail (Rivers)", key="btn_p4"):
            preset_data = {
                'name': "Emeka Okonkwo", 'id': "APP-HIGH-03", 'age': 28, 'state': "Rivers",
                'product': "Personal Loan", 'amount': 450000, 'tenor': 60, 'rate': 24.0,
                'income': 120000, 'dti': 58.5, 'credit_band': "E", 'employment': "Self-Employed", 'channel': "USSD"
            }
        elif col_preset5.button("⚖️ Incomplete Docs (Review)", key="btn_p5"):
            preset_data = {
                'name': "Folake Adebayo", 'id': "APP-REV-04", 'age': 36, 'state': "Oyo",
                'product': "Buy Now Pay Later", 'amount': 220000, 'tenor': 30, 'rate': 12.0,
                'income': 350000, 'dti': 22.0, 'credit_band': "A", 'employment': "Employed", 'channel': "Agent"
            }
            
        with st.form("appraisal_form"):
            st.markdown("#### 1. Applicant & Channel Information")
            c_app1, c_app2, c_app3, c_app4 = st.columns(4)
            with c_app1:
                app_name = st.text_input("Full Name", value=preset_data.get('name', "Aniefiok Bassey"))
            with c_app2:
                app_id = st.text_input("Application ID", value=preset_data.get('id', "APP-AKS-01"))
            with c_app3:
                applicant_age = st.number_input("Age", min_value=18, max_value=75, value=preset_data.get('age', 33))
            with c_app4:
                state_options = ["Akwa Ibom", "Lagos", "FCT", "Rivers", "Oyo", "Ogun", "Kaduna", "Kano", "Enugu"]
                default_state_idx = state_options.index(preset_data.get('state', "Akwa Ibom")) if preset_data.get('state', "Akwa Ibom") in state_options else 0
                state = st.selectbox("State of Residence", state_options, index=default_state_idx)
                
            st.markdown("#### 2. Financial Standing & Credit Bureau")
            c_fin1, c_fin2, c_fin3, c_fin4 = st.columns(4)
            with c_fin1:
                emp_options = ["Employed", "Self-Employed", "Student", "Unemployed"]
                default_emp_idx = emp_options.index(preset_data.get('employment', "Employed")) if preset_data.get('employment', "Employed") in emp_options else 0
                employment_type = st.selectbox("Employment Type", emp_options, index=default_emp_idx)
            with c_fin2:
                monthly_income = st.number_input("Monthly Income (₦)", min_value=10000, max_value=10000000, value=int(preset_data.get('income', 480000)), step=10000)
            with c_fin3:
                dti = st.number_input("Existing Debt-to-Income (DTI %)", min_value=0.0, max_value=100.0, value=float(preset_data.get('dti', 16.5)), step=1.0)
            with c_fin4:
                score_bands = ["A", "B", "C", "D", "E"]
                default_band_idx = score_bands.index(preset_data.get('credit_band', "A")) if preset_data.get('credit_band', "A") in score_bands else 0
                credit_score_band = st.selectbox("Credit Bureau Band", score_bands, index=default_band_idx, help="A: Prime (800), B: Near Prime (700), C: Standard (600), D: Subprime (500), E: Deep Subprime (400)")
                
            st.markdown("#### 3. Loan Application Details")
            c_loan1, c_loan2, c_loan3, c_loan4 = st.columns(4)
            with c_loan1:
                product_options = ["Salary Advance", "Personal Loan", "SME Loan", "Buy Now Pay Later"]
                default_prod_idx = product_options.index(preset_data.get('product', "Salary Advance")) if preset_data.get('product', "Salary Advance") in product_options else 0
                loan_product = st.selectbox("Loan Product", product_options, index=default_prod_idx)
            with c_loan2:
                loan_amount = st.number_input("Loan Amount (₦)", min_value=10000, max_value=5000000, value=int(preset_data.get('amount', 180000)), step=10000)
            with c_loan3:
                loan_tenor_days = st.number_input("Tenor (Days)", min_value=7, max_value=365, value=int(preset_data.get('tenor', 30)), step=7)
            with c_loan4:
                channel_options = ["Mobile App", "USSD", "Web", "Agent"]
                default_chan_idx = channel_options.index(preset_data.get('channel', "Mobile App")) if preset_data.get('channel', "Mobile App") in channel_options else 0
                channel = st.selectbox("Channel", channel_options, index=default_chan_idx)
                
            suggested_rate = preset_data.get('rate', 9.0)
            interest_rate_pct = st.number_input("Interest Rate (%)", min_value=1.0, max_value=60.0, value=float(suggested_rate), step=0.5)
            
            predict_btn = st.form_submit_button("⚡ Run Credit Risk Assessment", use_container_width=True)
            
        # Process Single Prediction
        input_payload = {
            'loan_product': loan_product,
            'loan_amount': loan_amount,
            'loan_tenor_days': loan_tenor_days,
            'interest_rate_pct': interest_rate_pct,
            'applicant_age': applicant_age,
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
        dec_color = assessment['decision_color']
        
        col_res1, col_res2, col_res3 = st.columns([1.2, 1, 1.2])
        
        with col_res1:
            st.markdown(f"""
            <div style='background-color: #f8f9fa; border: 2px solid {dec_color}; border-radius: 10px; padding: 18px; text-align:center;'>
                <span style='font-size: 0.85rem; color: #6c757d; font-weight:700;'>CLASSIFIED RISK LEVEL</span>
                <h2 style='color: {dec_color}; margin: 8px 0;'>{risk_label}</h2>
                <span style='font-size: 0.95rem; font-weight:600;'>Decision: {dec_label} ({rec_label})</span>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            def_prob = assessment['default_probability'] * 100.0
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = def_prob,
                title = {'text': "Default Risk Score (%)", 'font': {'size': 14}},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': dec_color},
                    'steps': [
                        {'range': [0, 35], 'color': "#d4edda"},
                        {'range': [35, 60], 'color': "#fff3cd"},
                        {'range': [60, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 3},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ))
            fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_res3:
            st.markdown("##### Model Outputs Breakdown:")
            st.write(f"**Random Forest**: {assessment['rf']['risk_class']} (Default Prob: {assessment['rf']['default_prob']*100:.1f}%)")
            st.write(f"**XGBoost**: {assessment['xgb']['risk_class']} (Default Prob: {assessment['xgb']['default_prob']*100:.1f}%)")
            derived = assessment['derived_features']
            st.write(f"**Monthly Installment**: ₦{derived['estimated_monthly_installment']:,.2f}")
            st.write(f"**Total Debt Service Ratio**: {derived['total_debt_service_ratio']:.1f}%")
            st.write(f"**Net Monthly Cash Cushion**: ₦{derived['disposable_income_after_loan']:,.2f}")

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown("#### 🔍 Risk Drivers")
            for driver in assessment['risk_drivers']:
                st.markdown(f"- {driver}")
                
        with col_exp2:
            st.markdown("#### 💡 Policy Recommendations")
            for rec in assessment['policy_recommendations']:
                st.markdown(f"- {rec}")
                
        st.divider()
        st.markdown("#### ✍️ Log Underwriter Decision to Audit Trail")
        
        with st.form("decision_storage_form"):
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                officer_decision = st.selectbox(
                    "Decision Outcome",
                    ["Approved", "Approved with Conditions (Guarantor/Reduced Amount)", "Under Further Review", "Declined"],
                    index=0 if dec_label == "Accept" else (1 if dec_label == "Review" else 3)
                )
            with col_act2:
                officer_notes = st.text_area(
                    "Underwriting Notes",
                    value=f"Automated risk appraisal: {risk_label} via {artifacts['metrics']['best_model']}. Default probability {def_prob:.1f}%."
                )
                
            save_btn = st.form_submit_button("💾 Save Decision to Database", use_container_width=True)
            
        if save_btn:
            decision_record = {
                'applicant_id': app_id,
                'applicant_name': app_name,
                'state': state,
                'loan_product': loan_product,
                'loan_amount': loan_amount,
                'loan_tenor_days': loan_tenor_days,
                'monthly_income': monthly_income,
                'debt_to_income_ratio': dti,
                'credit_score_band': credit_score_band,
                'channel': channel,
                'predicted_risk_level': risk_label,
                'predicted_approval_status': assessment['approval_status'],
                'default_probability': assessment['default_probability'],
                'system_recommendation': rec_label,
                'officer_decision': officer_decision,
                'officer_notes': officer_notes,
                'model_used': artifacts['metrics']['best_model']
            }
            database.save_decision(decision_record)
            st.success(f"🎉 Decision for applicant {app_id} successfully stored in audit database!")

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
                        color_discrete_map={'Low Risk': '#28a745', 'Medium Risk': '#ffc107', 'High Risk': '#dc3545'},
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
                            color_discrete_map={'Low Risk': '#28a745', 'Medium Risk': '#ffc107', 'High Risk': '#dc3545'},
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
                
                # Export and database persistence actions
                st.divider()
                act_col1, act_col2 = st.columns(2)
                with act_col1:
                    csv_export = classified_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Classified Results (CSV)",
                        data=csv_export,
                        file_name="classified_loan_applications_output.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
                with act_col2:
                    if st.button("💾 Save All Classifications to Audit Database", use_container_width=True):
                        records_to_save = classified_df.to_dict(orient="records")
                        for r in records_to_save:
                            r['model_used'] = artifacts['metrics']['best_model']
                        saved_cnt = database.save_batch_decisions(records_to_save)
                        st.success(f"🎉 Successfully saved {saved_cnt} applications and decisions to audit database!")

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
            marker_color='#28a745'
        ))
        fig_bar_comp.add_trace(go.Bar(
            name='XGBoost',
            x=comp_metrics,
            y=xgb_vals,
            text=[f"{v:.1f}%" for v in xgb_vals],
            textposition='auto',
            marker_color='#0d6efd'
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
            color_discrete_map={'3-Fold CV': '#6c757d', 'Holdout Test': '#0d6efd'},
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
            color_discrete_map={'Random Forest': '#28a745', 'XGBoost': '#0d6efd'},
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
            color_discrete_map={'Random Forest': '#28a745', 'XGBoost': '#0d6efd'},
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
            color_discrete_map={'par30_pct': '#ffc107', 'npl_pct': '#dc3545'}
        )
        st.plotly_chart(fig_prod, use_container_width=True)

# -------------------------------------------------------------
# 4. DECISIONS AUDIT LOG
# -------------------------------------------------------------
elif menu == "📋 Decisions Audit Log":
    st.markdown("<div class='main-header'>Decisions Audit Trail & Database</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Logged Underwriting Decisions and Model Predictions</div>", unsafe_allow_html=True)
    
    decisions_df = database.get_decisions()
    st.metric("Total Stored Underwriting Decisions", len(decisions_df))
    
    if len(decisions_df) > 0:
        st.dataframe(
            decisions_df[[
                'id', 'timestamp', 'applicant_id', 'applicant_name', 'state',
                'loan_product', 'loan_amount', 'predicted_risk_level',
                'system_recommendation', 'officer_decision', 'officer_notes', 'model_used'
            ]],
            use_container_width=True,
            height=420
        )
        
        db_csv = decisions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Decisions Log (CSV)",
            data=db_csv,
            file_name="credit_decisions_audit_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No underwriting decisions recorded yet. Test an applicant to log decisions.")
