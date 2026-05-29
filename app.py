import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Banking Risk Intelligence System",
    page_icon="🏦",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    fraud_model = joblib.load('fraud_model.pkl')
    loan_model = joblib.load('loan_model.pkl')
    loan_scaler = joblib.load('loan_scaler.pkl')
    loan_features = joblib.load('loan_features.pkl')
    return fraud_model, loan_model, loan_scaler, loan_features

fraud_model, loan_model, loan_scaler, loan_features = load_models()

# Header
st.title("🏦 Banking Risk Intelligence System")
st.markdown("### AI-Powered Fraud Detection & Loan Default Prediction")
st.markdown("---")

# Sidebar navigation
module = st.sidebar.selectbox(
    "🔍 Select Module",
    ["🏠 Home", "💳 Fraud Detection", "📋 Loan Default Prediction"]
)

# ─────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────
if module == "🏠 Home":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ## 💳 Fraud Detection
        - Analyzes transaction patterns
        - XGBoost ML Model
        - ROC-AUC Score: **0.9739**
        - Catches **88%** of all frauds
        """)
        if st.button("Go to Fraud Detection →"):
            st.info("Select 'Fraud Detection' from the sidebar")

    with col2:
        st.markdown("""
        ## 📋 Loan Default Prediction
        - Analyzes customer profile
        - XGBoost ML Model
        - Generates Risk Score (0-100%)
        - Risk Level: LOW / MEDIUM / HIGH
        """)
        if st.button("Go to Loan Default →"):
            st.info("Select 'Loan Default Prediction' from the sidebar")

    st.markdown("---")
    st.markdown("### 📊 System Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fraud ROC-AUC", "0.9739", "Excellent")
    col2.metric("Fraud Recall", "88%", "High")
    col3.metric("Frauds Caught", "86/98", "In test set")
    col4.metric("Loan Accuracy", "~82%", "Good")

# ─────────────────────────────────────────
# FRAUD DETECTION PAGE
# ─────────────────────────────────────────
elif module == "💳 Fraud Detection":
    st.header("💳 Transaction Fraud Detector")
    st.markdown("Enter real transaction details below")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧾 Transaction Info")
        amount = st.number_input(
            "Transaction Amount (₹)", 
            min_value=0.0, value=1000.0, step=100.0,
            help="The amount of this transaction")
        
        hour = st.slider(
            "Hour of Transaction (24hr format)", 
            0, 23, 14,
            help="0 = midnight, 14 = 2PM, 23 = 11PM")
        
        transaction_type = st.selectbox(
            "Transaction Type",
            ["Online Purchase", "ATM Withdrawal", 
             "POS/Swipe", "Wire Transfer", "International"])
        
        location_match = st.selectbox(
            "Location matches customer's city?",
            ["Yes — same city", 
             "No — different city", 
             "No — different country"])
        
        device_known = st.selectbox(
            "Device/IP recognized?",
            ["Yes — known device", 
             "No — new device", 
             "No — suspicious IP"])

    with col2:
        st.subheader("👤 Customer Profile")
        
        age = st.slider("Customer Age", 18, 80, 35)
        
        avg_transaction = st.number_input(
            "Customer's Usual Avg Transaction (₹)",
            min_value=0.0, value=2000.0, step=500.0,
            help="What does this customer usually spend?")
        
        transactions_today = st.number_input(
            "Number of transactions in last 1 hour",
            min_value=0, max_value=20, value=1,
            help="High frequency = suspicious")
        
        card_present = st.selectbox(
            "Card physically present?",
            ["Yes — card swiped", 
             "No — card number used online",
             "No — contactless"])
        
        previous_fraud = st.selectbox(
            "Previous fraud history on this account?",
            ["No previous fraud", 
             "1 suspicious transaction before",
             "Multiple fraud attempts"])

    if st.button("🔍 Analyze Transaction", use_container_width=True):
        
        # ── Convert user inputs into model features ──

        # Normalize amount (same way we trained)
        norm_amount = (amount - 88.35) / 250.12

        # Time feature (convert hour to seconds roughly)
        time_val = hour * 3600

        # Location risk score
        location_score = {
            "Yes — same city": 0.1,
            "No — different city": -2.5,
            "No — different country": -5.0
        }[location_match]

        # Device risk
        device_score = {
            "Yes — known device": 0.1,
            "No — new device": -2.0,
            "No — suspicious IP": -4.5
        }[device_known]

        # Transaction type risk
        type_score = {
            "Online Purchase": -1.0,
            "ATM Withdrawal": -0.5,
            "POS/Swipe": 0.1,
            "Wire Transfer": -3.0,
            "International": -4.0
        }[transaction_type]

        # Frequency risk
        freq_score = -1.0 * min(transactions_today, 10)

        # Amount vs average ratio
        amount_ratio = amount / (avg_transaction + 1)
        amount_anomaly = -2.0 if amount_ratio > 5 else \
                         -1.0 if amount_ratio > 3 else 0.1

        # Previous fraud
        fraud_history = {
            "No previous fraud": 0.1,
            "1 suspicious transaction before": -2.0,
            "Multiple fraud attempts": -5.0
        }[previous_fraud]

        # Night time risk (midnight to 5am = suspicious)
        night_risk = -2.5 if hour <= 5 else \
                     -1.0 if hour <= 7 else 0.1

        # Build 29-feature vector for model
        features = np.zeros(29)
        features[0]  = location_score        # maps to V1
        features[1]  = device_score          # maps to V2
        features[2]  = type_score            # maps to V3
        features[3]  = amount_anomaly        # maps to V4
        features[9]  = freq_score            # maps to V10
        features[10] = night_risk            # maps to V11
        features[11] = fraud_history         # maps to V12
        features[13] = location_score * 1.5  # maps to V14 (strongest)
        features[16] = device_score * 1.2    # maps to V17
        features[28] = norm_amount           # NormalizedAmount

        prediction = fraud_model.predict([features])[0]
        probability = fraud_model.predict_proba([features])[0][1]

        # ── Show Results ──
        st.markdown("---")
        st.subheader("🎯 Fraud Analysis Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{probability*100:.1f}%")
        col2.metric("Decision", "🚨 FRAUD" if prediction==1 else "✅ NORMAL")
        col3.metric("Amount", f"₹{amount:,.0f}")

        if prediction == 1:
            st.error("🚨 HIGH RISK: This transaction is likely FRAUDULENT!")
            st.markdown("**⚠️ Recommended Actions:**")
            st.markdown("- 🔒 Block this transaction immediately")
            st.markdown("- 📞 Call customer to verify")
            st.markdown("- 🔍 Flag account for review")
        else:
            st.success("✅ LOW RISK: This transaction appears LEGITIMATE")
            st.markdown("**✔️ Safe to proceed with transaction**")

        # Risk breakdown
        st.markdown("---")
        st.subheader("📊 Risk Factor Breakdown")
        
        risk_factors = {
            "Location Risk": abs(location_score),
            "Device Risk": abs(device_score),
            "Transaction Type": abs(type_score),
            "Night Time Risk": abs(night_risk),
            "Amount Anomaly": abs(amount_anomaly),
            "Fraud History": abs(fraud_history),
            "Frequency Risk": abs(freq_score)
        }
        
        for factor, score in sorted(
            risk_factors.items(), key=lambda x: x[1], reverse=True):
            level = "🔴" if score >= 3 else "🟡" if score >= 1.5 else "🟢"
            st.write(f"{level} **{factor}:** {score:.1f}/5.0")
# ─────────────────────────────────────────
# LOAN DEFAULT PAGE
# ─────────────────────────────────────────
elif module == "📋 Loan Default Prediction":
    st.header("📋 Loan Default Risk Assessor")
    st.markdown("Enter customer details to predict default risk")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Info")
        limit_bal = st.number_input("Credit Limit (₹)", 10000, 1000000, 50000, 10000)
        sex = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x==1 else "Female")
        education = st.selectbox("Education", [1,2,3,4], 
                    format_func=lambda x: {1:"Graduate",2:"University",3:"High School",4:"Others"}[x])
        marriage = st.selectbox("Marital Status", [1,2,3],
                    format_func=lambda x: {1:"Married",2:"Single",3:"Others"}[x])
        age = st.slider("Age", 18, 80, 30)

    with col2:
        st.subheader("Payment History")
        pay_0 = st.selectbox("Last Month Payment", [-1,0,1,2,3,4,5,6,7,8,9],
                    format_func=lambda x: "Paid on time" if x<=0 else f"{x} months delayed")
        pay_2 = st.selectbox("2 Months Ago", [-1,0,1,2,3],
                    format_func=lambda x: "Paid on time" if x<=0 else f"{x} months delayed")
        pay_3 = st.selectbox("3 Months Ago", [-1,0,1,2,3],
                    format_func=lambda x: "Paid on time" if x<=0 else f"{x} months delayed")
        bill_amt1 = st.number_input("Last Bill Amount (₹)", 0, 500000, 10000, 1000)
        bill_amt2 = st.number_input("Previous Bill Amount (₹)", 0, 500000, 10000, 1000)

    with col3:
        st.subheader("Payment Amounts")
        pay_amt1 = st.number_input("Last Payment Made (₹)", 0, 500000, 5000, 1000)
        pay_amt2 = st.number_input("Previous Payment Made (₹)", 0, 500000, 5000, 1000)
        pay_amt3 = st.number_input("Payment 3 Months Ago (₹)", 0, 500000, 5000, 1000)
        pay_4 = st.selectbox("4 Months Ago", [-1,0,1,2,3],
                    format_func=lambda x: "Paid on time" if x<=0 else f"{x} months delayed")
        pay_5 = st.selectbox("5 Months Ago", [-1,0,1,2,3],
                    format_func=lambda x: "Paid on time" if x<=0 else f"{x} months delayed")

    if st.button("🔍 Predict Default Risk", use_container_width=True):
        # Build full feature vector (23 features)
        input_data = pd.DataFrame([[
            limit_bal, sex, education, marriage, age,
            pay_0, pay_2, pay_3, pay_4, pay_5, 0,
            bill_amt1, bill_amt2, 0, 0, 0, 0,
            pay_amt1, pay_amt2, pay_amt3, 0, 0, 0
        ]], columns=loan_features)

        input_scaled = loan_scaler.transform(input_data)
        risk_score = loan_model.predict_proba(input_scaled)[0][1] * 100
        prediction = loan_model.predict(input_scaled)[0]

        st.markdown("---")
        st.subheader("🎯 Risk Assessment Result")

        if risk_score > 60:
            risk_level = "🔴 HIGH RISK"
            recommendation = "❌ Recommend REJECTING this loan application"
            st.error(f"{risk_level} — {recommendation}")
        elif risk_score > 30:
            risk_level = "🟡 MEDIUM RISK"
            recommendation = "⚠️ Recommend REVIEWING with additional documents"
            st.warning(f"{risk_level} — {recommendation}")
        else:
            risk_level = "🟢 LOW RISK"
            recommendation = "✅ Recommend APPROVING this loan application"
            st.success(f"{risk_level} — {recommendation}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Default Risk Score", f"{risk_score:.1f}%")
        col2.metric("Risk Level", risk_level.split()[1])
        col3.metric("Recommendation", "Approve" if risk_score < 30 else "Review/Reject")