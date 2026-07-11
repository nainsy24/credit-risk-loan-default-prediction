import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
import os
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Global Finance | Credit Risk", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
# We use custom CSS to override Streamlit's default components and inject our Google Font
# for a modern, sleek presentation aesthetic.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background-color: #ffffff;
    }
    
    h1, h2, h3 {
        color: #1f2937 !important;
        font-weight: 800 !important;
    }
    
    /* Hero Banner styling */
    .hero-container {
        padding: 3rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        text-align: left;
        margin-bottom: 2rem;
        border-left: 10px solid #1ebc3c;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .hero-title span {
        color: #1ebc3c;
    }
    .hero-subtitle {
        font-size: 1.8rem;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 0;
    }
    
    /* Metric styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        border-bottom: 4px solid #facc15;
    }
    [data-testid="stMetricValue"] {
        color: #1ebc3c !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #1ebc3c;
        color: #ffffff;
        border-radius: 30px;
        font-weight: 600;
        padding: 0.6em 2em;
        border: none;
        box-shadow: 0 4px 10px rgba(30, 188, 60, 0.3);
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1aa032;
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(30, 188, 60, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Inputs */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 8px !important;
        border: 1px solid #d1d5db !important;
        background-color: #f9fafb !important;
    }
    
    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# --- MATPLOTLIB STYLING (Light Theme) ---
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#e5e7eb",
    "axes.labelcolor": "#374151",
    "text.color": "#1f2937",
    "xtick.color": "#4b5563",
    "ytick.color": "#4b5563",
    "grid.color": "#e5e7eb",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
})

# --- DATA LOADING ---
@st.cache_data
def load_metrics():
    return pd.read_csv("model_comparison_results.csv", index_col=0)

@st.cache_data
def load_data():
    df = pd.read_csv("loans_for_dashboard.csv", low_memory=False)
    df["is_default"] = df["loan_status"].apply(lambda x: 1 if x in ["Charged Off", "Default"] else 0)
    return df

metrics_df = load_metrics()
df = load_data()

# --- SIDEBAR NAV ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/finance-document.png", width=60)
    st.title("Global Finance")
    st.markdown("**Credit Risk Analysis System**")
    st.markdown("---")
    
    page = option_menu(
        menu_title=None, 
        options=["Overview", "Exploratory Analysis", "Risk Analysis", "Model Performance", "Live Prediction"],
        icons=["house", "bar-chart-line", "shield-check", "graph-up-arrow", "calculator"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f8f9fa", "border": "none"},
            "icon": {"color": "#1ebc3c", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#e9ecef",
                "color": "#1f2937",
                "font-weight": "600",
                "border-radius": "10px"
            },
            "nav-link-selected": {
                "background-color": "#1ebc3c", 
                "color": "white", 
                "font-weight": "800",
                "border-radius": "10px"
            },
        }
    )
    
    st.markdown("---")
    st.markdown("⚙️ **Advanced Settings**")
    api_url = st.text_input("Prediction API URL", "https://credit-risk-loan-default-prediction-omsx.onrender.com/predict", help="Change to http://127.0.0.1:8000/predict for local testing.")

# --- PAGE ROUTING ---
if page == "Overview":
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Welcome to <span>Global Finance</span></h1>
        <p class="hero-subtitle">Make Your Ideas Come True!</p>
        <p style="margin-top:1rem; color: #4b5563;">Advanced Credit Risk Assessment and Default Prediction System.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Portfolio Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Loans Analyzed", f"{len(df):,}")
    col2.metric("Overall Default Rate", f"{df['is_default'].mean()*100:.1f}%")
    col3.metric("Best Model ROC-AUC", f"{metrics_df['roc_auc'].max():.3f}")
    
    st.markdown("---")
    st.markdown("### Executive Summary")
    st.markdown("""
    This platform analyzes historical loan data to uncover the primary drivers of loan default and leverages an advanced machine learning model (XGBoost) to predict default risk for new applicants.
    
    **Technology Stack:**
    - Python (Pandas, Scikit-Learn, XGBoost)
    - Streamlit (Web App & Visualizations)
    """)

elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("Understanding the distribution of our borrower profiles.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Interest Rate Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df["int_rate"], bins=40, kde=True, ax=ax, color="#1ebc3c")
        plt.xlabel("Interest Rate (%)")
        plt.ylabel("Frequency")
        st.pyplot(fig)
        
    with col2:
        st.markdown("#### Annual Income Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df["annual_inc"].clip(upper=300000), bins=40, kde=True, ax=ax, color="#facc15")
        plt.xlabel("Annual Income ($)")
        plt.ylabel("Frequency")
        st.pyplot(fig)
        
    st.markdown("---")
    st.markdown("### Feature Correlation Heatmap")
    num_cols = ["loan_amnt", "int_rate", "annual_inc", "dti", "installment", "is_default"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="Greens", fmt=".2f", ax=ax, 
                linewidths=.5, cbar_kws={"shrink": .8})
    st.pyplot(fig)

elif page == "Risk Analysis":
    st.title("Risk Analysis by Segment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        grade_default = df.groupby("grade")["is_default"].mean().sort_index() * 100
        fig, ax = plt.subplots(figsize=(6,5))
        grade_default.plot(kind="bar", ax=ax, color="#1ebc3c")
        ax.set_title("Default Rate by Credit Grade", fontweight='bold')
        ax.set_ylabel("Default Rate (%)")
        ax.set_xlabel("Lending Club Grade")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        
    with col2:
        purpose_default = df.groupby("purpose")["is_default"].mean().sort_values(ascending=True) * 100
        fig, ax = plt.subplots(figsize=(6,5))
        purpose_default.plot(kind="barh", ax=ax, color="#facc15")
        ax.set_title("Default Rate by Loan Purpose", fontweight='bold')
        ax.set_xlabel("Default Rate (%)")
        ax.set_ylabel("")
        st.pyplot(fig)

elif page == "Model Performance":
    st.title("Model Performance Comparison")
    
    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['roc_auc', 'accuracy', 'f1'], color='#dcfce7'))
    
    st.info("""
    **Note on Data Integrity:** `last_fico_range_low/high` and other post-origination fields
    were identified as data leakage and removed. This brought our ROC-AUC from an inflated 0.96
    to a realistic ~0.74, which is highly consistent with industry-standard credit risk models.
    """, icon="ℹ️")

elif page == "Live Prediction":
    st.title("Live Risk Assessment")
    st.markdown("Enter the applicant's loan details to instantly assess default risk using our trained XGBoost model.")

    with st.expander("📖 Glossary & Guidelines"):
        st.markdown("""
        - **Loan Amount** — Total dollar amount borrowed. Cap: **$40,000**.
        - **Term** — Repayment duration. Options: **36 or 60 months**.
        - **Interest Rate (%)** — Annual interest rate charged.
        - **Debt-to-Income Ratio (DTI)** — Monthly debt payments divided by monthly income (%).
        - **Annual Income** — Applicant's self-reported yearly income.
        - **Grade** — Internal risk grade (A = Safest, G = Riskiest).
        """)

    st.markdown("### Applicant Details")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            loan_amnt = st.number_input("💰 Loan Amount ($)", min_value=500.0, max_value=40000.0, value=10000.0)
            annual_inc = st.number_input("🏦 Annual Income ($)", min_value=0.0, value=65000.0)
        with col2:
            int_rate = st.number_input("📈 Interest Rate (%)", min_value=1.0, max_value=40.0, value=11.5, step=0.1)
            dti = st.number_input("⚖️ DTI Ratio (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        with col3:
            term = st.selectbox("📅 Term (months)", [36, 60])
            grade = st.selectbox("🏷️ Grade", ["A", "B", "C", "D", "E", "F", "G"])

    st.markdown("---")
    
    if st.button("🚀 Assess Risk Profile"):
        with st.spinner("Connecting to FastAPI backend..."):
            payload = {
                "loan_amnt": loan_amnt,
                "int_rate": int_rate,
                "dti": dti,
                "annual_inc": annual_inc,
                "term": term,
                "grade": grade
            }
            
            try:
                # We give the free-tier Render API 15 seconds to wake up
                response = requests.post(api_url, json=payload, timeout=15)
                response.raise_for_status()
                result = response.json()
                
                st.markdown("### Assessment Result")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric("Predicted Default Probability", f"{result['default_probability']*100:.2f}%")
                    
                with res_col2:
                    if result["risk_label"] == "High Risk":
                        st.error("⚠️ **High Risk Profile** - Manual review recommended.")
                    else:
                        st.success("✅ **Low Risk Profile** - Candidate for automatic approval.")
                        
            except requests.exceptions.Timeout:
                st.error("⏳ **API Timeout!** The Render prediction server is currently asleep and taking too long to wake up.")
                st.info("💡 **Pro-Tip**: During an interview, change the API URL in the sidebar to `http://127.0.0.1:8000/predict` and run `uvicorn creditrisk_api:app` locally for instant predictions!")
            except requests.exceptions.ConnectionError:
                st.error("🔌 **Connection Error!** Could not reach the API.")
                st.info("💡 **Pro-Tip**: Make sure your FastAPI server is running. Type `uvicorn creditrisk_api:app --reload` in your terminal.")
            except Exception as e:
                st.error(f"⚠️ **Prediction failed**: {e}")