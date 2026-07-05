import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Risk Analysis", layout="wide")

COLORS = {
    "bg_dark": "#0c0910",
    "brown_dark": "#401f12",
    "navy": "#1e2b47",
    "tan": "#88644a",
    "tan_light": "#a9897a",
    "grey_light": "#c8c0c0",
}

st.markdown(f"""
<style>
    .stApp {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['grey_light']};
    }}
    [data-testid="stSidebar"] {{
        background-color: {COLORS['brown_dark']};
    }}
    [data-testid="stSidebar"] * {{
        color: {COLORS['grey_light']} !important;
    }}
    h1, h2, h3 {{
        color: {COLORS['tan_light']} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {COLORS['tan']} !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLORS['grey_light']} !important;
    }}
    .stButton button {{
        background-color: {COLORS['navy']};
        color: {COLORS['grey_light']};
        border: 1px solid {COLORS['tan']};
    }}
    .stButton button:hover {{
        background-color: {COLORS['tan']};
        color: {COLORS['bg_dark']};
    }}
    [data-testid="stDataFrame"] {{
        background-color: {COLORS['brown_dark']};
    }}
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
        border-radius: 10px !important;
        border: 1px solid {COLORS['tan']} !important;
    }}
    .stSlider [data-baseweb="slider"] {{
        padding-top: 8px;
    }}
    div[data-testid="stExpander"] {{
        background-color: {COLORS['brown_dark']};
        border-radius: 10px;
        border: 1px solid {COLORS['tan']};
    }}
    div[data-testid="stMetric"] {{
        background-color: {COLORS['brown_dark']};
        border-radius: 12px;
        padding: 15px;
        border: 1px solid {COLORS['tan']};
    }}
    .stButton button {{
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5em 1.5em;
    }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": COLORS["bg_dark"],
    "axes.facecolor": COLORS["bg_dark"],
    "axes.edgecolor": COLORS["grey_light"],
    "axes.labelcolor": COLORS["grey_light"],
    "text.color": COLORS["grey_light"],
    "xtick.color": COLORS["grey_light"],
    "ytick.color": COLORS["grey_light"],
    "grid.color": COLORS["tan_light"],
    "grid.alpha": 0.2,
})

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

st.sidebar.title("Credit Risk Project")
page = st.sidebar.radio("Navigate", ["Overview", "EDA", "Risk Analysis", "Model Performance", "Live Prediction"])

if page == "Overview":
    st.title("Lending Club Credit Risk Analysis")
    st.markdown("""
    This project analyzes Lending Club loan data to understand what drives loan default
    and builds a machine learning model to predict default risk.
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Loans Analyzed", f"{len(df):,}")
    col2.metric("Overall Default Rate", f"{df['is_default'].mean()*100:.1f}%")
    col3.metric("Best Model ROC-AUC", f"{metrics_df['roc_auc'].max():.3f}")
    st.markdown("### Tools used")
    st.write("Python (pandas, scikit-learn, XGBoost) | PostgreSQL | Streamlit")

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df["int_rate"], bins=40, kde=True, ax=ax, color=COLORS["tan"])
        ax.set_title("Interest Rate Distribution")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(df["annual_inc"].clip(upper=300000), bins=40, kde=True, ax=ax, color=COLORS["navy"])
        ax.set_title("Annual Income Distribution (clipped at 300k)")
        st.pyplot(fig)
    st.markdown("### Correlation Heatmap")
    num_cols = ["loan_amnt", "int_rate", "annual_inc", "dti", "installment", "is_default"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="copper", fmt=".2f", ax=ax)
    st.pyplot(fig)

elif page == "Risk Analysis":
    st.title("Risk Analysis by Segment")
    grade_default = df.groupby("grade")["is_default"].mean().sort_index() * 100
    fig, ax = plt.subplots()
    grade_default.plot(kind="bar", ax=ax, color=COLORS["tan"])
    ax.set_title("Default Rate by Grade")
    ax.set_ylabel("Default Rate (%)")
    st.pyplot(fig)
    purpose_default = df.groupby("purpose")["is_default"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    purpose_default.plot(kind="barh", ax=ax, color=COLORS["navy"])
    ax.set_title("Default Rate by Purpose")
    ax.set_xlabel("Default Rate (%)")
    st.pyplot(fig)

elif page == "Model Performance":
    st.title("Model Performance Comparison")
    st.dataframe(metrics_df)
    st.markdown("""
    **Note on leakage:** `last_fico_range_low/high` and other post-origination fields
    were identified as data leakage and removed, dropping ROC-AUC from an inflated 0.96
    to a realistic ~0.74 - consistent with industry-standard credit risk models.
    """)

elif page == "Live Prediction":
    st.title("🔮 Predict Default Risk")
    st.write("Enter loan details to get a live default risk prediction from the trained XGBoost model.")

    with st.expander("📖 What do these terms mean? (Glossary)"):
        st.markdown("""
        - **Loan Amount** — total dollar amount borrowed. Lending Club capped personal loans at **$40,000**, so that's the real-world ceiling here, not an arbitrary limit.
        - **Term** — how many months the borrower has to repay. Lending Club only offered **36 or 60 month** terms as a fixed product choice — no other options existed.
        - **Interest Rate (%)** — the annual interest rate charged on the loan, mainly driven by the borrower's credit grade.
        - **Debt-to-Income Ratio (DTI)** — the borrower's total monthly debt payments divided by monthly income, expressed as a percentage. Higher DTI = more of their income is already committed to debt, which usually signals higher risk.
        - **Annual Income** — the borrower's self-reported yearly income.
        - **Grade** — Lending Club's own risk grade (A = safest, G = riskiest), which already reflects their internal credit assessment.
        """)

    st.markdown("### Loan Details")
    col1, col2 = st.columns(2)
    with col1:
        loan_amnt = st.number_input(
            "💰 Loan Amount ($)", min_value=500, max_value=40000, value=10000,
            help="Lending Club's max personal loan amount is $40,000 — this reflects their real product cap."
        )
        int_rate = st.slider(
            "📈 Interest Rate (%)", 5.0, 30.0, 12.0,
            help="Annual interest rate charged on the loan."
        )
        dti = st.slider(
            "⚖️ Debt-to-Income Ratio (%)", 0.0, 40.0, 18.0,
            help="Borrower's monthly debt payments as a percentage of monthly income."
        )
        annual_inc = st.number_input(
            "🏦 Annual Income ($)", min_value=0, value=60000,
            help="Borrower's self-reported yearly income."
        )
    with col2:
        term = st.selectbox(
            "📅 Term (months)", [36, 60],
            help="Lending Club only offers 36 or 60 month repayment terms — there's no other option in their product."
        )
        grade = st.selectbox(
            "🏷️ Grade", ["A", "B", "C", "D", "E", "F", "G"],
            help="Lending Club's internal risk grade — A is safest, G is riskiest."
        )

    st.markdown("---")
    API_URL = "http://127.0.0.1:8000/predict"

    if st.button("🚀 Predict Default Risk"):
        payload = {
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "dti": dti,
            "annual_inc": annual_inc,
            "term": term,
            "grade": grade
        }
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()

            st.metric("Predicted Default Probability", f"{result['default_probability']*100:.1f}%")
            if result["risk_label"] == "High Risk":
                st.error("High risk of default")
            else:
                st.success("Low risk of default")

        except requests.exceptions.ConnectionError:
            st.error("⚠️ Could not reach the prediction API. Make sure it's running: `uvicorn api:app --reload`")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")