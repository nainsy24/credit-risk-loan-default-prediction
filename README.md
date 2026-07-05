\#  RiskLens — Credit Risk Analysis \& Loan Default Prediction



> An end-to-end credit risk project: SQL-based analytics, machine learning, and a live interactive dashboard backed by a REST API.



\[!\[Streamlit App](https://static.streamlit.io/badges/streamlit\_badge\_black\_white.svg)](https://your-app.streamlit.app)

\&nbsp;

!\[Python](https://img.shields.io/badge/Python-3.10-blue)

!\[PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791)

!\[XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)

!\[FastAPI](https://img.shields.io/badge/API-FastAPI-green)



\---



\##  Live Demo



\*\*\[Open RiskLens Dashboard](https://your-app.streamlit.app)\*\*



> \*\*Note:\*\* The prediction API is hosted on Render's free tier and may take 30–60 seconds to wake up on first use after inactivity. This is expected — not a bug.



\---



\##  Dashboard Preview



\### Overview

!\[Overview](Creditrisk\_assets/dashboard\_overview.png)



\### Risk Analysis

!\[Risk Analysis](Creditrisk\_assets/dashboard\_risk.png)



\### Live Prediction

!\[Live Prediction](Creditrisk\_assets/dashboard\_prediction.png)



\---



\##  Project Overview



\*\*RiskLens\*\* analyzes \~100,000 sampled loans from the Lending Club dataset to understand what drives loan default, and predicts default risk on new, unseen loan applications in real time.



The project covers the \*\*full data science pipeline:\*\*



```

Raw Data → Cleaning → SQL Analytics → EDA → Feature Engineering

&#x20;      → Model Training → API Deployment → Live Dashboard

```



\---



\##  Key Findings



| Finding | Detail |

|---|---|

| Overall default rate | \~20% across completed loans |

| Safest loans (Grade A) | \~5% default rate |

| Riskiest loans (Grade G) | \~59% default rate |

| Top predictive features | Interest rate, Grade, Sub-grade, Term, DTI |

| Leakage detected \& removed | `last\_fico\_range\_low/high` inflated ROC-AUC from 0.96 → realistic 0.74 |



\---



\##  Visualizations



\### Default Rate by Grade

!\[Default Rate by Grade](Creditrisk\_assets/default\_rate\_by\_grade.png)



\### Default Rate by Loan Purpose

!\[Default Rate by Purpose](Creditrisk\_assets/default\_rate\_by\_purpose.png)



\### ROC Curve Comparison

!\[ROC Curve](Creditrisk\_assets/roc\_curve\_comparison.png)



\### Feature Importance (Random Forest)

!\[Feature Importance](Creditrisk\_assets/feature\_importance\_rf.png)



\### Correlation Heatmap

!\[Correlation Heatmap](Creditrisk\_assets/eda\_correlation\_heatmap.png)



\---



\##  Model Performance



| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |

|---|---|---|---|---|---|

| Logistic Regression | 0.68 | 0.35 | 0.65 | 0.45 | 0.74 |

| Random Forest | 0.69 | 0.35 | 0.62 | 0.45 | 0.73 |

| \*\*XGBoost \*\* | \*\*0.70\*\* | \*\*0.36\*\* | \*\*0.62\*\* | \*\*0.45\*\* | \*\*0.74\*\* |



\*\*XGBoost\*\* was selected as the final model. Given the class imbalance (\~20% defaults), all models were evaluated on precision, recall, F1, and ROC-AUC — not just accuracy, since accuracy alone is misleading on imbalanced datasets.



\*\*Key decision:\*\* Detected and removed post-origination leakage fields (`last\_fico\_range\_low`, `last\_fico\_range\_high`) which caused an artificially high ROC-AUC of 0.96. After removal, ROC-AUC settled at a realistic \*\*0.74\*\* — consistent with industry-standard credit risk models.



\---



\##  Project Pipeline



\### 1. Data Cleaning

\- Loaded 100,000 rows from the Lending Club accepted loans dataset

\- Fixed data types: percentage strings → float, date strings → datetime

\- Cleaned `emp\_length`, `term` fields with custom parsers

\- Filled missing values: median for numeric, "Unknown" for categorical



\### 2. SQL Analytics (PostgreSQL)

\- Default rate by grade, purpose, and state

\- Average interest rate trends by grade

\- \*\*Cohort/vintage analysis\*\* using SQL window functions — tracked how default rates shifted across loan origination years



\### 3. Exploratory Data Analysis

\- Distributions of key features (loan amount, interest rate, income, DTI)

\- Outlier detection via boxplots

\- Correlation heatmap vs default target

\- Default rate by income bracket



\### 4. Feature Engineering

\- Removed \*\*19 leakage columns\*\* (post-origination outcome fields)

\- Removed identifier/text columns

\- Label-encoded categorical features

\- 80/20 train/test stratified split



\### 5. Model Training \& Comparison

\- Logistic Regression (industry baseline, most interpretable)

\- Random Forest (captures non-linear interactions, gives feature importance)

\- XGBoost (best overall performance, industry-standard for tabular data)



\### 6. Deployment

\- Final XGBoost model saved as `.pkl` via `joblib`

\- Served via \*\*FastAPI\*\* REST endpoint (`/predict`)

\- \*\*Streamlit\*\* dashboard consumes the API for live predictions

\- Public deployment: Streamlit Cloud (frontend) + Render (API)



\---



\## 🛠️ Tech Stack



| Layer | Tools |

|---|---|

| Data processing | Python, pandas, numpy |

| Database | PostgreSQL, SQLAlchemy |

| Machine learning | scikit-learn, XGBoost |

| Visualization | matplotlib, seaborn |

| API | FastAPI, uvicorn |

| Dashboard | Streamlit |

| Deployment | Render (API), Streamlit Cloud (dashboard) |



\---



\##  Repository Structure



```

RiskLens/

├── Creditrisk\_assets/               # Charts and dashboard screenshots

├── creditrisk\_data\_cleaning.py      # Data loading and cleaning

├── creditrisk\_analysis.py           # Core SQL analytics queries

├── cohort\_vintage\_analysis.py       # SQL window functions - vintage analysis

├── eda\_analysis.py                  # Exploratory data analysis

├── feature\_engineering.py           # Leakage removal, encoding, train/test split

├── train\_models.py                  # Trains and compares 3 models

├── save\_final\_model.py              # Saves final XGBoost model as .pkl

├── save\_defaults.py                 # Computes realistic feature defaults

├── creditrisk\_api.py                # FastAPI model-serving endpoint

├── dashboard\_app.py                 # Streamlit dashboard

├── requirements.txt                 # Python dependencies

├── .gitignore

└── README.md

```



\---



\##  Running Locally



\### Prerequisites

\- Python 3.10+

\- PostgreSQL installed and running

\- Lending Club dataset (accepted loans CSV)



\### Setup



```bash

\# 1. Clone the repo

git clone https://github.com/nainsy24/credit-risk-loan-default-prediction.git

cd credit-risk-loan-default-prediction



\# 2. Install dependencies

pip install -r requirements.txt



\# 3. Set database password as environment variable (Windows)

setx DB\_PASSWORD "your\_postgres\_password"

\# Restart terminal after this



\# 4. Run pipeline scripts in order

python creditrisk\_data\_cleaning.py

python load\_to\_postgres.py

python creditrisk\_analysis.py

python cohort\_vintage\_analysis.py

python eda\_analysis.py

python feature\_engineering.py

python train\_models.py

python save\_final\_model.py

python save\_defaults.py



\# 5. Start the API (Terminal 1)

uvicorn creditrisk\_api:app --reload



\# 6. Start the dashboard (Terminal 2)

streamlit run dashboard\_app.py

```



\---



\##  Notes \& Limitations



\- \*\*Loan amount capped at $40,000\*\* and \*\*term restricted to 36/60 months\*\* — these are real Lending Club product constraints, not arbitrary limits in this analysis

\- Dataset is a \*\*100,000-row sample\*\* chosen to fit local hardware constraints (8GB RAM)

\- \*\*Precision on the default class is moderate (\~0.35)\*\* — the model prioritizes catching more true defaults (recall \~0.62) at the cost of some false positives, which is the standard trade-off in credit risk scoring

\- The API hosted on Render free tier \*\*sleeps after inactivity\*\* — first prediction after idle may take 30–60 seconds



\---



\##  Author



\*\*Nainsy\*\* — \[GitHub @nainsy24](https://github.com/nainsy24)



\---



\*Built as a portfolio project demonstrating end-to-end data science and analytics skills: SQL, Python, machine learning, API development, and dashboard deployment.\*

