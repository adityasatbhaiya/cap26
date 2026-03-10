import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Credit Card Default Risk Dashboard",
    layout="wide"
)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("default of credit card clients (1).csv")

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    return df


df = load_data()

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model


model = load_model()

# ------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------

bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
pay_amt_cols = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
pay_status_cols = ['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']

df_model = df.copy()

df_model['AVG_BILL_AMT'] = df_model[bill_cols].mean(axis=1)
df_model['AVG_PAY_AMT'] = df_model[pay_amt_cols].mean(axis=1)
df_model['UTILIZATION'] = df_model['AVG_BILL_AMT'] / (df_model['LIMIT_BAL'] + 1)
df_model['AVG_PAY_STATUS'] = df_model[pay_status_cols].mean(axis=1)
df_model['MAX_PAY_DELAY'] = df_model[pay_status_cols].max(axis=1)
df_model['TOTAL_BILL'] = df_model[bill_cols].sum(axis=1)
df_model['TOTAL_PAY'] = df_model[pay_amt_cols].sum(axis=1)
df_model['PAY_RATIO'] = df_model['TOTAL_PAY'] / (df_model['TOTAL_BILL'] + 1)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["EDA Dashboard", "Prediction"]
)

# =================================================
# PAGE 1 — EDA DASHBOARD
# =================================================

if page == "EDA Dashboard":

    st.title("💳 Credit Card Default Risk Analysis Dashboard")

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Total Features", df.shape[1]-1)
    col3.metric("Target Variable", "dpnm")

    st.dataframe(df.head())

    # Default distribution

    st.header("Default Distribution")

    fig, axes = plt.subplots(1,2, figsize=(12,5))

    vc = df["dpnm"].value_counts()

    axes[0].bar(["No Default","Default"], vc.values)
    axes[0].set_title("Default Distribution")

    axes[1].pie(vc.values, labels=["No Default","Default"], autopct="%1.1f%%")

    st.pyplot(fig)

    # Credit limit

    st.header("Credit Limit Distribution")

    fig, ax = plt.subplots()

    df[df["dpnm"]==0]["LIMIT_BAL"].hist(bins=40, alpha=0.6, label="No Default", ax=ax)
    df[df["dpnm"]==1]["LIMIT_BAL"].hist(bins=40, alpha=0.6, label="Default", ax=ax)

    ax.legend()

    st.pyplot(fig)

    # Payment delays

    st.header("Payment Delay Analysis")

    pay_cols = ["PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

    fig, axes = plt.subplots(2,3, figsize=(14,8))

    for ax,col in zip(axes.flatten(),pay_cols):

        default_rate = df.groupby(col)["dpnm"].mean()*100

        ax.bar(default_rate.index.astype(str), default_rate.values)

        ax.set_title(col)

    st.pyplot(fig)

    # Correlation heatmap

    st.header("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["int64","float64"])

    corr_matrix = numeric_df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(14,10))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdYlGn",
        center=0,
        ax=ax
    )

    st.pyplot(fig)

# =================================================
# PAGE 2 — PREDICTION
# =================================================

if page == "Prediction":

    st.title("🔮 Credit Default Prediction")

    st.write("Enter financial behaviour metrics to estimate default probability.")

    AVG_BILL_AMT = st.number_input("Average Bill Amount", value=50000.0)
    AVG_PAY_AMT = st.number_input("Average Payment Amount", value=20000.0)
    UTILIZATION = st.number_input("Credit Utilization", value=0.3)
    AVG_PAY_STATUS = st.number_input("Average Payment Status", value=0.0)
    MAX_PAY_DELAY = st.number_input("Maximum Payment Delay", value=0.0)
    TOTAL_BILL = st.number_input("Total Bill Amount", value=200000.0)
    TOTAL_PAY = st.number_input("Total Payment Amount", value=120000.0)
    PAY_RATIO = st.number_input("Payment Ratio", value=0.5)

    if st.button("Predict Default Risk"):

        input_df = pd.DataFrame({
            'AVG_BILL_AMT':[AVG_BILL_AMT],
            'AVG_PAY_AMT':[AVG_PAY_AMT],
            'UTILIZATION':[UTILIZATION],
            'AVG_PAY_STATUS':[AVG_PAY_STATUS],
            'MAX_PAY_DELAY':[MAX_PAY_DELAY],
            'TOTAL_BILL':[TOTAL_BILL],
            'TOTAL_PAY':[TOTAL_PAY],
            'PAY_RATIO':[PAY_RATIO]
        })

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Risk of Default\nProbability: {probability:.2%}")
        else:
            st.success(f"✅ Low Risk of Default\nProbability: {probability:.2%}")
