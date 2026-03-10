import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credit Card Default Risk Dashboard",
    layout="wide"
)

st.title("💳 Credit Card Default Risk Analysis Dashboard")

st.markdown("""
This dashboard presents **Exploratory Data Analysis (EDA)** for the Credit Card Default dataset.  
The objective is to understand the **demographic and financial factors influencing default risk**.
""")

# -------------------------
# Load Dataset
# -------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("default of credit card clients (1).csv")
    df.drop(columns=["ID"], inplace=True)
    return df

df = load_data()

# -------------------------
# Dataset Overview
# -------------------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{df.shape[0]:,}")
col2.metric("Total Features", df.shape[1]-1)
col3.metric("Target Variable", "dpnm")

st.write("Preview of Dataset")
st.dataframe(df.head())

# -------------------------
# Target Distribution
# -------------------------

st.header("🎯 Target Variable Distribution")

fig, ax = plt.subplots()

colors = ["#2ecc71", "#e74c3c"]

vc = df["dpnm"].value_counts()

ax.bar(["No Default", "Default"], vc.values, color=colors)
ax.set_title("Credit Card Default Distribution")
ax.set_ylabel("Count")

st.pyplot(fig)

st.info("⚠️ Class imbalance exists: ~78% Non-Default vs ~22% Default")

# -------------------------
# Demographic Analysis
# -------------------------

st.header("👥 Demographic Factors vs Default")

df["SEX_label"] = df["SEX"].map({1: "Male", 2: "Female"})
df["EDUCATION_label"] = df["EDUCATION"].map({
    1: "Grad School",
    2: "University",
    3: "High School",
    4: "Others"
})

fig, axes = plt.subplots(1,2, figsize=(12,5))

gender_default = df.groupby("SEX_label")["dpnm"].mean()*100
axes[0].bar(gender_default.index, gender_default.values)
axes[0].set_title("Default Rate by Gender")
axes[0].set_ylabel("Default Rate (%)")

edu_default = df.groupby("EDUCATION_label")["dpnm"].mean()*100
axes[1].bar(edu_default.index, edu_default.values)
axes[1].set_title("Default Rate by Education")

st.pyplot(fig)

# -------------------------
# Financial Features
# -------------------------

st.header("💰 Credit Limit vs Default")

fig, ax = plt.subplots()

df[df["dpnm"]==0]["LIMIT_BAL"].plot(
    kind="hist",
    bins=40,
    alpha=0.6,
    label="No Default",
    ax=ax
)

df[df["dpnm"]==1]["LIMIT_BAL"].plot(
    kind="hist",
    bins=40,
    alpha=0.6,
    label="Default",
    ax=ax
)

ax.legend()
ax.set_title("Credit Limit Distribution by Default Status")

st.pyplot(fig)

# -------------------------
# Payment History
# -------------------------

st.header("📅 Payment Delay vs Default")

pay_cols = ["PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

fig, axes = plt.subplots(2,3, figsize=(14,8))

for ax, col in zip(axes.flatten(), pay_cols):

    default_rate = df.groupby(col)["dpnm"].mean()*100

    ax.bar(default_rate.index.astype(str), default_rate.values)
    ax.set_title(f"{col} vs Default Rate")

st.pyplot(fig)

st.success(
"""
Key Insight:  
Customers with **recent payment delays (PAY_1)** show the **highest probability of default**.
"""
)

# -------------------------
# Age Distribution
# -------------------------

st.header("🎂 Age Distribution")

fig, ax = plt.subplots()

df[df["dpnm"]==0]["AGE"].plot(
    kind="hist",
    bins=30,
    alpha=0.6,
    label="No Default",
    ax=ax
)

df[df["dpnm"]==1]["AGE"].plot(
    kind="hist",
    bins=30,
    alpha=0.6,
    label="Default",
    ax=ax
)

ax.legend()
ax.set_title("Age Distribution by Default")

st.pyplot(fig)

# -------------------------
# Correlation Heatmap
# -------------------------

st.header("🔥 Feature Correlation Heatmap")

corr = df.corr()

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(
    corr,
    cmap="RdYlGn",
    center=0,
    ax=ax
)

st.pyplot(fig)

# -------------------------
# Business Insights
# -------------------------

st.header("💡 Business Insights")

st.markdown("""
### Key Drivers of Credit Card Default

**Payment Behavior**
- PAY_1 (recent payment delay) is the strongest predictor
- Multiple delayed payments significantly increase default risk

**Credit Utilization**
- Customers with high outstanding balances relative to credit limit show higher risk

**Demographics**
- Younger customers (21–30) tend to default more frequently
- Lower education levels correlate with higher default probability

### Recommendations

**1. Early Warning System**
Flag customers with **PAY_1 ≥ 1** for proactive outreach.

**2. Credit Limit Monitoring**
Reduce exposure for customers with **high utilization ratios**.

**3. Payment Behavior Tracking**
Trigger alerts after **two consecutive delayed payments**.

**4. Risk-Based Pricing**
Adjust interest rates or credit limits based on predicted risk segments.
""")

st.success("Dashboard Completed: Exploratory Data Analysis for Credit Default Risk")
