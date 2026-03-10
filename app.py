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
The objective is to understand demographic and financial factors contributing to **credit card default risk**.
""")

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("default of credit card clients (1).csv")

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    return df

df = load_data()

# ----------------------------------------------------
# Dataset Overview
# ----------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{df.shape[0]:,}")
col2.metric("Total Features", df.shape[1]-1)
col3.metric("Target Variable", "dpnm")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ----------------------------------------------------
# Target Distribution
# ----------------------------------------------------

st.header("🎯 Default Distribution")

fig, axes = plt.subplots(1,2, figsize=(12,5))

colors = ["#2ecc71","#e74c3c"]

vc = df["dpnm"].value_counts()

axes[0].bar(["No Default","Default"], vc.values, color=colors)
axes[0].set_title("Credit Card Default Distribution")
axes[0].set_ylabel("Count")

axes[1].pie(
    vc.values,
    labels=["No Default","Default"],
    autopct="%1.1f%%",
    colors=colors
)

axes[1].set_title("Default Proportion")

st.pyplot(fig)

st.info("⚠️ Class imbalance exists (~78% No Default vs ~22% Default)")

# ----------------------------------------------------
# Demographic Analysis
# ----------------------------------------------------

st.header("👥 Demographic Analysis")

df["SEX_label"] = df["SEX"].map({1:"Male",2:"Female"})

gender_default = df.groupby("SEX_label")["dpnm"].mean()*100

fig, ax = plt.subplots()

ax.bar(gender_default.index, gender_default.values)

ax.set_title("Default Rate by Gender")
ax.set_ylabel("Default Rate (%)")

st.pyplot(fig)

# ----------------------------------------------------
# Credit Limit Distribution
# ----------------------------------------------------

st.header("💰 Credit Limit Distribution")

fig, ax = plt.subplots()

df[df["dpnm"]==0]["LIMIT_BAL"].hist(
    bins=40,
    alpha=0.6,
    label="No Default",
    ax=ax
)

df[df["dpnm"]==1]["LIMIT_BAL"].hist(
    bins=40,
    alpha=0.6,
    label="Default",
    ax=ax
)

ax.legend()

ax.set_title("Credit Limit Distribution by Default")

st.pyplot(fig)

# ----------------------------------------------------
# Payment History Analysis
# ----------------------------------------------------

st.header("📅 Payment Delay vs Default")

pay_cols = ["PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

fig, axes = plt.subplots(2,3, figsize=(14,8))

for ax,col in zip(axes.flatten(),pay_cols):

    default_rate = df.groupby(col)["dpnm"].mean()*100

    ax.bar(default_rate.index.astype(str), default_rate.values)

    ax.set_title(col)

    ax.set_xlabel("Payment Status")
    ax.set_ylabel("Default Rate (%)")

st.pyplot(fig)

st.success("""
Key Insight: Customers with **recent payment delays (PAY_1)** have significantly higher default probability.
""")

# ----------------------------------------------------
# Age Distribution
# ----------------------------------------------------

st.header("🎂 Age Distribution")

fig, axes = plt.subplots(1,2, figsize=(12,5))

df[df["dpnm"]==0]["AGE"].hist(
    bins=30,
    alpha=0.6,
    label="No Default",
    ax=axes[0]
)

df[df["dpnm"]==1]["AGE"].hist(
    bins=30,
    alpha=0.6,
    label="Default",
    ax=axes[0]
)

axes[0].legend()
axes[0].set_title("Age Distribution by Default")

sns.boxplot(
    x="dpnm",
    y="AGE",
    data=df,
    ax=axes[1]
)

axes[1].set_title("Age Boxplot by Default")

st.pyplot(fig)

# ----------------------------------------------------
# Correlation Heatmap (FIXED)
# ----------------------------------------------------

st.header("🔥 Correlation Heatmap")

# Select only numeric columns to avoid errors
numeric_df = df.select_dtypes(include=["int64","float64"])

corr_matrix = numeric_df.corr()

# Mask upper triangle for cleaner heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(14,10))

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="RdYlGn",
    center=0,
    ax=ax
)

ax.set_title("Feature Correlation Heatmap")

st.pyplot(fig)

# ----------------------------------------------------
# Business Insights
# ----------------------------------------------------

st.header("💡 Business Insights")

st.markdown("""

### Key Risk Drivers

**Payment Behaviour**

• PAY_1 (recent payment delay) is the strongest predictor of default  
• Multiple delayed payments significantly increase risk  

**Credit Utilization**

• Customers with higher outstanding balances relative to their credit limit show higher risk  

**Demographics**

• Younger customers show slightly higher default probability  

---

### Business Recommendations

1️⃣ Implement **early warning systems for payment delays**  

2️⃣ Monitor **credit utilization ratios** for high-risk customers  

3️⃣ Introduce **risk-based credit limits and monitoring**

""")

st.success("✅ Dashboard Loaded Successfully")
