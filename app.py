import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Credit Card Default Risk Dashboard",
    layout="wide"
)

st.title("💳 Credit Card Default Risk Analysis Dashboard")

st.markdown("""
This dashboard presents **Exploratory Data Analysis (EDA)** for the Credit Card Default dataset.

Objective:
Identify demographic and financial patterns associated with **credit card default risk**.
""")

# -------------------------------------------------------
# Load Dataset from GitHub Repository
# -------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("credit_card_default.csv")

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    return df

df = load_data()

# -------------------------------------------------------
# Dataset Overview
# -------------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{df.shape[0]:,}")
col2.metric("Total Features", df.shape[1]-1)
col3.metric("Target Variable", "dpnm")

st.subheader("Dataset Preview")

st.dataframe(df.head())

# -------------------------------------------------------
# Target Distribution
# -------------------------------------------------------

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

# -------------------------------------------------------
# Demographic Analysis
# -------------------------------------------------------

st.header("👥 Demographic Factors vs Default")

df["SEX_label"] = df["SEX"].map({1:"Male",2:"Female"})
df["EDUCATION_label"] = df["EDUCATION"].map({
1:"Grad School",
2:"University",
3:"High School",
4:"Others"
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

# -------------------------------------------------------
# Credit Limit Analysis
# -------------------------------------------------------

st.header("💰 Credit Limit Distribution")

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

ax.set_title("Credit Limit Distribution by Default")

ax.set_xlabel("Credit Limit")

st.pyplot(fig)

# -------------------------------------------------------
# Payment History
# -------------------------------------------------------

st.header("📅 Payment Delay Analysis")

pay_cols = ["PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

fig, axes = plt.subplots(2,3, figsize=(14,8))

for ax,col in zip(axes.flatten(),pay_cols):

    default_rate = df.groupby(col)["dpnm"].mean()*100

    ax.bar(default_rate.index.astype(str),default_rate.values)

    ax.set_title(f"{col} vs Default Rate")

    ax.set_xlabel("Payment Status")

    ax.set_ylabel("Default Rate (%)")

st.pyplot(fig)

st.success(
"""
Key Insight:

Customers with **recent payment delays (PAY_1)** show the **highest probability of default**.
"""
)

# -------------------------------------------------------
# Age Analysis
# -------------------------------------------------------

st.header("🎂 Age Distribution")

fig, axes = plt.subplots(1,2, figsize=(12,5))

df[df["dpnm"]==0]["AGE"].plot(
kind="hist",
bins=30,
alpha=0.6,
label="No Default",
ax=axes[0]
)

df[df["dpnm"]==1]["AGE"].plot(
kind="hist",
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

# -------------------------------------------------------
# Correlation Heatmap
# -------------------------------------------------------

st.header("🔥 Feature Correlation Heatmap")

corr = df.corr()

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(
corr,
cmap="RdYlGn",
center=0,
ax=ax
)

ax.set_title("Feature Correlation")

st.pyplot(fig)

# -------------------------------------------------------
# Business Insights
# -------------------------------------------------------

st.header("💡 Business Insights")

st.markdown("""

### Key Risk Drivers

**Payment Behaviour**

• PAY_1 (most recent payment delay) is the strongest predictor of default  
• Multiple delayed payments dramatically increase risk  

**Credit Utilization**

• Customers with higher outstanding balances relative to their credit limit are more likely to default  

**Demographics**

• Younger customers (21–30) show higher default rates  
• Lower education levels correlate with higher credit risk  

---

### Recommendations

**1️⃣ Early Warning System**

Flag customers with **PAY_1 ≥ 1** for proactive outreach.

**2️⃣ Credit Limit Monitoring**

Adjust limits based on **utilization ratio and payment behaviour**.

**3️⃣ Risk-Based Monitoring**

Track customers with **2+ delayed payments across months**.

**4️⃣ Financial Literacy Programs**

Focus on younger and lower education customer segments.

""")

st.success("EDA Dashboard Successfully Loaded")
