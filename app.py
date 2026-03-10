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
""")

# ---------------------------------------------------
# LOAD DATA FROM GITHUB RAW LINK
# ---------------------------------------------------

@st.cache_data
def load_data():

    url = "https://raw.githubusercontent.com/YOUR_USERNAME/cap26/main/default%20of%20credit%20card%20clients%20(1).csv"

    df = pd.read_csv(url)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    return df


df = load_data()

# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{df.shape[0]:,}")
col2.metric("Total Features", df.shape[1]-1)
col3.metric("Target Variable", "dpnm")

st.dataframe(df.head())

# ---------------------------------------------------
# TARGET DISTRIBUTION
# ---------------------------------------------------

st.header("🎯 Default Distribution")

fig, axes = plt.subplots(1,2, figsize=(12,5))

colors = ["#2ecc71","#e74c3c"]

vc = df["dpnm"].value_counts()

axes[0].bar(["No Default","Default"], vc.values, color=colors)
axes[0].set_title("Default Distribution")

axes[1].pie(vc.values, labels=["No Default","Default"], autopct="%1.1f%%", colors=colors)

st.pyplot(fig)

# ---------------------------------------------------
# DEMOGRAPHIC ANALYSIS
# ---------------------------------------------------

st.header("👥 Demographic Analysis")

df["SEX_label"] = df["SEX"].map({1:"Male",2:"Female"})

gender_default = df.groupby("SEX_label")["dpnm"].mean()*100

fig, ax = plt.subplots()

ax.bar(gender_default.index, gender_default.values)

ax.set_title("Default Rate by Gender")
ax.set_ylabel("Default Rate %")

st.pyplot(fig)

# ---------------------------------------------------
# CREDIT LIMIT DISTRIBUTION
# ---------------------------------------------------

st.header("💰 Credit Limit Distribution")

fig, ax = plt.subplots()

df[df["dpnm"]==0]["LIMIT_BAL"].hist(bins=40, alpha=0.6, label="No Default", ax=ax)

df[df["dpnm"]==1]["LIMIT_BAL"].hist(bins=40, alpha=0.6, label="Default", ax=ax)

ax.legend()

st.pyplot(fig)

# ---------------------------------------------------
# PAYMENT HISTORY
# ---------------------------------------------------

st.header("📅 Payment Delay vs Default")

pay_cols = ["PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

fig, axes = plt.subplots(2,3, figsize=(14,8))

for ax,col in zip(axes.flatten(),pay_cols):

    default_rate = df.groupby(col)["dpnm"].mean()*100

    ax.bar(default_rate.index.astype(str), default_rate.values)

    ax.set_title(col)

st.pyplot(fig)

# ---------------------------------------------------
# AGE DISTRIBUTION
# ---------------------------------------------------

st.header("🎂 Age Distribution")

fig, ax = plt.subplots()

df[df["dpnm"]==0]["AGE"].hist(bins=30, alpha=0.6, label="No Default", ax=ax)
df[df["dpnm"]==1]["AGE"].hist(bins=30, alpha=0.6, label="Default", ax=ax)

ax.legend()

st.pyplot(fig)

# ---------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------

st.header("🔥 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(df.corr(), cmap="RdYlGn", center=0)

st.pyplot(fig)

# ---------------------------------------------------
# BUSINESS INSIGHTS
# ---------------------------------------------------

st.header("💡 Business Insights")

st.markdown("""

### Key Risk Drivers

• Payment delays are the strongest predictor of default  
• Higher credit utilization increases default probability  
• Younger customers show higher default risk  

### Recommendations

1️⃣ Monitor customers with **recent payment delays**  
2️⃣ Implement **credit utilization alerts**  
3️⃣ Introduce **risk-based credit policies**

""")

st.success("Dashboard Loaded Successfully")
