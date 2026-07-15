import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Segmentation - KMeans", layout="wide")

st.title("🛍️ Customer Segmentation using KMeans")
st.markdown("Different customers ka Age, Annual Income aur Spending Hours daalo, model unko clusters mein baantega, aur naye customer ka segment predict karega.")

# ---------- Session state initialize ----------
if "customers" not in st.session_state:
    st.session_state.customers = []

# ---------- Sidebar: Add customer data ----------
st.sidebar.header("Step 1: Customer Data Add Karo")

with st.sidebar.form("add_customer_form", clear_on_submit=True):
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    income = st.number_input("Money_Spent (in thousands)", min_value=0, value=50)
    spending = st.number_input("Spending Hours (per week)", min_value=0, value=5)
    submitted = st.form_submit_button("➕ Add Customer")
    if submitted:
        st.session_state.customers.append(
            {"Age": age, "Money_Spent ": income, "spending_hours": spending}
        )
        st.sidebar.success(f"Customer {len(st.session_state.customers)} added!")

if st.sidebar.button("🗑️ Clear All Customers"):
    st.session_state.customers = []
    st.sidebar.info("Saara data clear ho gaya.")

st.sidebar.markdown("---")
n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=6, value=3)

# ---------- Optional: CSV upload for bulk data ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Ya CSV upload karo (bulk data)")
uploaded_file = st.sidebar.file_uploader("CSV file with columns: Age, Money_Spent , spending_hours", type=["csv"])

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif len(st.session_state.customers) > 0:
    df = pd.DataFrame(st.session_state.customers)

# ---------- Main area ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Customer Data")
    if df is not None and len(df) > 0:
        st.dataframe(df, use_container_width=True)
        st.caption(f"Total customers: {len(df)}")
    else:
        st.info("Abhi tak koi customer data nahi hai. Sidebar se add karo ya CSV upload karo.")

# ---------- Run KMeans only if enough data ----------
if df is not None and len(df) >= n_clusters:
    X = df[["Age", "Money_Spent ", "spending_hours"]]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(x_scaled)

    # Dynamically label clusters based on avg income + spending (not hardcoded!)
    cluster_summary = df.groupby("Cluster")[["Money_Spent ", "spending_hours"]].mean()
    cluster_summary["score"] = cluster_summary["Money_Spent "] + cluster_summary["spending_hours"]
    ranked = cluster_summary["score"].rank(ascending=True).astype(int)

    label_map = {}
    n_labels = ["Low segment customer", "Medium segment customer", "Premium customer"]
    if n_clusters <= 3:
        sorted_clusters = cluster_summary["score"].sort_values().index.tolist()
        for idx, c in enumerate(sorted_clusters):
            label_map[c] = n_labels[idx] if idx < len(n_labels) else f"Segment {idx+1}"
    else:
        sorted_clusters = cluster_summary["score"].sort_values().index.tolist()
        for idx, c in enumerate(sorted_clusters):
            label_map[c] = f"Segment {idx+1} (score rank {idx+1})"

    df["Segment"] = df["Cluster"].map(label_map)

    with col2:
        st.subheader("📊 Cluster Visualization")
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(df["Age"], df["Money_Spent "], c=df["Cluster"], cmap="viridis", s=80)
        ax.set_xlabel("Age")
        ax.set_ylabel("Annual Income")
        ax.set_title("Customer Segmentation (Age vs Income)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.subheader("📈 Clustered Data with Segment Labels")
    st.dataframe(df, use_container_width=True)

    # ---------- Predict new customer ----------
    st.markdown("---")
    st.subheader("🔮 Naye Customer Ka Segment Predict Karo")

    p1, p2, p3, p4 = st.columns([1, 1, 1, 1])
    with p1:
        new_age = st.number_input("Age", min_value=1, max_value=100, value=30, key="predict_age")
    with p2:
        new_income = st.number_input("Annual Income (k)", min_value=0, value=50, key="predict_income")
    with p3:
        new_spending = st.number_input("Spending Hours", min_value=0, value=5, key="predict_spending")
    with p4:
        st.write("")
        st.write("")
        predict_btn = st.button("🎯 Predict Segment", use_container_width=True)

    if predict_btn:
        user_input = pd.DataFrame([[new_age, new_income, new_spending]],
                                   columns=["Age", "Money_Spent ", "spending_hours"])
        user_scaled = scaler.transform(user_input)
        pred_cluster = kmeans.predict(user_scaled)[0]
        pred_label = label_map[pred_cluster]

        st.success(f"✅ Ye customer **{pred_label}** hai (Cluster {pred_cluster})")

elif df is not None and len(df) > 0:
    st.warning(f"KMeans ke liye kam se kam {n_clusters} customers chahiye. Abhi {len(df)} hain — thoda aur data add karo.")

st.markdown("---")
st.caption("Note: Cluster labels (Premium/Medium/Low) actual cluster centers ke avg income+spending ke hisaab se dynamically assign hote hain — hardcoded nahi, jaisa original script mein tha.")