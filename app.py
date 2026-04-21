import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Segment names (adjust according to your clusters)
cluster_names = {
    0: "Budget Customers",
    1: "Regular Customers",
    2: "High Income Low Spending",
    3: "VIP Customers",
    4: "Low Income High Spending"
}

st.title("Customer Segmentation App")
st.write("Enter customer Annual Income and Spending Score to find their segment.")

# Input fields
income = st.number_input("Annual Income (k$)", min_value=0.0, max_value=200.0, step=1.0)
spending = st.number_input("Spending Score (1-100)", min_value=0.0, max_value=100.0, step=1.0)

# Predict button
if st.button("Predict Segment"):

    # Convert input into array
    user_data = np.array([[income, spending]])

    # Scale input data
    user_scaled = scaler.transform(user_data)

    # Predict cluster
    cluster = kmeans.predict(user_scaled)[0]

    # Show result
    st.success(f"Customer belongs to Cluster: {cluster}")
    st.info(f"Segment: {cluster_names.get(cluster, 'Unknown Segment')}")