import streamlit as st
import pandas as pd
import joblib
from utils.preprocessing import preprocess_input

# -------------------------------
# Load model assets
# -------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("model/cancellation_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_assets()

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Order Cancellation Risk System",
    layout="centered"
)

st.title("📦 Order Cancellation Risk System")
st.write("Predict the risk of order cancellation and take action early.")

# -------------------------------
# Sidebar mode selection
# -------------------------------
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Single Order Prediction", "Bulk CSV Prediction"]
)

# -------------------------------
# Helper: preprocessing
# (temporary – we will move this to utils/preprocessing.py next)
# -------------------------------
'''def preprocess_input(df):
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    df_scaled = scaler.transform(df_encoded)
    return df_scaled'''

# -------------------------------
# Mode 1: Single Order Prediction
# -------------------------------
if mode == "Single Order Prediction":

    st.subheader("🔍 Single Order Check")

    order_amount = st.number_input("Order Amount", min_value=0.0)
    delivery_distance = st.number_input("Delivery Distance (km)", min_value=0.0)
    previous_cancellations = st.number_input(
        "Previous Cancellations", min_value=0, step=1
    )

    payment_method = st.selectbox(
        "Payment Method",
        ["Cash", "Card", "Online"]
    )

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([{
            "order_amount": order_amount,
            "delivery_distance_km": delivery_distance,
            "previous_cancellations": previous_cancellations,
            "payment_method": payment_method
        }])

        X_processed = preprocess_input(input_df, feature_columns, scaler)
        prob = model.predict_proba(X_processed)[0][1]

        st.write(f"### Risk Score: **{prob:.2f}**")

        if prob >= 0.7:
            st.error("🚨 High Risk – Call customer immediately")
        elif prob >= 0.4:
            st.warning("⚠️ Medium Risk – Monitor order")
        else:
            st.success("✅ Low Risk – Safe order")

# -------------------------------
# Mode 2: Bulk CSV Prediction
# -------------------------------
else:
    st.subheader("📂 Bulk CSV Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        X_processed = preprocess_input(df, feature_columns, scaler)

        df["risk_score"] = model.predict_proba(X_processed)[:, 1]

        def risk_label(prob):
            if prob >= 0.7:
                return "High Risk"
            elif prob >= 0.4:
                return "Medium Risk"
            else:
                return "Low Risk"

        df["risk_level"] = df["risk_score"].apply(risk_label)

        st.dataframe(df)

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            file_name="order_risk_results.csv",
            mime="text/csv"
        )
