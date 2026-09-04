import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Artefaktları yükle (cache'lenir, sayfa her yenilendiğinde tekrar okunmaz)

@st.cache_resource
def load_artifacts():
    label_encoders = joblib.load("models/label_encoders.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    scaler = joblib.load("models/scaler_supervised.pkl")
    model = joblib.load("models/best_model.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    return label_encoders, preprocessor, feature_columns, scaler, model, explainer


label_encoders, preprocessor, feature_columns, scaler, model, explainer = load_artifacts()

binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
nominal_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaymentMethod", "Contract",
]

# preprocessor'ın fit edildiği orijinal df kolon sırası (Churn dahil, customerID hariç)
ORIGINAL_COLUMN_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


# Kullanıcı girişi
st.title("📉 Customer Churn Prediction")
st.caption("Model: Logistic Regression (tuned) — SHAP ile açıklamalı tahmin")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (ay)", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=840.0, step=1.0)

    submitted = st.form_submit_button("Churn Tahmini Yap")

# Tahmin + SHAP açıklaması
if submitted:
    raw = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": "No",  # dummy — preprocessor bunu bekliyor, tahminden önce atılacak
    }
    input_df = pd.DataFrame([raw])[ORIGINAL_COLUMN_ORDER]

    # 1) binary kolonları eğitimdeki LabelEncoder'larla encode et
    for col in binary_cols:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # 2) nominal kolonları eğitimdeki preprocessor (OneHotEncoder) ile encode et
    encoded = preprocessor.transform(input_df)
    onehot_cols = preprocessor.named_transformers_["onehot"].get_feature_names_out(nominal_cols)
    remainder_cols = [c for c in ORIGINAL_COLUMN_ORDER if c not in nominal_cols]
    all_cols = list(onehot_cols) + remainder_cols
    encoded_df = pd.DataFrame(encoded, columns=all_cols)

    # 3) Churn'u at, eğitimdeki feature sırasına göre hizala, ölçekle
    X_new = encoded_df.drop(columns=["Churn"])[feature_columns]
    X_new_scaled = scaler.transform(X_new)

    # 4) Tahmin
    pred = model.predict(X_new_scaled)[0]
    proba = model.predict_proba(X_new_scaled)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"Müşteri **churn edecek** olarak tahmin edildi — olasılık: **{proba:.1%}**")
    else:
        st.success(f"Müşteri **kalacak** olarak tahmin edildi — churn olasılığı: **{proba:.1%}**")

    # 5) SHAP ile bu tekil tahminin açıklaması
    st.subheader("Bu tahmin neden verildi? (SHAP)")
    shap_value_single = explainer(X_new_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_value_single[0], show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Kırmızı çubuklar churn olasılığını artıran, mavi çubuklar azaltan "
        "özellikleri gösterir. Çubuk uzunluğu etkinin büyüklüğüdür."
    )