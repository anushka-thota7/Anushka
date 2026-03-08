# ======================================================
# DEMAND FORECASTING STREAMLIT APP
# ======================================================

# 1️⃣ Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 2️⃣ Page Configuration
# ------------------------------------------------------
st.set_page_config(
    page_title="Demand Forecasting App",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Demand Forecasting using Machine Learning")
st.write("Predict product demand using a trained LightGBM model")

# ------------------------------------------------------
# 3️⃣ Load Model and Scaler
# ------------------------------------------------------
@st.cache_resource
def load_files():
    model = joblib.load("LightGBM_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_files()

# ------------------------------------------------------
# 4️⃣ Sidebar Inputs
# ------------------------------------------------------
st.sidebar.header("Enter Feature Values")

date_id = st.sidebar.number_input(
    "Date ID",
    min_value=1,
    max_value=5000,
    value=100
)

pc1 = st.sidebar.number_input(
    "PC1",
    value=0.0,
    format="%.4f"
)

pc2 = st.sidebar.number_input(
    "PC2",
    value=0.0,
    format="%.4f"
)

# ------------------------------------------------------
# 5️⃣ Feature Engineering Function
# ------------------------------------------------------
def create_features(date_id, pc1, pc2):

    day_sin = np.sin(2 * np.pi * date_id / 7)
    day_cos = np.cos(2 * np.pi * date_id / 7)

    month_sin = np.sin(2 * np.pi * date_id / 30.5)
    month_cos = np.cos(2 * np.pi * date_id / 30.5)

    data = pd.DataFrame({
        "date_id":[date_id],
        "PC1":[pc1],
        "PC2":[pc2],
        "day_sin":[day_sin],
        "day_cos":[day_cos],
        "month_sin":[month_sin],
        "month_cos":[month_cos]
    })

    return data


# ------------------------------------------------------
# 6️⃣ Prediction
# ------------------------------------------------------
st.subheader("🔮 Predict Demand")

if st.button("Predict Demand"):

    # Create input features
    input_df = create_features(date_id, pc1, pc2)

    # Scale features
    features_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_df)

    st.success(f"📊 Predicted Demand: {prediction[0]:.2f}")


# ------------------------------------------------------
# 7️⃣ Optional Dataset Visualization
# ------------------------------------------------------
st.subheader("📈 Historical Demand Visualization")

try:

    df = pd.read_feather("dataset/dimred_df.feather")

    fig = plt.figure(figsize=(10,4))

    plt.plot(df["value"].head(200))
    plt.xlabel("Samples")
    plt.ylabel("Demand")
    plt.title("Historical Demand Trend")

    st.pyplot(fig)

except:
    st.warning("Dataset not found for visualization.")


# ------------------------------------------------------
# 8️⃣ Footer
# ------------------------------------------------------
st.markdown("---")
st.write("ML Project: Demand Forecasting using LightGBM 🚀")