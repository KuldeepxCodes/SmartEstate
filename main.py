#  All code will be here
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="House Price Prediction (No Sklearn)", layout="wide")

st.title("🏠 SmartEstate: Intelligent House Price Prediction System")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # st.subheader("📌 Dataset Preview")
    # st.dataframe(df.head())

    # st.subheader("📈 Stats Summary")
    # st.write(df.describe())

    # Pairplot
    # st.subheader("📊 Pairplot")
    # fig1 = sns.pairplot(df)
    # st.pyplot(fig1)

    # Correlation Heatmap
    # st.subheader("🔥 Correlation Heatmap")
    # fig2, ax2 = plt.subplots(figsize=(10, 6))
    # sns.heatmap(df.drop('location', axis=1).corr(), annot=True, cmap="coolwarm", ax=ax2)
    # st.pyplot(fig2)

    st.subheader("🤖 NumPy Linear Regression Model")

    # One-hot encode location
    df_encoded = pd.get_dummies(df, columns=["location"])

    # Ensure all columns except target are numeric
    for col in df_encoded.columns:
        if col != "price_lakh":
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')

    X = df_encoded.drop("price_lakh", axis=1)
    y = df_encoded["price_lakh"]

    # Convert to numpy
    X_np = X.values
    y_np = y.values.reshape(-1, 1)

    # Replace NaN with 0 (required for Pyodide)
    X_np = np.nan_to_num(X_np)
    y_np = np.nan_to_num(y_np)

    # Add bias term
    X_np = np.hstack([np.ones((X_np.shape[0], 1)), X_np])

    # FORCE EVERYTHING INTO FLOAT
    X_np = X_np.astype(float)
    y_np = y_np.astype(float)

    # Train-test split
    split = int(0.8 * len(X_np))
    X_train, X_test = X_np[:split], X_np[split:]
    y_train, y_test = y_np[:split], y_np[split:]

    # ---------------------------------------
    # SAFE REGRESSION using Pseudo-Inverse
    # ---------------------------------------
    theta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train

    # Predictions
    y_pred = X_test @ theta

    # Metrics
    mae = np.mean(np.abs(y_test - y_pred))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    st.write(f"📌 **R² Score:** {r2:.3f}")
    st.write(f"📌 **Mean Absolute Error:** {mae:.3f}")

    # ---------------------------------------
    # Prediction Section
    # ---------------------------------------
    st.subheader("🔮 Predict House Price")

    col1, col2, col3 = st.columns(3)

    with col1:
        area_sqft = st.number_input("Area (sqft)", value=1500)

    with col2:
        bedrooms = st.number_input("Bedrooms", value=3)

    with col3:
        bathrooms = st.number_input("Bathrooms", value=2)

    age_years = st.number_input("Age (years)", value=5)
    location = st.selectbox("Location", df["location"].unique())

    if st.button("Predict Price"):

        row = {
            "area_sqft": area_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
        }

        for loc in df["location"].unique():
            row[f"location_{loc}"] = 1 if location == loc else 0

        sample_df = pd.DataFrame([row])

        sample_df = sample_df.reindex(columns=X.columns, fill_value=0)

        sample_np = sample_df.values.astype(float)
        sample_np = np.hstack([np.ones((1, 1)), sample_np])

        pred = float(sample_np @ theta)

        st.success(f"🏷️ **Predicted Price: ₹ {pred:.2f} Lakhs**")

else:
    st.info("📤 Upload a CSV file to start.")



https://streamlit.io/playground
