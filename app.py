import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load dataset
file_path = "real_estate_data.csv"
df = pd.read_csv(file_path)
df.drop(columns=["Unnamed: 0"], errors='ignore', inplace=True)

# Feature Engineering
df["Price_per_sqft"] = df["Price (Lakhs)"] * 100000 / df["Total Area (Sq.Ft.)"]

# Define features and target
X = df.drop(columns=["Price (Lakhs)"])
y = df["Price (Lakhs)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if models already exist, else train them
if not os.path.exists("scaler.joblib") or not os.path.exists("random_forest.joblib") or not os.path.exists("gradient_boosting.joblib"):
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.joblib")

    # Train models
    rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=5, random_state=42)
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, "random_forest.joblib")

    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    gb.fit(X_train_scaled, y_train)
    joblib.dump(gb, "gradient_boosting.joblib")

    st.write("✅ Models trained and saved!")

# Load trained models
scaler = joblib.load("scaler.joblib")
rf_model = joblib.load("random_forest.joblib")
gb_model = joblib.load("gradient_boosting.joblib")

def predict_price(features):
    """Predicts price using trained models."""
    features_df = pd.DataFrame([features])
    features_scaled = scaler.transform(features_df)

    price_rf = rf_model.predict(features_scaled)[0]
    price_gb = gb_model.predict(features_scaled)[0]

    return price_rf, price_gb

# Streamlit UI
st.title("Real Estate Price Prediction")
st.write("Enter property details to predict the price.")

# User Inputs
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

if st.button("Predict Price"):
    price_rf, price_gb = predict_price(user_input)

    st.success(f"Predicted Price (Random Forest): ₹{price_rf:.2f} Lakhs")
    st.success(f"Predicted Price (Gradient Boosting): ₹{price_gb:.2f} Lakhs")

    # Feature Importance
    st.subheader("Feature Importance - Random Forest")
    fig, ax = plt.subplots()
    sns.barplot(x=rf_model.feature_importances_, y=X.columns, ax=ax)
    st.pyplot(fig)
