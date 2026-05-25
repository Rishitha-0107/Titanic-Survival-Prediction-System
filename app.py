import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# =====================================
# LOAD MODEL & SCALER
# =====================================

model = joblib.load("titanic_model.pkl")

scaler = joblib.load("scaler.pkl")

# =====================================
# HEADER SECTION
# =====================================

st.markdown("""
# 🚢 Titanic Survival Prediction System
### Machine Learning Based Passenger Survival Prediction
""")

# =====================================
# DESCRIPTION
# =====================================

st.markdown("""
## 📘 Project Description

This application predicts whether a passenger survived
during the Titanic disaster using Machine Learning.
""")

st.divider()

# =====================================
# INPUT SECTION
# =====================================

st.subheader("Passenger Information")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        25
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=50.0
    )

# =====================================
# PREDICTION
# =====================================

if st.button("Predict Survival"):

    input_data = np.array([[pclass, age, fare]])

    input_scaled = scaler.transform(input_data)

    probability = model.predict_proba(input_scaled)[0][1]

    survival_prob = probability * 100
    nonsurvival_prob = (1 - probability) * 100

    st.divider()

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        if probability > 0.5:
            st.success("Passenger Survived")
        else:
            st.error("Passenger Not Survived")

    with col2:
        st.metric(
            "Survival Probability",
            f"{survival_prob:.2f}%"
        )

    with col3:
        confidence = max(
            survival_prob,
            nonsurvival_prob
        )

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

    # =====================================
    # VISUALIZATION
    # =====================================

    st.subheader("Probability Visualization")

    labels = ['Survived', 'Not Survived']
    values = [survival_prob, nonsurvival_prob]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_ylabel("Probability (%)")

    st.pyplot(fig)