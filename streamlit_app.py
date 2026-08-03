import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

# Model Load Karein
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

model = load_model()

st.title("💳 Credit Card Fraud Detection System")
st.write("Enter the transaction details below to check if it is safe or fraudulent.")

# Input fields
time = st.number_input("Transaction Time", value=0.0)
amount = st.number_input("Transaction Amount ($)", value=0.0)

if st.button("Predict Transaction", type="primary"):
    # Prediction aur Probability (agar model predict_proba support karta hai)
    features = np.array([[time, amount]])
    prediction = model.predict(features)
    
    try:
        proba = model.predict_proba(features)[0][1] * 100 # Fraud probability in %
    except:
        proba = 100.0 if prediction[0] == 1 else 0.0

    if prediction[0] == 0:
        st.success("✅ **Safe Transaction:** This is a Legitimate Transaction.")
    else:
        st.error("⚠️ **Fraud Alert:** Fraudulent Transaction Detected!")

    # --- Bar Chart with Vertical Risk Score (Y-axis) ---
    st.subheader("📊 Transaction Risk Analysis (Vertical Bar Chart)")
    
    fig, ax = plt.subplots(figsize=(5, 3))
    categories = ['Safe', 'Fraud Risk']
    
    # Agar safe hai toh safe ka score high, warna fraud ka score high
    safe_score = 100 - proba
    fraud_score = proba
    
    scores = [safe_score, fraud_score]
    bar_colors = ['#28a745', '#dc3545']
    
    ax.bar(categories, scores, color=bar_colors, width=0.5)
    ax.set_ylabel("Probability (%)")  # Y-axis vertical label
    ax.set_ylim(0, 100)
    ax.set_title("Transaction Confidence Score")
    
    st.pyplot(fig)
