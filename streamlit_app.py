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
    # Prediction aur Probability
    features = np.array([[time, amount]])
    prediction = model.predict(features)
    
    # Probabilities nikalne ki koshish (agar model support karta hai)
    try:
        proba_scores = model.predict_proba(features)[0]
        safe_prob = proba_scores[0] * 100
        fraud_prob = proba_scores[1] * 100
    except:
        # Fallback agar predict_proba kaam na kare
        if prediction[0] == 0:
            safe_prob = 99.0
            fraud_prob = 1.0
        else:
            safe_prob = 1.0
            fraud_prob = 99.0

    if prediction[0] == 0:
        st.success("✅ **Safe Transaction:** This is a Legitimate Transaction.")
    else:
        st.error("⚠️ **Fraud Alert:** Fraudulent Transaction Detected!")

    # --- Pie Chart with Model Confidence ---
    st.subheader("📊 Transaction Risk Probability Distribution")
    
    # Pie Chart Data
    labels = ['Safe Probability', 'Fraud Risk']
    sizes = [safe_prob, fraud_prob]
    colors_list = ['#28a745', '#dc3545'] # Green for Safe, Red for Fraud
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_list, shadow=True)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    st.pyplot(fig)
