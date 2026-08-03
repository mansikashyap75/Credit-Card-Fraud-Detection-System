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
    # Prediction
    features = np.array([[time, amount]])
    prediction = model.predict(features)
    
    if prediction[0] == 0:
        st.success("✅ **Safe Transaction:** This is a Legitimate Transaction.")
    else:
        st.error("⚠️ **Fraud Alert:** Fraudulent Transaction Detected!")

    # --- Pie Chart Add Karein ---
    st.subheader("📊 Transaction Risk Breakdown (Pie Chart)")
    
    # Example distribution data for visualization based on prediction
    if prediction[0] == 0:
        labels = ['Safe Probability', 'Fraud Risk']
        sizes = [95.0, 5.0]
        colors_list = ['#28a745', '#dc3545']
    else:
        labels = ['Safe Probability', 'Fraud Risk']
        sizes = [15.0, 85.0]
        colors_list = ['#28a745', '#dc3545']

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_list)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    st.pyplot(fig)
