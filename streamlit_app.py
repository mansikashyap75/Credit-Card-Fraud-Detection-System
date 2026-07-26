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

    # --- Graph / Chart Add Karein ---
    st.subheader("📊 Transaction Analysis Chart")
    
    # Example Bar Chart for Visualization
    fig, ax = plt.subplots(figsize=(5, 3))
    categories = ['Transaction Amount']
    values = [amount]
    
    ax.bar(categories, values, color='#3f72af', width=0.4)
    ax.set_ylabel("Amount ($)")
    ax.set_title("Input Transaction Overview")
    
    st.pyplot(fig)
