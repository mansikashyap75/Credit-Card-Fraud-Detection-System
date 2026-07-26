from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

safe_count = 0
fraud_count = 0

@app.route('/')
def home():
    return render_template('index.html', safe_count=safe_count, fraud_count=fraud_count)

@app.route('/predict', methods=['POST'])
def predict():
    global safe_count, fraud_count
    try:
        time = float(request.form['time'])
        amount = float(request.form['amount'])
        
        features = np.array([[time, amount]])
        prediction = model.predict(features)
        
        if prediction[0] == 0:
            result = "Safe: This is a Legitimate Transaction."
            safe_count += 1
        else:
            result = "Alert: Fraudulent Transaction Detected!"
            fraud_count += 1
        
        return render_template('index.html', result=result, safe_count=safe_count, fraud_count=fraud_count)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}", safe_count=safe_count, fraud_count=fraud_count)

if __name__ == '__main__':
    app.run(debug=True)