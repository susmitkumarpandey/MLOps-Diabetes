# Flask app for Model Inference
# app/app.py
from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load the model
with open('app/diabetes_model.pkl', 'rb') as file:
    # we are using scaler because our data is preprocessed using scaler.
    scaler, model=pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Diabetes Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([data['features']]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({"prediction": int(prediction[0])})
    
if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)