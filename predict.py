"""
Business Failure Predictor - Make Predictions
Load the trained model and predict bankruptcy risk for new companies
"""

import pandas as pd
import pickle
import numpy as np

print("=" * 70)
print("BUSINESS FAILURE RISK PREDICTOR")
print("=" * 70)

# Load model and scaler
with open('model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"\n✅ Loaded {metadata['model_type']} model")
print(f"   AUC Score: {metadata['auc_score']:.4f}")

# Example: Predict on test companies
df = pd.read_csv('data/american_bankruptcy.csv')

# Get a few example companies
examples = df.sample(5, random_state=42)

print("\n" + "=" * 70)
print("EXAMPLE PREDICTIONS")
print("=" * 70)

for idx, row in examples.iterrows():
    # Extract features
    features = [row[f'X{i}'] for i in range(1, 19)]
    features_scaled = scaler.transform([features])
    
    # Make prediction
    probability = model.predict_proba(features_scaled)[0][1]
    prediction = "HIGH RISK" if probability > 0.5 else "LOW RISK"
    actual = "FAILED" if row['status_label'] == 'failed' else "ALIVE"
    
    print(f"\nCompany: {row['company_name']}")
    print(f"Year: {row['year']}")
    print(f"Failure Risk Score: {probability:.1%}")
    print(f"Prediction: {prediction}")
    print(f"Actual Status: {actual}")
    print(f"Match: {'✓' if (probability > 0.5 and actual == 'FAILED') or (probability <= 0.5 and actual == 'ALIVE') else '✗'}")

print("\n" + "=" * 70)
print("Model ready for integration into web applications!")
print("=" * 70)
