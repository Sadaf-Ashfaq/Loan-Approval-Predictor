# Test the trained model with sample input
import pandas as pd
import joblib

# Load saved model and encoders
model = joblib.load('loan_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')

print("Loaded encoders:", label_encoders.keys())
print("Feature names:", feature_names)

# Sample input data - Test Case 1: Should predict Approved
sample_data = {
    'no_of_dependents': 2,
    'education': 'Graduate',
    'self_employed': 'No',
    'income_annum': 9600000,
    'loan_amount': 29900000,
    'loan_term': 12,
    'cibil_score': 778,
    'residential_assets_value': 2400000,
    'commercial_assets_value': 17600000,
    'luxury_assets_value': 22700000,
    'bank_asset_value': 8000000
}

# Create DataFrame
input_df = pd.DataFrame([sample_data])

# Clean input data
for col in input_df.select_dtypes(include=['object']).columns:
    input_df[col] = input_df[col].str.strip()

print("\nBefore encoding:")
print(input_df)
print(input_df.dtypes)

# Encode categorical variables
for col in input_df.columns:
    if col in label_encoders:
        print(f"Encoding {col}: {input_df[col].values[0]} -> ", end="")
        input_df[col] = label_encoders[col].transform(input_df[col])
        print(f"{input_df[col].values[0]}")

print("\nAfter encoding:")
print(input_df)
print(input_df.dtypes)

# Ensure correct column order
input_df = input_df[feature_names]

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Decode prediction
result = target_encoder.inverse_transform(prediction)[0]

print("\n" + "="*50)
print("PREDICTION RESULT")
print("="*50)
print(f"Prediction: {result}")
print(f"Confidence: {max(prediction_proba[0]) * 100:.2f}%")
print(f"Probabilities: Approved={prediction_proba[0][0]*100:.2f}%, Rejected={prediction_proba[0][1]*100:.2f}%")