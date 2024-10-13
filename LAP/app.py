from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and label encoders
model = joblib.load('loan_model.pkl')
encoders = {
    'Gender': joblib.load('Gender_encoder.pkl'),
    'Married': joblib.load('Married_encoder.pkl'),
    'Dependents': joblib.load('Dependents_encoder.pkl'),
    'Education': joblib.load('Education_encoder.pkl'),
    'Self_Employed': joblib.load('Self_Employed_encoder.pkl'),
    'Property_Area': joblib.load('Property_Area_encoder.pkl'),
    'target': joblib.load('target_encoder.pkl')
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from the form
        input_data = {
            'Gender': str(request.form['Gender']).strip(),
            'Married': str(request.form['Married']).strip(),
            'Dependents': str(request.form['Dependents']).strip(),
            'Education': str(request.form['Education']).strip(),
            'Self_Employed': str(request.form['Self_Employed']).strip(),
            'ApplicantIncome': float(request.form['ApplicantIncome']),
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'LoanAmount': float(request.form['LoanAmount']),
            'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
            'Credit_History': float(request.form['Credit_History']),
            'Property_Area': str(request.form['Property_Area']).strip()
        }

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables using the loaded label encoders
        for column, encoder in encoders.items():
            if column in input_df.columns:
                try:
                    input_df[column] = encoder.transform(input_df[column].astype(str))
                except ValueError as e:
                    valid_values = encoder.classes_
                    return render_template('index.html', prediction_text=f"Invalid input for {column}: {input_df[column].values[0]}. Valid values are: {', '.join(valid_values)}")

        # Predict the loan status
        prediction = model.predict(input_df)
        
        # Decode the prediction
        loan_status = encoders['target'].inverse_transform(prediction)[0]

        return render_template('index.html', prediction_text=f'Loan Status: {loan_status}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)