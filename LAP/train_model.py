import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('loan_data.csv')

# Convert all categorical columns to string type
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_columns:
    data[col] = data[col].astype(str)

# Fill missing values
numeric_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Impute numeric columns
imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Impute categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])

# Split the dataset into features and target
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Create a label encoder
le = LabelEncoder()

# Encode categorical variables for the entire dataset
for column in categorical_columns[:-1]:  # Exclude 'Loan_Status'
    X[column] = le.fit_transform(X[column])
    joblib.dump(le, f'{column}_encoder.pkl')  # Save each encoder separately

# Encode the target variable
y = le.fit_transform(y)
joblib.dump(le, 'target_encoder.pkl')  # Save the target encoder

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'loan_model.pkl')

# Print model accuracy
print(f"Model accuracy: {model.score(X_test, y_test):.2f}")

# Print unique values for each categorical column
for column in categorical_columns:
    print(f"Unique values in {column}: {data[column].unique()}")