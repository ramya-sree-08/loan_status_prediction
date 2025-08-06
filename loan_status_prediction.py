import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('loan_data.csv')

# Encode categorical columns
label = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_columns:
    data[col] = label.fit_transform(data[col])

# Features and target
X = data[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History', 'Education', 'Self_Employed', 'Property_Area']]
y = data['Loan_Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict a new example
new_applicant = [[1, 1, 5000, 150, 1, 0, 0, 2]]  # example data
prediction = model.predict(new_applicant)
print("Loan Status Prediction:", "Approved" if prediction[0] == 1 else "Not Approved")
