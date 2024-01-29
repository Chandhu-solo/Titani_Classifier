import pandas as pd

# Creating a synthetic Titanic dataset
data = {
    'Pclass': [1, 2, 3, 1, 2],
    'Gender': ['female', 'male', 'female', 'male', 'female'],
    'Age': [25, 30, 22, 35, 28],
    'SibSp': [1, 0, 3, 1, 0],
    'Parch': [2, 1, 0, 1, 0],
    'Fare': [200.0, 50.0, 10.0, 150.0, 80.0],
    'Survived': [1, 0, 1, 0, 1]
}

titanic_data = pd.DataFrame(data)

# Save the synthetic dataset to a CSV file
u=titanic_data.to_csv('titanic.csv', index=False)
print(u)

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Titanic dataset (you can download it from various sources, e.g., Kaggle)
# For this example, I'll assume you have a CSV file named 'titanic.csv'
titanic_data = pd.read_csv('titanic.csv')

# Explore the dataset
print(titanic_data.head())

# Data Preprocessing
# Drop irrelevant columns and handle missing values
titanic_data = titanic_data.drop(['Pclass', 'Gender', 'Age', 'SibSp', 'Parch','Fare','Survived'], axis=1)
titanic_data['Gender'] = titanic_data['Gender'].map({'male': 0, 'female': 1})
titanic_data = titanic_data.dropna()

# Split the data into features (X) and target variable (y)
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Classification report and confusion matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
