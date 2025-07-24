import os
import sys
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Load model
model = joblib.load("sklearn_model.joblib")

# Load data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"R² Score (verify): {r2_score(y_test, y_pred)}")
