import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read the CSV file
df = pd.read_csv('Phishing_Legitimate_full.csv')
df = df.drop(columns=['id'])
df.head()
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
# Separate the features (X) and target variable (y)
X = df.drop("CLASS_LABEL", axis=1)  # Replace 'target_variable_name' with the actual column name
y = df['CLASS_LABEL']  # Replace 'target_variable_name' with the actual column name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost classifier
xgb_model = xgb.XGBClassifier()

# Fit the model on the training data
xgb_model.fit(X_train, y_train)

# Use feature importance to select top 30 features
importance = xgb_model.feature_importances_
feature_names = X.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
top_features = feature_importance.nlargest(30, 'Importance')['Feature'].values

# Select only the top 30 features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Print the selected features
print("Selected features:")
print(top_features)
from sklearn.metrics import accuracy_score, confusion_matrix

# Create a new XGBoost classifier with the selected features
xgb_model_selected = xgb.XGBClassifier()

# Fit the model on the selected training data
xgb_model_selected.fit(X_train_selected, y_train)

# Predict the target variable for the test data
y_pred = xgb_model_selected.predict(X_test_selected)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)
from sklearn.metrics import classification_report

# Generate the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
