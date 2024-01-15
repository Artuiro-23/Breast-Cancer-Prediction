# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# # Load data from CSV file
# file_path = 'data.csv'  # Replace with your actual file path
# df = pd.read_csv(file_path)

# # Split the data into features (X) and target variable (y)
# X = df.drop('diagnosis', axis=1)
# y = df['diagnosis']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a Random Forest model to get feature importances
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)

# # Get feature importances from the model
# feature_importances = rf_model.feature_importances_

# # Create a DataFrame with feature names and importances
# feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# # Sort features by importance in descending order
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# # Display the top N most important features
# top_n_features = 10
# print(f"Top {top_n_features} most important features:")
# print(feature_importance_df.head(top_n_features))

# # Plotting feature importances
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance_df['Feature'][:top_n_features], feature_importance_df['Importance'][:top_n_features])
# plt.xlabel('Importance')
# plt.title('Top Feature Importances for Breast Cancer Prediction')
# plt.show()

# # Feature engineering: Example - creating a new feature representing the mean radius squared
# X['mean_radius_squared'] = X['mean radius'] ** 2

# # Retrain the model with the new feature
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# rf_model.fit(X_train, y_train)

# # Evaluate the performance on the test set
# accuracy_with_new_feature = rf_model.score(X_test, y_test)
# print(f"Accuracy with the new feature: {accuracy_with_new_feature}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load data from CSV file
file_path = 'data.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Feature engineering: Adding a new feature (mean radius squared)
df['mean_radius_squared'] = df['mean radius'] ** 2

# Split the data into features (X) and target variable (y)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using Random Forest feature importances
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Display feature importances
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)
print("Top features based on importance:")
print(feature_importances.head(10))

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances.head(10).plot(kind='barh')
plt.xlabel('Importance')
plt.title('Top Feature Importances for Breast Cancer Prediction')
plt.show()

# Select top features based on importance
sfm = SelectFromModel(rf_model, threshold=0.03)  # Adjust the threshold based on your preference
sfm.fit(X_train, y_train)

# Transform the data to include only selected features
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

# Retrain the model using selected features
rf_model_selected = RandomForestClassifier(random_state=42)
rf_model_selected.fit(X_train_selected, y_train)

# Evaluate the performance on the test set with selected features
accuracy_with_selected_features = rf_model_selected.score(X_test_selected, y_test)
print(f"Accuracy with selected features: {accuracy_with_selected_features}")

