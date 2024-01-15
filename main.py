import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load the Breast Cancer Wisconsin (Diagnostic) Dataset from a CSV file
file_path = 'data.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Introduce some missing values
import numpy as np
np.random.seed(42)
mask = np.random.rand(df.shape[0], df.shape[1]) < 0.02  # 2% missing values
df[mask] = np.nan

# Display the number of missing values in each column
print("Number of missing values in each column:")
print(df.isnull().sum())

# Handling missing values (you can choose a different strategy)
df.fillna(df.mean(),inplace=True)


# Use Isolation Forest to identify outliers
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
iso_forest = IsolationForest(contamination=0.05)  # Adjust the contamination parameter based on your data
outliers = iso_forest.fit_predict(X_train)

# Visualize outliers
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=outliers, cmap='viridis')
plt.title('Outliers Detection using Isolation Forest')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Remove outliers
X_train_clean = X_train[outliers == 1]

# Display the shape before and after removing outliers
print("Shape before removing outliers:", X_train.shape)
print("Shape after removing outliers:", X_train_clean.shape)
df.to_csv('breast_cancer_data_processed.csv', index=False)
