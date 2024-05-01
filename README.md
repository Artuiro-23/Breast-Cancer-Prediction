##Breast Cancer Prediction using SVM Model

This repository contains code for predicting breast cancer using a Support Vector Machine (SVM) model. The SVM model is a supervised machine learning algorithm that is commonly used for classification tasks like this one.

## Dataset
The dataset used for training and testing the SVM model is the Breast Cancer Wisconsin (Diagnostic) Dataset. This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, and the task is to predict whether the mass is benign or malignant.


## Requirements
To run the code in this repository, you'll need the following dependencies:
- Python 3.x
- NumPy
- Pandas
- Scikit-learn

You can install these dependencies using pip:
```
pip install numpy pandas scikit-learn
```

## Usage
1. Clone this repository to your local machine.
2. Navigate to the directory containing the code.
3. Ensure that you have the dataset (in CSV format) downloaded and placed in the same directory.
4. Run the `breast_cancer_prediction.py` file using Python:
   ```
   python breast_cancer_prediction.py
   ```

## Code Overview
- `breast_cancer_prediction.py`: This script contains the code for loading the dataset, preprocessing the data, training the SVM model, and evaluating its performance.
- `data.csv`: This file contains the dataset used for training and testing the model.

## Model Evaluation
The SVM model's performance is evaluated using commonly used metrics such as accuracy, precision, recall, and F1-score. These metrics provide insights into how well the model is performing in classifying breast masses as benign or malignant.

## Future Improvements
- Hyperparameter tuning: Fine-tuning the parameters of the SVM model to improve its performance.
- Feature engineering: Experimenting with different feature engineering techniques to enhance the predictive power of the model.
- Ensemble methods: Exploring ensemble learning techniques to further boost the model's performance.

