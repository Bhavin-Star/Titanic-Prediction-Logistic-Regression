Titanic Survival Prediction using Logistic Regression

A machine learning project that predicts whether a passenger survived the Titanic disaster by using a Logistic Regression model trained on the Titanic dataset from Kaggle. Logistic Regression is a supervised learning algorithm commonly used for binary classification tasks. ([Kaggle][1])

Project Overview

This project loads the Titanic dataset, performs data preprocessing, trains a Logistic Regression model, evaluates its performance, and generates predictions. The goal is to classify passengers as survived (1) or not survived (0) based on features such as passenger class, age, sex, fare, and others. ([Kaggle][1])

Files

train.csv
The training dataset with passenger features and survival labels.

test.csv
The test dataset with passenger features but without survival labels.

gender_submission.csv
A sample submission file demonstrating the expected format.

train.py
Script to preprocess data and train the Logistic Regression model.

test.py
Script to load the trained model and make predictions on test data.

logreg_model.pkl
Saved Logistic Regression model file.

scaler.pkl
Saved scaler object used to scale features during preprocessing.

train_columns.pkl
Saved list of training feature column names.

submission.csv
Output predictions for the test dataset.

How to Run

1. Clone the repository
   git clone [https://github.com/Bhavin-Star/Titanic-Prediction-Logistic-Regression.git](https://github.com/Bhavin-Star/Titanic-Prediction-Logistic-Regression.git)

2. Navigate to the project directory
   cd Titanic-Prediction-Logistic-Regression

3. Install required Python libraries (e.g., pandas, scikit-learn)

4. Train the model
   python train.py

5. Make predictions
   python test.py

Outputs the `submission.csv` file with predicted survival status.

Logistic Regression
Logistic Regression is used here as a binary classifier to estimate the probability of survival for passengers. The model outputs values between 0 and 1 indicating the probability of belonging to the survived class, and a threshold (e.g., 0.5) is used to decide the final classification. ([Kaggle][1])


Future Improvements

* Add data visualization and exploratory data analysis (EDA).
* Introduce feature engineering to improve model accuracy.
* Save evaluation metrics.
* Add documentation and usage examples.

Author: 
Bhavin Shah
GitHub: [https://github.com/Bhavin-Star](https://github.com/Bhavin-Star)
