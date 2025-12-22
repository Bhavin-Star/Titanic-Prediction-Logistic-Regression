import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("train.csv")
print(df.info())

print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
print(df.isnull().sum())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
print(df.isnull().sum())


print(df.info())

df['Deck'] = df['Cabin'].str[0]
df['Deck'].fillna('Unknown', inplace=True)

df = pd.get_dummies(df, columns=['Deck'], drop_first=True)
df.drop('Cabin', axis=1, inplace=True)

print(df.columns)

df['Sex'] = df['Sex'].map({"female": 0, "male": 1})
print(df.info())

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
print(df.info())

df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

X = df.drop('Survived', axis=1)
y =  df['Survived']

from sklearn.model_selection import  train_test_split

X_train, X_test, y_train, y_test  =  train_test_split(
    X,y,test_size=0.20,random_state=23
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import joblib
joblib.dump(model, "logreg_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns, "train_columns.pkl")  # save training feature columns