import pandas as pd
import joblib

test_df = pd.read_csv("test.csv")

model = joblib.load("logreg_model.pkl")
scaler = joblib.load("scaler.pkl")
train_columns = joblib.load("train_columns.pkl")  


test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())  
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())  
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])  


test_df['Sex'] = test_df['Sex'].map({"female": 0, "male": 1})


test_df['Deck'] = test_df['Cabin'].str[0]
test_df['Deck'].fillna('Unknown', inplace=True)
test_df = pd.get_dummies(test_df, columns=['Deck', 'Embarked'], drop_first=True)


for col in train_columns:
    if col not in test_df.columns:
        test_df[col] = 0
test_df = test_df[train_columns]


test_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True, errors='ignore')

X_test_final = scaler.transform(test_df)
y_test_pred = model.predict(X_test_final)

submission = pd.DataFrame({
    "PassengerId": pd.read_csv("test.csv")["PassengerId"],
    "Survived": y_test_pred
})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
