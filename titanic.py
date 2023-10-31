import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb


def main():
    df = pd.read_csv('train.csv')
    test_df = pd.read_csv("test.csv")
    print(df.tail())
    print(test_df.tail())
    # ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    # ['num_family_onboard', 'age_category_code', 'is_alone']
    features = ['Pclass', 'Sex_code', 'age_category_code', 'is_alone', 'title_code']

    df = feature_engineer(df)
    test_df = feature_engineer(test_df)
    df, test_df = encode_data(df, test_df, None)

    print(df.tail())
    print(test_df.tail())
    X = df[features]
    y = df["Survived"]

    X_test = test_df[features]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # lgb.LGBMClassifier
    models = [lgb.LGBMClassifier, LogisticRegression, xgb.XGBClassifier, RandomForestClassifier, DecisionTreeClassifier]

    for m in models:
        model = m()
        model.fit(X, y)
        # model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # print(f"{m.__name__}: Acc: {accuracy_score(y_test, pred)}")
        # print(f"{m.__name__}: Prec: {precision_score(y_test, pred)}")
        # print(f"{m.__name__}: Recall: {recall_score(y_test, pred)}")
        # print(f"{m.__name__}: F1: {f1_score(y_test, pred)}")
        # print("-"*100)
        # print(f"{m.__name__}: Con: {confusion_matrix(y_test, pred)}")

        # Create a DataFrame with PassengerId and the corresponding predictions
        results = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': np.round(pred).astype(int)})

        # Save the DataFrame to a CSV file
        results.to_csv(f'{m.__name__}_predictions.csv', index=False)


def encode_data(df, test_df, features):
    features = ["Sex", 'Embarked', 'age_category', 'title']
    encoder = LabelEncoder()
    for feature in features:
        encoder.fit(pd.concat([df[feature], test_df[feature]], axis=0))
        df[f'{feature}_code'] = encoder.transform(df[feature])
        test_df[f'{feature}_code'] = encoder.transform(test_df[feature])

    return df, test_df


def feature_engineer(df):
    df['num_family_onboard'] = df['SibSp'] + df['Parch']
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['title'] = df['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    df['is_alone'] = df['num_family_onboard'].apply(lambda x: int(x > 0))
    df['age_category'] = df['Age'].map(map_age_to_weight_category)

    return df


def map_age_to_weight_category(age):
    if age < 19:
        return 'child'
    elif 19 <= age < 30:
        return 'young_adult'
    elif 30 <= age < 60:
        return 'adult'
    else:
        return 'senior'
    

# def create_model(num_features):
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(128, activation="relu", input_shape=(num_features,)),
#         tf.keras.layers.Dense(64, activation="relu"),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(32, activation="relu"),
#         tf.keras.layers.Dense(1, activation="sigmoid"),
#     ])

#     model.compile(
#         loss="binary_crossentropy",
#         optimizer="adam",
#         metrics=["accuracy"]
#     )
    
#     return model


if __name__ == '__main__':
    main()