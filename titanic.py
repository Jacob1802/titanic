import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb


def main():
    df = pd.read_csv('train.csv')
    test_df = pd.read_csv("test.csv")
    
    # ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    # ['num_family_onboard', 'age_category_code', 'is_alone']
    features = ['Pclass', 'sex_code', 'age_category_code', 'is_alone']

    df = feature_engineer(df)
    df = encode_data(df)

    test_df = feature_engineer(test_df)
    test_df = encode_data(test_df)

    X = df[features]
    y = df["Survived"]

    X_test = test_df[features]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # lgb.LGBMClassifier
    models = [lgb.LGBMClassifier, LogisticRegression, xgb.XGBClassifier, RandomForestClassifier, DecisionTreeClassifier]

    for m in models:
        model = m()
        model.fit(X, y)
        pred = model.predict(X_test)

        # print(f"{m.__name__}: {accuracy_score(y_test, pred)}")

        # Create a DataFrame with PassengerId and the corresponding predictions
        results = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': np.round(pred).astype(int)})

        # Save the DataFrame to a CSV file
        results.to_csv(f'{m.__name__}_predictions.csv', index=False)


def encode_data(df):

    encoder = LabelEncoder()
    df['name_code'] = encoder.fit_transform(df['Name'])
    df['sex_code'] = encoder.fit_transform(df['Sex'])
    df['ticket_code'] = encoder.fit_transform(df['Ticket'])
    df['cabin_code'] = encoder.fit_transform(df['Cabin'])
    df['embarked_code'] = encoder.fit_transform(df['Embarked'])
    df['age_category_code'] = encoder.fit_transform(df['age_category'])

    return df


def feature_engineer(df):
    df['num_family_onboard'] = df['SibSp'] + df['Parch']
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['title'] = ... # extarct title from name
    df['is_alone'] = df['num_family_onboard'].apply(lambda x: int(x > 0))
    df['age_category'] = df['Age'].map(map_age_to_weight_category)

    return df


def map_age_to_weight_category(age):

    if age < 19:
        return 'child'
    elif 19 <= age < 25:
        return 'young_adult'
    elif 25 <= age < 55:
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