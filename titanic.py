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
from sklearn.model_selection import GridSearchCV


def main():
    df = pd.read_csv('train.csv')
    test_df = pd.read_csv("test.csv")

    # ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    # ['num_family_onboard', 'age_category_code', 'is_alone']
    features = ['Pclass', 'Sex_code', 'is_alone']

    df = feature_engineer(df)
    test_df = feature_engineer(test_df)
    df, test_df = encode_data(df, test_df, None)

    X = df[features]
    y = df["Survived"]

    X_test = test_df[features]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the RandomForestClassifier
    rf = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100, min_samples_split=2)

    rf.fit(X, y)
    pred = rf.predict(X_test)
    # Save the DataFrame to a CSV file
    results = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': np.round(pred).astype(int)})
    results.to_csv(f'rf_predictions.csv', index=False)


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
    df['Age'].fillna(df['Age'].mean(), inplace=True)
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