import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import sys
import os


def main():
    train_df = pd.read_csv('titanic/data/train.csv')
    test_df = pd.read_csv("titanic/data/test.csv")

    train_df['dataset'] = "train"
    test_df['dataset'] = "test"

    # ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    # ['num_family_onboard', 'age_cat_code', 'is_alone']
    features = ['Pclass', 'Sex_code', 'num_family_onboard', 'title_code']
    
    joined_df = pd.concat([train_df, test_df])
    joined_df = feature_engineer(joined_df)
    
    joined_df = encode_data(joined_df, ["Sex", 'Embarked', 'age_cat', 'title'])
    
    encoder = LabelEncoder()
    joined_df['Cabin_code'] = encoder.fit_transform(joined_df['Cabin'])


    train_df = joined_df[joined_df['dataset'] == 'train'].drop(columns=['dataset'])
    test_df = joined_df[joined_df['dataset'] == 'test'].drop(columns=['dataset'])

    X = train_df[features]
    y = train_df["Survived"]

    X_test = test_df[features]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the RandomForestClassifier
    # rf = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100, min_samples_split=2)
    model = create_model(X.shape[1])
    xg = xgb.XGBClassifier(random_state=42, max_depth=10, n_estimators=100,)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

    model.fit(X, y, epochs=20, batch_size=40, callbacks=[early_stopping])
    xg.fit(X, y)
    # xg.fit(X_train, y_train)
    importances = pd.Series(xg.feature_importances_, index = features)
    importances.plot(kind = 'barh', figsize = (12, 8))
    plt.show()

    pred = xg.predict(X_test)
    
    # print(accuracy_score(y_test, pred))
    # Save the DataFrame to a CSV file
    results = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': np.round(pred).astype(int)})
    results.to_csv(f'titanic/data/rf_predictions.csv', index=False)
 

def encode_data(df, features):
    encoder = LabelEncoder()

    for feature in features:
        df[f'{feature}_code'] = encoder.fit_transform(df[feature])

    return df


def feature_engineer(df):
    df['num_family_onboard'] = df['SibSp'] + df['Parch']
    df['is_alone'] = df['num_family_onboard'].apply(lambda x: int(x > 0))

    df['title'] = df['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    df["Age"] = df['Age'].fillna(df.groupby('Sex')['Age'].transform('mean'))
    df["age_cat"] = pd.cut(df['Age'], bins = [0.0, 19.0, 40.0, 60.0, 80.0], labels = ['0 - 19', '20 - 40', '41 - 61', '61 - 80'])

    df['Embarked'] = df["Embarked"].fillna(df["Embarked"].mode())
    df['Fare'] = df["Fare"].fillna(df["Fare"].mean())

    return df


def create_model(num_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(num_features,)),
        # tf.keras.layers.Dropout(0.5),  # Optional dropout layer for regularization
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Optional dropout layer for regularization
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification, so using sigmoid activation
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()