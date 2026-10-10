
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. File paths
X_train = "data/processed/X_train_cleaned.csv"
X_test = "data/processed/X_test_cleaned.csv"



# 2. Load data
def load_data():
    X_train = pd.read_csv(X_train)
    X_test = pd.read_csv(X_test)

    return X_train, X_test


# 3. Create preprocessing pipeline
def create_preprocessor(X_train):

    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor

import os
output_path = os.path.join("data/external")

# 4. Preprocess data
def preprocess_data():

    X_train, X_test = load_data()

    preprocessor = create_preprocessor(X_train)

    # Fit on training data
    X_train_array = preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_array = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_array,
        columns=feature_names
    )

    X_test_processed = pd.DataFrame(
        X_test_array,
        columns=feature_names
    )
    

    # 5. Save processed data
    X_train_processed.to_csv(
        os.path.join(output_path,"X_train_processed.csv"),
        index=False
    )

    X_test_processed.to_csv(
        os.path.join(output_path,"X_test_processed.csv"),
        index=False
    )


# 7. Entry point
if __name__ == "__main__":
    preprocess_data()
