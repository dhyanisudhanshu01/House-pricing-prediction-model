import os
import joblib

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

Model_File = "model.pkl"
Pipeline_File = "pipeline.pkl"
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "housing.csv")
TARGET_COL = "median_house_value"

def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

#If Model file does not exist, build the model and save it. Otherwise, load the model from file.

if not os.path.exists(Model_File):
    #let's train the model ans save it to file.

    #Load the data
    df = pd.read_csv(CSV_PATH)

    #Split the data into features and target
    #Separate features and target variable
    X = df.drop(columns=[TARGET_COL])
    Y = df[TARGET_COL]

    #train testsplit
    X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=RANDOM_STATE)
    pd.concat([X_test, Y_test], axis=1).to_csv(os.path.join(BASE_DIR, "data", "test_data.csv"), index=False)
    #Identify numerical and categorical features
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include="object").columns.tolist()

    #Build the preprocessing pipeline
    pipeline = build_pipeline(numerical_features, categorical_features)

    #Fit and transform the training data
    X_train_processed = pipeline.fit_transform(X_train)
    
    model = RandomForestRegressor(random_state=RANDOM_STATE)
    model.fit(X_train_processed, Y_train)
    #Save the model and pipeline to file
    joblib.dump(model, Model_File)
    joblib.dump(pipeline, Pipeline_File)
else:
    #Load the model and pipeline from file
    model = joblib.load(Model_File)
    pipeline = joblib.load(Pipeline_File)

    #Load the test data
    test_df = pd.read_csv(os.path.join(BASE_DIR, "data", "test_data.csv"))
    X_test = test_df.drop(columns=[TARGET_COL])
    Y_test = test_df[TARGET_COL]

    #Preprocess the test data
    X_test_processed = pipeline.transform(X_test)
    #Make predictions
    predictions = model.predict(X_test_processed)
    #Evaluate the model
    rmse = root_mean_squared_error(Y_test, predictions)
    print(f"Test RMSE: {rmse}")

    test_df["Predicted_" + TARGET_COL] = predictions
    test_df.to_csv(os.path.join(BASE_DIR, "data", "test_data_with_predictions.csv"), index=False)

    print("Predictions saved to ..data/test_data_with_predictions.csv")




