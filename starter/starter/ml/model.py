from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from typing import Iterable
import pickle
from ml.data import process_data

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Define the parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    # Create the model
    model = RandomForestClassifier()

    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def save_model(file: str, model):
    """Save `model` to `file`.

    Args:
        file (str): _description_
        model (_type_): _description_
    """
    with open(file, "wb") as f:
        pickle.dump(model, f)

def load_model(file: str):
    """Load a model from `file`

    Args:
        file (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(file, "rb") as f:
        model=pickle.load(f)

    return model




def compute_slice(model, df:pd.DataFrame,
                  feature: str,
                  encoder,
                  lb) -> Iterable:
    """Computes metrics  for each gender category

    Args:
        model: Trained model
        y (np.array): true target
        preds (np.array): Predicted value
        feature (str): Categorical feature to slice the data on.
        encoder: Categorical features encoder
        lb:  label binarizer
    Returns:
        fbeta_score: Dict
            Dictionary where keys references a gender and values,
            the fbeta scores
    """
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
    #cat_features.remove(feature)
    X_test, y_test, _, _ = process_data(df, categorical_features=cat_features,
                                    training=False,
                                    encoder=encoder,
                                    lb=lb,
                                    label="salary")
    unique_values = df[feature].unique()

    slice_metric = dict()
    for value in unique_values:
        mask = df[feature]==value
        x_slice, y_slice = X_test[mask], y_test[mask]

        y_pred = inference(model, x_slice)
        precision, recall, f1_score = compute_model_metrics(y_slice, y_pred)


        slice_metric[value] = {"precision":precision,
                               "recall": recall,
                               "f1_score": f1_score}
    return slice_metric

