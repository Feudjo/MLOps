from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from typing import Iterable
import pickle

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




def slice_gender(df:pd.DataFrame, y:np.array, preds:np.array) -> Iterable:
    """Computes f1 score for each gender category

    Args:
        df (pd.DataFrame): input dataframe
        y (_type_): trained model
        preds ()
    Returns:
        fbeta_score: Dict
            Dictionary where keys references a gender and values,
            the fbeta scores
    """
    f1_dict = dict()
    for gender in df["sex"].unique():
        df_temp = df[df["sex"]==gender]
        f1 = compute_model_metrics(y, preds)
        f1_dict[gender]=f1
    return f1_dict

