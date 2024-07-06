import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from starter.ml.data import process_data

from starter.ml import model

def test_load_model():
    """Import data"""
    mdl = model.load_model("starter/model/rf.pkl")
    assert isinstance(type(mdl), type(RandomForestClassifier))



def test_train_model():
    data = pd.read_csv("starter/data/census.csv")
    data = data.copy()[:20]
    # Optional enhancement, use K-fold cross validation instead of a train-test split.

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
    X_train, y_train, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )


    # Train and save a model.

    rfc =  model.train_model(X_train, y_train)
    assert isinstance(type(rfc), type(RandomForestClassifier))