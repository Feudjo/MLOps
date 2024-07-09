import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ml import model
from ml.data import process_data

@pytest.fixture
def load_data():
    return pd.read_csv("starter/data/census.csv")

def test_load_model():
    """Import data"""
    mdl = model.load_model("starter/model/rf.pkl")
    assert isinstance(type(mdl), type(RandomForestClassifier))



def test_train_model(load_data):
    data = load_data.copy()[:20]
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


def test_compute_slice(load_data):
    data = load_data.copy()[:20]
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
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )


    # Train and save a model.

    rfc =  model.train_model(X_train, y_train)
    r = model.compute_slice(rfc, data, "sex", encoder, lb)
    assert set(r.keys())== set(("Male", "Female"))