from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_api_root():
    r = client.get("/")
    assert r.status_code==200


def test_missing_fields():
    response = client.post(
    "/record",
    json={
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        # "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
)
    assert response.status_code == 422

def test_invalid_datatype():
    response = client.post(
        "/record",
        json={
            "age": "thirty-nine",  # Invalid datatype
            "workclass": "State-gov",
            "fnlwgt": "seventy-seven thousand",  # Invalid datatype
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": "two thousand one hundred seventy-four",  # Invalid datatype
            "capital_loss": 0,
            "hours_per_week": "forty",  # Invalid datatype
            "native_country": "United-States"
        }
    )
    assert response.status_code == 422  # Unprocessable Entity
