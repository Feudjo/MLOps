import requests
import json

url = "https://salary-classifier-0f42b9dd9022.herokuapp.com/record/"
data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "dm-clerica",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
response = requests.post(url,
                         data=json.dumps(data)
                         )


print(response.status_code)
print(response.json())