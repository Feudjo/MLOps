from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from src.ml.data import process_data
from src.ml import model
import pandas as pd
from fastapi.responses import JSONResponse
import numpy as np
from typing import Annotated

import json


from pydantic import BaseModel

class CensusDataRecord(BaseModel):

    age:int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain:int
    capital_loss: int
    hours_per_week: int
    native_country: str


app  = FastAPI()

@app.get("/")
async def read_root():
    html_content = """
                <html>
                    <head>
                        <title>Welcome to Income Predictor API</title>
                    </head>
                    <body>
                        <h1>Welcome to the Income Predictor API!</h1>
                        <p>Submit census data to find out if income exceeds $50K/year.</p>
                    </body>
                </html>
                 """
    return HTMLResponse(content=html_content)

@app.post("/record")
async def predict_income(item: Annotated[CensusDataRecord,
                                         Body(
                                             examples=[
                                                {
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
                                             ]
                                         )]):
    df = pd.DataFrame([item.dict()])
    df.rename(columns={"marital_status":"marital-status",
                       "native_country":"native-country"},
              inplace=True)
    encoder = model.load_model("model/encoder.pkl")
    rf = model.load_model("model/rf.pkl")

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
    X,_,_,_= process_data(df,training=False,
                             encoder=encoder,
                             categorical_features=cat_features)

    result = rf.predict(X)
    interpretation  = np.where(result==0, "Salary does not exceed 50k",
                               "Salary exceeds 50k")
    interpretationlist = interpretation.tolist()
    return json.dumps({"Prediction": interpretationlist})
