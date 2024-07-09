from fastapi import FastAPI
from fastapi.responses import HTMLResponse

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
async def predict_income(item: CensusDataRecord):
    pass