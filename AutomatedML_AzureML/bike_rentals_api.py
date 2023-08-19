# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("bike_rentals_api")

# Create input/output pydantic models
input_model = create_model("bike_rentals_api_input", **{'day': 15.0, 'mnth': 5.0, 'year': 2012.0, 'season': 2.0, 'holiday': 0.0, 'weekday': 2.0, 'workingday': 1.0, 'weathersit': 2.0, 'temp': 0.6116669774055481, 'atemp': 0.5764039754867554, 'hum': 0.7945830225944519, 'windspeed': 0.14739200472831726})
output_model = create_model("bike_rentals_api_output", prediction=331)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
