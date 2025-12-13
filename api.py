from fastapi import FastAPI, UploadFile
import pandas as pd
import joblib
import json

app = FastAPI()

# Load model + columns
model = joblib.load("churn_model.pkl")
columns = json.load(open("columns.json"))

@app.post("/predict-row")
async def predict_row(data: dict):
    df = pd.DataFrame([data])
    df = df.reindex(columns=columns, fill_value=0)
    
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }

@app.post("/predict-csv")
async def predict_csv(file: UploadFile):
    df = pd.read_csv(file.file)
    df = df.reindex(columns=columns, fill_value=0)

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df["prediction"] = preds
    df["probability"] = probs

    return df.to_dict(orient="records")
