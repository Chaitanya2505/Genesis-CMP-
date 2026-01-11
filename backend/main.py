from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI()

# Load model
model_path = os.path.join(os.path.dirname(__file__), "../models/winner_model.pkl")
feature_path = os.path.join(os.path.dirname(__file__), "../models/feature_columns.pkl")
encoder_path = os.path.join(os.path.dirname(__file__), "../models/label_encoder.pkl")

model = pickle.load(open(model_path, "rb"))
feature_columns = pickle.load(open(feature_path, "rb"))
label_encoder = pickle.load(open(encoder_path, "rb"))

class MatchInput(BaseModel):
    team1: str
    team2: str
    toss_winner: str
    batting_first: str
    first_innings_score: int

@app.post("/predict")
def predict_winner(data: MatchInput):
    # Dummy example (later connect team stats properly)
    input_data = {
        "toss_winner_team1": int(data.toss_winner == data.team1),
        "batting_first_team1": int(data.batting_first == data.team1),
        "first_innings_score": data.first_innings_score
    }

    df = pd.DataFrame([input_data])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    pred = model.predict(df)[0]
    winner = label_encoder.inverse_transform([pred])[0]

    return {"predicted_winner": winner}