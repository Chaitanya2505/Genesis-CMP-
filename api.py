from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# --- 1. ENABLE CORS (Allow Frontend to talk to Backend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LOAD DATA & MODELS ---
try:
    winner_model = pickle.load(open("models/winner_model.pkl", "rb"))
    score_1 = pickle.load(open("models/first_innings_model.pkl", "rb"))
    score_2 = pickle.load(open("models/second_innings_model.pkl", "rb"))
    feature_cols = pickle.load(open("models/feature_columns.pkl", "rb"))
    
    teams_df = pd.read_csv("data/team_features_normalized.csv", encoding="utf-8-sig")
    teams_df['team'] = teams_df['team'].astype(str).str.strip()
    
    unique_teams = sorted(teams_df['team'].unique().tolist())
    
    # Rebuild Encoder
    le = pickle.load(open("models/label_encoder.pkl", "rb"))


except Exception as e:
    print(f"Error loading models: {e}")

# --- 3. DEFINE INPUT FORMAT ---
class PredictionInput(BaseModel):
    team1: str
    team2: str
    toss_winner: str
    toss_decision: str

# --- 4. API ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "IPL Predictor API is Running!"}

@app.get("/teams")
def get_teams():
    # Send list of teams to Frontend for Dropdown
    return {"teams": unique_teams}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # A. Encode Inputs
        t1_enc = le.transform([data.team1])[0]
        t2_enc = le.transform([data.team2])[0]
        toss_enc = le.transform([data.toss_winner])[0]
        
        if data.toss_decision == "Bat":
            bat_first = data.toss_winner
        else:
            bat_first = data.team1 if data.toss_winner == data.team2 else data.team2
            
        bat_first_enc = 1 if bat_first == data.team1 else 0

        # B. Get Stats
        row_t1 = teams_df[teams_df['team'] == data.team1].iloc[0]
        row_t2 = teams_df[teams_df['team'] == data.team2].iloc[0]

        # C. Build DataFrame
        input_data = {
            'team_1_enc': t1_enc,
            'team_2_enc': t2_enc,
            'toss_winner_enc': toss_enc,
            'batting_first_enc': bat_first_enc,
            'batting_rating_t1': row_t1['batting_rating'],
            'bowling_rating_t1': row_t1['bowling_rating'],
            'avg_score_last_5_t1': row_t1['avg_score_last_5'],
            'batting_rating_t2': row_t2['batting_rating'],
            'bowling_rating_t2': row_t2['bowling_rating'],
            'avg_score_last_5_t2': row_t2['avg_score_last_5']
        }

        df = pd.DataFrame([input_data])
        
        # Add missing columns
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]

        # D. Predict
        # win_pred = winner_model.predict(df)[0]
        # win_prob = float(winner_model.predict_proba(df).max() * 100)
        # winner_name = le.inverse_transform([win_pred])[0]
        winner_name = winner_model.predict(df)[0]
        proba = winner_model.predict_proba(df)[0]
        win_prob = round(float(max(proba) * 100), 2)


        
        s1 = int(score_1.predict(df)[0])
        s2 = int(score_2.predict(df)[0])

        return {
            "winner": winner_name,
            "probability": win_prob,
            "score_inning_1": s1,
            "score_inning_2": s2
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))