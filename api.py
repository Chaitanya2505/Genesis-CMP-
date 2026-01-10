# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder

# app = FastAPI()

# # --- 1. ENABLE CORS (Allow Frontend to talk to Backend) ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins (for development)
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- 2. LOAD DATA & MODELS ---
# try:
#     winner_model = pickle.load(open("models/winner_model.pkl", "rb"))
#     score_1 = pickle.load(open("models/first_innings_model.pkl", "rb"))
#     score_2 = pickle.load(open("models/second_innings_model.pkl", "rb"))
#     feature_cols = pickle.load(open("models/feature_columns.pkl", "rb"))
    
#     teams_df = pd.read_csv("data/team_features_normalized.csv", encoding="utf-8-sig")
#     teams_df['team'] = teams_df['team'].astype(str).str.strip()
    
#     unique_teams = sorted(teams_df['team'].unique().tolist())
    
#     # Rebuild Encoder
#     le = pickle.load(open("models/label_encoder.pkl", "rb"))


# except Exception as e:
#     print(f"Error loading models: {e}")

# # --- 3. DEFINE INPUT FORMAT ---
# class PredictionInput(BaseModel):
#     team1: str
#     team2: str
#     toss_winner: str
#     toss_decision: str

# # --- 4. API ENDPOINTS ---

# @app.get("/")
# def home():
#     return {"message": "IPL Predictor API is Running!"}

# @app.get("/teams")
# def get_teams():
#     # Send list of teams to Frontend for Dropdown
#     return {"teams": unique_teams}

# @app.post("/predict")
# def predict(data: PredictionInput):
#     try:
#         # ---------------- ENCODE TEAMS ----------------
#         team1 = data.team1.strip()
#         team2 = data.team2.strip()
#         toss_winner = data.toss_winner.strip()

#         t1 = teams_df[teams_df["team"] == team1].iloc[0]
#         t2 = teams_df[teams_df["team"] == team2].iloc[0]

#         input_df = pd.DataFrame([{
#             "team_1_enc": le.transform([team1])[0],
#             "team_2_enc": le.transform([team2])[0],
#             "toss_winner_enc": le.transform([toss_winner])[0],
#             "batting_first_enc": 1 if data.toss_decision == "Bat" else 0,

#             "batting_rating_t1": t1["batting_rating"],
#             "bowling_rating_t1": t1["bowling_rating"],
#             "avg_score_last_5_t1": t1["avg_score_last_5"],

#             "batting_rating_t2": t2["batting_rating"],
#             "bowling_rating_t2": t2["bowling_rating"],
#             "avg_score_last_5_t2": t2["avg_score_last_5"],
#         }])

#         # ---------------- WINNER MODEL ----------------
#         winner_input = input_df[feature_cols]
#         winner = winner_model.predict(winner_input)[0]
#         prob = round(float(max(winner_model.predict_proba(winner_input)[0]) * 100), 2)

#         # ---------------- SCORE MODELS ----------------
#         score_features = [
#             "batting_rating_t1", "bowling_rating_t2",
#             "batting_rating_t2", "bowling_rating_t1",
#             "avg_score_last_5_t1", "avg_score_last_5_t2"
#         ]

#         score_input = input_df[score_features]
#         s1 = int(score_1.predict(score_input)[0])
#         s2 = int(score_2.predict(score_input)[0])

#         return {
#             "winner": winner,
#             "probability": prob,
#             "score_inning_1": s1,
#             "score_inning_2": s2
#         }
    
    

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
    

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

# ---------------- APP INIT ----------------
app = FastAPI(title="IPL Match Predictor API")

# ---------------- ENABLE CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # use specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODELS & DATA ----------------
try:
    # Models
    winner_model = pickle.load(open("models/winner_model.pkl", "rb"))
    score_1 = pickle.load(open("models/first_innings_model.pkl", "rb"))
    score_2 = pickle.load(open("models/second_innings_model.pkl", "rb"))
    feature_cols = pickle.load(open("models/feature_columns.pkl", "rb"))
    le = pickle.load(open("models/label_encoder.pkl", "rb"))

    # Team stats
    teams_df = pd.read_csv(
        "data/team_features_normalized.csv", encoding="utf-8-sig"
    )
    teams_df["team"] = teams_df["team"].astype(str).str.strip()

    unique_teams = sorted(teams_df["team"].unique().tolist())

    print("✅ Models & data loaded successfully")

except Exception as e:
    print("❌ Error loading models/data:", e)

# ---------------- INPUT SCHEMA ----------------
class PredictionInput(BaseModel):
    team1: str
    team2: str
    toss_winner: str
    toss_decision: str  # Bat / Field

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"message": "IPL Predictor API is Running 🚀"}

@app.get("/teams")
def get_teams():
    return {"teams": unique_teams}

# ---------------- PREDICT ENDPOINT ----------------
@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # ---------- CLEAN INPUTS ----------
        team1 = data.team1.strip()
        team2 = data.team2.strip()
        toss_winner = data.toss_winner.strip()
        toss_decision = data.toss_decision

        # ---------- TEAM ROWS ----------
        t1 = teams_df[teams_df["team"] == team1].iloc[0]
        t2 = teams_df[teams_df["team"] == team2].iloc[0]

        # ---------- MASTER INPUT DF ----------
        input_df = pd.DataFrame([{
            "team_1_enc": le.transform([team1])[0],
            "team_2_enc": le.transform([team2])[0],
            "toss_winner_enc": le.transform([toss_winner])[0],
            "batting_first_enc": 1 if toss_decision == "Bat" else 0,

            "batting_rating_t1": t1["batting_rating"],
            "bowling_rating_t1": t1["bowling_rating"],
            "avg_score_last_5_t1": t1["avg_score_last_5"],

            "batting_rating_t2": t2["batting_rating"],
            "bowling_rating_t2": t2["bowling_rating"],
            "avg_score_last_5_t2": t2["avg_score_last_5"],
        }])

        # ---------- WINNER PREDICTION ----------
        winner_input = input_df[feature_cols]
        winner = winner_model.predict(winner_input)[0]

        proba = winner_model.predict_proba(winner_input)[0]
        win_prob = round(float(max(proba) * 100), 2)

        # ---------- SCORE PREDICTION ----------
        score_features = [
            "batting_rating_t1", "bowling_rating_t2",
            "batting_rating_t2", "bowling_rating_t1",
            "avg_score_last_5_t1", "avg_score_last_5_t2"
        ]

        score_input = input_df[score_features]
        score_inning_1 = int(score_1.predict(score_input)[0])
        score_inning_2 = int(score_2.predict(score_input)[0])

        # ---------- PYTHON-POWERED ANALYSIS ----------
        team1_strength = round(
            t1["batting_rating"] * 0.6 + t1["bowling_rating"] * 0.4, 2
        )
        team2_strength = round(
            t2["batting_rating"] * 0.6 + t2["bowling_rating"] * 0.4, 2
        )

        win_split = {
            team1: round(100 - win_prob, 2),
            team2: win_prob
        }

        analysis = {
            "team1": team1,
            "team2": team2,
            "team1_strength": team1_strength,
            "team2_strength": team2_strength,
            "win_split": win_split
        }

        # ---------- FINAL RESPONSE ----------
        return {
            "winner": winner,
            "probability": win_prob,
            "score_inning_1": score_inning_1,
            "score_inning_2": score_inning_2,
            "analysis": analysis
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
