import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model & helpers
# -----------------------------
model = joblib.load("models/winner_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

team_df = pd.read_csv("data/team_features_normalized.csv")

teams = sorted(team_df["team"].unique())

st.set_page_config(page_title="IPL Match Predictor", layout="centered")
st.title("🏏 IPL Match Winner Predictor")

# -----------------------------
# Team Selection
# -----------------------------
st.subheader("Select Teams")

team1 = st.selectbox("Team 1", teams)
team2 = st.selectbox("Team 2", [t for t in teams if t != team1])

# -----------------------------
# Match Conditions
# -----------------------------
st.subheader("Match Conditions")

toss_winner = st.selectbox("Toss Winner", [team1, team2])
batting_first = st.selectbox("Batting First", [team1, team2])
first_innings_score = st.slider("First Innings Score", 100, 260, 160)

# -----------------------------
# Fetch team stats
# -----------------------------
team1_stats = team_df[team_df["team"] == team1].add_prefix("team1_")
team2_stats = team_df[team_df["team"] == team2].add_prefix("team2_")

input_df = pd.concat([team1_stats.reset_index(drop=True),
                      team2_stats.reset_index(drop=True)], axis=1)

# -----------------------------
# Add match-level features
# -----------------------------
input_df["toss_winner_team1"] = int(toss_winner == team1)
input_df["batting_first_team1"] = int(batting_first == team1)
input_df["first_innings_score"] = first_innings_score

# -----------------------------
# Match training feature order
# -----------------------------
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Winner"):
    pred = model.predict(input_df)[0]
    winner = label_encoder.inverse_transform([pred])[0]
    st.success(f"🏆 Predicted Winner: **{winner}**")