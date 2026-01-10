import streamlit as st
import pandas as pd
import pickle

# ---------------- LOAD MODELS ----------------
winner_model = pickle.load(open("models/winner_model.pkl", "rb"))
le = pickle.load(open("models/label_encoder.pkl", "rb"))
winner_features = pickle.load(open("models/feature_columns.pkl", "rb"))

score_1 = pickle.load(open("models/first_innings_model.pkl", "rb"))
score_2 = pickle.load(open("models/second_innings_model.pkl", "rb"))

# ---------------- LOAD DATA ----------------
players_df = pd.read_csv("data/player_stats_normalized.csv")
teams_df = pd.read_csv("data/team_features_normalized.csv")

# ---------------- UI ----------------
st.title("🏏 IPL Match Predictor")

team1 = st.selectbox("Select Team 1", teams_df["team"].unique())
team2 = st.selectbox("Select Team 2", teams_df["team"].unique())

toss_winner = st.selectbox("Toss Winner", [team1, team2])
batting_first = st.selectbox("Batting First", [team1, team2])

st.subheader("Select Playing XI (Not used yet)")

xi_team1 = st.multiselect("Playing XI - Team 1", players_df["Player"].unique(), max_selections=11)
xi_team2 = st.multiselect("Playing XI - Team 2", players_df["Player"].unique(), max_selections=11)

# ---------------- PREDICTION ----------------
if st.button("Predict Match"):
    t1 = teams_df[teams_df["team"] == team1].iloc[0]
    t2 = teams_df[teams_df["team"] == team2].iloc[0]

    # ✅ 1. MASTER INPUT DF (ALL FEATURES)
    input_df = pd.DataFrame([{
        "team_1_enc": le.transform([team1])[0],
        "team_2_enc": le.transform([team2])[0],
        "toss_winner_enc": le.transform([toss_winner])[0],
        "batting_first_enc": 1 if batting_first == team1 else 0,

        "batting_rating_t1": t1["batting_rating"],
        "bowling_rating_t1": t1["bowling_rating"],
        "batting_rating_t2": t2["batting_rating"],
        "bowling_rating_t2": t2["bowling_rating"],

        "avg_score_last_5_t1": t1["avg_score_last_5"],
        "avg_score_last_5_t2": t2["avg_score_last_5"],
    }])

    # ✅ 2. WINNER MODEL INPUT (ONLY TRAINED FEATURES)
    winner_input = input_df[winner_features]

    winner = winner_model.predict(winner_input)[0]
    prob = max(winner_model.predict_proba(winner_input)[0]) * 100

    # ✅ 3. SCORE MODEL INPUT (ONLY TRAINED FEATURES)
    score_features = [
        "batting_rating_t1", "bowling_rating_t2",
        "batting_rating_t2", "bowling_rating_t1",
        "avg_score_last_5_t1", "avg_score_last_5_t2"
    ]

    score_input = input_df[score_features]

    s1 = int(score_1.predict(score_input)[0])
    s2 = int(score_2.predict(score_input)[0])
    # ---------------- CRICKET LOGIC FIX ----------------

    # Identify batting second team
    batting_second = team2 if batting_first == team1 else team1

    # If chasing team wins, second innings must be higher
    if winner == batting_second and s2 <= s1:
        s2 = s1 + int(abs(s1 - s2) * 0.6) + 1

    # ---------------- OUTPUT ----------------
    st.success(f"🏆 Predicted Winner: {winner}")
    st.info(f"📊 Winning Probability: {prob:.2f}%")
    st.write(f"🏏 First Innings Score: {s1}")
    st.write(f"🏏 Second Innings Score: {s2}")