import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYER_PATH = os.path.join(BASE_DIR, "data", "player_stats_normalized.csv")

players_df = pd.read_csv(PLAYER_PATH)

def build_team_features(playing_xi):
    df = players_df[players_df["Player"].isin(playing_xi)]

    features = {
        "avg_batting_avg": df["Avg"].mean(),
        "avg_strike_rate": df["SR"].mean(),
        "total_runs": df["Runs"].sum(),
        "avg_bowling_econ": df["B_Econ"].mean(),
        "total_wickets": df["B_Wkts"].sum(),
        "experience": df["Mat"].sum()
    }

    return features