import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MATCH_PATH = os.path.join(BASE_DIR, "data", "match_level_dataset.csv")
TEAM_PATH = os.path.join(BASE_DIR, "data", "team_features_normalized.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)
# --------------------------------

match_df = pd.read_csv(MATCH_PATH)
team_df = pd.read_csv(TEAM_PATH)

df = match_df.merge(team_df, left_on="team_1", right_on="team", how="left")
df = df.merge(team_df, left_on="team_2", right_on="team", how="left", suffixes=("_t1", "_t2"))

features = [
    "batting_rating_t1", "bowling_rating_t2",
    "batting_rating_t2", "bowling_rating_t1",
    "avg_score_last_5_t1", "avg_score_last_5_t2"
]

X = df[features]

y1 = df["first_innings_score"]
y2 = df["second_innings_score"]

model_1 = RandomForestRegressor(n_estimators=200, random_state=42)
model_2 = RandomForestRegressor(n_estimators=200, random_state=42)

model_1.fit(X, y1)
model_2.fit(X, y2)

pickle.dump(model_1, open(os.path.join(MODEL_DIR, "first_innings_model.pkl"), "wb"))
pickle.dump(model_2, open(os.path.join(MODEL_DIR, "second_innings_model.pkl"), "wb"))

print("✅ Score models trained & saved")