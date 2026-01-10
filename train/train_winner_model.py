import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MATCH_PATH = os.path.join(BASE_DIR, "data", "match_level_dataset.csv")
TEAM_PATH = os.path.join(BASE_DIR, "data", "team_features_normalized.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)
# --------------------------------

match_df = pd.read_csv(MATCH_PATH)
team_df = pd.read_csv(TEAM_PATH)

# Merge team features
df = match_df.merge(team_df, left_on="team_1", right_on="team", how="left")
df = df.merge(team_df, left_on="team_2", right_on="team", how="left", suffixes=("_t1", "_t2"))

# Encode categorical columns
le_team = LabelEncoder()
df["team_1_enc"] = le_team.fit_transform(df["team_1"])
df["team_2_enc"] = le_team.transform(df["team_2"])
df["toss_winner_enc"] = le_team.transform(df["toss_winner"])

df["batting_first_enc"] = (df["batting_first"] == df["team_1"]).astype(int)

features = [
    "team_1_enc", "team_2_enc",
    "toss_winner_enc", "batting_first_enc",
    "batting_rating_t1", "bowling_rating_t1",
    "batting_rating_t2", "bowling_rating_t2"
]

X = df[features]
y = df["winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save everything
with open(os.path.join(MODEL_DIR, "winner_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le_team, f)

with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "wb") as f:
    pickle.dump(features, f)

print("✅ Winner model trained & saved")