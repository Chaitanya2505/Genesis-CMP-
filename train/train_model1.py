import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Setup
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# Load datasets
# -----------------------------
match_df = pd.read_csv("data/match_level_dataset.csv")
team_df = pd.read_csv("data/team_features_normalized.csv")

# -----------------------------
# Merge Team 1 stats
# -----------------------------
df = match_df.merge(
    team_df.add_prefix("team1_"),
    left_on="team_1",
    right_on="team1_team",
    how="left"
)

# -----------------------------
# Merge Team 2 stats
# -----------------------------
df = df.merge(
    team_df.add_prefix("team2_"),
    left_on="team_2",
    right_on="team2_team",
    how="left"
)

# -----------------------------
# Encode match-level categorical info
# -----------------------------
df["toss_winner_team1"] = (df["toss_winner"] == df["team_1"]).astype(int)
df["batting_first_team1"] = (df["batting_first"] == df["team_1"]).astype(int)

# -----------------------------
# Encode target
# -----------------------------
le = LabelEncoder()
df["winner_encoded"] = le.fit_transform(df["winner"])

# -----------------------------
# Drop non-numeric / leakage columns
# -----------------------------
drop_cols = [
    "match_id", "season", "match_date",
    "team_1", "team_2",
    "toss_winner", "toss_decision",
    "batting_first", "winner",
    "team1_team", "team2_team",
    "second_innings_score"  # ❌ remove leakage
]

df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# -----------------------------
# Features & target
# -----------------------------
X = df.drop(columns=["winner_encoded"])
y = df["winner_encoded"]

# Save feature order
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
joblib.dump(le, "models/label_encoder.pkl")

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "models/winner_model.pkl")

print("✅ Model trained and saved successfully")