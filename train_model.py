import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =============================
# 1. Load datasets
# =============================
team_df = pd.read_csv("data/team_features_normalized.csv")
match_df = pd.read_csv("data/match_level_dataset.csv")

# =============================
# 2. Merge Team 1 stats
# =============================
df = match_df.merge(
    team_df.add_prefix("team1_"),
    left_on="team_1",
    right_on="team1_team",
    how="left"
)

# =============================
# 3. Merge Team 2 stats
# =============================
df = df.merge(
    team_df.add_prefix("team2_"),
    left_on="team_2",
    right_on="team2_team",
    how="left"
)

# =============================
# 4. Encode categorical match info
# =============================
df["toss_winner_team1"] = (df["toss_winner"] == df["team_1"]).astype(int)
df["batting_first_team1"] = (df["batting_first"] == df["team_1"]).astype(int)

# =============================
# 5. Encode target variable
# =============================
le = LabelEncoder()
df["winner_encoded"] = le.fit_transform(df["winner"])

# =============================
# 6. Drop non-numeric / ID columns
# =============================
drop_cols = [
    "match_id",
    "match_date",
    "team_1",
    "team_2",
    "toss_winner",
    "batting_first",
    "winner",
    "team1_team",
    "team2_team",
    "season"
]

df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# =============================
# 7. Split features & target
# =============================
X = df.drop(columns=["winner_encoded"])
y = df["winner_encoded"]

# Save feature order
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
joblib.dump(le, "models/label_encoder.pkl")

# =============================
# 8. Train-test split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 9. Train model
# =============================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# =============================
# 10. Save model
# =============================
joblib.dump(model, "models/winner_model.pkl")

print("✅ Model trained successfully")