import pandas as pd
from sklearn.model_selection import train_test_split
from data import load_activities
from features import feature_engineer
from model import ModelRunner

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "distance_km",
    "avg_hr",
    "avg_cadence",
    "elevation_per_km",
    "weekly_km",
    "rolling_pace",
    "hr_percent_max",
    "effort_pace"
]

TARGET_COLUMN = "total_time_min"

def train_model(test_size=0.2):
    raw_df = load_activities()
    feature_engineer("../data/raw/runs.csv")

    df = pd.read_csv("../data/processed/features.csv")
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    runner = ModelRunner()
    results_df = runner.train_and_evaluate(X_train, y_train, X_val, y_val)

    best_model_name = results_df.iloc[0]["Model"]
    #feat_imp = runner.get_feature_importance(best_model_name, FEATURE_COLUMNS)

    return {
        "results": results_df,
        "best_model": best_model_name,
        "models": runner.trained_models,
        #"feature_importance": feat_imp,
        "X_val": X_val,
        "y_val": y_val
    }