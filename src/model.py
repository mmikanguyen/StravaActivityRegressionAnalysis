import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


class ModelRunner:
    def __init__(self, random_state=42):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        }
        self.trained_models = {}

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        results = []

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)

            results.append({
                "Model": name,
                "RMSE": rmse,
                "R2": r2
            })

            self.trained_models[name] = model

        return pd.DataFrame(results).sort_values(by="RMSE")

    def get_feature_importance(self, model_name, feature_names):
        model = self.trained_models.get(model_name)

        if hasattr(model, "feature_importances_"):
            return pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

        return None