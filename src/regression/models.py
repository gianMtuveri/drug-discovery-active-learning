from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge


def make_regression_model(model_name="random_forest", random_state=42):
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            random_state=random_state,
        )

    if model_name == "bayesian_ridge":
        return BayesianRidge()

    raise ValueError(
        "model_name must be one of: "
        "'random_forest', 'gradient_boosting', 'bayesian_ridge'"
    )