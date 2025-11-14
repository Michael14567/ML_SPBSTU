from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
DATA_PATH = Path("data.csv")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the raw CSV and create helper columns."""
    df = pd.read_csv(path)
    rename_map = {
        "Whole weight.1": "Shucked weight",
        "Whole weight.2": "Viscera weight",
    }
    df = df.rename(columns=rename_map)
    df["Age"] = df["Rings"] + 1.5  # Abalone age proxy (years)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-inspired ratios and volumetric metrics."""
    enriched = df.copy()
    eps = 1e-6
    enriched["Volume"] = (
        enriched["Length"] * enriched["Diameter"] * enriched["Height"]
    )
    enriched["Height_to_Length"] = enriched["Height"] / (
        enriched["Length"] + eps
    )
    enriched["Diameter_to_Length"] = enriched["Diameter"] / (
        enriched["Length"] + eps
    )
    enriched["Shell_to_Whole"] = enriched["Shell weight"] / (
        enriched["Whole weight"] + eps
    )
    enriched["Viscera_to_Whole"] = enriched["Viscera weight"] / (
        enriched["Whole weight"] + eps
    )
    enriched["Shucked_to_Whole"] = enriched["Shucked weight"] / (
        enriched["Whole weight"] + eps
    )
    enriched["Density"] = enriched["Whole weight"] / (
        enriched["Volume"] + eps
    )
    enriched["Mass_per_Length"] = enriched["Whole weight"] / (
        enriched["Length"] + eps
    )
    enriched["Shell_mass_per_Length"] = enriched["Shell weight"] / (
        enriched["Length"] + eps
    )

    derived_cols = [
        "Volume",
        "Height_to_Length",
        "Diameter_to_Length",
        "Shell_to_Whole",
        "Viscera_to_Whole",
        "Shucked_to_Whole",
        "Density",
        "Mass_per_Length",
        "Shell_mass_per_Length",
    ]
    enriched[derived_cols] = enriched[derived_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    enriched[derived_cols] = enriched[derived_cols].fillna(0.0)
    return enriched


def compute_permutation_mae(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 5,
    random_state: int = SEED,
) -> pd.DataFrame:
    """Estimate feature contribution via MAE increase on shuffled columns."""
    rng = np.random.default_rng(random_state)
    baseline = mean_absolute_error(y, model.predict(X))
    X_perm = X.copy()
    importances: list[tuple[str, float]] = []

    for feature in X.columns:
        original = X[feature].to_numpy(copy=True)
        deltas: list[float] = []
        for _ in range(n_repeats):
            shuffled = original.copy()
            rng.shuffle(shuffled)
            X_perm[feature] = shuffled
            mae = mean_absolute_error(y, model.predict(X_perm))
            deltas.append(mae - baseline)
        X_perm[feature] = original
        importances.append((feature, float(np.mean(deltas))))

    return (
        pd.DataFrame(importances, columns=["feature", "mae_increase"])
        .sort_values("mae_increase", ascending=False)
        .reset_index(drop=True)
    )


def run_eda(df: pd.DataFrame) -> None:
    """Print quick exploratory insights."""
    print("=== Dataset snapshot ===")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(df.head())
    print("\n=== Missing values ===")
    print(df.isna().sum())
    print("\n=== Sex distribution ===")
    print(df["Sex"].value_counts(normalize=True).mul(100).round(2))
    print("\n=== Age statistics (years) ===")
    print(df["Age"].describe())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(["Age"])
    corr = df[numeric_cols].corrwith(df["Age"]).sort_values(ascending=False)
    print("\n=== Top correlations with Age ===")
    print(corr.head(15))


def train_and_evaluate(df: pd.DataFrame) -> None:
    """Train MLP regressor with hyper-parameter search and report metrics."""
    feature_df = df.drop(columns=["Age", "Rings", "id"])
    target = df["Age"]
    numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = feature_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = MLPRegressor(
        random_state=SEED,
        solver="adam",
        max_iter=250,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    param_grid = {
        "model__hidden_layer_sizes": [(64, 32), (96, 48)],
        "model__activation": ["relu", "tanh"],
        "model__alpha": [1e-4, 1e-3],
        "model__learning_rate_init": [0.001],
        "model__batch_size": [128],
    }

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        target,
        test_size=0.2,
        random_state=SEED,
        stratify=pd.qcut(target, q=10, duplicates="drop"),
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)
    print("\n=== Best hyper-parameters ===")
    print(grid.best_params_)
    print(f"Best CV MAE: {-grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    print("\n=== Hold-out performance ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")

    residuals = preds - y_test
    print("\n=== Residual diagnostics ===")
    print(f"Residual mean: {residuals.mean():.4f}")
    print(f"Residual std : {residuals.std():.4f}")

    top_results = (
        pd.DataFrame(grid.cv_results_)
        .sort_values("rank_test_score")
        .head(5)[
            [
                "rank_test_score",
                "mean_test_score",
                "param_model__hidden_layer_sizes",
                "param_model__activation",
                "param_model__alpha",
                "param_model__learning_rate_init",
                "param_model__batch_size",
            ]
        ]
    )
    print("\n=== Top-5 CV configurations (higher is better) ===")
    print(top_results)

    importances = compute_permutation_mae(
        best_model, X_test, y_test, n_repeats=5, random_state=SEED
    )
    print("\n=== Feature contributions (MAE change when permuted) ===")
    print(importances.head(15))


def main() -> None:
    df = load_dataset(DATA_PATH)
    df = engineer_features(df)
    run_eda(df)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
