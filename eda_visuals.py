from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis import (
    SEED,
    compute_permutation_mae,
    engineer_features,
    load_dataset,
)

DATA_PATH = Path("data.csv")
FIG_DIR = Path("figures")


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    features = df.drop(columns=["Age", "Rings", "id"])
    target = df["Age"]
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(include=["object", "category"]).columns.tolist()
    return features, target, numeric_features, categorical_features


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="tanh",
        alpha=1e-3,
        batch_size=256,
        learning_rate_init=1e-3,
        solver="adam",
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=SEED,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    out_path = FIG_DIR / name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_age_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Age"], bins=40, kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Age distribution (years)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    save_fig(fig, "age_distribution.png")


def plot_volume_vs_weight(df: pd.DataFrame) -> None:
    sample = df.sample(n=min(5000, len(df)), random_state=SEED)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=sample,
        x="Volume",
        y="Whole weight",
        hue="Sex",
        alpha=0.5,
        ax=ax,
    )
    ax.set_title("Whole weight vs. engineered Volume")
    save_fig(fig, "volume_vs_weight.png")


def plot_density_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Density"], bins=40, kde=True, color="#55A868", ax=ax)
    ax.set_title("Density distribution (Whole weight / Volume)")
    save_fig(fig, "density_distribution.png")


def plot_residuals(preds: pd.Series, residuals: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=preds, y=residuals, alpha=0.4, ax=ax, color="#C44E52")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted age")
    ax.set_ylabel("Residual (y - y_hat)")
    ax.set_title("Residuals vs. predictions")
    save_fig(fig, "residuals_vs_predictions.png")


def plot_feature_importance(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    sample = X.sample(n=min(4000, len(X)), random_state=SEED)
    y_sample = y.loc[sample.index]
    importance_df = compute_permutation_mae(
        model, sample, y_sample, n_repeats=5, random_state=SEED
    ).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=importance_df,
        y="feature",
        x="mae_increase",
        ax=ax,
        palette="viridis",
    )
    ax.set_title("Permutation importance (ΔMAE)")
    ax.set_xlabel("MAE increase when permuted")
    ax.set_ylabel("")
    save_fig(fig, "feature_importance.png")


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = engineer_features(load_dataset(DATA_PATH))
    X, y, num_cols, cat_cols = prepare_features(df)
    pipeline = build_pipeline(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=pd.qcut(y, q=10, duplicates="drop"),
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    residuals = y_test - preds

    plot_age_distribution(df)
    plot_volume_vs_weight(df)
    plot_density_hist(df)
    plot_residuals(pd.Series(preds, index=y_test.index), residuals)
    plot_feature_importance(pipeline, X_test, y_test)


if __name__ == "__main__":
    main()
