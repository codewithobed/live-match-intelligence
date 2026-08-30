"""
Train and compare match outcome models for LiveMatch Intelligence.

Input:
    data/historical_match_states.csv

Models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting

Evaluation:
    - Accuracy
    - Macro F1
    - Log Loss

Important:
    The train/test split is performed at MATCH level so all six snapshots
    from a match remain together. This prevents leakage.

Outputs:
    models/best_match_outcome_model.joblib
    models/match_outcome_model_metadata.joblib
    data/model_comparison_results.csv
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "historical_match_states.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_FILE = PROJECT_ROOT / "data" / "model_comparison_results.csv"
MODEL_FILE = MODELS_DIR / "best_match_outcome_model.joblib"
METADATA_FILE = MODELS_DIR / "match_outcome_model_metadata.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ---------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "snapshot_minute",

    "home_goals",
    "away_goals",
    "goal_difference",

    "home_xg",
    "away_xg",
    "xg_difference",

    "home_shots",
    "away_shots",
    "shot_difference",

    "home_passes",
    "away_passes",
    "pass_difference",

    "home_pass_completion",
    "away_pass_completion",

    "home_pressures",
    "away_pressures",
    "pressure_difference",

    "home_carries",
    "away_carries",

    "home_recoveries",
    "away_recoveries",

    "home_interceptions",
    "away_interceptions",

    "home_recent_xg",
    "away_recent_xg",

    "home_recent_shots",
    "away_recent_shots",

    "home_recent_pressures",
    "away_recent_pressures",

    "home_momentum",
    "away_momentum",
    "momentum_difference",
]

TARGET_COLUMN = "target"
GROUP_COLUMN = "match_id"

CLASS_ORDER = ["Home Win", "Draw", "Away Win"]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def validate_input(df):
    required = set(
        FEATURE_COLUMNS
        + [TARGET_COLUMN, GROUP_COLUMN]
    )

    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            f"Dataset is missing required columns: {missing}"
        )

    if df[TARGET_COLUMN].isna().any():
        raise RuntimeError("Target column contains missing values.")

    if df[GROUP_COLUMN].isna().any():
        raise RuntimeError("match_id contains missing values.")


def make_match_level_split(df):
    """
    Split by match_id so every snapshot from the same match is in
    either train or test, never both.
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            y=df[TARGET_COLUMN],
            groups=df[GROUP_COLUMN],
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_matches = set(
        train_df[GROUP_COLUMN].unique()
    )
    test_matches = set(
        test_df[GROUP_COLUMN].unique()
    )

    overlap = train_matches.intersection(test_matches)

    if overlap:
        raise RuntimeError(
            "Leakage detected: some match IDs are in both train and test."
        )

    return train_df, test_df


def build_logistic_regression_pipeline():
    """
    Scale numeric features for Logistic Regression.
    """
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_random_forest_pipeline():
    """
    Tree model with median imputation.
    """
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_gradient_boosting_pipeline():
    """
    Gradient Boosting with median imputation.
    """
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def align_probabilities(model, raw_probabilities):
    """
    Reorder model probabilities to:
        Home Win, Draw, Away Win
    """
    model_classes = list(
        model.named_steps["model"].classes_
    )

    aligned = np.zeros(
        (raw_probabilities.shape[0], len(CLASS_ORDER)),
        dtype=float,
    )

    for target_index, class_name in enumerate(CLASS_ORDER):
        if class_name in model_classes:
            source_index = model_classes.index(class_name)
            aligned[:, target_index] = raw_probabilities[:, source_index]

    return aligned


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    raw_probabilities = model.predict_proba(
        X_test
    )

    probabilities = align_probabilities(
        model,
        raw_probabilities,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    loss = log_loss(
        y_test,
        probabilities,
        labels=CLASS_ORDER,
    )

    print(
        f"  Accuracy : {accuracy:.4f}"
    )
    print(
        f"  Macro F1 : {macro_f1:.4f}"
    )
    print(
        f"  Log Loss : {loss:.4f}"
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Log Loss": loss,
        "Pipeline": model,
    }


def score_model_for_selection(result):
    """
    Selection rule:
    prioritize probability quality while still rewarding classification quality.

    Lower log loss is better.
    Higher macro F1 and accuracy are better.
    """
    return (
        result["Macro F1"] * 0.45
        + result["Accuracy"] * 0.25
        + (1.0 / (1.0 + result["Log Loss"])) * 0.30
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")

    print_section(
        "LIVE MATCH INTELLIGENCE — MODEL TRAINING"
    )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    validate_input(df)

    print(
        f"Loaded {len(df):,} snapshot rows "
        f"from {df[GROUP_COLUMN].nunique():,} matches."
    )

    print("\nTarget distribution:")
    print(
        df[TARGET_COLUMN]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # Match-level train/test split
    # -----------------------------------------------------

    print_section("MATCH-LEVEL TRAIN / TEST SPLIT")

    train_df, test_df = make_match_level_split(
        df
    )

    train_match_count = train_df[GROUP_COLUMN].nunique()
    test_match_count = test_df[GROUP_COLUMN].nunique()

    print(
        f"Training matches: {train_match_count}"
    )
    print(
        f"Test matches:     {test_match_count}"
    )
    print(
        f"Training rows:    {len(train_df)}"
    )
    print(
        f"Test rows:        {len(test_df)}"
    )

    print("\nTraining target distribution:")
    print(
        train_df[TARGET_COLUMN]
        .value_counts()
        .to_string()
    )

    print("\nTest target distribution:")
    print(
        test_df[TARGET_COLUMN]
        .value_counts()
        .to_string()
    )

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    # -----------------------------------------------------
    # Train models
    # -----------------------------------------------------

    print_section("TRAINING MODELS")

    models = {
        "Logistic Regression":
            build_logistic_regression_pipeline(),

        "Random Forest":
            build_random_forest_pipeline(),

        "Gradient Boosting":
            build_gradient_boosting_pipeline(),
    }

    results = []

    for name, model in models.items():
        result = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        result["Selection Score"] = (
            score_model_for_selection(result)
        )
        results.append(result)

    # -----------------------------------------------------
    # Comparison table
    # -----------------------------------------------------

    print_section("MODEL COMPARISON")

    comparison_df = pd.DataFrame(
        [
            {
                "Model": result["Model"],
                "Accuracy": result["Accuracy"],
                "Macro F1": result["Macro F1"],
                "Log Loss": result["Log Loss"],
                "Selection Score": result["Selection Score"],
            }
            for result in results
        ]
    )

    comparison_df = comparison_df.sort_values(
        by=[
            "Selection Score",
            "Macro F1",
            "Accuracy",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    print(
        comparison_df.round(4).to_string(index=False)
    )

    best_model_name = comparison_df.iloc[0]["Model"]

    best_result = next(
        result
        for result in results
        if result["Model"] == best_model_name
    )

    best_model = best_result["Pipeline"]

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    print_section("SAVING MODEL")

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    joblib.dump(
        best_model,
        MODEL_FILE,
    )

    metadata = {
        "best_model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "class_order": CLASS_ORDER,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_match_count": int(train_match_count),
        "test_match_count": int(test_match_count),
        "train_row_count": int(len(train_df)),
        "test_row_count": int(len(test_df)),
        "metrics": {
            row["Model"]: {
                "accuracy": float(row["Accuracy"]),
                "macro_f1": float(row["Macro F1"]),
                "log_loss": float(row["Log Loss"]),
                "selection_score": float(
                    row["Selection Score"]
                ),
            }
            for _, row in comparison_df.iterrows()
        },
    }

    joblib.dump(
        metadata,
        METADATA_FILE,
    )

    # Also write human-readable JSON metadata.
    metadata_json_file = (
        MODELS_DIR
        / "match_outcome_model_metadata.json"
    )

    with metadata_json_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"Best model: {best_model_name}"
    )
    print(
        f"Saved model: {MODEL_FILE}"
    )
    print(
        f"Saved metadata: {METADATA_FILE}"
    )
    print(
        f"Saved comparison: {RESULTS_FILE}"
    )

    print("\nBest-model metrics:")
    print(
        f"  Accuracy : "
        f"{best_result['Accuracy']:.4f}"
    )
    print(
        f"  Macro F1 : "
        f"{best_result['Macro F1']:.4f}"
    )
    print(
        f"  Log Loss : "
        f"{best_result['Log Loss']:.4f}"
    )

    print_section("TRAINING COMPLETE")

    print(
        "The best model has been saved successfully."
    )
    print(
        "Do not connect it to the dashboard yet."
    )
    print(
        "Next step: inspect model performance by snapshot minute "
        "and check probability behaviour before replacing the "
        "rule-based predictive prototype."
    )


if __name__ == "__main__":
    main()
