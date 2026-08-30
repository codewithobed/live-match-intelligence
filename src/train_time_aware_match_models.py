"""
Train time-aware match outcome models for LiveMatch Intelligence.

Instead of forcing one classifier to learn every match state from 15' to 85',
this script trains a separate model for each snapshot minute:

    15, 30, 45, 60, 75, 85

For every minute it compares:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting

Evaluation:
    - Accuracy
    - Macro F1
    - Log Loss

Important:
    The SAME match-level 80/20 split is used at every minute.
    A match never appears in both train and test data.

Outputs:
    models/time_aware/
        match_outcome_15.joblib
        match_outcome_30.joblib
        match_outcome_45.joblib
        match_outcome_60.joblib
        match_outcome_75.joblib
        match_outcome_85.joblib
        time_aware_metadata.joblib
        time_aware_metadata.json

    data/time_aware_model_results.csv
"""

from pathlib import Path
import json
import sys
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
# Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "historical_match_states.csv"

MODELS_DIR = PROJECT_ROOT / "models" / "time_aware"

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "time_aware_model_results.csv"
)

METADATA_FILE = (
    MODELS_DIR
    / "time_aware_metadata.joblib"
)

METADATA_JSON_FILE = (
    MODELS_DIR
    / "time_aware_metadata.json"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

SNAPSHOT_MINUTES = [15, 30, 45, 60, 75, 85]

CLASS_ORDER = [
    "Home Win",
    "Draw",
    "Away Win",
]


# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

# snapshot_minute is excluded because each model is trained at only one minute.
FEATURE_COLUMNS = [
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


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def validate_dataset(df):
    required = set(
        FEATURE_COLUMNS
        + [
            "match_id",
            "snapshot_minute",
            "target",
        ]
    )

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Dataset is missing required columns: {missing}"
        )

    observed_minutes = sorted(
        df["snapshot_minute"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    missing_minutes = [
        minute
        for minute in SNAPSHOT_MINUTES
        if minute not in observed_minutes
    ]

    if missing_minutes:
        raise RuntimeError(
            f"Dataset is missing snapshot minutes: {missing_minutes}"
        )


def make_match_split(df):
    """
    Create one global train/test split at match level.

    We split a one-row-per-match table so that the exact same matches
    are used for training/testing at every snapshot minute.
    """
    match_table = (
        df[
            [
                "match_id",
                "target",
            ]
        ]
        .drop_duplicates("match_id")
        .reset_index(drop=True)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            match_table,
            y=match_table["target"],
            groups=match_table["match_id"],
        )
    )

    train_matches = set(
        match_table.iloc[train_idx]["match_id"]
        .astype(int)
        .tolist()
    )

    test_matches = set(
        match_table.iloc[test_idx]["match_id"]
        .astype(int)
        .tolist()
    )

    if train_matches.intersection(test_matches):
        raise RuntimeError(
            "Leakage detected: train/test match IDs overlap."
        )

    return train_matches, test_matches


def make_preprocessor(scale=False):
    steps = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]

    if scale:
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=steps
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )


def build_logistic_regression():
    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(
                    scale=True
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest():
    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(
                    scale=False
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=600,
                    max_depth=8,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_gradient_boosting():
    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(
                    scale=False
                ),
            ),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=220,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def model_candidates():
    return {
        "Logistic Regression":
            build_logistic_regression(),

        "Random Forest":
            build_random_forest(),

        "Gradient Boosting":
            build_gradient_boosting(),
    }


def align_probabilities(model, raw_probabilities):
    model_classes = list(
        model.named_steps["model"].classes_
    )

    aligned = np.zeros(
        (
            raw_probabilities.shape[0],
            len(CLASS_ORDER),
        ),
        dtype=float,
    )

    for target_index, class_name in enumerate(CLASS_ORDER):
        if class_name in model_classes:
            source_index = model_classes.index(class_name)
            aligned[:, target_index] = (
                raw_probabilities[:, source_index]
            )

    return aligned


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    probabilities = align_probabilities(
        model,
        model.predict_proba(X_test),
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

    confidence = probabilities.max(axis=1)
    correct_mask = (
        predictions == y_test.to_numpy()
    )

    wrong_confidence = (
        float(
            confidence[
                ~correct_mask
            ].mean()
        )
        if (~correct_mask).any()
        else 0.0
    )

    return {
        "Model": name,
        "Accuracy": float(accuracy),
        "Macro F1": float(macro_f1),
        "Log Loss": float(loss),
        "Wrong Confidence": wrong_confidence,
        "Pipeline": model,
    }


def selection_score(result):
    """
    Combined score for choosing a practical dashboard model.

    Classification quality matters, but probability quality is especially
    important because the dashboard displays percentages.
    """
    probability_quality = (
        1.0 / (1.0 + result["Log Loss"])
    )

    return (
        result["Macro F1"] * 0.40
        + result["Accuracy"] * 0.20
        + probability_quality * 0.40
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")

    section(
        "LIVE MATCH INTELLIGENCE — TIME-AWARE MODEL TRAINING"
    )

    if not DATA_FILE.exists():
        print(
            f"[FAIL] Dataset not found: {DATA_FILE}"
        )
        sys.exit(1)

    df = pd.read_csv(
        DATA_FILE
    )

    validate_dataset(df)

    print(
        f"Rows: {len(df):,}"
    )
    print(
        f"Unique matches: "
        f"{df['match_id'].nunique():,}"
    )

    train_matches, test_matches = make_match_split(
        df
    )

    print(
        f"Training matches: {len(train_matches):,}"
    )
    print(
        f"Test matches: {len(test_matches):,}"
    )

    section(
        "TRAINING SEPARATE MODELS BY MATCH MINUTE"
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []
    minute_metadata = {}

    for minute in SNAPSHOT_MINUTES:

        print(
            f"\n{'-' * 78}"
        )
        print(
            f"SNAPSHOT MINUTE: {minute}'"
        )
        print(
            f"{'-' * 78}"
        )

        minute_df = df[
            df["snapshot_minute"] == minute
        ].copy()

        train_df = minute_df[
            minute_df["match_id"].isin(
                train_matches
            )
        ].copy()

        test_df = minute_df[
            minute_df["match_id"].isin(
                test_matches
            )
        ].copy()

        if train_df.empty or test_df.empty:
            raise RuntimeError(
                f"No train/test rows for minute {minute}."
            )

        X_train = train_df[
            FEATURE_COLUMNS
        ].copy()

        y_train = train_df[
            "target"
        ].copy()

        X_test = test_df[
            FEATURE_COLUMNS
        ].copy()

        y_test = test_df[
            "target"
        ].copy()

        print(
            f"Training rows: {len(train_df)} | "
            f"Test rows: {len(test_df)}"
        )

        print(
            "Test target distribution:"
        )
        print(
            y_test
            .value_counts()
            .to_string()
        )

        minute_results = []

        for name, model in model_candidates().items():

            result = evaluate_model(
                name,
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )

            result[
                "Selection Score"
            ] = selection_score(
                result
            )

            minute_results.append(
                result
            )

            print(
                f"\n{name}"
            )
            print(
                f"  Accuracy         : "
                f"{result['Accuracy']:.4f}"
            )
            print(
                f"  Macro F1         : "
                f"{result['Macro F1']:.4f}"
            )
            print(
                f"  Log Loss         : "
                f"{result['Log Loss']:.4f}"
            )
            print(
                f"  Wrong Confidence : "
                f"{result['Wrong Confidence']:.4f}"
            )

        # Choose best model for this minute.
        best_result = max(
            minute_results,
            key=selection_score,
        )

        best_model = best_result[
            "Pipeline"
        ]

        best_model_name = best_result[
            "Model"
        ]

        model_path = (
            MODELS_DIR
            / f"match_outcome_{minute}.joblib"
        )

        joblib.dump(
            best_model,
            model_path,
        )

        print(
            f"\nBEST MODEL AT {minute}': "
            f"{best_model_name}"
        )
        print(
            f"Saved: {model_path}"
        )

        minute_metadata[str(minute)] = {
            "best_model": best_model_name,
            "accuracy": best_result["Accuracy"],
            "macro_f1": best_result["Macro F1"],
            "log_loss": best_result["Log Loss"],
            "wrong_confidence": (
                best_result[
                    "Wrong Confidence"
                ]
            ),
            "selection_score": (
                best_result[
                    "Selection Score"
                ]
            ),
            "model_file": str(
                model_path.name
            ),
        }

        for result in minute_results:
            all_results.append(
                {
                    "Snapshot Minute": minute,
                    "Model": result["Model"],
                    "Accuracy": result["Accuracy"],
                    "Macro F1": result["Macro F1"],
                    "Log Loss": result["Log Loss"],
                    "Wrong Confidence": (
                        result[
                            "Wrong Confidence"
                        ]
                    ),
                    "Selection Score": (
                        result[
                            "Selection Score"
                        ]
                    ),
                    "Selected": (
                        result["Model"]
                        == best_model_name
                    ),
                }
            )

    # -----------------------------------------------------
    # Save results / metadata
    # -----------------------------------------------------

    section(
        "TIME-AWARE MODEL SUMMARY"
    )

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    selected_df = results_df[
        results_df["Selected"]
    ].copy()

    selected_df = selected_df.sort_values(
        "Snapshot Minute"
    )

    print(
        selected_df[
            [
                "Snapshot Minute",
                "Model",
                "Accuracy",
                "Macro F1",
                "Log Loss",
                "Wrong Confidence",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "class_order": CLASS_ORDER,
        "snapshot_minutes": SNAPSHOT_MINUTES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_match_count": len(
            train_matches
        ),
        "test_match_count": len(
            test_matches
        ),
        "minutes": minute_metadata,
    }

    joblib.dump(
        metadata,
        METADATA_FILE,
    )

    with METADATA_JSON_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # Interpretation
    # -----------------------------------------------------

    section(
        "INTERPRETATION"
    )

    early = selected_df[
        selected_df[
            "Snapshot Minute"
        ] <= 30
    ]

    late = selected_df[
        selected_df[
            "Snapshot Minute"
        ] >= 75
    ]

    early_accuracy = float(
        early["Accuracy"].mean()
    )

    late_accuracy = float(
        late["Accuracy"].mean()
    )

    early_log_loss = float(
        early["Log Loss"].mean()
    )

    late_log_loss = float(
        late["Log Loss"].mean()
    )

    print(
        f"Average early accuracy (15'-30'): "
        f"{early_accuracy:.4f}"
    )
    print(
        f"Average late accuracy (75'-85'): "
        f"{late_accuracy:.4f}"
    )

    print(
        f"Average early log loss (15'-30'): "
        f"{early_log_loss:.4f}"
    )
    print(
        f"Average late log loss (75'-85'): "
        f"{late_log_loss:.4f}"
    )

    if late_accuracy > early_accuracy:
        print(
            "\n[PASS] Time-aware classification "
            "improves later in matches."
        )
    else:
        print(
            "\n[WARNING] Time-aware classification "
            "does not improve later in matches."
        )

    if late_log_loss < early_log_loss:
        print(
            "[PASS] Time-aware probability quality "
            "improves later in matches."
        )
    else:
        print(
            "[WARNING] Time-aware probability quality "
            "does not improve later in matches."
        )

    section(
        "FINAL RECOMMENDATION"
    )

    # This threshold is deliberately conservative.
    if (
        late_accuracy >= 0.65
        and late_log_loss <= 1.10
    ):
        print(
            "[PASS] The time-aware late-match models "
            "show promising predictive behaviour."
        )
        print(
            "Next step: calibrate each selected minute model "
            "and compare against the current benchmark."
        )
    else:
        print(
            "[PARTIAL RESULT] The time-aware models were "
            "trained successfully, but performance still needs "
            "improvement before dashboard replacement."
        )
        print(
            "Use the results table to identify which minutes "
            "and model families perform best before the next iteration."
        )

    print(
        f"\nSaved full results:"
    )
    print(
        f"  {RESULTS_FILE}"
    )

    print(
        f"\nSaved time-aware models:"
    )

    for minute in SNAPSHOT_MINUTES:
        print(
            f"  {MODELS_DIR / f'match_outcome_{minute}.joblib'}"
        )


if __name__ == "__main__":
    main()
