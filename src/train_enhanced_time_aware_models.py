"""
Train enhanced time-aware match outcome models for LiveMatch Intelligence.

Input:
    data/enhanced_historical_match_states.csv

Benchmark:
    data/time_aware_model_results.csv

Models trained at each snapshot minute:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting

Evaluation:
    - Accuracy
    - Macro F1
    - Log Loss
    - Wrong-prediction confidence

Important:
    The same global match-level 80/20 split (RANDOM_STATE=42) is used so the
    enhanced models can be compared fairly against the existing benchmark.

Outputs:
    models/enhanced_time_aware/
        match_outcome_15.joblib
        match_outcome_30.joblib
        match_outcome_45.joblib
        match_outcome_60.joblib
        match_outcome_75.joblib
        match_outcome_85.joblib
        enhanced_time_aware_metadata.joblib
        enhanced_time_aware_metadata.json

    data/enhanced_time_aware_model_results.csv
    data/enhanced_vs_benchmark_comparison.csv
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "enhanced_historical_match_states.csv"
)

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "data"
    / "time_aware_model_results.csv"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "enhanced_time_aware"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "enhanced_time_aware_model_results.csv"
)

COMPARISON_FILE = (
    PROJECT_ROOT
    / "data"
    / "enhanced_vs_benchmark_comparison.csv"
)

METADATA_FILE = (
    MODELS_DIR
    / "enhanced_time_aware_metadata.joblib"
)

METADATA_JSON_FILE = (
    MODELS_DIR
    / "enhanced_time_aware_metadata.json"
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
# Columns excluded from model input
# ---------------------------------------------------------

EXCLUDED_COLUMNS = {
    # identifiers / metadata
    "match_id",
    "match_date",
    "competition",
    "season",
    "home_team",
    "away_team",

    # supervised target / future information
    "target",
    "final_home_goals",
    "final_away_goals",

    # each model is trained at one fixed minute
    "snapshot_minute",
}


def section(title):
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)


def determine_feature_columns(df):
    """
    Use all numeric non-leakage columns from the enhanced dataset.
    """
    candidate = df.drop(
        columns=[
            col
            for col in EXCLUDED_COLUMNS
            if col in df.columns
        ],
        errors="ignore",
    )

    numeric_columns = (
        candidate
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    if not numeric_columns:
        raise RuntimeError(
            "No numeric feature columns were found."
        )

    return numeric_columns


def validate_dataset(df):
    required = {
        "match_id",
        "snapshot_minute",
        "target",
    }

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

    for minute in SNAPSHOT_MINUTES:
        if minute not in observed_minutes:
            raise RuntimeError(
                f"Snapshot minute {minute} is missing."
            )


def make_match_split(df):
    """
    Recreate the same global split used by the benchmark.
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


def make_preprocessor(feature_columns, scale=False):
    steps = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
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
                feature_columns,
            )
        ],
        remainder="drop",
    )


def build_logistic_regression(feature_columns):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(
                    feature_columns,
                    scale=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest(feature_columns):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(
                    feature_columns,
                    scale=False,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=700,
                    max_depth=9,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_gradient_boosting(feature_columns):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(
                    feature_columns,
                    scale=False,
                ),
            ),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=260,
                    learning_rate=0.04,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def model_candidates(feature_columns):
    return {
        "Logistic Regression":
            build_logistic_regression(
                feature_columns
            ),

        "Random Forest":
            build_random_forest(
                feature_columns
            ),

        "Gradient Boosting":
            build_gradient_boosting(
                feature_columns
            ),
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
            source_index = model_classes.index(
                class_name
            )
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

    accuracy = float(
        accuracy_score(
            y_test,
            predictions,
        )
    )

    macro_f1 = float(
        f1_score(
            y_test,
            predictions,
            average="macro",
        )
    )

    loss = float(
        log_loss(
            y_test,
            probabilities,
            labels=CLASS_ORDER,
        )
    )

    confidence = probabilities.max(
        axis=1
    )

    correct = (
        predictions
        == y_test.to_numpy()
    )

    wrong_confidence = (
        float(
            confidence[
                ~correct
            ].mean()
        )
        if (~correct).any()
        else 0.0
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Log Loss": loss,
        "Wrong Confidence": wrong_confidence,
        "Pipeline": model,
    }


def selection_score(result):
    """
    Probability quality receives high weight because the dashboard
    ultimately displays probabilities.
    """
    probability_quality = (
        1.0
        / (
            1.0
            + result["Log Loss"]
        )
    )

    return (
        result["Macro F1"] * 0.35
        + result["Accuracy"] * 0.20
        + probability_quality * 0.45
    )


def load_selected_benchmark():
    """
    Read the selected baseline time-aware model at each minute.
    """
    if not BENCHMARK_FILE.exists():
        return None

    benchmark = pd.read_csv(
        BENCHMARK_FILE
    )

    if "Selected" not in benchmark.columns:
        return None

    selected = benchmark[
        benchmark["Selected"].astype(str).str.lower().isin(
            ["true", "1"]
        )
    ].copy()

    if selected.empty:
        return None

    return selected


def main():
    warnings.filterwarnings("ignore")

    section(
        "LIVE MATCH INTELLIGENCE — ENHANCED TIME-AWARE MODEL TRAINING"
    )

    if not DATA_FILE.exists():
        print(
            f"[FAIL] Enhanced dataset not found: {DATA_FILE}"
        )
        sys.exit(1)

    df = pd.read_csv(
        DATA_FILE
    )

    validate_dataset(
        df
    )

    feature_columns = determine_feature_columns(
        df
    )

    print(
        f"Rows: {len(df):,}"
    )
    print(
        f"Unique matches: "
        f"{df['match_id'].nunique():,}"
    )
    print(
        f"Numeric model features: "
        f"{len(feature_columns):,}"
    )

    print(
        "\nLeakage exclusions:"
    )
    for col in sorted(
        EXCLUDED_COLUMNS
    ):
        print(
            f"  - {col}"
        )

    train_matches, test_matches = make_match_split(
        df
    )

    print(
        f"\nTraining matches: "
        f"{len(train_matches):,}"
    )
    print(
        f"Test matches: "
        f"{len(test_matches):,}"
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []
    minute_metadata = {}

    section(
        "TRAINING ENHANCED MODELS BY MATCH MINUTE"
    )

    for minute in SNAPSHOT_MINUTES:

        print(
            f"\n{'-' * 84}"
        )
        print(
            f"SNAPSHOT MINUTE: {minute}'"
        )
        print(
            f"{'-' * 84}"
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

        X_train = train_df[
            feature_columns
        ].copy()

        y_train = train_df[
            "target"
        ].copy()

        X_test = test_df[
            feature_columns
        ].copy()

        y_test = test_df[
            "target"
        ].copy()

        print(
            f"Training rows: {len(train_df)} | "
            f"Test rows: {len(test_df)}"
        )

        minute_results = []

        for name, model in model_candidates(
            feature_columns
        ).items():

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
            f"\nBEST ENHANCED MODEL AT {minute}': "
            f"{best_model_name}"
        )
        print(
            f"Saved: {model_path}"
        )

        minute_metadata[
            str(minute)
        ] = {
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
            "model_file": (
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

    section(
        "ENHANCED TIME-AWARE MODEL SUMMARY"
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
        "feature_columns": feature_columns,
        "class_order": CLASS_ORDER,
        "snapshot_minutes": SNAPSHOT_MINUTES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_match_count": (
            len(train_matches)
        ),
        "test_match_count": (
            len(test_matches)
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
    # Compare to existing benchmark
    # -----------------------------------------------------

    section(
        "ENHANCED VS CURRENT BENCHMARK"
    )

    benchmark = load_selected_benchmark()

    comparison_rows = []

    if benchmark is None:
        print(
            "[WARNING] Existing benchmark results could not be loaded."
        )
        print(
            "Enhanced models were trained successfully, "
            "but automatic before/after comparison was skipped."
        )
    else:
        benchmark = benchmark.copy()

        for minute in SNAPSHOT_MINUTES:
            enhanced_row = selected_df[
                selected_df[
                    "Snapshot Minute"
                ] == minute
            ]

            benchmark_row = benchmark[
                benchmark[
                    "Snapshot Minute"
                ] == minute
            ]

            if (
                enhanced_row.empty
                or benchmark_row.empty
            ):
                continue

            e = enhanced_row.iloc[0]
            b = benchmark_row.iloc[0]

            comparison_rows.append(
                {
                    "Snapshot Minute": minute,
                    "Benchmark Model": (
                        b["Model"]
                    ),
                    "Enhanced Model": (
                        e["Model"]
                    ),

                    "Benchmark Accuracy": (
                        float(
                            b["Accuracy"]
                        )
                    ),
                    "Enhanced Accuracy": (
                        float(
                            e["Accuracy"]
                        )
                    ),
                    "Accuracy Change": (
                        float(
                            e["Accuracy"]
                        )
                        - float(
                            b["Accuracy"]
                        )
                    ),

                    "Benchmark Macro F1": (
                        float(
                            b["Macro F1"]
                        )
                    ),
                    "Enhanced Macro F1": (
                        float(
                            e["Macro F1"]
                        )
                    ),
                    "Macro F1 Change": (
                        float(
                            e["Macro F1"]
                        )
                        - float(
                            b["Macro F1"]
                        )
                    ),

                    "Benchmark Log Loss": (
                        float(
                            b["Log Loss"]
                        )
                    ),
                    "Enhanced Log Loss": (
                        float(
                            e["Log Loss"]
                        )
                    ),
                    "Log Loss Change": (
                        float(
                            e["Log Loss"]
                        )
                        - float(
                            b["Log Loss"]
                        )
                    ),
                }
            )

        comparison_df = pd.DataFrame(
            comparison_rows
        )

        comparison_df.to_csv(
            COMPARISON_FILE,
            index=False,
        )

        print(
            comparison_df[
                [
                    "Snapshot Minute",
                    "Benchmark Model",
                    "Enhanced Model",
                    "Benchmark Accuracy",
                    "Enhanced Accuracy",
                    "Accuracy Change",
                    "Benchmark Log Loss",
                    "Enhanced Log Loss",
                    "Log Loss Change",
                ]
            ]
            .round(4)
            .to_string(index=False)
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
        f"Enhanced early accuracy (15'-30'): "
        f"{early_accuracy:.4f}"
    )
    print(
        f"Enhanced late accuracy (75'-85'): "
        f"{late_accuracy:.4f}"
    )
    print(
        f"Enhanced early log loss (15'-30'): "
        f"{early_log_loss:.4f}"
    )
    print(
        f"Enhanced late log loss (75'-85'): "
        f"{late_log_loss:.4f}"
    )

    if comparison_rows:
        comparison_df = pd.DataFrame(
            comparison_rows
        )

        late_comparison = comparison_df[
            comparison_df[
                "Snapshot Minute"
            ] >= 75
        ]

        avg_late_accuracy_change = float(
            late_comparison[
                "Accuracy Change"
            ].mean()
        )

        avg_late_logloss_change = float(
            late_comparison[
                "Log Loss Change"
            ].mean()
        )

        print(
            f"\nAverage late accuracy change vs benchmark: "
            f"{avg_late_accuracy_change:+.4f}"
        )
        print(
            f"Average late log-loss change vs benchmark: "
            f"{avg_late_logloss_change:+.4f}"
        )

        if avg_late_accuracy_change > 0:
            print(
                "[PASS] Enhanced features improved late-match accuracy."
            )
        else:
            print(
                "[WARNING] Enhanced features did not improve "
                "late-match accuracy on average."
            )

        if avg_late_logloss_change < 0:
            print(
                "[PASS] Enhanced features improved late-match "
                "probability quality."
            )
        else:
            print(
                "[WARNING] Enhanced features did not improve "
                "late-match probability quality on average."
            )

    section(
        "FINAL RECOMMENDATION"
    )

    if (
        late_accuracy >= 0.70
        and late_log_loss <= 1.50
    ):
        print(
            "[PASS] Enhanced time-aware models show a meaningful "
            "combination of late-match classification and probability quality."
        )
        print(
            "Next step: calibrate the enhanced minute-specific models "
            "and assess whether they are ready for experimental dashboard use."
        )
    elif (
        late_accuracy >= 0.65
    ):
        print(
            "[PARTIAL PASS] Enhanced models retain useful late-match "
            "classification performance, but probability quality still "
            "needs improvement."
        )
        print(
            "Next step: calibrate the enhanced models and compare "
            "before/after log loss."
        )
    else:
        print(
            "[PARTIAL RESULT] The enhanced feature set did not produce "
            "enough improvement for dashboard replacement."
        )
        print(
            "Keep the current benchmark and inspect feature importance "
            "and model errors before further expansion."
        )

    print(
        f"\nSaved enhanced model results:"
    )
    print(
        f"  {RESULTS_FILE}"
    )

    if comparison_rows:
        print(
            f"\nSaved benchmark comparison:"
        )
        print(
            f"  {COMPARISON_FILE}"
        )

    print(
        f"\nSaved enhanced models:"
    )

    for minute in SNAPSHOT_MINUTES:
        print(
            f"  {MODELS_DIR / f'match_outcome_{minute}.joblib'}"
        )


if __name__ == "__main__":
    main()
