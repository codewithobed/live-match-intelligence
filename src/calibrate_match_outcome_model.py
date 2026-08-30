"""
Calibrate the saved match outcome model for LiveMatch Intelligence.

Purpose:
    Improve the quality of predicted probabilities before dashboard use.

Inputs:
    data/historical_match_states.csv
    models/best_match_outcome_model.joblib
    models/match_outcome_model_metadata.joblib

Method:
    - Recreate the same match-level train/test split.
    - Fit sigmoid and isotonic probability calibration using training matches.
    - Compare uncalibrated vs calibrated models on untouched test matches.
    - Save the best calibrated model only if calibration improves log loss.

Outputs:
    models/calibrated_match_outcome_model.joblib
    models/calibrated_match_outcome_metadata.joblib
    models/calibrated_match_outcome_metadata.json
    data/calibration_comparison_results.csv
"""

from pathlib import Path
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "historical_match_states.csv"
BASE_MODEL_FILE = PROJECT_ROOT / "models" / "best_match_outcome_model.joblib"
BASE_METADATA_FILE = PROJECT_ROOT / "models" / "match_outcome_model_metadata.joblib"

CALIBRATED_MODEL_FILE = (
    PROJECT_ROOT / "models" / "calibrated_match_outcome_model.joblib"
)
CALIBRATED_METADATA_FILE = (
    PROJECT_ROOT / "models" / "calibrated_match_outcome_metadata.joblib"
)
CALIBRATED_METADATA_JSON_FILE = (
    PROJECT_ROOT / "models" / "calibrated_match_outcome_metadata.json"
)
RESULTS_FILE = (
    PROJECT_ROOT / "data" / "calibration_comparison_results.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASS_ORDER = ["Home Win", "Draw", "Away Win"]


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def make_match_level_split(df):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            y=df["target"],
            groups=df["match_id"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_matches = set(train_df["match_id"].unique())
    test_matches = set(test_df["match_id"].unique())

    if train_matches.intersection(test_matches):
        raise RuntimeError(
            "Leakage detected: match IDs overlap between train and test."
        )

    return train_df, test_df


def align_probabilities(classes, raw_probabilities):
    aligned = np.zeros(
        (raw_probabilities.shape[0], len(CLASS_ORDER)),
        dtype=float,
    )

    classes = list(classes)

    for target_index, class_name in enumerate(CLASS_ORDER):
        if class_name in classes:
            source_index = classes.index(class_name)
            aligned[:, target_index] = raw_probabilities[:, source_index]

    return aligned


def get_classes(model):
    if hasattr(model, "classes_"):
        return model.classes_

    if hasattr(model, "named_steps"):
        return model.named_steps["model"].classes_

    raise RuntimeError("Could not determine model class ordering.")


def evaluate_model(model, X, y, name):
    predictions = model.predict(X)
    raw_probabilities = model.predict_proba(X)

    probabilities = align_probabilities(
        get_classes(model),
        raw_probabilities,
    )

    confidence = probabilities.max(axis=1)
    correct = predictions == y.to_numpy()

    wrong_confidence = (
        confidence[~correct].mean()
        if (~correct).any()
        else 0.0
    )

    correct_confidence = (
        confidence[correct].mean()
        if correct.any()
        else 0.0
    )

    return {
        "Model": name,
        "Accuracy": float(
            accuracy_score(y, predictions)
        ),
        "Macro F1": float(
            f1_score(
                y,
                predictions,
                average="macro",
            )
        ),
        "Log Loss": float(
            log_loss(
                y,
                probabilities,
                labels=CLASS_ORDER,
            )
        ),
        "Mean Confidence": float(confidence.mean()),
        "Correct Confidence": float(correct_confidence),
        "Wrong Confidence": float(wrong_confidence),
    }


def evaluate_by_minute(
    model,
    test_df,
    feature_columns,
    name,
):
    rows = []

    minutes = sorted(
        test_df["snapshot_minute"]
        .dropna()
        .astype(int)
        .unique()
    )

    for minute in minutes:
        subset = test_df[
            test_df["snapshot_minute"] == minute
        ].copy()

        if subset.empty:
            continue

        metrics = evaluate_model(
            model,
            subset[feature_columns],
            subset["target"],
            name,
        )

        rows.append(
            {
                "Snapshot Minute": minute,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore")

    section("LIVE MATCH INTELLIGENCE — PROBABILITY CALIBRATION")

    required_files = [
        DATA_FILE,
        BASE_MODEL_FILE,
        BASE_METADATA_FILE,
    ]

    for path in required_files:
        if not path.exists():
            print(f"[FAIL] Missing required file: {path}")
            sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    base_model = joblib.load(BASE_MODEL_FILE)
    base_metadata = joblib.load(BASE_METADATA_FILE)

    feature_columns = base_metadata["feature_columns"]
    best_model_name = base_metadata["best_model_name"]

    print(f"Base model: {best_model_name}")
    print(f"Rows: {len(df):,}")
    print(f"Unique matches: {df['match_id'].nunique():,}")

    train_df, test_df = make_match_level_split(df)

    X_train = train_df[feature_columns].copy()
    y_train = train_df["target"].copy()

    X_test = test_df[feature_columns].copy()
    y_test = test_df["target"].copy()

    print(
        f"Training matches: {train_df['match_id'].nunique()}"
    )
    print(
        f"Test matches: {test_df['match_id'].nunique()}"
    )

    section("BASE MODEL PERFORMANCE")

    base_results = evaluate_model(
        base_model,
        X_test,
        y_test,
        f"{best_model_name} — Uncalibrated",
    )

    print(f"Accuracy : {base_results['Accuracy']:.4f}")
    print(f"Macro F1 : {base_results['Macro F1']:.4f}")
    print(f"Log Loss : {base_results['Log Loss']:.4f}")
    print(
        "Wrong-prediction confidence: "
        f"{base_results['Wrong Confidence']:.4f}"
    )

    section("FITTING CALIBRATED MODELS")

    sigmoid_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
    )

    print("Training sigmoid calibration...")
    sigmoid_model.fit(X_train, y_train)

    sigmoid_results = evaluate_model(
        sigmoid_model,
        X_test,
        y_test,
        f"{best_model_name} — Sigmoid",
    )

    print(f"  Accuracy : {sigmoid_results['Accuracy']:.4f}")
    print(f"  Macro F1 : {sigmoid_results['Macro F1']:.4f}")
    print(f"  Log Loss : {sigmoid_results['Log Loss']:.4f}")
    print(
        "  Wrong-prediction confidence: "
        f"{sigmoid_results['Wrong Confidence']:.4f}"
    )

    isotonic_model = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=5,
    )

    print("\nTraining isotonic calibration...")
    isotonic_model.fit(X_train, y_train)

    isotonic_results = evaluate_model(
        isotonic_model,
        X_test,
        y_test,
        f"{best_model_name} — Isotonic",
    )

    print(f"  Accuracy : {isotonic_results['Accuracy']:.4f}")
    print(f"  Macro F1 : {isotonic_results['Macro F1']:.4f}")
    print(f"  Log Loss : {isotonic_results['Log Loss']:.4f}")
    print(
        "  Wrong-prediction confidence: "
        f"{isotonic_results['Wrong Confidence']:.4f}"
    )

    section("CALIBRATION COMPARISON")

    comparison = pd.DataFrame(
        [
            base_results,
            sigmoid_results,
            isotonic_results,
        ]
    )

    comparison = comparison.sort_values(
        by=["Log Loss", "Macro F1", "Accuracy"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    print(
        comparison.round(4).to_string(index=False)
    )

    comparison.to_csv(
        RESULTS_FILE,
        index=False,
    )

    best_row = comparison.iloc[0]
    best_name = best_row["Model"]

    if "Sigmoid" in best_name:
        best_calibrated_model = sigmoid_model
        method = "sigmoid"
    elif "Isotonic" in best_name:
        best_calibrated_model = isotonic_model
        method = "isotonic"
    else:
        best_calibrated_model = None
        method = "uncalibrated"

    section("PERFORMANCE BY SNAPSHOT MINUTE")

    minute_comparison = pd.concat(
        [
            evaluate_by_minute(
                base_model,
                test_df,
                feature_columns,
                "Uncalibrated",
            ),
            evaluate_by_minute(
                sigmoid_model,
                test_df,
                feature_columns,
                "Sigmoid",
            ),
            evaluate_by_minute(
                isotonic_model,
                test_df,
                feature_columns,
                "Isotonic",
            ),
        ],
        ignore_index=True,
    )

    for minute in sorted(
        minute_comparison["Snapshot Minute"].unique()
    ):
        subset = minute_comparison[
            minute_comparison["Snapshot Minute"] == minute
        ]

        print(f"\nMinute {int(minute)}:")
        print(
            subset[
                [
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

    section("FINAL DECISION")

    base_log_loss = base_results["Log Loss"]
    best_log_loss = float(best_row["Log Loss"])
    improvement = base_log_loss - best_log_loss

    if best_calibrated_model is None:
        print(
            "Calibration did not improve overall test-set log loss."
        )
        print(
            "Keep the current saved model as the benchmark."
        )
        print(
            "Do NOT replace the dashboard baseline yet."
        )
    else:
        joblib.dump(
            best_calibrated_model,
            CALIBRATED_MODEL_FILE,
        )

        metadata = {
            "base_model_name": best_model_name,
            "calibration_method": method,
            "feature_columns": feature_columns,
            "class_order": CLASS_ORDER,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "base_metrics": base_results,
            "calibrated_metrics": best_row.to_dict(),
            "log_loss_improvement": float(improvement),
        }

        joblib.dump(
            metadata,
            CALIBRATED_METADATA_FILE,
        )

        with CALIBRATED_METADATA_JSON_FILE.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )

        print(f"Best calibration method: {method}")
        print(
            f"Uncalibrated log loss: {base_log_loss:.4f}"
        )
        print(
            f"Calibrated log loss:   {best_log_loss:.4f}"
        )
        print(
            f"Improvement:            {improvement:.4f}"
        )
        print(
            f"\nSaved calibrated model:\n  "
            f"{CALIBRATED_MODEL_FILE}"
        )

        if (
            improvement >= 0.05
            and best_log_loss <= 1.10
        ):
            print(
                "\n[PASS] Calibration materially improved probability quality."
            )
            print(
                "Next step: connect the calibrated ML model to "
                "Predictive Intelligence in the dashboard."
            )
        elif improvement > 0:
            print(
                "\n[PARTIAL PASS] Calibration improved probability quality, "
                "but probabilities are still not strong enough for "
                "production-style claims."
            )
            print(
                "Keep the calibrated model as the ML benchmark and "
                "improve the modelling pipeline before dashboard replacement."
            )
        else:
            print(
                "\n[WARNING] Calibration did not materially improve "
                "probability quality."
            )
            print(
                "Do NOT replace the dashboard baseline yet."
            )

    print(
        f"\nSaved comparison table: {RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()
