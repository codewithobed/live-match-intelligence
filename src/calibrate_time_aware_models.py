"""
Calibrate the six time-aware match outcome models.

Inputs:
    data/historical_match_states.csv
    models/time_aware/match_outcome_15.joblib
    models/time_aware/match_outcome_30.joblib
    models/time_aware/match_outcome_45.joblib
    models/time_aware/match_outcome_60.joblib
    models/time_aware/match_outcome_75.joblib
    models/time_aware/match_outcome_85.joblib
    models/time_aware/time_aware_metadata.joblib

Method:
    - Recreate the same global match-level 80/20 split.
    - At each snapshot minute, compare:
        * Uncalibrated saved model
        * Sigmoid calibration
        * Isotonic calibration
    - Select the lowest-log-loss option for that minute.
    - Save a calibrated model only when calibration beats the base model.

Outputs:
    models/time_aware/calibrated/
        calibrated_match_outcome_15.joblib
        ...
        calibrated_match_outcome_85.joblib
        calibrated_time_aware_metadata.joblib
        calibrated_time_aware_metadata.json

    data/time_aware_calibration_results.csv
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

TIME_AWARE_DIR = PROJECT_ROOT / "models" / "time_aware"

TIME_AWARE_METADATA_FILE = (
    TIME_AWARE_DIR / "time_aware_metadata.joblib"
)

CALIBRATED_DIR = (
    TIME_AWARE_DIR / "calibrated"
)

CALIBRATED_METADATA_FILE = (
    CALIBRATED_DIR / "calibrated_time_aware_metadata.joblib"
)

CALIBRATED_METADATA_JSON_FILE = (
    CALIBRATED_DIR / "calibrated_time_aware_metadata.json"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "time_aware_calibration_results.csv"
)

SNAPSHOT_MINUTES = [15, 30, 45, 60, 75, 85]

CLASS_ORDER = ["Home Win", "Draw", "Away Win"]

RANDOM_STATE = 42
TEST_SIZE = 0.20


def section(title):
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


def validate_inputs():
    required = [
        DATA_FILE,
        TIME_AWARE_METADATA_FILE,
    ]

    for minute in SNAPSHOT_MINUTES:
        required.append(
            TIME_AWARE_DIR / f"match_outcome_{minute}.joblib"
        )

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        print("[FAIL] Missing required files:")
        for path in missing:
            print(f"  {path}")
        sys.exit(1)


def make_global_match_split(df):
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


def get_classes(model):
    if hasattr(model, "classes_"):
        return list(model.classes_)

    if hasattr(model, "named_steps"):
        return list(
            model.named_steps["model"].classes_
        )

    raise RuntimeError(
        "Could not determine model class ordering."
    )


def align_probabilities(
    classes,
    raw_probabilities,
):
    aligned = np.zeros(
        (
            raw_probabilities.shape[0],
            len(CLASS_ORDER),
        ),
        dtype=float,
    )

    for target_index, class_name in enumerate(CLASS_ORDER):
        if class_name in classes:
            source_index = classes.index(class_name)
            aligned[:, target_index] = (
                raw_probabilities[:, source_index]
            )

    return aligned


def evaluate_model(
    model,
    X,
    y,
    label,
):
    predictions = model.predict(X)

    raw_probabilities = model.predict_proba(X)

    probabilities = align_probabilities(
        get_classes(model),
        raw_probabilities,
    )

    confidence = probabilities.max(axis=1)

    correct_mask = (
        predictions == y.to_numpy()
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
        "Variant": label,
        "Accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
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
        "Wrong Confidence": wrong_confidence,
    }


def build_calibrated(
    base_model,
    method,
):
    return CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv=5,
    )


def main():
    warnings.filterwarnings("ignore")

    section(
        "LIVE MATCH INTELLIGENCE — TIME-AWARE PROBABILITY CALIBRATION"
    )

    validate_inputs()

    df = pd.read_csv(
        DATA_FILE
    )

    metadata = joblib.load(
        TIME_AWARE_METADATA_FILE
    )

    feature_columns = metadata[
        "feature_columns"
    ]

    train_matches, test_matches = (
        make_global_match_split(df)
    )

    print(
        f"Rows: {len(df):,}"
    )
    print(
        f"Unique matches: "
        f"{df['match_id'].nunique():,}"
    )
    print(
        f"Training matches: "
        f"{len(train_matches):,}"
    )
    print(
        f"Test matches: "
        f"{len(test_matches):,}"
    )

    CALIBRATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []
    calibrated_metadata = {}

    section(
        "CALIBRATING EACH SNAPSHOT MODEL"
    )

    for minute in SNAPSHOT_MINUTES:

        print(
            f"\n{'-' * 82}"
        )
        print(
            f"SNAPSHOT MINUTE: {minute}'"
        )
        print(
            f"{'-' * 82}"
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

        base_model_path = (
            TIME_AWARE_DIR
            / f"match_outcome_{minute}.joblib"
        )

        base_model = joblib.load(
            base_model_path
        )

        base_result = evaluate_model(
            base_model,
            X_test,
            y_test,
            "Uncalibrated",
        )

        print(
            "\nUncalibrated"
        )
        print(
            f"  Accuracy         : "
            f"{base_result['Accuracy']:.4f}"
        )
        print(
            f"  Macro F1         : "
            f"{base_result['Macro F1']:.4f}"
        )
        print(
            f"  Log Loss         : "
            f"{base_result['Log Loss']:.4f}"
        )
        print(
            f"  Wrong Confidence : "
            f"{base_result['Wrong Confidence']:.4f}"
        )

        sigmoid_model = build_calibrated(
            base_model,
            "sigmoid",
        )

        print(
            "\nTraining sigmoid calibration..."
        )

        sigmoid_model.fit(
            X_train,
            y_train,
        )

        sigmoid_result = evaluate_model(
            sigmoid_model,
            X_test,
            y_test,
            "Sigmoid",
        )

        print(
            f"  Accuracy         : "
            f"{sigmoid_result['Accuracy']:.4f}"
        )
        print(
            f"  Macro F1         : "
            f"{sigmoid_result['Macro F1']:.4f}"
        )
        print(
            f"  Log Loss         : "
            f"{sigmoid_result['Log Loss']:.4f}"
        )
        print(
            f"  Wrong Confidence : "
            f"{sigmoid_result['Wrong Confidence']:.4f}"
        )

        isotonic_model = build_calibrated(
            base_model,
            "isotonic",
        )

        print(
            "\nTraining isotonic calibration..."
        )

        isotonic_model.fit(
            X_train,
            y_train,
        )

        isotonic_result = evaluate_model(
            isotonic_model,
            X_test,
            y_test,
            "Isotonic",
        )

        print(
            f"  Accuracy         : "
            f"{isotonic_result['Accuracy']:.4f}"
        )
        print(
            f"  Macro F1         : "
            f"{isotonic_result['Macro F1']:.4f}"
        )
        print(
            f"  Log Loss         : "
            f"{isotonic_result['Log Loss']:.4f}"
        )
        print(
            f"  Wrong Confidence : "
            f"{isotonic_result['Wrong Confidence']:.4f}"
        )

        variants = [
            (
                "Uncalibrated",
                base_model,
                base_result,
            ),
            (
                "Sigmoid",
                sigmoid_model,
                sigmoid_result,
            ),
            (
                "Isotonic",
                isotonic_model,
                isotonic_result,
            ),
        ]

        best_variant_name, best_model, best_result = min(
            variants,
            key=lambda item: item[2]["Log Loss"],
        )

        improvement = (
            base_result["Log Loss"]
            - best_result["Log Loss"]
        )

        saved_model_name = None

        if best_variant_name != "Uncalibrated":
            saved_model_name = (
                f"calibrated_match_outcome_{minute}.joblib"
            )

            saved_model_path = (
                CALIBRATED_DIR
                / saved_model_name
            )

            joblib.dump(
                best_model,
                saved_model_path,
            )

        print(
            f"\nBEST PROBABILITY MODEL AT {minute}': "
            f"{best_variant_name}"
        )
        print(
            f"Base log loss: "
            f"{base_result['Log Loss']:.4f}"
        )
        print(
            f"Best log loss: "
            f"{best_result['Log Loss']:.4f}"
        )
        print(
            f"Improvement: "
            f"{improvement:.4f}"
        )

        if saved_model_name:
            print(
                f"Saved calibrated model: "
                f"{CALIBRATED_DIR / saved_model_name}"
            )
        else:
            print(
                "Calibration did not beat the base model; "
                "no calibrated replacement was saved for this minute."
            )

        calibrated_metadata[
            str(minute)
        ] = {
            "best_variant": (
                best_variant_name
            ),
            "base_model_file": (
                base_model_path.name
            ),
            "calibrated_model_file": (
                saved_model_name
            ),
            "base_accuracy": (
                base_result["Accuracy"]
            ),
            "base_macro_f1": (
                base_result["Macro F1"]
            ),
            "base_log_loss": (
                base_result["Log Loss"]
            ),
            "best_accuracy": (
                best_result["Accuracy"]
            ),
            "best_macro_f1": (
                best_result["Macro F1"]
            ),
            "best_log_loss": (
                best_result["Log Loss"]
            ),
            "log_loss_improvement": float(
                improvement
            ),
            "wrong_confidence": (
                best_result[
                    "Wrong Confidence"
                ]
            ),
        }

        for variant_name, _, result in variants:
            all_rows.append(
                {
                    "Snapshot Minute": minute,
                    "Variant": variant_name,
                    "Accuracy": (
                        result["Accuracy"]
                    ),
                    "Macro F1": (
                        result["Macro F1"]
                    ),
                    "Log Loss": (
                        result["Log Loss"]
                    ),
                    "Wrong Confidence": (
                        result[
                            "Wrong Confidence"
                        ]
                    ),
                    "Selected": (
                        variant_name
                        == best_variant_name
                    ),
                    "Log Loss Improvement vs Base": (
                        base_result["Log Loss"]
                        - result["Log Loss"]
                    ),
                }
            )

    section(
        "TIME-AWARE CALIBRATION SUMMARY"
    )

    results_df = pd.DataFrame(
        all_rows
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
                "Variant",
                "Accuracy",
                "Macro F1",
                "Log Loss",
                "Wrong Confidence",
                "Log Loss Improvement vs Base",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    full_metadata = {
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
        "minutes": (
            calibrated_metadata
        ),
    }

    joblib.dump(
        full_metadata,
        CALIBRATED_METADATA_FILE,
    )

    with CALIBRATED_METADATA_JSON_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            full_metadata,
            f,
            indent=2,
        )

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

    early_log_loss = float(
        early["Log Loss"].mean()
    )

    late_log_loss = float(
        late["Log Loss"].mean()
    )

    early_accuracy = float(
        early["Accuracy"].mean()
    )

    late_accuracy = float(
        late["Accuracy"].mean()
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

    improved_minutes = selected_df[
        selected_df[
            "Log Loss Improvement vs Base"
        ] > 0
    ]

    print(
        f"\nCalibration improved "
        f"{len(improved_minutes)} of "
        f"{len(SNAPSHOT_MINUTES)} snapshot models."
    )

    section(
        "FINAL RECOMMENDATION"
    )

    if (
        late_accuracy >= 0.65
        and late_log_loss <= 1.10
    ):
        print(
            "[PASS] The calibrated time-aware late-match "
            "models now have promising classification and "
            "probability behaviour."
        )
        print(
            "Next step: integrate the appropriate minute-specific "
            "model into the dashboard replay."
        )
    elif (
        late_accuracy >= 0.65
        and late_log_loss < 1.50
    ):
        print(
            "[PARTIAL PASS] Late-match classification is strong "
            "and calibration materially improved probability quality."
        )
        print(
            "The calibrated time-aware system is suitable as an "
            "experimental ML probability layer, but should remain "
            "clearly labelled as a research prototype."
        )
    else:
        print(
            "[PARTIAL RESULT] Calibration helped some minute models, "
            "but probability quality is still not strong enough "
            "for unqualified dashboard probability claims."
        )
        print(
            "Keep these models as the strongest ML benchmark and "
            "continue improving features/model validation before "
            "full dashboard replacement."
        )

    print(
        f"\nSaved full calibration results:"
    )
    print(
        f"  {RESULTS_FILE}"
    )

    print(
        f"\nSaved calibration metadata:"
    )
    print(
        f"  {CALIBRATED_METADATA_FILE}"
    )


if __name__ == "__main__":
    main()
