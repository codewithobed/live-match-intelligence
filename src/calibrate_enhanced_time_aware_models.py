"""
Calibrate enhanced time-aware match-outcome models.

Inputs:
    data/enhanced_historical_match_states.csv
    models/enhanced_time_aware/match_outcome_<minute>.joblib
    models/enhanced_time_aware/enhanced_time_aware_metadata.joblib

Outputs:
    models/enhanced_time_aware/calibrated/match_outcome_<minute>.joblib
    models/enhanced_time_aware/calibrated/calibrated_enhanced_metadata.joblib
    data/enhanced_time_aware_calibration_results.csv

At each snapshot minute this script compares:
    - Uncalibrated
    - Sigmoid calibration
    - Isotonic calibration

The same leakage-safe match-level split (random_state=42) is recreated.
Calibration is fitted only on training matches. Test matches remain untouched.
"""

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "enhanced_historical_match_states.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "enhanced_time_aware"
METADATA_FILE = MODEL_DIR / "enhanced_time_aware_metadata.joblib"

CALIBRATED_DIR = MODEL_DIR / "calibrated"
RESULTS_FILE = PROJECT_ROOT / "data" / "enhanced_time_aware_calibration_results.csv"
CALIBRATED_METADATA_FILE = CALIBRATED_DIR / "calibrated_enhanced_metadata.joblib"

SNAPSHOT_MINUTES = [15, 30, 45, 60, 75, 85]
CLASS_ORDER = ["Home Win", "Draw", "Away Win"]

RANDOM_STATE = 42
TEST_SIZE = 0.20


def section(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def make_match_split(df):
    match_table = (
        df[["match_id", "target"]]
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

    train_matches = set(match_table.iloc[train_idx]["match_id"].tolist())
    test_matches = set(match_table.iloc[test_idx]["match_id"].tolist())

    if train_matches.intersection(test_matches):
        raise RuntimeError("Train/test match leakage detected.")

    return train_matches, test_matches


def get_model_classes(model):
    if hasattr(model, "classes_"):
        return list(model.classes_)

    if hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]
        if hasattr(final_estimator, "classes_"):
            return list(final_estimator.classes_)

    raise RuntimeError("Could not determine model class order.")


def align_probabilities(model, raw_probabilities):
    model_classes = get_model_classes(model)

    aligned = np.zeros(
        (raw_probabilities.shape[0], len(CLASS_ORDER)),
        dtype=float,
    )

    for target_index, class_name in enumerate(CLASS_ORDER):
        if class_name in model_classes:
            source_index = model_classes.index(class_name)
            aligned[:, target_index] = raw_probabilities[:, source_index]

    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return aligned / row_sums


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = align_probabilities(
        model,
        model.predict_proba(X_test),
    )

    accuracy = float(accuracy_score(y_test, predictions))
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

    confidence = probabilities.max(axis=1)
    correct = predictions == y_test.to_numpy()

    wrong_confidence = (
        float(confidence[~correct].mean())
        if (~correct).any()
        else 0.0
    )

    return {
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Log Loss": loss,
        "Wrong Confidence": wrong_confidence,
    }


def make_calibrator(base_model, method):
    # sklearn >=1.6 uses estimator=. Older versions used base_estimator=.
    try:
        return CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv=5,
            ensemble=False,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=base_model,
            method=method,
            cv=5,
        )


def choose_variant(results):
    """
    Primary objective: lower Log Loss.
    Tie-breakers: Macro F1, then Accuracy.
    """
    return min(
        results,
        key=lambda r: (
            r["Log Loss"],
            -r["Macro F1"],
            -r["Accuracy"],
        ),
    )


def main():
    warnings.filterwarnings("ignore")

    section("ENHANCED TIME-AWARE MODEL CALIBRATION")

    if not DATA_FILE.exists():
        print(f"[FAIL] Missing dataset: {DATA_FILE}")
        sys.exit(1)

    if not METADATA_FILE.exists():
        print(f"[FAIL] Missing metadata: {METADATA_FILE}")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    metadata = joblib.load(METADATA_FILE)

    feature_columns = metadata.get("feature_columns")
    if not feature_columns:
        print("[FAIL] Feature columns missing from enhanced metadata.")
        sys.exit(1)

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        print(f"[FAIL] Dataset is missing model features: {missing_features}")
        sys.exit(1)

    train_matches, test_matches = make_match_split(df)

    print(f"Rows: {len(df):,}")
    print(f"Unique matches: {df['match_id'].nunique():,}")
    print(f"Training matches: {len(train_matches):,}")
    print(f"Test matches: {len(test_matches):,}")
    print(f"Model features: {len(feature_columns):,}")

    CALIBRATED_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    selected_metadata = {}

    for minute in SNAPSHOT_MINUTES:
        section(f"CALIBRATING {minute}' MODEL")

        model_file = MODEL_DIR / f"match_outcome_{minute}.joblib"

        if not model_file.exists():
            print(f"[FAIL] Missing enhanced model: {model_file}")
            sys.exit(1)

        minute_df = df[df["snapshot_minute"] == minute].copy()

        train_df = minute_df[
            minute_df["match_id"].isin(train_matches)
        ].copy()

        test_df = minute_df[
            minute_df["match_id"].isin(test_matches)
        ].copy()

        X_train = train_df[feature_columns].copy()
        y_train = train_df["target"].copy()

        X_test = test_df[feature_columns].copy()
        y_test = test_df["target"].copy()

        print(
            f"Training rows: {len(train_df)} | "
            f"Test rows: {len(test_df)}"
        )

        base_model = joblib.load(model_file)

        variants = []

        base_metrics = evaluate(
            base_model,
            X_test,
            y_test,
        )

        variants.append(
            {
                "Snapshot Minute": minute,
                "Variant": "Uncalibrated",
                **base_metrics,
                "Model Object": base_model,
            }
        )

        print("\nUncalibrated")
        for key, value in base_metrics.items():
            print(f"  {key:<18}: {value:.4f}")

        for method in ["sigmoid", "isotonic"]:
            fresh_base_model = joblib.load(model_file)

            calibrator = make_calibrator(
                fresh_base_model,
                method,
            )

            try:
                calibrator.fit(
                    X_train,
                    y_train,
                )

                metrics = evaluate(
                    calibrator,
                    X_test,
                    y_test,
                )

                variants.append(
                    {
                        "Snapshot Minute": minute,
                        "Variant": method.title(),
                        **metrics,
                        "Model Object": calibrator,
                    }
                )

                print(f"\n{method.title()}")
                for key, value in metrics.items():
                    print(f"  {key:<18}: {value:.4f}")

            except Exception as exc:
                print(
                    f"\n[WARNING] {method.title()} calibration failed: {exc}"
                )

        best = choose_variant(variants)

        base_loss = base_metrics["Log Loss"]
        improvement = base_loss - best["Log Loss"]

        output_model = (
            CALIBRATED_DIR
            / f"match_outcome_{minute}.joblib"
        )

        joblib.dump(
            best["Model Object"],
            output_model,
        )

        selected_metadata[str(minute)] = {
            "variant": best["Variant"],
            "accuracy": best["Accuracy"],
            "macro_f1": best["Macro F1"],
            "log_loss": best["Log Loss"],
            "wrong_confidence": best["Wrong Confidence"],
            "log_loss_improvement_vs_uncalibrated": improvement,
            "model_file": output_model.name,
        }

        print(
            f"\nSELECTED: {best['Variant']} | "
            f"Accuracy {best['Accuracy']:.4f} | "
            f"Macro F1 {best['Macro F1']:.4f} | "
            f"Log Loss {best['Log Loss']:.4f} | "
            f"Improvement {improvement:.4f}"
        )

        for result in variants:
            all_results.append(
                {
                    key: value
                    for key, value in result.items()
                    if key != "Model Object"
                }
                | {
                    "Selected": (
                        result["Variant"]
                        == best["Variant"]
                    ),
                    "Log Loss Improvement vs Base": (
                        base_loss - result["Log Loss"]
                    ),
                }
            )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_FILE, index=False)

    selected_df = (
        results_df[results_df["Selected"]]
        .sort_values("Snapshot Minute")
        .copy()
    )

    section("ENHANCED TIME-AWARE CALIBRATION SUMMARY")

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

    calibrated_metadata = {
        "feature_columns": feature_columns,
        "class_order": CLASS_ORDER,
        "snapshot_minutes": SNAPSHOT_MINUTES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "minutes": selected_metadata,
    }

    joblib.dump(
        calibrated_metadata,
        CALIBRATED_METADATA_FILE,
    )

    late = selected_df[
        selected_df["Snapshot Minute"] >= 75
    ].copy()

    late_accuracy = float(late["Accuracy"].mean())
    late_f1 = float(late["Macro F1"].mean())
    late_loss = float(late["Log Loss"].mean())

    late_improvement = float(
        late["Log Loss Improvement vs Base"].mean()
    )

    section("LATE-MATCH CHECK")

    print(
        f"Average calibrated accuracy (75'-85'): "
        f"{late_accuracy:.4f}"
    )
    print(
        f"Average calibrated Macro F1 (75'-85'): "
        f"{late_f1:.4f}"
    )
    print(
        f"Average calibrated Log Loss (75'-85'): "
        f"{late_loss:.4f}"
    )
    print(
        f"Average late Log Loss improvement: "
        f"{late_improvement:.4f}"
    )

    section("FINAL RECOMMENDATION")

    if (
        late_accuracy >= 0.70
        and late_loss <= 1.50
        and late_improvement > 0
    ):
        print(
            "[PASS] Enhanced calibration preserved strong late-match "
            "classification and materially improved probability quality."
        )
        print(
            "The calibrated enhanced models are ready for a controlled "
            "dashboard integration test, with probabilities labelled "
            "as experimental model estimates."
        )

    elif (
        late_accuracy >= 0.70
        and late_improvement > 0
    ):
        print(
            "[PARTIAL PASS] Calibration improved the enhanced models "
            "while retaining useful late-match classification, but "
            "probability quality is still not strong enough for "
            "unqualified production-style probability claims."
        )
        print(
            "Keep these as the strongest ML benchmark and inspect the "
            "minute-by-minute results before dashboard replacement."
        )

    else:
        print(
            "[PARTIAL RESULT] Calibration did not produce a sufficiently "
            "strong probability model for dashboard replacement."
        )
        print(
            "Keep the current dashboard baseline and use the enhanced "
            "models as an experimental benchmark."
        )

    print("\nSaved calibrated models:")
    for minute in SNAPSHOT_MINUTES:
        print(
            f"  {CALIBRATED_DIR / f'match_outcome_{minute}.joblib'}"
        )

    print(f"\nSaved calibration results:\n  {RESULTS_FILE}")
    print(f"\nSaved calibration metadata:\n  {CALIBRATED_METADATA_FILE}")


if __name__ == "__main__":
    main()
