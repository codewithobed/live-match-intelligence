"""
Evaluate the saved match outcome model by snapshot minute.

Inputs:
    data/historical_match_states.csv
    models/best_match_outcome_model.joblib
    models/match_outcome_model_metadata.joblib

This script rebuilds the same match-level 80/20 test split used during
training (RANDOM_STATE=42), then evaluates the saved best model separately
at 15, 30, 45, 60, 75 and 85 minutes.

Metrics:
    - Accuracy
    - Macro F1
    - Log Loss
    - Mean confidence
    - Wrong-prediction confidence
    - Number of test rows per snapshot

Outputs:
    data/model_performance_by_minute.csv
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "historical_match_states.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "best_match_outcome_model.joblib"
METADATA_FILE = PROJECT_ROOT / "models" / "match_outcome_model_metadata.joblib"

OUTPUT_FILE = PROJECT_ROOT / "data" / "model_performance_by_minute.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

EXPECTED_MINUTES = [15, 30, 45, 60, 75, 85]


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def make_test_split(df):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    _, test_idx = next(
        splitter.split(
            df,
            y=df["target"],
            groups=df["match_id"],
        )
    )

    return df.iloc[test_idx].copy()


def align_probabilities(model, raw_probabilities, class_order):
    model_classes = list(
        model.named_steps["model"].classes_
    )

    aligned = np.zeros(
        (raw_probabilities.shape[0], len(class_order)),
        dtype=float,
    )

    for target_index, class_name in enumerate(class_order):
        if class_name in model_classes:
            source_index = model_classes.index(class_name)
            aligned[:, target_index] = raw_probabilities[:, source_index]

    return aligned


def evaluate_subset(model, df_subset, feature_columns, class_order):
    X = df_subset[feature_columns].copy()
    y = df_subset["target"].copy()

    predictions = model.predict(X)
    raw_probabilities = model.predict_proba(X)

    probabilities = align_probabilities(
        model,
        raw_probabilities,
        class_order,
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )

    loss = log_loss(
        y,
        probabilities,
        labels=class_order,
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
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Log Loss": loss,
        "Mean Confidence": confidence.mean(),
        "Correct Prediction Confidence": correct_confidence,
        "Wrong Prediction Confidence": wrong_confidence,
        "Rows": len(df_subset),
    }


def main():
    section("LIVE MATCH INTELLIGENCE — MODEL EVALUATION BY MINUTE")

    for path in [DATA_FILE, MODEL_FILE, METADATA_FILE]:
        if not path.exists():
            print(f"[FAIL] Missing required file: {path}")
            sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    metadata = joblib.load(METADATA_FILE)

    feature_columns = metadata["feature_columns"]
    class_order = metadata["class_order"]
    best_model_name = metadata["best_model_name"]

    print(f"Best saved model: {best_model_name}")
    print(f"Dataset rows: {len(df):,}")
    print(f"Unique matches: {df['match_id'].nunique():,}")

    test_df = make_test_split(df)

    print(
        f"Test matches: {test_df['match_id'].nunique()}"
    )
    print(
        f"Test rows: {len(test_df)}"
    )

    section("PERFORMANCE BY SNAPSHOT MINUTE")

    rows = []

    for minute in EXPECTED_MINUTES:
        subset = test_df[
            test_df["snapshot_minute"] == minute
        ].copy()

        if subset.empty:
            print(
                f"{minute:>2}' — no test rows available"
            )
            continue

        metrics = evaluate_subset(
            model,
            subset,
            feature_columns,
            class_order,
        )

        rows.append(
            {
                "Snapshot Minute": minute,
                **metrics,
            }
        )

        print(
            f"{minute:>2}' | "
            f"Accuracy {metrics['Accuracy']:.3f} | "
            f"Macro F1 {metrics['Macro F1']:.3f} | "
            f"Log Loss {metrics['Log Loss']:.3f} | "
            f"Mean Confidence {metrics['Mean Confidence']:.3f} | "
            f"Wrong Confidence {metrics['Wrong Prediction Confidence']:.3f}"
        )

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        print("[FAIL] No evaluation rows were created.")
        sys.exit(1)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    section("INTERPRETATION")

    best_accuracy_row = results_df.loc[
        results_df["Accuracy"].idxmax()
    ]

    best_logloss_row = results_df.loc[
        results_df["Log Loss"].idxmin()
    ]

    worst_logloss_row = results_df.loc[
        results_df["Log Loss"].idxmax()
    ]

    print(
        f"Highest accuracy: "
        f"{int(best_accuracy_row['Snapshot Minute'])}' "
        f"({best_accuracy_row['Accuracy']:.3f})"
    )

    print(
        f"Best log loss: "
        f"{int(best_logloss_row['Snapshot Minute'])}' "
        f"({best_logloss_row['Log Loss']:.3f})"
    )

    print(
        f"Worst log loss: "
        f"{int(worst_logloss_row['Snapshot Minute'])}' "
        f"({worst_logloss_row['Log Loss']:.3f})"
    )

    early = results_df[
        results_df["Snapshot Minute"] <= 30
    ]

    late = results_df[
        results_df["Snapshot Minute"] >= 75
    ]

    if not early.empty and not late.empty:
        print(
            f"\nAverage early accuracy (15'-30'): "
            f"{early['Accuracy'].mean():.3f}"
        )
        print(
            f"Average late accuracy (75'-85'): "
            f"{late['Accuracy'].mean():.3f}"
        )

        print(
            f"Average early log loss (15'-30'): "
            f"{early['Log Loss'].mean():.3f}"
        )
        print(
            f"Average late log loss (75'-85'): "
            f"{late['Log Loss'].mean():.3f}"
        )

        if (
            late["Accuracy"].mean()
            > early["Accuracy"].mean()
        ):
            print(
                "\n[PASS] Classification accuracy improves later in matches."
            )
        else:
            print(
                "\n[WARNING] Accuracy does not improve later in matches."
            )

        if (
            late["Log Loss"].mean()
            < early["Log Loss"].mean()
        ):
            print(
                "[PASS] Probability quality improves later in matches."
            )
        else:
            print(
                "[WARNING] Probability quality does not improve later in matches."
            )

    section("CONFIDENCE CHECK")

    high_wrong = results_df[
        results_df["Wrong Prediction Confidence"] >= 0.70
    ]

    if high_wrong.empty:
        print(
            "[PASS] No snapshot minute has average wrong-prediction "
            "confidence above 70%."
        )
    else:
        print(
            "[WARNING] The model is overconfident when wrong at:"
        )
        for _, row in high_wrong.iterrows():
            print(
                f"  {int(row['Snapshot Minute'])}' — "
                f"{row['Wrong Prediction Confidence']:.3f}"
            )

    section("FINAL RECOMMENDATION")

    late_accuracy = (
        late["Accuracy"].mean()
        if not late.empty
        else 0.0
    )

    late_logloss = (
        late["Log Loss"].mean()
        if not late.empty
        else np.inf
    )

    if late_accuracy >= 0.65 and late_logloss <= 1.10:
        print(
            "The saved model shows promising late-match behaviour."
        )
        print(
            "Next step: probability calibration and dashboard integration."
        )
    else:
        print(
            "The current model should NOT replace the dashboard baseline yet."
        )
        print(
            "Next step: increase the historical training sample and retrain."
        )

    print(
        f"\nSaved evaluation table: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
