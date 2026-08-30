"""
Build an enhanced match-state training dataset for LiveMatch Intelligence.

Input:
    data/historical_match_states.csv

Output:
    data/enhanced_historical_match_states.csv

Purpose:
    Add stronger contextual features to the already validated historical
    snapshot dataset without altering or overwriting the original CSV.

The enhanced features focus on:
    - Score state (leading / drawing / trailing)
    - Time remaining
    - Score x time interactions
    - xG relative to current score
    - Shot and pressure context
    - Recent attacking pressure
    - Late-match state indicators
    - Comeback / protection context

This script DOES NOT retrain models.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "historical_match_states.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "enhanced_historical_match_states.csv"
)

MATCH_END_MINUTE = 90.0


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def require_columns(df, columns):
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )


def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    ).fillna(0.0)

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    ).fillna(0.0)

    return np.where(
        np.abs(denominator) > 1e-9,
        numerator / denominator,
        0.0,
    )


def main():
    section(
        "LIVE MATCH INTELLIGENCE — ENHANCED FEATURE ENGINEERING"
    )

    if not INPUT_FILE.exists():
        print(
            f"[FAIL] Input dataset not found: {INPUT_FILE}"
        )
        sys.exit(1)

    df = pd.read_csv(
        INPUT_FILE
    )

    required = [
        "match_id",
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
        "home_pressures",
        "away_pressures",
        "pressure_difference",
        "home_recent_xg",
        "away_recent_xg",
        "home_recent_shots",
        "away_recent_shots",
        "home_recent_pressures",
        "away_recent_pressures",
        "home_momentum",
        "away_momentum",
        "momentum_difference",
        "target",
    ]

    require_columns(
        df,
        required,
    )

    print(
        f"Loaded rows: {len(df):,}"
    )
    print(
        f"Unique matches: "
        f"{df['match_id'].nunique():,}"
    )

    enhanced = df.copy()

    # -----------------------------------------------------
    # Time context
    # -----------------------------------------------------

    minute = pd.to_numeric(
        enhanced["snapshot_minute"],
        errors="coerce",
    ).fillna(0.0)

    time_remaining = (
        MATCH_END_MINUTE - minute
    ).clip(lower=0.0)

    enhanced[
        "time_remaining"
    ] = time_remaining

    enhanced[
        "match_progress"
    ] = (
        minute / MATCH_END_MINUTE
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    enhanced[
        "late_match_flag"
    ] = (
        minute >= 75
    ).astype(int)

    enhanced[
        "very_late_match_flag"
    ] = (
        minute >= 85
    ).astype(int)

    enhanced[
        "second_half_flag"
    ] = (
        minute >= 45
    ).astype(int)

    # -----------------------------------------------------
    # Score state
    # -----------------------------------------------------

    goal_diff = pd.to_numeric(
        enhanced["goal_difference"],
        errors="coerce",
    ).fillna(0.0)

    enhanced[
        "home_leading_flag"
    ] = (
        goal_diff > 0
    ).astype(int)

    enhanced[
        "draw_flag"
    ] = (
        goal_diff == 0
    ).astype(int)

    enhanced[
        "home_trailing_flag"
    ] = (
        goal_diff < 0
    ).astype(int)

    enhanced[
        "abs_goal_difference"
    ] = np.abs(
        goal_diff
    )

    # -----------------------------------------------------
    # Score x time interactions
    # -----------------------------------------------------

    enhanced[
        "goal_diff_x_progress"
    ] = (
        goal_diff
        * enhanced[
            "match_progress"
        ]
    )

    enhanced[
        "goal_diff_x_time_remaining"
    ] = (
        goal_diff
        * enhanced[
            "time_remaining"
        ]
    )

    enhanced[
        "lead_protection_pressure"
    ] = (
        enhanced[
            "home_leading_flag"
        ]
        * enhanced[
            "match_progress"
        ]
    )

    enhanced[
        "comeback_pressure"
    ] = (
        enhanced[
            "home_trailing_flag"
        ]
        * enhanced[
            "match_progress"
        ]
    )

    enhanced[
        "late_draw_pressure"
    ] = (
        enhanced[
            "draw_flag"
        ]
        * enhanced[
            "late_match_flag"
        ]
    )

    # -----------------------------------------------------
    # xG relative to score
    # -----------------------------------------------------

    home_xg = pd.to_numeric(
        enhanced["home_xg"],
        errors="coerce",
    ).fillna(0.0)

    away_xg = pd.to_numeric(
        enhanced["away_xg"],
        errors="coerce",
    ).fillna(0.0)

    home_goals = pd.to_numeric(
        enhanced["home_goals"],
        errors="coerce",
    ).fillna(0.0)

    away_goals = pd.to_numeric(
        enhanced["away_goals"],
        errors="coerce",
    ).fillna(0.0)

    enhanced[
        "home_xg_minus_goals"
    ] = (
        home_xg - home_goals
    )

    enhanced[
        "away_xg_minus_goals"
    ] = (
        away_xg - away_goals
    )

    enhanced[
        "xg_score_gap_difference"
    ] = (
        enhanced[
            "home_xg_minus_goals"
        ]
        - enhanced[
            "away_xg_minus_goals"
        ]
    )

    enhanced[
        "xg_diff_x_progress"
    ] = (
        pd.to_numeric(
            enhanced[
                "xg_difference"
            ],
            errors="coerce",
        ).fillna(0.0)
        * enhanced[
            "match_progress"
        ]
    )

    # -----------------------------------------------------
    # Shot quality and efficiency
    # -----------------------------------------------------

    home_shots = pd.to_numeric(
        enhanced["home_shots"],
        errors="coerce",
    ).fillna(0.0)

    away_shots = pd.to_numeric(
        enhanced["away_shots"],
        errors="coerce",
    ).fillna(0.0)

    enhanced[
        "home_xg_per_shot"
    ] = safe_divide(
        home_xg,
        home_shots,
    )

    enhanced[
        "away_xg_per_shot"
    ] = safe_divide(
        away_xg,
        away_shots,
    )

    enhanced[
        "xg_per_shot_difference"
    ] = (
        enhanced[
            "home_xg_per_shot"
        ]
        - enhanced[
            "away_xg_per_shot"
        ]
    )

    enhanced[
        "home_goal_conversion"
    ] = safe_divide(
        home_goals,
        home_shots,
    )

    enhanced[
        "away_goal_conversion"
    ] = safe_divide(
        away_goals,
        away_shots,
    )

    enhanced[
        "goal_conversion_difference"
    ] = (
        enhanced[
            "home_goal_conversion"
        ]
        - enhanced[
            "away_goal_conversion"
        ]
    )

    # -----------------------------------------------------
    # Recent attacking pressure
    # -----------------------------------------------------

    home_recent_xg = pd.to_numeric(
        enhanced[
            "home_recent_xg"
        ],
        errors="coerce",
    ).fillna(0.0)

    away_recent_xg = pd.to_numeric(
        enhanced[
            "away_recent_xg"
        ],
        errors="coerce",
    ).fillna(0.0)

    home_recent_shots = pd.to_numeric(
        enhanced[
            "home_recent_shots"
        ],
        errors="coerce",
    ).fillna(0.0)

    away_recent_shots = pd.to_numeric(
        enhanced[
            "away_recent_shots"
        ],
        errors="coerce",
    ).fillna(0.0)

    home_recent_pressures = pd.to_numeric(
        enhanced[
            "home_recent_pressures"
        ],
        errors="coerce",
    ).fillna(0.0)

    away_recent_pressures = pd.to_numeric(
        enhanced[
            "away_recent_pressures"
        ],
        errors="coerce",
    ).fillna(0.0)

    enhanced[
        "home_recent_attack_index"
    ] = (
        10.0 * home_recent_xg
        + 4.0 * home_recent_shots
        + 0.35 * home_recent_pressures
    )

    enhanced[
        "away_recent_attack_index"
    ] = (
        10.0 * away_recent_xg
        + 4.0 * away_recent_shots
        + 0.35 * away_recent_pressures
    )

    enhanced[
        "recent_attack_difference"
    ] = (
        enhanced[
            "home_recent_attack_index"
        ]
        - enhanced[
            "away_recent_attack_index"
        ]
    )

    enhanced[
        "recent_xg_difference"
    ] = (
        home_recent_xg
        - away_recent_xg
    )

    enhanced[
        "recent_shot_difference"
    ] = (
        home_recent_shots
        - away_recent_shots
    )

    enhanced[
        "recent_pressure_difference"
    ] = (
        home_recent_pressures
        - away_recent_pressures
    )

    # -----------------------------------------------------
    # Momentum / state interactions
    # -----------------------------------------------------

    momentum_diff = pd.to_numeric(
        enhanced[
            "momentum_difference"
        ],
        errors="coerce",
    ).fillna(0.0)

    enhanced[
        "momentum_x_progress"
    ] = (
        momentum_diff
        * enhanced[
            "match_progress"
        ]
    )

    enhanced[
        "leading_but_under_pressure"
    ] = (
        enhanced[
            "home_leading_flag"
        ]
        * (
            enhanced[
                "recent_attack_difference"
            ] < 0
        ).astype(int)
    )

    enhanced[
        "trailing_with_pressure"
    ] = (
        enhanced[
            "home_trailing_flag"
        ]
        * (
            enhanced[
                "recent_attack_difference"
            ] > 0
        ).astype(int)
    )

    enhanced[
        "draw_with_home_pressure"
    ] = (
        enhanced[
            "draw_flag"
        ]
        * (
            enhanced[
                "recent_attack_difference"
            ] > 0
        ).astype(int)
    )

    enhanced[
        "draw_with_away_pressure"
    ] = (
        enhanced[
            "draw_flag"
        ]
        * (
            enhanced[
                "recent_attack_difference"
            ] < 0
        ).astype(int)
    )

    # -----------------------------------------------------
    # Relative dominance features
    # -----------------------------------------------------

    pressure_diff = pd.to_numeric(
        enhanced[
            "pressure_difference"
        ],
        errors="coerce",
    ).fillna(0.0)

    shot_diff = pd.to_numeric(
        enhanced[
            "shot_difference"
        ],
        errors="coerce",
    ).fillna(0.0)

    xg_diff = pd.to_numeric(
        enhanced[
            "xg_difference"
        ],
        errors="coerce",
    ).fillna(0.0)

    enhanced[
        "overall_dominance_index"
    ] = (
        2.5 * goal_diff
        + 1.5 * xg_diff
        + 0.20 * shot_diff
        + 0.02 * pressure_diff
        + 0.01 * momentum_diff
    )

    enhanced[
        "late_dominance_index"
    ] = (
        enhanced[
            "overall_dominance_index"
        ]
        * enhanced[
            "match_progress"
        ]
    )

    # -----------------------------------------------------
    # Validate newly created features
    # -----------------------------------------------------

    new_columns = [
        column
        for column in enhanced.columns
        if column not in df.columns
    ]

    numeric_new = enhanced[
        new_columns
    ].select_dtypes(
        include=[
            np.number,
        ]
    )

    inf_count = int(
        np.isinf(
            numeric_new.to_numpy(
                dtype=float
            )
        ).sum()
    )

    nan_count = int(
        numeric_new.isna().sum().sum()
    )

    section(
        "ENHANCED DATASET VALIDATION"
    )

    print(
        f"New engineered features: "
        f"{len(new_columns)}"
    )

    print(
        f"Infinite numeric values: "
        f"{inf_count}"
    )

    print(
        f"Missing engineered numeric values: "
        f"{nan_count}"
    )

    if inf_count > 0:
        print(
            "[FAIL] Infinite values detected in engineered features."
        )
        sys.exit(1)

    if nan_count > 0:
        print(
            "[FAIL] Missing values detected in engineered features."
        )
        sys.exit(1)

    # Original row identity must remain unchanged.
    original_keys = (
        df[
            [
                "match_id",
                "snapshot_minute",
            ]
        ]
        .astype(str)
        .agg(
            "|".join,
            axis=1,
        )
        .tolist()
    )

    enhanced_keys = (
        enhanced[
            [
                "match_id",
                "snapshot_minute",
            ]
        ]
        .astype(str)
        .agg(
            "|".join,
            axis=1,
        )
        .tolist()
    )

    if original_keys != enhanced_keys:
        print(
            "[FAIL] Row identity/order changed during feature engineering."
        )
        sys.exit(1)

    if not (
        enhanced["target"].astype(str)
        .equals(
            df["target"].astype(str)
        )
    ):
        print(
            "[FAIL] Target labels changed unexpectedly."
        )
        sys.exit(1)

    enhanced.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    section(
        "ENHANCED TRAINING DATASET CREATED"
    )

    print(
        f"File: {OUTPUT_FILE}"
    )
    print(
        f"Rows: {len(enhanced):,}"
    )
    print(
        f"Unique matches: "
        f"{enhanced['match_id'].nunique():,}"
    )
    print(
        f"Original columns: "
        f"{len(df.columns)}"
    )
    print(
        f"Enhanced columns: "
        f"{len(enhanced.columns)}"
    )
    print(
        f"Added features: "
        f"{len(new_columns)}"
    )

    print(
        "\nNew feature names:"
    )

    for column in new_columns:
        print(
            f"  - {column}"
        )

    print(
        "\nNext step:"
    )
    print(
        "Train a new time-aware model pipeline using "
        "enhanced_historical_match_states.csv and compare "
        "it directly against the current benchmark."
    )


if __name__ == "__main__":
    main()
