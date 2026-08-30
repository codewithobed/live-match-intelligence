"""
Test the LiveMatch Intelligence calibrated match-outcome predictor.

This script feeds a realistic Borussia Dortmund vs Bayer Leverkusen
late-match state into src.match_outcome_predictor and prints the resulting
experimental Home Win / Draw / Away Win probabilities.

Run:
    python src\test_match_outcome_predictor.py
"""

from src.match_outcome_predictor import predict_match_outcome


def main():
    # Realistic late-match state based on the Dortmund vs Leverkusen
    # replay used throughout the dashboard development.
    #
    # At approximately minute 93:
    # Dortmund lead 1-0.
    # Leverkusen have the stronger cumulative xG and more shots.
    # Recent momentum slightly favours Leverkusen.

    base_features = {
        "snapshot_minute": 93,

        # Score
        "home_goals": 1,
        "away_goals": 0,
        "goal_difference": 1,

        # xG
        "home_xg": 0.55,
        "away_xg": 1.21,
        "xg_difference": -0.66,

        # Shots
        "home_shots": 8,
        "away_shots": 11,
        "shot_difference": -3,

        # Passing
        "home_passes": 500,
        "away_passes": 545,
        "pass_difference": -45,

        "home_pass_completion": 85.0,
        "away_pass_completion": 86.0,

        # Defensive / activity metrics
        "home_pressures": 198,
        "away_pressures": 145,
        "pressure_difference": 53,

        "home_carries": 370,
        "away_carries": 455,

        "home_recoveries": 46,
        "away_recoveries": 49,

        "home_interceptions": 10,
        "away_interceptions": 5,

        # Recent 10-minute state
        "home_recent_xg": 0.00,
        "away_recent_xg": 0.00,

        "home_recent_shots": 0,
        "away_recent_shots": 0,

        "home_recent_pressures": 11,
        "away_recent_pressures": 12,

        # Rolling momentum
        "home_momentum": 45.3,
        "away_momentum": 54.7,
        "momentum_difference": -9.4,
    }

    result = predict_match_outcome(
        match_minute=93,
        base_features=base_features,
    )

    print("\n" + "=" * 64)
    print("LIVE MATCH INTELLIGENCE — PREDICTOR TEST")
    print("=" * 64)

    print(
        f"Model stage used: "
        f"{result['model_minute']}'"
    )

    print(
        f"Model variant: "
        f"{result['model_variant']}"
    )

    print("\nExperimental ML probabilities:")

    print(
        f"  Borussia Dortmund Win : "
        f"{result['Home Win'] * 100:.1f}%"
    )

    print(
        f"  Draw                  : "
        f"{result['Draw'] * 100:.1f}%"
    )

    print(
        f"  Bayer Leverkusen Win  : "
        f"{result['Away Win'] * 100:.1f}%"
    )

    print("\nValidation metrics for this model stage:")

    accuracy = result.get(
        "validation_accuracy"
    )

    macro_f1 = result.get(
        "validation_macro_f1"
    )

    log_loss = result.get(
        "validation_log_loss"
    )

    if accuracy is not None:
        print(
            f"  Accuracy : "
            f"{accuracy:.4f}"
        )

    if macro_f1 is not None:
        print(
            f"  Macro F1 : "
            f"{macro_f1:.4f}"
        )

    if log_loss is not None:
        print(
            f"  Log Loss : "
            f"{log_loss:.4f}"
        )

    total_probability = (
        result["Home Win"]
        + result["Draw"]
        + result["Away Win"]
    )

    print(
        f"\nProbability total: "
        f"{total_probability * 100:.1f}%"
    )

    if abs(total_probability - 1.0) > 1e-6:
        raise RuntimeError(
            "Prediction probabilities do not sum to 100%."
        )

    print(
        "\nSUCCESS: full match-state prediction pipeline is working."
    )


if __name__ == "__main__":
    main()
