"""
Test the Tactical Change Detector using the Borussia Dortmund vs Bayer Leverkusen
StatsBomb match used throughout LiveMatch Intelligence.

Run:
    python -m src.test_tactical_change_detector
"""

from statsbombpy import sb

from src.tactical_change_detector import detect_tactical_changes


MATCH_ID = 3895309

TEAM_NAMES = [
    "Borussia Dortmund",
    "Bayer Leverkusen",
]


def main():
    print("\n" + "=" * 72)
    print("LIVE MATCH INTELLIGENCE — TACTICAL CHANGE DETECTOR TEST")
    print("=" * 72)

    print(
        f"\nLoading StatsBomb events for match {MATCH_ID}..."
    )

    events = sb.events(
        match_id=MATCH_ID,
        split=False,
        flatten_attrs=True,
    )

    print(
        f"Loaded {len(events):,} events."
    )

    result = detect_tactical_changes(
        events,
        team_names=TEAM_NAMES,
    )

    team_comparisons = result[
        "team_comparisons"
    ]

    player_comparisons = result[
        "player_comparisons"
    ]

    alerts = result[
        "alerts"
    ]

    print("\n" + "=" * 72)
    print("TEAM HALF-BY-HALF COMPARISON")
    print("=" * 72)

    if team_comparisons.empty:
        print(
            "No team comparison data was produced."
        )
    else:
        display_columns = [
            "Team",
            "First Half Avg X",
            "Second Half Avg X",
            "Average X Change",
            "First Half Pressures",
            "Second Half Pressures",
            "Pressure Change %",
            "First Half Shots",
            "Second Half Shots",
            "Shot Change",
            "First Half xG",
            "Second Half xG",
            "xG Change",
            "Attack Index Change %",
        ]

        available = [
            column
            for column in display_columns
            if column in team_comparisons.columns
        ]

        print(
            team_comparisons[
                available
            ]
            .round(3)
            .to_string(
                index=False
            )
        )

    print("\n" + "=" * 72)
    print("LARGEST PLAYER POSITION SHIFTS")
    print("=" * 72)

    if player_comparisons.empty:
        print(
            "No player positional comparison data was produced."
        )
    else:
        player_table = (
            player_comparisons
            .copy()
        )

        player_table[
            "Absolute X Change"
        ] = (
            player_table[
                "X Change"
            ]
            .abs()
        )

        player_table = player_table.sort_values(
            "Absolute X Change",
            ascending=False,
        )

        display_columns = [
            "Player",
            "Team",
            "First Half X",
            "Second Half X",
            "X Change",
            "First Half Involvements",
            "Second Half Involvements",
        ]

        print(
            player_table[
                display_columns
            ]
            .head(12)
            .round(2)
            .to_string(
                index=False
            )
        )

    print("\n" + "=" * 72)
    print("TACTICAL ALERTS")
    print("=" * 72)

    if alerts.empty:
        print(
            "No alert thresholds were triggered."
        )
    else:
        for index, row in alerts.iterrows():
            alert_type = row.get(
                "Type",
                "Signal",
            )

            team = row.get(
                "Team",
                "",
            )

            message = row.get(
                "Message",
                "",
            )

            print(
                f"\n[{index + 1}] "
                f"{alert_type}"
            )

            if team:
                print(
                    f"    Team: {team}"
                )

            print(
                f"    {message}"
            )

    print("\n" + "=" * 72)
    print("TEST RESULT")
    print("=" * 72)

    if (
        team_comparisons.empty
        and player_comparisons.empty
    ):
        raise RuntimeError(
            "Detector returned no comparison data."
        )

    print(
        "SUCCESS: tactical change detection pipeline is working."
    )

    print(
        "\nReminder: these are event-data-derived tactical signals, "
        "not confirmed coaching instructions or tracking-data formations."
    )


if __name__ == "__main__":
    main()
