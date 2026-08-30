"""
Test the Pass Network & Build-up Analyzer using the Borussia Dortmund vs
Bayer Leverkusen StatsBomb match used throughout LiveMatch Intelligence.

Run:
    python -m src.test_pass_network_analyzer
"""

from statsbombpy import sb

from src.pass_network_analyzer import analyze_pass_networks


MATCH_ID = 3895309

TEAM_NAMES = [
    "Borussia Dortmund",
    "Bayer Leverkusen",
]


def main():
    print("\n" + "=" * 76)
    print("LIVE MATCH INTELLIGENCE — PASS NETWORK ANALYZER TEST")
    print("=" * 76)

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

    result = analyze_pass_networks(
        events,
        team_names=TEAM_NAMES,
        period_label="Full Match",
    )

    summaries = result[
        "summaries"
    ]

    nodes = result[
        "nodes"
    ]

    edges = result[
        "edges"
    ]

    half_comparisons = result[
        "half_comparisons"
    ]

    insights = result[
        "insights"
    ]

    print("\n" + "=" * 76)
    print("TEAM PASS-NETWORK SUMMARY")
    print("=" * 76)

    if summaries.empty:
        print(
            "No pass-network summary was produced."
        )
    else:
        display_columns = [
            "Team",
            "Passes Attempted",
            "Passes Completed",
            "Pass Completion %",
            "Most Involved Player",
            "Top Passer",
            "Top Receiver",
            "Strongest Link",
            "Strongest Link Passes",
            "Average Network X",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in summaries.columns
        ]

        print(
            summaries[
                available_columns
            ]
            .round(
                {
                    "Pass Completion %": 2,
                    "Average Network X": 2,
                }
            )
            .to_string(
                index=False
            )
        )

    print("\n" + "=" * 76)
    print("TOP PLAYER NETWORK NODES")
    print("=" * 76)

    if nodes.empty:
        print(
            "No player network nodes were produced."
        )
    else:
        node_display = (
            nodes
            .sort_values(
                [
                    "Team",
                    "Network Involvement",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .groupby(
                "Team",
                group_keys=False,
            )
            .head(8)
        )

        display_columns = [
            "Player",
            "Team",
            "Passes Attempted",
            "Passes Completed",
            "Pass Completion %",
            "Passes Received",
            "Network Involvement",
            "Average X",
            "Average Y",
        ]

        print(
            node_display[
                display_columns
            ]
            .round(
                {
                    "Pass Completion %": 1,
                    "Average X": 1,
                    "Average Y": 1,
                }
            )
            .to_string(
                index=False
            )
        )

    print("\n" + "=" * 76)
    print("STRONGEST PASSING CONNECTIONS")
    print("=" * 76)

    if edges.empty:
        print(
            "No completed passing connections met the threshold."
        )
    else:
        edge_display = (
            edges
            .sort_values(
                [
                    "Team",
                    "Pass Count",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .groupby(
                "Team",
                group_keys=False,
            )
            .head(10)
        )

        print(
            edge_display[
                [
                    "Passer",
                    "Recipient",
                    "Team",
                    "Pass Count",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print("\n" + "=" * 76)
    print("FIRST-HALF VS SECOND-HALF BUILD-UP")
    print("=" * 76)

    if half_comparisons.empty:
        print(
            "No half-by-half build-up comparison was produced."
        )
    else:
        display_columns = [
            "Team",
            "First Half Passes",
            "Second Half Passes",
            "Pass Volume Change",
            "First Half Completion %",
            "Second Half Completion %",
            "Completion Change",
            "First Half Network X",
            "Second Half Network X",
            "Network X Change",
            "First Half Most Involved",
            "Second Half Most Involved",
            "First Half Strongest Link",
            "Second Half Strongest Link",
        ]

        print(
            half_comparisons[
                display_columns
            ]
            .round(
                {
                    "First Half Completion %": 2,
                    "Second Half Completion %": 2,
                    "Completion Change": 2,
                    "First Half Network X": 2,
                    "Second Half Network X": 2,
                    "Network X Change": 2,
                }
            )
            .to_string(
                index=False
            )
        )

    print("\n" + "=" * 76)
    print("ANALYST INSIGHTS")
    print("=" * 76)

    if insights.empty:
        print(
            "No analyst insights were produced."
        )
    else:
        for index, row in insights.iterrows():
            print(
                f"\n[{index + 1}] "
                f"{row.get('Type', 'Insight')} — "
                f"{row.get('Team', '')}"
            )

            print(
                f"    {row.get('Message', '')}"
            )

    print("\n" + "=" * 76)
    print("TEST RESULT")
    print("=" * 76)

    if (
        summaries.empty
        or nodes.empty
    ):
        raise RuntimeError(
            "Pass-network analyzer returned incomplete output."
        )

    print(
        "SUCCESS: pass-network analysis pipeline is working."
    )

    print(
        "\nReminder: pass-network positions are derived from event locations, "
        "not optical tracking data or exact formation coordinates."
    )


if __name__ == "__main__":
    main()
