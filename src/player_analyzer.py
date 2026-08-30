from statsbombpy import sb
import pandas as pd


MATCH_ID = 3895309


def load_events(match_id):
    """Load event data for the selected match."""
    return sb.events(match_id=match_id)


def calculate_player_stats(events):
    """Calculate core statistics for every player in the match."""

    players = events["player"].dropna().unique()

    player_stats = []

    for player in players:

        player_events = events[events["player"] == player]

        team = player_events["team"].dropna().iloc[0]

        passes = player_events[
            player_events["type"] == "Pass"
        ]

        shots = player_events[
            player_events["type"] == "Shot"
        ]

        carries = player_events[
            player_events["type"] == "Carry"
        ]

        pressures = player_events[
            player_events["type"] == "Pressure"
        ]

        interceptions = player_events[
            player_events["type"] == "Interception"
        ]

        ball_recoveries = player_events[
            player_events["type"] == "Ball Recovery"
        ]

        completed_passes = passes["pass_outcome"].isna().sum()

        total_passes = len(passes)

        pass_completion = (
            completed_passes / total_passes * 100
            if total_passes > 0
            else 0
        )

        xg = shots["shot_statsbomb_xg"].fillna(0).sum()

        goals = shots["shot_outcome"].eq("Goal").sum()

        player_stats.append(
            {
                "Player": player,
                "Team": team,
                "Passes": total_passes,
                "Pass Completion %": round(pass_completion, 1),
                "Carries": len(carries),
                "Shots": len(shots),
                "xG": round(xg, 2),
                "Goals": goals,
                "Pressures": len(pressures),
                "Interceptions": len(interceptions),
                "Recoveries": len(ball_recoveries),
            }
        )

    return pd.DataFrame(player_stats)


if __name__ == "__main__":

    print("=" * 80)
    print("LIVEMATCH INTELLIGENCE")
    print("PLAYER INTELLIGENCE ENGINE")
    print("=" * 80)

    print("\nLoading player events...")

    events = load_events(MATCH_ID)

    player_stats = calculate_player_stats(events)

    print("\nPLAYER PERFORMANCE TABLE\n")

    player_stats = player_stats.sort_values(
        by="Passes",
        ascending=False
    )

    print(player_stats.to_string(index=False))