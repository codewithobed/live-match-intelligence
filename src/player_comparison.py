from statsbombpy import sb
import pandas as pd


MATCH_ID = 3895309

PLAYER_1 = "Florian Wirtz"
PLAYER_2 = "Julian Brandt"


def load_events(match_id):
    """Load event data for the selected match."""
    return sb.events(match_id=match_id)


def get_player_metrics(events, player_name):
    """Calculate core match metrics for one player."""

    player_events = events[events["player"] == player_name]

    if player_events.empty:
        return None

    team = player_events["team"].dropna().iloc[0]

    passes = player_events[player_events["type"] == "Pass"]
    shots = player_events[player_events["type"] == "Shot"]
    carries = player_events[player_events["type"] == "Carry"]
    pressures = player_events[player_events["type"] == "Pressure"]
    interceptions = player_events[player_events["type"] == "Interception"]
    recoveries = player_events[player_events["type"] == "Ball Recovery"]

    completed_passes = passes["pass_outcome"].isna().sum()
    total_passes = len(passes)

    pass_completion = (
        completed_passes / total_passes * 100
        if total_passes > 0
        else 0
    )

    total_xg = shots["shot_statsbomb_xg"].fillna(0).sum()
    goals = shots["shot_outcome"].eq("Goal").sum()

    return {
        "Player": player_name,
        "Team": team,
        "Passes": total_passes,
        "Completed Passes": completed_passes,
        "Pass Completion %": round(pass_completion, 1),
        "Carries": len(carries),
        "Shots": len(shots),
        "xG": round(total_xg, 2),
        "Goals": goals,
        "Pressures": len(pressures),
        "Interceptions": len(interceptions),
        "Recoveries": len(recoveries),
    }


def compare_players(events, player_1, player_2):
    """Create a side-by-side comparison for two players."""

    first = get_player_metrics(events, player_1)
    second = get_player_metrics(events, player_2)

    if first is None:
        print(f"Player not found: {player_1}")
        return

    if second is None:
        print(f"Player not found: {player_2}")
        return

    comparison = pd.DataFrame(
        {
            player_1: first,
            player_2: second,
        }
    )

    return comparison


if __name__ == "__main__":

    print("=" * 80)
    print("LIVEMATCH INTELLIGENCE")
    print("PLAYER COMPARISON ENGINE")
    print("=" * 80)

    print(f"\nComparing {PLAYER_1} vs {PLAYER_2}...\n")

    events = load_events(MATCH_ID)

    comparison = compare_players(
        events,
        PLAYER_1,
        PLAYER_2
    )

    if comparison is not None:
        print(comparison.to_string())