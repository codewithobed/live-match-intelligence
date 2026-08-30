from statsbombpy import sb
import pandas as pd


MATCH_ID = 3895309


def load_match_events(match_id):
    """Load StatsBomb event data for a match."""
    print(f"Loading events for match {match_id}...")
    events = sb.events(match_id=match_id)
    print(f"Loaded {len(events)} events.\n")
    return events


def get_teams(events):
    """Return the teams participating in the match."""
    teams = events["team"].dropna().unique().tolist()
    return teams


def calculate_team_stats(events, team):
    """Calculate core match statistics for one team."""

    team_events = events[events["team"] == team]

    passes = team_events[team_events["type"] == "Pass"]
    shots = team_events[team_events["type"] == "Shot"]
    pressures = team_events[team_events["type"] == "Pressure"]
    interceptions = team_events[team_events["type"] == "Interception"]
    carries = team_events[team_events["type"] == "Carry"]

    completed_passes = passes["pass_outcome"].isna().sum()

    total_passes = len(passes)

    pass_completion = (
        completed_passes / total_passes * 100
        if total_passes > 0
        else 0
    )

    total_xg = shots["shot_statsbomb_xg"].fillna(0).sum()

    goals = (
        shots["shot_outcome"]
        .eq("Goal")
        .sum()
    )

    return {
        "Team": team,
        "Goals": goals,
        "Shots": len(shots),
        "xG": round(total_xg, 2),
        "Passes": total_passes,
        "Completed Passes": completed_passes,
        "Pass Completion %": round(pass_completion, 1),
        "Carries": len(carries),
        "Pressures": len(pressures),
        "Interceptions": len(interceptions),
    }


def build_match_summary(events):
    """Create a comparison table for both teams."""

    teams = get_teams(events)

    summaries = []

    for team in teams:
        summaries.append(
            calculate_team_stats(events, team)
        )

    return pd.DataFrame(summaries)


if __name__ == "__main__":

    print("=" * 70)
    print("LIVEMATCH INTELLIGENCE")
    print("MATCH ANALYSIS ENGINE")
    print("=" * 70)

    events = load_match_events(MATCH_ID)

    summary = build_match_summary(events)

    print("TEAM PERFORMANCE COMPARISON\n")

    print(summary.to_string(index=False))