"""
Build historical match-state snapshots for LiveMatch Intelligence.

Output:
    data/historical_match_states.csv

Each row represents one historical match at one snapshot minute.
Only information available up to that minute is used as a feature.
The target is the actual final result: Home Win, Draw, or Away Win.

Data source:
    StatsBomb Open Data via statsbombpy.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from statsbombpy import sb


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "historical_match_states.csv"

SNAPSHOT_MINUTES = [15, 30, 45, 60, 75, 85]

# Larger multi-competition sample for the production modelling stage.
# The builder combines multiple StatsBomb Open Data competition/seasons
# until it has up to this many UNIQUE historical matches.
TARGET_MATCHES = 500

# Minimum number of matches required before we accept the expanded sample.
# If StatsBomb Open Data exposes fewer, the script still uses everything
# available and prints a warning.
MIN_ACCEPTABLE_MATCHES = 250

RANDOM_STATE = 42


# ---------------------------------------------------------
# Small utility functions
# ---------------------------------------------------------

def safe_numeric(series):
    """Convert a pandas Series to numeric values safely."""
    return pd.to_numeric(series, errors="coerce").fillna(0)


def normalise_name(value):
    """Return a clean string representation of a StatsBomb value."""
    if isinstance(value, dict):
        return str(value.get("name", ""))
    if pd.isna(value):
        return ""
    return str(value)


def event_team_series(events):
    """Return event team names as strings."""
    if "team" not in events.columns:
        return pd.Series("", index=events.index)
    return events["team"].apply(normalise_name)


def event_type_series(events):
    """Return event type names as strings."""
    if "type" not in events.columns:
        return pd.Series("", index=events.index)
    return events["type"].apply(normalise_name)


def event_minute_series(events):
    """Return event minutes as numeric values."""
    if "minute" not in events.columns:
        return pd.Series(0, index=events.index, dtype=float)
    return safe_numeric(events["minute"])


def event_outcome_series(events, column):
    """Return a named outcome column as strings when present."""
    if column not in events.columns:
        return pd.Series("", index=events.index)
    return events[column].apply(normalise_name)


def xg_series(events):
    """Return StatsBomb shot xG safely."""
    if "shot_statsbomb_xg" not in events.columns:
        return pd.Series(0.0, index=events.index)
    return safe_numeric(events["shot_statsbomb_xg"])


# ---------------------------------------------------------
# Match-state feature engineering
# ---------------------------------------------------------

def count_goals(events, team_name):
    """
    Count goals scored by a team.

    StatsBomb own goals are handled separately because an own-goal event
    is credited to the opposition rather than the team responsible for it.
    """
    if events.empty:
        return 0

    teams = event_team_series(events)
    types = event_type_series(events)
    shot_outcomes = event_outcome_series(events, "shot_outcome")

    regular_goals = (
        (teams == team_name)
        & (types == "Shot")
        & (shot_outcomes == "Goal")
    ).sum()

    # Own Goal For is already recorded for the benefiting team.
    own_goal_for = (
        (teams == team_name)
        & (types == "Own Goal For")
    ).sum()

    return int(regular_goals + own_goal_for)


def team_metrics(events, team_name):
    """Calculate cumulative metrics for one team from the supplied events."""
    if events.empty:
        return {
            "Goals": 0,
            "xG": 0.0,
            "Shots": 0,
            "Passes": 0,
            "Completed Passes": 0,
            "Pressures": 0,
            "Carries": 0,
            "Recoveries": 0,
            "Interceptions": 0,
        }

    teams = event_team_series(events)
    types = event_type_series(events)
    team_mask = teams == team_name

    team_events = events.loc[team_mask].copy()
    team_types = types.loc[team_mask]

    shots_mask = team_types == "Shot"
    passes_mask = team_types == "Pass"

    shots = int(shots_mask.sum())
    passes = int(passes_mask.sum())

    # In StatsBomb, a missing pass_outcome normally means the pass completed.
    if "pass_outcome" in team_events.columns:
        pass_outcomes = event_outcome_series(team_events, "pass_outcome")
        completed_passes = int(
            (
                (event_type_series(team_events) == "Pass")
                & (pass_outcomes == "")
            ).sum()
        )
    else:
        completed_passes = passes

    if "shot_statsbomb_xg" in team_events.columns:
        team_xg = float(
            xg_series(team_events)
            .where(event_type_series(team_events) == "Shot", 0.0)
            .sum()
        )
    else:
        team_xg = 0.0

    return {
        "Goals": count_goals(events, team_name),
        "xG": team_xg,
        "Shots": shots,
        "Passes": passes,
        "Completed Passes": completed_passes,
        "Pressures": int((team_types == "Pressure").sum()),
        "Carries": int((team_types == "Carry").sum()),
        "Recoveries": int((team_types == "Ball Recovery").sum()),
        "Interceptions": int((team_types == "Interception").sum()),
    }


def momentum_points(metrics):
    """
    Transparent rolling activity score.

    This mirrors the current dashboard prototype so the historical dataset
    contains a feature comparable with the live replay view.
    """
    return (
        metrics["Shots"] * 4.0
        + metrics["xG"] * 10.0
        + metrics["Pressures"] * 0.35
        + metrics["Carries"] * 0.08
        + metrics["Recoveries"] * 0.20
        + metrics["Passes"] * 0.03
    )


def relative_momentum(team_a_metrics, team_b_metrics):
    """Convert two activity scores to relative 0-100 momentum shares."""
    a = momentum_points(team_a_metrics)
    b = momentum_points(team_b_metrics)
    total = a + b

    if total <= 0:
        return 50.0, 50.0

    return (100.0 * a / total, 100.0 * b / total)


def final_result(home_goals, away_goals):
    """Create the three-class target label."""
    if home_goals > away_goals:
        return "Home Win"
    if home_goals < away_goals:
        return "Away Win"
    return "Draw"


def create_match_snapshots(events, match_row):
    """Create historical feature rows for one match."""
    if events is None or events.empty:
        return []

    home_team = normalise_name(match_row["home_team"])
    away_team = normalise_name(match_row["away_team"])

    if not home_team or not away_team:
        return []

    minutes = event_minute_series(events)

    final_home_goals = int(match_row["home_score"])
    final_away_goals = int(match_row["away_score"])
    target = final_result(final_home_goals, final_away_goals)

    rows = []

    for snapshot_minute in SNAPSHOT_MINUTES:
        cumulative = events.loc[minutes <= snapshot_minute].copy()

        window_start = max(0, snapshot_minute - 10)
        recent = events.loc[
            (minutes > window_start)
            & (minutes <= snapshot_minute)
        ].copy()

        home = team_metrics(cumulative, home_team)
        away = team_metrics(cumulative, away_team)

        home_recent = team_metrics(recent, home_team)
        away_recent = team_metrics(recent, away_team)

        home_momentum, away_momentum = relative_momentum(
            home_recent,
            away_recent,
        )

        home_pass_completion = (
            100.0 * home["Completed Passes"] / home["Passes"]
            if home["Passes"] > 0
            else 0.0
        )
        away_pass_completion = (
            100.0 * away["Completed Passes"] / away["Passes"]
            if away["Passes"] > 0
            else 0.0
        )

        rows.append(
            {
                "match_id": int(match_row["match_id"]),
                "match_date": str(match_row.get("match_date", "")),
                "competition": normalise_name(
                    match_row.get("competition", "")
                ),
                "season": normalise_name(
                    match_row.get("season", "")
                ),
                "home_team": home_team,
                "away_team": away_team,
                "snapshot_minute": snapshot_minute,

                # Current score
                "home_goals": home["Goals"],
                "away_goals": away["Goals"],
                "goal_difference": home["Goals"] - away["Goals"],

                # Chance creation
                "home_xg": round(home["xG"], 4),
                "away_xg": round(away["xG"], 4),
                "xg_difference": round(home["xG"] - away["xG"], 4),
                "home_shots": home["Shots"],
                "away_shots": away["Shots"],
                "shot_difference": home["Shots"] - away["Shots"],

                # Ball use
                "home_passes": home["Passes"],
                "away_passes": away["Passes"],
                "pass_difference": home["Passes"] - away["Passes"],
                "home_pass_completion": round(home_pass_completion, 2),
                "away_pass_completion": round(away_pass_completion, 2),

                # Activity / defensive actions
                "home_pressures": home["Pressures"],
                "away_pressures": away["Pressures"],
                "pressure_difference": (
                    home["Pressures"] - away["Pressures"]
                ),
                "home_carries": home["Carries"],
                "away_carries": away["Carries"],
                "home_recoveries": home["Recoveries"],
                "away_recoveries": away["Recoveries"],
                "home_interceptions": home["Interceptions"],
                "away_interceptions": away["Interceptions"],

                # Recent 10-minute state
                "home_recent_xg": round(home_recent["xG"], 4),
                "away_recent_xg": round(away_recent["xG"], 4),
                "home_recent_shots": home_recent["Shots"],
                "away_recent_shots": away_recent["Shots"],
                "home_recent_pressures": home_recent["Pressures"],
                "away_recent_pressures": away_recent["Pressures"],
                "home_momentum": round(home_momentum, 2),
                "away_momentum": round(away_momentum, 2),
                "momentum_difference": round(
                    home_momentum - away_momentum,
                    2,
                ),

                # Supervised-learning target
                "final_home_goals": final_home_goals,
                "final_away_goals": final_away_goals,
                "target": target,
            }
        )

    return rows


# ---------------------------------------------------------
# Competition collection
# ---------------------------------------------------------

def collect_open_matches():
    """
    Collect matches across multiple StatsBomb Open Data competition/seasons.

    Preference:
        - men's competitions where available
        - newer seasons first
        - unique match IDs only

    Returns:
        DataFrame containing up to TARGET_MATCHES unique matches.
    """
    print("Loading available StatsBomb open competitions...")
    competitions = sb.competitions()

    if competitions.empty:
        raise RuntimeError("StatsBomb returned no open competitions.")

    candidates = competitions.copy()

    if "competition_gender" in candidates.columns:
        male = candidates[
            candidates["competition_gender"]
            .astype(str)
            .str.lower()
            .eq("male")
        ]
        if not male.empty:
            candidates = male

    # Sort recent-looking season labels first where possible.
    sort_columns = [
        col for col in ["season_name", "competition_name"]
        if col in candidates.columns
    ]
    if sort_columns:
        candidates = candidates.sort_values(
            sort_columns,
            ascending=False,
        )

    collected = []
    seen_match_ids = set()

    print(
        f"Target sample: up to {TARGET_MATCHES} unique matches "
        "across multiple open competition/seasons.\n"
    )

    for _, comp in candidates.iterrows():
        if len(seen_match_ids) >= TARGET_MATCHES:
            break

        competition_id = int(comp["competition_id"])
        season_id = int(comp["season_id"])

        competition_name = normalise_name(
            comp.get("competition_name", competition_id)
        )
        season_name = normalise_name(
            comp.get("season_name", season_id)
        )

        try:
            matches = sb.matches(
                competition_id=competition_id,
                season_id=season_id,
            )
        except Exception as exc:
            print(
                f"Skipping {competition_name} — {season_name}: {exc}"
            )
            continue

        if matches is None or matches.empty:
            continue

        matches = matches.dropna(
            subset=["match_id", "home_score", "away_score"]
        ).copy()

        if matches.empty:
            continue

        matches["competition"] = competition_name
        matches["season"] = season_name
        matches["_competition_id"] = competition_id
        matches["_season_id"] = season_id

        before = len(seen_match_ids)

        for _, row in matches.iterrows():
            match_id = int(row["match_id"])

            if match_id in seen_match_ids:
                continue

            collected.append(row.to_dict())
            seen_match_ids.add(match_id)

            if len(seen_match_ids) >= TARGET_MATCHES:
                break

        added = len(seen_match_ids) - before

        if added:
            print(
                f"Added {added:>3} matches from "
                f"{competition_name} — {season_name} "
                f"(total {len(seen_match_ids)})"
            )

    if not collected:
        raise RuntimeError(
            "Could not collect any usable StatsBomb open matches."
        )

    all_matches = pd.DataFrame(collected)

    # Reproducibly shuffle the multi-competition sample so one competition
    # does not dominate the processing order.
    all_matches = all_matches.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    if len(all_matches) > TARGET_MATCHES:
        all_matches = all_matches.iloc[:TARGET_MATCHES].copy()

    if len(all_matches) < MIN_ACCEPTABLE_MATCHES:
        print(
            f"\nWARNING: only {len(all_matches)} unique matches were "
            f"available, below the preferred minimum "
            f"of {MIN_ACCEPTABLE_MATCHES}."
        )

    print(
        f"\nCollected {len(all_matches)} unique historical matches "
        "for event processing."
    )

    return all_matches


# ---------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------

def main():
    warnings.filterwarnings(
        "ignore",
        message="credentials were not supplied",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    matches = collect_open_matches()

    print(
        f"\nBuilding snapshots for {len(matches)} matches..."
    )
    print(
        f"Snapshot minutes: {SNAPSHOT_MINUTES}"
    )
    print(
        f"Expected maximum rows: "
        f"{len(matches) * len(SNAPSHOT_MINUTES)}\n"
    )

    all_rows = []
    failed_matches = []

    for index, match_row in matches.iterrows():
        match_id = int(match_row["match_id"])
        home_team = normalise_name(match_row["home_team"])
        away_team = normalise_name(match_row["away_team"])

        competition_name = normalise_name(
            match_row.get("competition", "")
        )
        season_name = normalise_name(
            match_row.get("season", "")
        )

        print(
            f"[{index + 1}/{len(matches)}] "
            f"{home_team} vs {away_team} "
            f"| {competition_name} {season_name} "
            f"| match {match_id}"
        )

        try:
            events = sb.events(
                match_id=match_id,
                split=False,
                flatten_attrs=True,
            )

            rows = create_match_snapshots(
                events,
                match_row,
            )

            if len(rows) != len(SNAPSHOT_MINUTES):
                raise RuntimeError(
                    f"Expected {len(SNAPSHOT_MINUTES)} snapshots, "
                    f"created {len(rows)}."
                )

            all_rows.extend(rows)

        except Exception as exc:
            failed_matches.append(
                {
                    "match_id": match_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "error": str(exc),
                }
            )
            print(f"  Skipped: {exc}")

        # Small pause to avoid hammering the public open-data endpoint.
        time.sleep(0.05)

    if not all_rows:
        raise RuntimeError(
            "No training rows were created. "
            "Check the StatsBomb connection and terminal output."
        )

    dataset = pd.DataFrame(all_rows)

    # Remove any accidental duplicate match/minute pairs defensively.
    before_dedup = len(dataset)
    dataset = dataset.drop_duplicates(
        subset=["match_id", "snapshot_minute"],
        keep="first",
    ).copy()

    removed = before_dedup - len(dataset)
    if removed:
        print(
            f"\nRemoved {removed} duplicate match/minute rows."
        )

    # Keep only complete matches containing all six expected snapshots.
    snapshot_counts = (
        dataset.groupby("match_id")["snapshot_minute"]
        .nunique()
    )

    complete_match_ids = snapshot_counts[
        snapshot_counts == len(SNAPSHOT_MINUTES)
    ].index

    dataset = dataset[
        dataset["match_id"].isin(complete_match_ids)
    ].copy()

    # Sort for readability while preserving match IDs for grouped splitting.
    dataset = dataset.sort_values(
        ["match_date", "match_id", "snapshot_minute"]
    ).reset_index(drop=True)

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    unique_matches = dataset["match_id"].nunique()

    print("\n" + "=" * 72)
    print("EXPANDED TRAINING DATASET CREATED")
    print("=" * 72)
    print(f"File: {OUTPUT_FILE}")
    print(f"Rows: {len(dataset):,}")
    print(f"Unique complete matches: {unique_matches:,}")
    print(f"Features/columns: {len(dataset.columns):,}")

    if unique_matches < TARGET_MATCHES:
        print(
            f"Note: target was {TARGET_MATCHES} matches; "
            f"{unique_matches} complete matches were retained."
        )

    print("\nTarget distribution:")
    print(
        dataset["target"]
        .value_counts()
        .to_string()
    )

    print("\nRows by snapshot minute:")
    print(
        dataset["snapshot_minute"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if "competition" in dataset.columns:
        print("\nCompetition distribution:")
        print(
            dataset[
                ["match_id", "competition"]
            ]
            .drop_duplicates("match_id")
            ["competition"]
            .value_counts()
            .head(15)
            .to_string()
        )

    if failed_matches:
        print(
            f"\nWarning: {len(failed_matches)} "
            "matches could not be processed."
        )

        failed_file = DATA_DIR / "failed_match_downloads.csv"
        pd.DataFrame(failed_matches).to_csv(
            failed_file,
            index=False,
        )

        print(
            f"Failure log saved to: {failed_file}"
        )

    print(
        "\nNext step:"
    )
    print(
        "1. Run: python src\\validate_training_dataset.py"
    )
    print(
        "2. If validation passes, run: "
        "python src\\train_match_outcome_model.py"
    )
    print(
        "3. Then run: python src\\evaluate_match_outcome_model.py"
    )


if __name__ == "__main__":
    main()
