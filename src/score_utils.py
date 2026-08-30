
"""
Score utilities for LiveMatch Intelligence.

Separates normal/extra-time goals from penalty-shootout kicks.
StatsBomb shootout events are normally period 5.
"""

from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd


def _normalise_name(value):
    if isinstance(value, dict):
        return str(value.get("name", ""))
    if pd.isna(value):
        return ""
    return str(value)


def match_score_breakdown(
    events: pd.DataFrame,
    team_1: str,
    team_2: str,
) -> Dict[str, object]:
    """
    Return match score excluding shootout kicks plus shootout score separately.

    Returns keys:
      team_1_goals
      team_2_goals
      has_shootout
      team_1_penalties
      team_2_penalties
      winner_on_penalties
    """
    if events is None or events.empty:
        return {
            "team_1_goals": 0,
            "team_2_goals": 0,
            "has_shootout": False,
            "team_1_penalties": 0,
            "team_2_penalties": 0,
            "winner_on_penalties": None,
        }

    types = (
        events.get(
            "type",
            pd.Series("", index=events.index),
        )
        .apply(_normalise_name)
    )

    teams = (
        events.get(
            "team",
            pd.Series("", index=events.index),
        )
        .apply(_normalise_name)
    )

    outcomes = (
        events.get(
            "shot_outcome",
            pd.Series("", index=events.index),
        )
        .apply(_normalise_name)
    )

    periods = pd.to_numeric(
        events.get(
            "period",
            pd.Series(0, index=events.index),
        ),
        errors="coerce",
    ).fillna(0)

    shot_goals = (
        (types == "Shot")
        & (outcomes == "Goal")
    )

    non_shootout = shot_goals & (periods != 5)
    shootout = shot_goals & (periods == 5)

    team_1_goals = int(
        (
            non_shootout
            & (teams == str(team_1))
        ).sum()
    )

    team_2_goals = int(
        (
            non_shootout
            & (teams == str(team_2))
        ).sum()
    )

    team_1_penalties = int(
        (
            shootout
            & (teams == str(team_1))
        ).sum()
    )

    team_2_penalties = int(
        (
            shootout
            & (teams == str(team_2))
        ).sum()
    )

    has_shootout = bool(
        (periods == 5).any()
    )

    winner_on_penalties = None

    if has_shootout:
        if team_1_penalties > team_2_penalties:
            winner_on_penalties = str(team_1)
        elif team_2_penalties > team_1_penalties:
            winner_on_penalties = str(team_2)

    return {
        "team_1_goals": team_1_goals,
        "team_2_goals": team_2_goals,
        "has_shootout": has_shootout,
        "team_1_penalties": team_1_penalties,
        "team_2_penalties": team_2_penalties,
        "winner_on_penalties": winner_on_penalties,
    }


def exclude_shootout_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return normal + extra-time events only.

    StatsBomb penalty-shootout events are normally period 5 and should not
    inflate match shots, xG, goals, player statistics or tactical indicators.
    """
    if (
        events is None
        or events.empty
        or "period" not in events.columns
    ):
        return events.copy()

    periods = pd.to_numeric(
        events["period"],
        errors="coerce",
    ).fillna(0)

    return events.loc[
        periods != 5
    ].copy()
