
"""
StatsBomb Open Data match catalogue for LiveMatch Intelligence.

Provides:
- available competitions/seasons
- available matches for a selected competition/season
- clean display labels for Streamlit selectors
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from statsbombpy import sb


def load_open_competitions() -> pd.DataFrame:
    competitions = sb.competitions().copy()

    if competitions.empty:
        return competitions

    wanted = [
        "competition_id",
        "season_id",
        "country_name",
        "competition_name",
        "season_name",
        "competition_gender",
        "competition_youth",
        "competition_international",
    ]

    cols = [
        c for c in wanted
        if c in competitions.columns
    ]

    competitions = competitions[cols].copy()

    competitions["competition_id"] = pd.to_numeric(
        competitions["competition_id"],
        errors="coerce",
    ).astype("Int64")

    competitions["season_id"] = pd.to_numeric(
        competitions["season_id"],
        errors="coerce",
    ).astype("Int64")

    competitions = competitions.dropna(
        subset=[
            "competition_id",
            "season_id",
            "competition_name",
            "season_name",
        ]
    )

    competitions["competition_label"] = (
        competitions["competition_name"].astype(str)
        + " — "
        + competitions["season_name"].astype(str)
    )

    competitions = competitions.sort_values(
        [
            "competition_name",
            "season_name",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)

    return competitions


def load_open_matches(
    competition_id: int,
    season_id: int,
) -> pd.DataFrame:
    matches = sb.matches(
        competition_id=int(competition_id),
        season_id=int(season_id),
    ).copy()

    if matches.empty:
        return matches

    wanted = [
        "match_id",
        "match_date",
        "kick_off",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "competition_stage",
        "stadium",
    ]

    cols = [
        c for c in wanted
        if c in matches.columns
    ]

    matches = matches[cols].copy()

    matches["match_id"] = pd.to_numeric(
        matches["match_id"],
        errors="coerce",
    ).astype("Int64")

    matches = matches.dropna(
        subset=[
            "match_id",
            "home_team",
            "away_team",
        ]
    )

    if "match_date" in matches.columns:
        matches["match_date"] = pd.to_datetime(
            matches["match_date"],
            errors="coerce",
        )

    def score_text(row):
        if (
            "home_score" in row.index
            and "away_score" in row.index
            and pd.notna(row["home_score"])
            and pd.notna(row["away_score"])
        ):
            return (
                f"{int(row['home_score'])}"
                f"–"
                f"{int(row['away_score'])}"
            )
        return "vs"

    def date_text(value):
        if pd.isna(value):
            return ""
        try:
            return pd.Timestamp(value).strftime("%d %b %Y")
        except Exception:
            return str(value)

    matches["match_label"] = matches.apply(
        lambda row: (
            f"{date_text(row.get('match_date'))} | "
            f"{row['home_team']} {score_text(row)} {row['away_team']}"
        ),
        axis=1,
    )

    if "match_date" in matches.columns:
        matches = matches.sort_values(
            "match_date",
            ascending=False,
        )

    return matches.reset_index(drop=True)


def find_match_row(
    matches: pd.DataFrame,
    match_id: int,
) -> Optional[pd.Series]:
    if matches.empty:
        return None

    rows = matches[
        matches["match_id"].astype(int)
        == int(match_id)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]
