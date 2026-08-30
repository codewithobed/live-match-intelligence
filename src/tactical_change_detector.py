"""
Tactical Change Detector for LiveMatch Intelligence.

Purpose:
    Compare first-half and second-half event behaviour and identify
    meaningful tactical or positional shifts.

This module detects:
    - Team average action-position change
    - Pressure-intensity change
    - Shot/xG change
    - Recent attacking-activity change
    - Momentum-style activity change
    - Player average-position shifts

Important:
    These are data-derived tactical signals based on StatsBomb event
    locations and event counts. They are NOT definitive evidence of a
    manager's tactical instruction or true tracking-data formation change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.image_fetcher import get_team_badge


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TEAM_POSITION_SHIFT_THRESHOLD = 4.0
TEAM_PRESSURE_CHANGE_THRESHOLD = 0.20
TEAM_XG_CHANGE_THRESHOLD = 0.20
TEAM_SHOT_CHANGE_THRESHOLD = 2
TEAM_ATTACK_INDEX_CHANGE_THRESHOLD = 0.20
PLAYER_POSITION_SHIFT_THRESHOLD = 7.0
MIN_PLAYER_INVOLVEMENTS = 8


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------

def _normalise_name(value):
    if isinstance(value, dict):
        return str(value.get("name", ""))

    if pd.isna(value):
        return ""

    return str(value)


def _event_types(events: pd.DataFrame) -> pd.Series:
    if "type" not in events.columns:
        return pd.Series("", index=events.index)

    return events["type"].apply(_normalise_name)


def _event_teams(events: pd.DataFrame) -> pd.Series:
    if "team" not in events.columns:
        return pd.Series("", index=events.index)

    return events["team"].apply(_normalise_name)


def _event_players(events: pd.DataFrame) -> pd.Series:
    if "player" not in events.columns:
        return pd.Series("", index=events.index)

    return events["player"].apply(_normalise_name)


def _minutes(events: pd.DataFrame) -> pd.Series:
    if "minute" not in events.columns:
        return pd.Series(0.0, index=events.index)

    return pd.to_numeric(
        events["minute"],
        errors="coerce",
    ).fillna(0.0)


def _period_filter(
    events: pd.DataFrame,
    period_label: str,
) -> pd.DataFrame:
    """
    Return first-half or second-half events.

    Preference:
        use StatsBomb 'period' when present.
    Fallback:
        minute <=45 for first half, >45 for second half.
    """
    label = str(period_label).strip().lower()

    if label not in {
        "first half",
        "second half",
    }:
        raise ValueError(
            "period_label must be 'First Half' or 'Second Half'."
        )

    if "period" in events.columns:
        period_values = pd.to_numeric(
            events["period"],
            errors="coerce",
        )

        if label == "first half":
            return events.loc[
                period_values == 1
            ].copy()

        return events.loc[
            period_values == 2
        ].copy()

    minute_values = _minutes(events)

    if label == "first half":
        return events.loc[
            minute_values <= 45
        ].copy()

    return events.loc[
        minute_values > 45
    ].copy()


def _extract_xy(location):
    if isinstance(location, (list, tuple)) and len(location) >= 2:
        try:
            return float(location[0]), float(location[1])
        except (TypeError, ValueError):
            return np.nan, np.nan

    return np.nan, np.nan


def _safe_pct_change(
    old_value: float,
    new_value: float,
) -> float:
    """
    Relative percentage change.

    If old_value is zero:
      - returns 0 when both are zero
      - returns 1.0 when new value is positive
    """
    old = float(old_value)
    new = float(new_value)

    if abs(old) < 1e-9:
        if abs(new) < 1e-9:
            return 0.0
        return 1.0

    return (
        new - old
    ) / abs(old)


# ---------------------------------------------------------
# Team metrics
# ---------------------------------------------------------

def build_team_period_metrics(
    events: pd.DataFrame,
    team_name: str,
    period_label: str,
) -> Dict[str, float]:
    """
    Calculate one team's tactical/event metrics for a half.
    """
    period_events = _period_filter(
        events,
        period_label,
    )

    teams = _event_teams(
        period_events
    )

    types = _event_types(
        period_events
    )

    team_mask = (
        teams == team_name
    )

    team_events = period_events.loc[
        team_mask
    ].copy()

    team_types = types.loc[
        team_mask
    ]

    shots = int(
        (
            team_types == "Shot"
        ).sum()
    )

    pressures = int(
        (
            team_types == "Pressure"
        ).sum()
    )

    carries = int(
        (
            team_types == "Carry"
        ).sum()
    )

    recoveries = int(
        (
            team_types == "Ball Recovery"
        ).sum()
    )

    passes = int(
        (
            team_types == "Pass"
        ).sum()
    )

    xg = 0.0

    if (
        "shot_statsbomb_xg"
        in team_events.columns
    ):
        shot_mask = (
            _event_types(
                team_events
            )
            == "Shot"
        )

        xg = float(
            pd.to_numeric(
                team_events.loc[
                    shot_mask,
                    "shot_statsbomb_xg",
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )

    avg_x = np.nan
    avg_y = np.nan
    located_events = 0

    if (
        "location"
        in team_events.columns
    ):
        coords = team_events[
            "location"
        ].apply(
            _extract_xy
        )

        x_values = [
            xy[0]
            for xy in coords
            if not np.isnan(
                xy[0]
            )
        ]

        y_values = [
            xy[1]
            for xy in coords
            if not np.isnan(
                xy[1]
            )
        ]

        located_events = len(
            x_values
        )

        if x_values:
            avg_x = float(
                np.mean(
                    x_values
                )
            )

        if y_values:
            avg_y = float(
                np.mean(
                    y_values
                )
            )

    attack_index = (
        shots * 4.0
        + xg * 10.0
        + pressures * 0.35
        + carries * 0.08
        + recoveries * 0.20
        + passes * 0.03
    )

    return {
        "Team": team_name,
        "Period": period_label,

        "Average X":
            avg_x,

        "Average Y":
            avg_y,

        "Located Events":
            located_events,

        "Shots":
            shots,

        "xG":
            xg,

        "Pressures":
            pressures,

        "Carries":
            carries,

        "Recoveries":
            recoveries,

        "Passes":
            passes,

        "Attack Index":
            attack_index,
    }


# ---------------------------------------------------------
# Player average positions
# ---------------------------------------------------------

def build_player_period_positions(
    events: pd.DataFrame,
    team_name: str,
    period_label: str,
) -> pd.DataFrame:
    """
    Estimate player average action positions from event locations.
    """
    period_events = _period_filter(
        events,
        period_label,
    )

    if (
        "location"
        not in period_events.columns
        or "player"
        not in period_events.columns
    ):
        return pd.DataFrame(
            columns=[
                "Player",
                "Team",
                "Average X",
                "Average Y",
                "Involvements",
            ]
        )

    teams = _event_teams(
        period_events
    )

    players = _event_players(
        period_events
    )

    mask = (
        (teams == team_name)
        & (players != "")
    )

    team_events = period_events.loc[
        mask
    ].copy()

    if team_events.empty:
        return pd.DataFrame(
            columns=[
                "Player",
                "Team",
                "Average X",
                "Average Y",
                "Involvements",
            ]
        )

    team_events[
        "Player Name"
    ] = _event_players(
        team_events
    )

    xy = team_events[
        "location"
    ].apply(
        _extract_xy
    )

    team_events[
        "_x"
    ] = [
        point[0]
        for point in xy
    ]

    team_events[
        "_y"
    ] = [
        point[1]
        for point in xy
    ]

    team_events = team_events.dropna(
        subset=[
            "_x",
            "_y",
        ]
    )

    if team_events.empty:
        return pd.DataFrame(
            columns=[
                "Player",
                "Team",
                "Average X",
                "Average Y",
                "Involvements",
            ]
        )

    result = (
        team_events
        .groupby(
            "Player Name",
            as_index=False,
        )
        .agg(
            {
                "_x": "mean",
                "_y": "mean",
                "location": "count",
            }
        )
        .rename(
            columns={
                "Player Name":
                    "Player",
                "_x":
                    "Average X",
                "_y":
                    "Average Y",
                "location":
                    "Involvements",
            }
        )
    )

    result[
        "Team"
    ] = team_name

    return result[
        [
            "Player",
            "Team",
            "Average X",
            "Average Y",
            "Involvements",
        ]
    ].copy()


# ---------------------------------------------------------
# Tactical comparisons
# ---------------------------------------------------------

def compare_team_halves(
    events: pd.DataFrame,
    team_name: str,
) -> Dict[str, float]:
    first = build_team_period_metrics(
        events,
        team_name,
        "First Half",
    )

    second = build_team_period_metrics(
        events,
        team_name,
        "Second Half",
    )

    avg_x_change = np.nan

    if (
        not np.isnan(
            first["Average X"]
        )
        and not np.isnan(
            second["Average X"]
        )
    ):
        avg_x_change = (
            second["Average X"]
            - first["Average X"]
        )

    return {
        "Team":
            team_name,

        "First Half Avg X":
            first["Average X"],

        "Second Half Avg X":
            second["Average X"],

        "Average X Change":
            avg_x_change,

        "First Half Pressures":
            first["Pressures"],

        "Second Half Pressures":
            second["Pressures"],

        "Pressure Change %":
            _safe_pct_change(
                first["Pressures"],
                second["Pressures"],
            ),

        "First Half Shots":
            first["Shots"],

        "Second Half Shots":
            second["Shots"],

        "Shot Change":
            (
                second["Shots"]
                - first["Shots"]
            ),

        "First Half xG":
            first["xG"],

        "Second Half xG":
            second["xG"],

        "xG Change":
            (
                second["xG"]
                - first["xG"]
            ),

        "First Half Attack Index":
            first["Attack Index"],

        "Second Half Attack Index":
            second["Attack Index"],

        "Attack Index Change %":
            _safe_pct_change(
                first["Attack Index"],
                second["Attack Index"],
            ),
    }


def compare_player_halves(
    events: pd.DataFrame,
    team_name: str,
) -> pd.DataFrame:
    first = build_player_period_positions(
        events,
        team_name,
        "First Half",
    )

    second = build_player_period_positions(
        events,
        team_name,
        "Second Half",
    )

    if (
        first.empty
        or second.empty
    ):
        return pd.DataFrame(
            columns=[
                "Player",
                "Team",
                "First Half X",
                "Second Half X",
                "X Change",
                "First Half Y",
                "Second Half Y",
                "Y Change",
                "First Half Involvements",
                "Second Half Involvements",
            ]
        )

    merged = first.merge(
        second,
        on=[
            "Player",
            "Team",
        ],
        suffixes=(
            " First",
            " Second",
        ),
    )

    merged[
        "X Change"
    ] = (
        merged[
            "Average X Second"
        ]
        - merged[
            "Average X First"
        ]
    )

    merged[
        "Y Change"
    ] = (
        merged[
            "Average Y Second"
        ]
        - merged[
            "Average Y First"
        ]
    )

    return pd.DataFrame(
        {
            "Player":
                merged["Player"],

            "Team":
                merged["Team"],

            "First Half X":
                merged[
                    "Average X First"
                ],

            "Second Half X":
                merged[
                    "Average X Second"
                ],

            "X Change":
                merged[
                    "X Change"
                ],

            "First Half Y":
                merged[
                    "Average Y First"
                ],

            "Second Half Y":
                merged[
                    "Average Y Second"
                ],

            "Y Change":
                merged[
                    "Y Change"
                ],

            "First Half Involvements":
                merged[
                    "Involvements First"
                ],

            "Second Half Involvements":
                merged[
                    "Involvements Second"
                ],
        }
    )


# ---------------------------------------------------------
# Alert generation
# ---------------------------------------------------------

def build_team_tactical_alerts(
    team_comparison: Dict[str, float],
) -> List[Dict[str, str]]:
    alerts = []

    team = team_comparison[
        "Team"
    ]

    avg_x_change = team_comparison[
        "Average X Change"
    ]

    if (
        not np.isnan(
            avg_x_change
        )
        and abs(
            avg_x_change
        )
        >= TEAM_POSITION_SHIFT_THRESHOLD
    ):
        if avg_x_change > 0:
            message = (
                f"{team} are operating higher up the pitch in the "
                f"second half (+{avg_x_change:.1f} average action-position units)."
            )
        else:
            message = (
                f"{team} are operating deeper in the second half "
                f"({avg_x_change:.1f} average action-position units)."
            )

        alerts.append(
            {
                "Type":
                    "Team Shape",

                "Team":
                    team,

                "Message":
                    message,
            }
        )

    pressure_change = team_comparison[
        "Pressure Change %"
    ]

    if abs(
        pressure_change
    ) >= TEAM_PRESSURE_CHANGE_THRESHOLD:
        if pressure_change > 0:
            message = (
                f"{team}'s pressure-event volume increased by "
                f"{pressure_change * 100:.0f}% in the second half."
            )
        else:
            message = (
                f"{team}'s pressure-event volume decreased by "
                f"{abs(pressure_change) * 100:.0f}% in the second half."
            )

        alerts.append(
            {
                "Type":
                    "Pressure Shift",

                "Team":
                    team,

                "Message":
                    message,
            }
        )

    shot_change = team_comparison[
        "Shot Change"
    ]

    xg_change = team_comparison[
        "xG Change"
    ]

    if (
        abs(
            shot_change
        )
        >= TEAM_SHOT_CHANGE_THRESHOLD
        or abs(
            xg_change
        )
        >= TEAM_XG_CHANGE_THRESHOLD
    ):
        if (
            shot_change > 0
            or xg_change > 0
        ):
            message = (
                f"{team}'s attacking output increased after half-time: "
                f"shot change {shot_change:+d}, xG change {xg_change:+.2f}."
            )
        else:
            message = (
                f"{team}'s attacking output reduced after half-time: "
                f"shot change {shot_change:+d}, xG change {xg_change:+.2f}."
            )

        alerts.append(
            {
                "Type":
                    "Attacking Shift",

                "Team":
                    team,

                "Message":
                    message,
            }
        )

    attack_change = team_comparison[
        "Attack Index Change %"
    ]

    if abs(
        attack_change
    ) >= TEAM_ATTACK_INDEX_CHANGE_THRESHOLD:
        if attack_change > 0:
            message = (
                f"{team}'s overall event-based attacking activity increased "
                f"by approximately {attack_change * 100:.0f}% in the second half."
            )
        else:
            message = (
                f"{team}'s overall event-based attacking activity decreased "
                f"by approximately {abs(attack_change) * 100:.0f}% in the second half."
            )

        alerts.append(
            {
                "Type":
                    "Activity Shift",

                "Team":
                    team,

                "Message":
                    message,
            }
        )

    return alerts


def build_player_tactical_alerts(
    player_comparison: pd.DataFrame,
) -> List[Dict[str, str]]:
    alerts = []

    if player_comparison.empty:
        return alerts

    for _, row in player_comparison.iterrows():

        first_involvements = int(
            row[
                "First Half Involvements"
            ]
        )

        second_involvements = int(
            row[
                "Second Half Involvements"
            ]
        )

        if (
            first_involvements
            < MIN_PLAYER_INVOLVEMENTS
            or second_involvements
            < MIN_PLAYER_INVOLVEMENTS
        ):
            continue

        x_change = float(
            row[
                "X Change"
            ]
        )

        if abs(
            x_change
        ) < PLAYER_POSITION_SHIFT_THRESHOLD:
            continue

        player = row[
            "Player"
        ]

        team = row[
            "Team"
        ]

        if x_change > 0:
            message = (
                f"{player} ({team}) is appearing substantially higher "
                f"in second-half event locations (+{x_change:.1f} X units)."
            )
        else:
            message = (
                f"{player} ({team}) is appearing substantially deeper "
                f"in second-half event locations ({x_change:.1f} X units)."
            )

        alerts.append(
            {
                "Type":
                    "Player Position Shift",

                "Team":
                    team,

                "Player":
                    player,

                "Message":
                    message,
            }
        )

    return alerts



# ---------------------------------------------------------
# Tactical-analysis visualisation
# ---------------------------------------------------------

def _safe_metric(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if np.isnan(number) or np.isinf(number):
        return default

    return number


def _resolve_team_badge_path(team_name: str):
    try:
        badge = get_team_badge(
            str(team_name).strip()
        )

        if (
            badge is not None
            and Path(badge).exists()
        ):
            return Path(badge)

    except Exception:
        pass

    return None


def _add_team_badge(
    fig,
    team_name: str,
    left: float,
    bottom: float,
    width: float = 0.055,
    height: float = 0.055,
) -> bool:
    badge_path = _resolve_team_badge_path(
        team_name
    )

    if badge_path is None:
        return False

    try:
        from PIL import Image as PILImage

        image = PILImage.open(
            badge_path
        ).convert("RGBA")

        badge_ax = fig.add_axes(
            [
                left,
                bottom,
                width,
                height,
            ]
        )

        badge_ax.imshow(
            image
        )
        badge_ax.axis(
            "off"
        )

        return True

    except Exception:
        return False


def _metric_direction_label(
    value: float,
    *,
    percent: bool = False,
) -> str:
    value = _safe_metric(
        value
    )

    if abs(value) < 1e-9:
        return "No material change"

    arrow = "↑" if value > 0 else "↓"

    if percent:
        return (
            f"{arrow} {abs(value) * 100:.0f}%"
        )

    return (
        f"{arrow} {abs(value):.2f}"
    )


def _top_player_shifts(
    player_df: pd.DataFrame,
    team_name: str,
    n: int = 3,
) -> pd.DataFrame:
    if (
        player_df is None
        or player_df.empty
        or "Team" not in player_df.columns
    ):
        return pd.DataFrame()

    rows = player_df.loc[
        player_df["Team"] == team_name
    ].copy()

    if rows.empty:
        return rows

    rows["Absolute X Change"] = pd.to_numeric(
        rows.get(
            "X Change",
            pd.Series(
                0.0,
                index=rows.index,
            ),
        ),
        errors="coerce",
    ).abs()

    return (
        rows
        .sort_values(
            "Absolute X Change",
            ascending=False,
        )
        .head(n)
    )


def build_tactical_analysis_figure(
    events: pd.DataFrame,
    team_names: Optional[List[str]] = None,
):
    """
    Professional event-derived tactical comparison graphic.
    Designed to work dynamically with any two teams in the selected match.
    """
    result = detect_tactical_changes(events, team_names=team_names)

    team_df = result.get("team_comparisons", pd.DataFrame()).copy()
    player_df = result.get("player_comparisons", pd.DataFrame()).copy()
    alerts_df = result.get("alerts", pd.DataFrame()).copy()

    if team_df.empty:
        fig, ax = plt.subplots(figsize=(15.5, 9.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No tactical comparison data available",
                ha="center", va="center", fontsize=15, weight="bold")
        return fig

    if team_names is None:
        team_names = team_df["Team"].dropna().astype(str).tolist()

    team_names = list(team_names or [])[:2]
    if len(team_names) < 2:
        fig, ax = plt.subplots(figsize=(15.5, 9.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "Two teams are required for tactical comparison",
                ha="center", va="center", fontsize=15, weight="bold")
        return fig

    team_1, team_2 = team_names

    def team_row(team):
        rows = team_df.loc[team_df["Team"] == team]
        return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)

    r1, r2 = team_row(team_1), team_row(team_2)

    fig = plt.figure(figsize=(16, 10.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header
    ax.text(0.5, 0.965, "TACTICAL ANALYSIS",
            fontsize=22, weight="bold", ha="center", va="top")
    ax.text(0.5, 0.925, "FIRST HALF  →  SECOND HALF",
            fontsize=10.5, weight="bold", ha="center", va="center")

    _add_team_badge(fig, team_1, 0.065, 0.875, 0.060, 0.060)
    _add_team_badge(fig, team_2, 0.875, 0.875, 0.060, 0.060)

    ax.text(0.135, 0.905, team_1, fontsize=15, weight="bold",
            ha="left", va="center")
    ax.text(0.865, 0.905, team_2, fontsize=15, weight="bold",
            ha="right", va="center")

    # Summary cards
    metrics = [
        ("AVG ACTION X", "First Half Avg X", "Second Half Avg X",
         "Average X Change", False, "territorial position"),
        ("PRESSURES", "First Half Pressures", "Second Half Pressures",
         "Pressure Change %", True, "pressure volume"),
        ("SHOTS", "First Half Shots", "Second Half Shots",
         "Shot Change", False, "shot volume"),
        ("xG", "First Half xG", "Second Half xG",
         "xG Change", False, "chance quality"),
        ("ATTACK INDEX", "First Half Attack Index", "Second Half Attack Index",
         "Attack Index Change %", True, "attacking activity"),
    ]

    card_lefts = [0.055, 0.235, 0.415, 0.595, 0.775]
    card_w = 0.165

    for i, (label, first_col, second_col, change_col, pct, _) in enumerate(metrics):
        left = card_lefts[i]
        box = FancyBboxPatch(
            (left, 0.735), card_w, 0.105,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.0, fill=False
        )
        ax.add_patch(box)

        v11 = _safe_metric(r1.get(first_col, 0.0))
        v12 = _safe_metric(r1.get(second_col, 0.0))
        c1 = _safe_metric(r1.get(change_col, 0.0))
        v21 = _safe_metric(r2.get(first_col, 0.0))
        v22 = _safe_metric(r2.get(second_col, 0.0))
        c2 = _safe_metric(r2.get(change_col, 0.0))

        ax.text(left + card_w/2, 0.817, label, fontsize=8.5,
                weight="bold", ha="center")

        fmt = lambda a, b: (
            f"{int(round(a))} → {int(round(b))}"
            if label in {"PRESSURES", "SHOTS"}
            else f"{a:.2f} → {b:.2f}"
        )

        ax.text(left + 0.012, 0.782, fmt(v11, v12),
                fontsize=9.2, weight="bold", ha="left")
        ax.text(left + card_w - 0.012, 0.782, fmt(v21, v22),
                fontsize=9.2, weight="bold", ha="right")

        ax.text(left + 0.012, 0.754, _metric_direction_label(c1, percent=pct),
                fontsize=7.5, ha="left")
        ax.text(left + card_w - 0.012, 0.754,
                _metric_direction_label(c2, percent=pct),
                fontsize=7.5, ha="right")

    # Tactical momentum graphic
    ax.text(0.055, 0.690, "TACTICAL MOMENTUM",
            fontsize=12, weight="bold", ha="left")

    def momentum_values(row):
        return [
            _safe_metric(row.get("Average X Change", 0)),
            _safe_metric(row.get("Pressure Change %", 0)) * 10,
            _safe_metric(row.get("Shot Change", 0)),
            _safe_metric(row.get("xG Change", 0)) * 5,
            _safe_metric(row.get("Attack Index Change %", 0)) * 10,
        ]

    labels = ["Position", "Pressure", "Shots", "xG", "Attack"]
    m1, m2 = momentum_values(r1), momentum_values(r2)
    max_abs = max([abs(x) for x in m1 + m2] + [1])

    chart = fig.add_axes([0.075, 0.515, 0.85, 0.145])
    x = np.arange(len(labels))
    width = 0.32
    chart.axhline(0, linewidth=0.8)
    chart.bar(x - width/2, m1, width, label=team_1, alpha=0.75)
    chart.bar(x + width/2, m2, width, label=team_2, alpha=0.75)
    chart.set_xticks(x)
    chart.set_xticklabels(labels, fontsize=8)
    chart.set_ylabel("Relative change signal", fontsize=8)
    chart.set_ylim(-max_abs * 1.35, max_abs * 1.35)
    chart.tick_params(axis="y", labelsize=7)
    chart.spines["top"].set_visible(False)
    chart.spines["right"].set_visible(False)
    chart.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)

    # Player shifts
    ax.text(0.055, 0.475, "KEY PLAYER POSITION SHIFTS",
            fontsize=12, weight="bold", ha="left")

    for team, left in ((team_1, 0.055), (team_2, 0.525)):
        shifts = _top_player_shifts(player_df, team, n=3)
        card = FancyBboxPatch(
            (left, 0.285), 0.420, 0.155,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.0, fill=False
        )
        ax.add_patch(card)
        ax.text(left + 0.018, 0.415, team.upper(),
                fontsize=9.5, weight="bold", ha="left")

        if shifts.empty:
            ax.text(left + 0.018, 0.375, "No qualifying positional shifts.",
                    fontsize=8.2, ha="left")
        else:
            y = 0.372
            for _, row in shifts.iterrows():
                name = _normalise_name(row.get("Player", ""))
                change = _safe_metric(row.get("X Change", 0.0))
                direction = "higher" if change > 0 else "deeper"
                arrow = "↑" if change > 0 else "↓"
                ax.text(left + 0.018, y,
                        f"{arrow} {name}: {abs(change):.1f} X units {direction}",
                        fontsize=8.0, ha="left", va="top")
                y -= 0.034

    # Analyst interpretation / signals
    ax.text(0.055, 0.245, "ANALYST INTERPRETATION",
            fontsize=12, weight="bold", ha="left")

    interpretation_box = FancyBboxPatch(
        (0.055, 0.095), 0.890, 0.120,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=1.0, fill=False
    )
    ax.add_patch(interpretation_box)

    messages = []
    if alerts_df is not None and not alerts_df.empty:
        type_series = alerts_df.get(
            "Type", pd.Series("", index=alerts_df.index)
        ).astype(str)
        team_alerts = alerts_df.loc[type_series.ne("Player Position Shift")]
        messages = team_alerts.get(
            "Message", pd.Series(dtype=str)
        ).dropna().astype(str).head(3).tolist()

    if not messages:
        # Safe automatic fallback from the actual calculated team changes.
        for team, row in ((team_1, r1), (team_2, r2)):
            x_change = _safe_metric(row.get("Average X Change", 0))
            pressure_change = _safe_metric(row.get("Pressure Change %", 0))
            messages.append(
                f"{team}: average event position changed by {x_change:+.1f} X units; "
                f"pressure volume changed by {pressure_change:+.0%}."
            )

    y = 0.185
    for message in messages[:3]:
        ax.text(0.075, y, f"• {message}",
                fontsize=8.1, ha="left", va="top")
        y -= 0.036

    # Dedicated footer so nothing collides.
    ax.plot([0.055, 0.945], [0.060, 0.060], linewidth=0.8)
    ax.text(
        0.5, 0.037,
        "Event-derived tactical intelligence | First-half vs second-half comparison | "
        "Signals describe recorded event behaviour, not confirmed coaching instructions.",
        fontsize=7.4, ha="center", va="center"
    )
    ax.text(
        0.5, 0.017,
        "Average X uses StatsBomb pitch coordinates; player shifts are based on changes in average event location.",
        fontsize=7.2, ha="center", va="center"
    )

    return fig

# ---------------------------------------------------------
# Public interface
# ---------------------------------------------------------

def detect_tactical_changes(
    events: pd.DataFrame,
    team_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Run the full tactical-change detection pipeline.

    Returns:
        {
            "team_comparisons": DataFrame,
            "player_comparisons": DataFrame,
            "alerts": DataFrame,
        }
    """

    if events is None or events.empty:
        return {
            "team_comparisons":
                pd.DataFrame(),

            "player_comparisons":
                pd.DataFrame(),

            "alerts":
                pd.DataFrame(),
        }

    if team_names is None:
        team_values = (
            _event_teams(
                events
            )
            .replace(
                "",
                np.nan,
            )
            .dropna()
            .unique()
            .tolist()
        )

        team_names = [
            str(
                team
            )
            for team in team_values
        ]

    team_rows = []
    player_frames = []
    alerts = []

    for team_name in team_names:
        team_comparison = compare_team_halves(
            events,
            team_name,
        )

        team_rows.append(
            team_comparison
        )

        team_alerts = build_team_tactical_alerts(
            team_comparison
        )

        alerts.extend(
            team_alerts
        )

        player_comparison = compare_player_halves(
            events,
            team_name,
        )

        if not player_comparison.empty:
            player_frames.append(
                player_comparison
            )

            player_alerts = build_player_tactical_alerts(
                player_comparison
            )

            alerts.extend(
                player_alerts
            )

    team_df = pd.DataFrame(
        team_rows
    )

    if player_frames:
        player_df = pd.concat(
            player_frames,
            ignore_index=True,
        )
    else:
        player_df = pd.DataFrame()

    alerts_df = pd.DataFrame(
        alerts
    )

    return {
        "team_comparisons":
            team_df,

        "player_comparisons":
            player_df,

        "alerts":
            alerts_df,
    }
