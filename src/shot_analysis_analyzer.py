"""
Shot Analysis Intelligence for LiveMatch Intelligence.

Purpose:
    Derive event-based shooting and chance-quality indicators from StatsBomb
    match events in a reusable module for the dashboard and focused reports.

Important:
    - Full Match excludes StatsBomb period 5 penalty-shootout events.
    - xG uses StatsBomb's shot_statsbomb_xg field when available.
    - Shot locations use StatsBomb's 120 x 80 coordinate system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.image_fetcher import get_team_badge


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _normalise_name(value) -> str:
    if isinstance(value, dict):
        return str(value.get("name", ""))

    if pd.isna(value):
        return ""

    return str(value)


def _normalise_series(series: pd.Series) -> pd.Series:
    return series.apply(_normalise_name)


def _extract_xy(location):
    if isinstance(location, (list, tuple)) and len(location) >= 2:
        try:
            return float(location[0]), float(location[1])
        except (TypeError, ValueError):
            return np.nan, np.nan

    return np.nan, np.nan


def _period_filter(
    events: pd.DataFrame,
    period_label: str,
) -> pd.DataFrame:
    label = str(period_label).strip().lower()

    if label == "full match":
        # Exclude penalty-shootout actions (StatsBomb period 5).
        if "period" in events.columns:
            periods = pd.to_numeric(
                events["period"],
                errors="coerce",
            )

            return events.loc[
                periods.ne(5)
                | periods.isna()
            ].copy()

        return events.copy()

    if label not in {
        "first half",
        "second half",
    }:
        raise ValueError(
            "period_label must be Full Match, First Half, or Second Half."
        )

    if "period" in events.columns:
        periods = pd.to_numeric(
            events["period"],
            errors="coerce",
        )

        if label == "first half":
            return events.loc[
                periods == 1
            ].copy()

        return events.loc[
            periods == 2
        ].copy()

    minutes = pd.to_numeric(
        events.get(
            "minute",
            pd.Series(0, index=events.index),
        ),
        errors="coerce",
    ).fillna(0)

    if label == "first half":
        return events.loc[
            minutes <= 45
        ].copy()

    return events.loc[
        minutes > 45
    ].copy()


# ---------------------------------------------------------
# Shot extraction
# ---------------------------------------------------------

def extract_shots(
    events: pd.DataFrame,
    team_name: Optional[str] = None,
    period_label: str = "Full Match",
) -> pd.DataFrame:
    """
    Return shot events with standardised fields.

    Returned columns include:
        X, Y, Player, Team, Minute, xG, Outcome, Is Goal, On Target
    """
    period_events = _period_filter(
        events,
        period_label,
    )

    if (
        "type" not in period_events.columns
        or "team" not in period_events.columns
    ):
        return pd.DataFrame()

    event_types = _normalise_series(
        period_events["type"]
    )

    teams = _normalise_series(
        period_events["team"]
    )

    mask = event_types.eq("Shot")

    if team_name:
        mask &= teams.eq(str(team_name))

    shots = period_events.loc[
        mask
    ].copy()

    if shots.empty:
        return pd.DataFrame(
            columns=[
                "X",
                "Y",
                "Player",
                "Team",
                "Minute",
                "xG",
                "Outcome",
                "Is Goal",
                "On Target",
            ]
        )

    coords = shots.get(
        "location",
        pd.Series(index=shots.index, dtype=object),
    ).apply(_extract_xy)

    shots["X"] = [
        point[0]
        for point in coords
    ]

    shots["Y"] = [
        point[1]
        for point in coords
    ]

    shots["Player"] = (
        shots.get(
            "player",
            pd.Series("", index=shots.index),
        )
        .apply(_normalise_name)
    )

    shots["Team"] = (
        shots.get(
            "team",
            pd.Series("", index=shots.index),
        )
        .apply(_normalise_name)
    )

    shots["Minute"] = pd.to_numeric(
        shots.get(
            "minute",
            pd.Series(0, index=shots.index),
        ),
        errors="coerce",
    ).fillna(0).astype(int)

    shots["xG"] = pd.to_numeric(
        shots.get(
            "shot_statsbomb_xg",
            pd.Series(0.0, index=shots.index),
        ),
        errors="coerce",
    ).fillna(0.0)

    shots["Outcome"] = (
        shots.get(
            "shot_outcome",
            pd.Series("", index=shots.index),
        )
        .apply(_normalise_name)
    )

    shots["Is Goal"] = (
        shots["Outcome"]
        .str.casefold()
        .eq("goal")
    )

    # StatsBomb shot outcomes generally treat Goal and Saved as on-target.
    # Saved To Post is included when present.
    shots["On Target"] = (
        shots["Outcome"]
        .str.casefold()
        .isin(
            {
                "goal",
                "saved",
                "saved to post",
            }
        )
    )

    return shots[
        [
            "X",
            "Y",
            "Player",
            "Team",
            "Minute",
            "xG",
            "Outcome",
            "Is Goal",
            "On Target",
        ]
    ].copy()


# ---------------------------------------------------------
# Team and player summaries
# ---------------------------------------------------------

def build_team_shot_summary(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> Dict[str, object]:
    shots = extract_shots(
        events,
        team_name,
        period_label,
    )

    if shots.empty:
        return {
            "Team": team_name,
            "Period": period_label,
            "Shots": 0,
            "Shots on Target": 0,
            "Goals": 0,
            "xG": 0.0,
            "Average xG / Shot": 0.0,
            "High Quality Chances": 0,
            "Top Shooter": None,
            "Top xG Player": None,
        }

    player_counts = (
        shots.loc[
            shots["Player"] != "",
            "Player",
        ]
        .value_counts()
    )

    player_xg = (
        shots.loc[
            shots["Player"] != ""
        ]
        .groupby("Player")["xG"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    return {
        "Team": team_name,
        "Period": period_label,
        "Shots": int(len(shots)),
        "Shots on Target": int(shots["On Target"].sum()),
        "Goals": int(shots["Is Goal"].sum()),
        "xG": float(shots["xG"].sum()),
        "Average xG / Shot": float(shots["xG"].mean()),
        # Transparent convenience threshold for report interpretation only.
        "High Quality Chances": int((shots["xG"] >= 0.20).sum()),
        "Top Shooter": (
            str(player_counts.index[0])
            if not player_counts.empty
            else None
        ),
        "Top xG Player": (
            str(player_xg.index[0])
            if not player_xg.empty
            else None
        ),
    }


def build_player_shot_summary(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> pd.DataFrame:
    shots = extract_shots(
        events,
        team_name,
        period_label,
    )

    if shots.empty:
        return pd.DataFrame(
            columns=[
                "Player",
                "Team",
                "Shots",
                "Shots on Target",
                "Goals",
                "xG",
                "Average xG / Shot",
            ]
        )

    valid = shots.loc[
        shots["Player"] != ""
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    rows = []

    for player, player_shots in valid.groupby("Player"):
        rows.append(
            {
                "Player": player,
                "Team": team_name,
                "Shots": int(len(player_shots)),
                "Shots on Target": int(
                    player_shots["On Target"].sum()
                ),
                "Goals": int(
                    player_shots["Is Goal"].sum()
                ),
                "xG": float(
                    player_shots["xG"].sum()
                ),
                "Average xG / Shot": float(
                    player_shots["xG"].mean()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "xG",
                "Shots",
                "Goals",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# Visualisation
# ---------------------------------------------------------

def _resolve_team_badge_path(team_name: str):
    try:
        badge = get_team_badge(str(team_name).strip())

        if (
            badge is not None
            and Path(badge).exists()
        ):
            return Path(badge)

    except Exception:
        pass

    return None


def _add_team_badge_to_figure(
    fig,
    team_name: str,
    left: float,
    bottom: float,
    width: float = 0.050,
    height: float = 0.050,
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

        badge_ax.imshow(image)
        badge_ax.axis("off")
        return True

    except Exception:
        return False


def _draw_attacking_half_pitch(ax) -> None:
    ax.set_xlim(60, 120)
    ax.set_ylim(80, 0)

    ax.plot(
        [60, 120, 120, 60, 60],
        [0, 0, 80, 80, 0],
        linewidth=1.2,
    )

    ax.plot(
        [102, 120, 120, 102],
        [18, 18, 62, 62],
        linewidth=0.9,
    )

    ax.plot(
        [114, 120, 120, 114],
        [30, 30, 50, 50],
        linewidth=0.9,
    )

    ax.scatter(
        [108],
        [40],
        s=9,
        zorder=5,
    )

    # Portion of centre circle visible in attacking half.
    theta = np.linspace(
        -np.pi / 2,
        np.pi / 2,
        100,
    )
    ax.plot(
        60 + 10 * np.cos(theta),
        40 + 10 * np.sin(theta),
        linewidth=0.9,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_team_shot_panel(
    ax,
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> None:
    shots = extract_shots(
        events,
        team_name,
        period_label,
    )

    summary = build_team_shot_summary(
        events,
        team_name,
        period_label,
    )

    _draw_attacking_half_pitch(ax)

    if not shots.empty:
        non_goals = shots.loc[
            ~shots["Is Goal"]
        ]

        goals = shots.loc[
            shots["Is Goal"]
        ]

        if not non_goals.empty:
            ax.scatter(
                non_goals["X"],
                non_goals["Y"],
                s=38 + non_goals["xG"] * 520,
                alpha=0.52,
                marker="o",
                linewidth=0.7,
                edgecolors="black",
                label="Shot",
                zorder=4,
            )

        if not goals.empty:
            ax.scatter(
                goals["X"],
                goals["Y"],
                s=90 + goals["xG"] * 560,
                alpha=0.95,
                marker="*",
                linewidth=0.8,
                edgecolors="black",
                label="Goal",
                zorder=6,
            )

    # Small xG-size key for rapid interpretation.
    legend_y = 73.5
    legend_xs = [66.0, 72.0, 79.0]
    legend_xgs = [0.10, 0.30, 0.50]

    for x_pos, xg_value in zip(
        legend_xs,
        legend_xgs,
    ):
        ax.scatter(
            [x_pos],
            [legend_y],
            s=38 + xg_value * 520,
            alpha=0.45,
            marker="o",
            linewidth=0.7,
            edgecolors="black",
            zorder=3,
        )

        ax.text(
            x_pos,
            legend_y - 4.4,
            f"{xg_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
        )

    ax.text(
        72.5,
        legend_y + 4.4,
        "xG size",
        ha="center",
        va="top",
        fontsize=7.0,
        weight="bold",
    )

    ax.set_title(
        f"{team_name} — Shot Map",
        fontsize=14,
        weight="bold",
        pad=14,
    )

    ax.text(
        0.5,
        1.012,
        (
            f"{period_label} | "
            f"{summary['Shots']} shots | "
            f"{summary['Shots on Target']} on target | "
            f"{summary['Goals']} goals | "
            f"{summary['xG']:.2f} xG"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.7,
    )

    if not shots.empty:
        ax.legend(
            loc="upper left",
            fontsize=7.5,
            frameon=False,
        )


def build_shot_analysis_figure(
    events: pd.DataFrame,
    team_names: Optional[List[str]] = None,
    period_label: str = "Full Match",
):
    """
    Build a professional two-team shot analysis figure.

    Marker size represents StatsBomb xG.
    Star markers identify goals.
    """
    if events is None or events.empty:
        fig, ax = plt.subplots(
            figsize=(11.5, 6.8)
        )
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No shot data available",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
        )
        return fig

    if team_names is None:
        if "team" not in events.columns:
            team_names = []
        else:
            team_names = (
                _normalise_series(
                    events["team"]
                )
                .replace("", np.nan)
                .dropna()
                .unique()
                .tolist()
            )

    team_names = list(
        team_names or []
    )[:2]

    if len(team_names) < 2:
        fig, ax = plt.subplots(
            figsize=(11.5, 6.8)
        )
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Two teams are required for shot comparison",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
        )
        return fig

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.7),
    )

    for ax, team_name in zip(
        axes,
        team_names,
    ):
        _draw_team_shot_panel(
            ax,
            events,
            team_name,
            period_label,
        )

    s1 = build_team_shot_summary(
        events,
        team_names[0],
        period_label,
    )

    s2 = build_team_shot_summary(
        events,
        team_names[1],
        period_label,
    )

    fig.suptitle(
        "Shot Analysis",
        fontsize=19,
        weight="bold",
        y=0.985,
    )

    _add_team_badge_to_figure(
        fig,
        team_names[0],
        0.145,
        0.842,
        0.050,
        0.050,
    )

    _add_team_badge_to_figure(
        fig,
        team_names[1],
        0.645,
        0.842,
        0.050,
        0.050,
    )

    fig.text(
        0.5,
        0.060,
        (
            f"{team_names[0]}: {s1['Shots']} shots | "
            f"{s1['Shots on Target']} on target | "
            f"{s1['xG']:.2f} xG | "
            f"Avg xG/shot {s1['Average xG / Shot']:.3f}"
            "     ||     "
            f"{team_names[1]}: {s2['Shots']} shots | "
            f"{s2['Shots on Target']} on target | "
            f"{s2['xG']:.2f} xG | "
            f"Avg xG/shot {s2['Average xG / Shot']:.3f}"
        ),
        ha="center",
        va="center",
        fontsize=9.0,
    )

    fig.text(
        0.5,
        0.030,
        (
            "Marker size and xG-size key represent StatsBomb xG | "
            "Star = goal | Full Match excludes penalty-shootout period 5"
        ),
        ha="center",
        va="center",
        fontsize=8.2,
    )

    fig.text(
        0.5,
        0.010,
        (
            "Shot locations use the StatsBomb 120 × 80 coordinate system; "
            "xG represents estimated chance quality, not goal certainty."
        ),
        ha="center",
        va="bottom",
        fontsize=7.8,
    )

    fig.tight_layout(
        rect=[
            0.02,
            0.105,
            0.98,
            0.94,
        ]
    )

    return fig


# ---------------------------------------------------------
# Public interface
# ---------------------------------------------------------

def analyze_shots(
    events: pd.DataFrame,
    team_names: Optional[List[str]] = None,
    period_label: str = "Full Match",
) -> Dict[str, object]:
    if events is None or events.empty:
        return {
            "summaries": pd.DataFrame(),
            "players": pd.DataFrame(),
            "shots": pd.DataFrame(),
        }

    if team_names is None:
        if "team" not in events.columns:
            team_names = []
        else:
            team_names = (
                _normalise_series(
                    events["team"]
                )
                .replace("", np.nan)
                .dropna()
                .unique()
                .tolist()
            )

    summary_rows = []
    player_frames = []
    shot_frames = []

    for team_name in team_names:
        summary_rows.append(
            build_team_shot_summary(
                events,
                team_name,
                period_label,
            )
        )

        player_summary = (
            build_player_shot_summary(
                events,
                team_name,
                period_label,
            )
        )

        if not player_summary.empty:
            player_frames.append(
                player_summary
            )

        team_shots = extract_shots(
            events,
            team_name,
            period_label,
        )

        if not team_shots.empty:
            shot_frames.append(
                team_shots
            )

    return {
        "summaries": pd.DataFrame(
            summary_rows
        ),
        "players": (
            pd.concat(
                player_frames,
                ignore_index=True,
            )
            if player_frames
            else pd.DataFrame()
        ),
        "shots": (
            pd.concat(
                shot_frames,
                ignore_index=True,
            )
            if shot_frames
            else pd.DataFrame()
        ),
    }
