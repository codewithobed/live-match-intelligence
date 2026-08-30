"""
Possession & Territory Intelligence for LiveMatch Intelligence.

Purpose:
    Derive event-based territorial and possession-style indicators from
    StatsBomb match events.

This module provides:
    - Event share by team
    - Activity by defensive / middle / attacking thirds
    - Attacking-third presence
    - Average event position
    - Progressive-entry counts
    - Touch/event density by field zone
    - First-half vs second-half territorial shifts
    - Analyst-style territory insights

Important:
    These are event-data-derived territorial indicators.
    They are NOT optical-tracking possession maps and should not be described
    as exact spatial control or continuous player occupancy.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.image_fetcher import get_team_badge


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEFENSIVE_THIRD_MAX_X = 40.0
MIDDLE_THIRD_MAX_X = 80.0
ATTACKING_THIRD_MIN_X = 80.0

FINAL_THIRD_ENTRY_X = 80.0
BOX_ENTRY_X = 102.0


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _normalise_name(value):
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
    """
    Filter by:
        Full Match
        First Half
        Second Half
    """
    label = str(period_label).strip().lower()

    if label == "full match":
        # Exclude penalty-shootout events (StatsBomb period 5) so
        # territorial metrics reflect match play rather than shootout actions.
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
            pd.Series(
                0,
                index=events.index,
            ),
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


def _with_coordinates(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add numeric X/Y columns from StatsBomb location.
    """
    df = events.copy()

    if "location" not in df.columns:
        df["X"] = np.nan
        df["Y"] = np.nan
        return df

    coords = df[
        "location"
    ].apply(
        _extract_xy
    )

    df["X"] = [
        point[0]
        for point in coords
    ]

    df["Y"] = [
        point[1]
        for point in coords
    ]

    return df


def _field_third(x):
    if pd.isna(x):
        return "Unknown"

    if x < DEFENSIVE_THIRD_MAX_X:
        return "Defensive Third"

    if x < MIDDLE_THIRD_MAX_X:
        return "Middle Third"

    return "Attacking Third"


def _channel(y):
    if pd.isna(y):
        return "Unknown"

    if y < 26.7:
        return "Left"

    if y < 53.3:
        return "Centre"

    return "Right"


# ---------------------------------------------------------
# Team territory metrics
# ---------------------------------------------------------

def build_team_territory_metrics(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> Dict[str, object]:
    """
    Build event-based territorial metrics for one team.
    """
    period_events = _period_filter(
        events,
        period_label,
    )

    if "team" not in period_events.columns:
        raise RuntimeError(
            "Events do not contain a team column."
        )

    period_events = _with_coordinates(
        period_events
    )

    teams = _normalise_series(
        period_events[
            "team"
        ]
    )

    team_events = period_events.loc[
        teams == team_name
    ].copy()

    located_team_events = team_events.dropna(
        subset=[
            "X",
            "Y",
        ]
    ).copy()

    all_located_events = period_events.dropna(
        subset=[
            "X",
            "Y",
        ]
    ).copy()

    total_team_events = len(
        team_events
    )

    total_period_events = len(
        period_events
    )

    event_share = (
        total_team_events
        / total_period_events
        * 100.0
        if total_period_events > 0
        else 0.0
    )

    located_count = len(
        located_team_events
    )

    avg_x = (
        float(
            located_team_events[
                "X"
            ].mean()
        )
        if located_count > 0
        else np.nan
    )

    avg_y = (
        float(
            located_team_events[
                "Y"
            ].mean()
        )
        if located_count > 0
        else np.nan
    )

    thirds = (
        located_team_events[
            "X"
        ]
        .apply(
            _field_third
        )
        .value_counts()
    )

    defensive_events = int(
        thirds.get(
            "Defensive Third",
            0,
        )
    )

    middle_events = int(
        thirds.get(
            "Middle Third",
            0,
        )
    )

    attacking_events = int(
        thirds.get(
            "Attacking Third",
            0,
        )
    )

    attacking_third_share = (
        attacking_events
        / located_count
        * 100.0
        if located_count > 0
        else 0.0
    )

    channels = (
        located_team_events[
            "Y"
        ]
        .apply(
            _channel
        )
        .value_counts()
    )

    left_events = int(
        channels.get(
            "Left",
            0,
        )
    )

    centre_events = int(
        channels.get(
            "Centre",
            0,
        )
    )

    right_events = int(
        channels.get(
            "Right",
            0,
        )
    )

    final_third_events = int(
        (
            located_team_events[
                "X"
            ]
            >= FINAL_THIRD_ENTRY_X
        ).sum()
    )

    box_zone_events = int(
        (
            located_team_events[
                "X"
            ]
            >= BOX_ENTRY_X
        ).sum()
    )

    # Territory share compares located events in the attacking half.
    attacking_half_events = int(
        (
            located_team_events[
                "X"
            ]
            >= 60.0
        ).sum()
    )

    total_attacking_half_events_all = int(
        (
            all_located_events[
                "X"
            ]
            >= 60.0
        ).sum()
    )

    attacking_half_share_of_match = (
        attacking_half_events
        / total_attacking_half_events_all
        * 100.0
        if total_attacking_half_events_all > 0
        else 0.0
    )

    # Rough event-based territory index.
    # Higher average X and more attacking-third activity increase it.
    territory_index = (
        (
            0.60
            * (
                avg_x / 120.0
                if pd.notna(avg_x)
                else 0.0
            )
        )
        + (
            0.40
            * (
                attacking_third_share
                / 100.0
            )
        )
    ) * 100.0

    return {
        "Team":
            team_name,

        "Period":
            period_label,

        "Events":
            total_team_events,

        "Event Share %":
            event_share,

        "Located Events":
            located_count,

        "Average X":
            avg_x,

        "Average Y":
            avg_y,

        "Defensive Third Events":
            defensive_events,

        "Middle Third Events":
            middle_events,

        "Attacking Third Events":
            attacking_events,

        "Attacking Third Share %":
            attacking_third_share,

        "Left Channel Events":
            left_events,

        "Centre Channel Events":
            centre_events,

        "Right Channel Events":
            right_events,

        "Final Third Events":
            final_third_events,

        "Box Zone Events":
            box_zone_events,

        "Attacking Half Event Share %":
            attacking_half_share_of_match,

        "Territory Index":
            territory_index,
    }


# ---------------------------------------------------------
# Zone distributions
# ---------------------------------------------------------

def build_zone_distribution(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> pd.DataFrame:
    """
    Return field-third and channel counts for one team.
    """
    period_events = _period_filter(
        events,
        period_label,
    )

    if (
        "team" not in period_events.columns
        or "location" not in period_events.columns
    ):
        return pd.DataFrame()

    period_events = _with_coordinates(
        period_events
    )

    teams = _normalise_series(
        period_events[
            "team"
        ]
    )

    team_events = period_events.loc[
        teams == team_name
    ].dropna(
        subset=[
            "X",
            "Y",
        ]
    ).copy()

    if team_events.empty:
        return pd.DataFrame()

    team_events[
        "Third"
    ] = team_events[
        "X"
    ].apply(
        _field_third
    )

    team_events[
        "Channel"
    ] = team_events[
        "Y"
    ].apply(
        _channel
    )

    zone_df = (
        team_events
        .groupby(
            [
                "Third",
                "Channel",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "Event Count",
            }
        )
    )

    zone_df[
        "Team"
    ] = team_name

    zone_df[
        "Period"
    ] = period_label

    total = zone_df[
        "Event Count"
    ].sum()

    zone_df[
        "Zone Share %"
    ] = np.where(
        total > 0,
        (
            zone_df[
                "Event Count"
            ]
            / total
            * 100.0
        ),
        0.0,
    )

    return zone_df


# ---------------------------------------------------------
# Half-by-half territory comparison
# ---------------------------------------------------------

def compare_territory_halves(
    events: pd.DataFrame,
    team_name: str,
) -> Dict[str, object]:
    first = build_team_territory_metrics(
        events,
        team_name,
        "First Half",
    )

    second = build_team_territory_metrics(
        events,
        team_name,
        "Second Half",
    )

    return {
        "Team":
            team_name,

        "First Half Average X":
            first[
                "Average X"
            ],

        "Second Half Average X":
            second[
                "Average X"
            ],

        "Average X Change":
            (
                second[
                    "Average X"
                ]
                - first[
                    "Average X"
                ]
                if (
                    pd.notna(
                        first[
                            "Average X"
                        ]
                    )
                    and pd.notna(
                        second[
                            "Average X"
                        ]
                    )
                )
                else np.nan
            ),

        "First Half Attacking Third Share %":
            first[
                "Attacking Third Share %"
            ],

        "Second Half Attacking Third Share %":
            second[
                "Attacking Third Share %"
            ],

        "Attacking Third Share Change":
            (
                second[
                    "Attacking Third Share %"
                ]
                - first[
                    "Attacking Third Share %"
                ]
            ),

        "First Half Territory Index":
            first[
                "Territory Index"
            ],

        "Second Half Territory Index":
            second[
                "Territory Index"
            ],

        "Territory Index Change":
            (
                second[
                    "Territory Index"
                ]
                - first[
                    "Territory Index"
                ]
            ),

        "First Half Final Third Events":
            first[
                "Final Third Events"
            ],

        "Second Half Final Third Events":
            second[
                "Final Third Events"
            ],

        "Final Third Event Change":
            (
                second[
                    "Final Third Events"
                ]
                - first[
                    "Final Third Events"
                ]
            ),

        "First Half Box Zone Events":
            first[
                "Box Zone Events"
            ],

        "Second Half Box Zone Events":
            second[
                "Box Zone Events"
            ],

        "Box Zone Event Change":
            (
                second[
                    "Box Zone Events"
                ]
                - first[
                    "Box Zone Events"
                ]
            ),
    }


# ---------------------------------------------------------
# Analyst insights
# ---------------------------------------------------------

def build_territory_insights(
    metrics: Dict[str, object],
) -> List[str]:
    insights = []

    team = metrics[
        "Team"
    ]

    avg_x = metrics[
        "Average X"
    ]

    attacking_share = metrics[
        "Attacking Third Share %"
    ]

    final_third_events = metrics[
        "Final Third Events"
    ]

    box_events = metrics[
        "Box Zone Events"
    ]

    territory_index = metrics[
        "Territory Index"
    ]

    if pd.notna(
        avg_x
    ):
        insights.append(
            f"{team}'s average event position was "
            f"{avg_x:.1f} on the StatsBomb X-axis."
        )

    insights.append(
        f"{attacking_share:.1f}% of {team}'s located events "
        f"occurred in the attacking third."
    )

    insights.append(
        f"{team} recorded {final_third_events} events in the final third "
        f"and {box_events} events in the box-zone area."
    )

    insights.append(
        f"{team}'s event-based territory index was "
        f"{territory_index:.1f}/100."
    )

    return insights


def build_territory_change_insights(
    comparison: Dict[str, object],
) -> List[str]:
    insights = []

    team = comparison[
        "Team"
    ]

    avg_x_change = comparison[
        "Average X Change"
    ]

    attacking_change = comparison[
        "Attacking Third Share Change"
    ]

    territory_change = comparison[
        "Territory Index Change"
    ]

    final_third_change = comparison[
        "Final Third Event Change"
    ]

    box_change = comparison[
        "Box Zone Event Change"
    ]

    if pd.notna(
        avg_x_change
    ) and abs(
        avg_x_change
    ) >= 3.0:
        if avg_x_change > 0:
            insights.append(
                f"{team}'s average event position moved "
                f"{avg_x_change:.1f} X units higher in the second half."
            )
        else:
            insights.append(
                f"{team}'s average event position moved "
                f"{abs(avg_x_change):.1f} X units deeper in the second half."
            )

    if abs(
        attacking_change
    ) >= 3.0:
        direction = (
            "increased"
            if attacking_change > 0
            else "decreased"
        )

        insights.append(
            f"{team}'s attacking-third event share {direction} by "
            f"{abs(attacking_change):.1f} percentage points."
        )

    if abs(
        territory_change
    ) >= 3.0:
        direction = (
            "improved"
            if territory_change > 0
            else "declined"
        )

        insights.append(
            f"{team}'s territory index {direction} by "
            f"{abs(territory_change):.1f} points after half-time."
        )

    if (
        abs(
            final_third_change
        )
        >= 5
        or abs(
            box_change
        )
        >= 3
    ):
        insights.append(
            f"Second-half advanced-area activity changed for {team}: "
            f"final-third events {final_third_change:+d}, "
            f"box-zone events {box_change:+d}."
        )

    return insights




def _resolve_team_badge_path(team_name: str):
    try:
        badge = get_team_badge(str(team_name).strip())
        if badge is not None and Path(badge).exists():
            return Path(badge)
    except Exception:
        pass

    return None


def _add_team_badge_to_figure(
    fig,
    team_name: str,
    left: float,
    bottom: float,
    width: float = 0.035,
    height: float = 0.035,
) -> bool:
    badge_path = _resolve_team_badge_path(team_name)

    if badge_path is None:
        return False

    try:
        from PIL import Image as PILImage

        image = PILImage.open(badge_path).convert("RGBA")

        badge_ax = fig.add_axes(
            [left, bottom, width, height]
        )
        badge_ax.imshow(image)
        badge_ax.axis("off")
        return True

    except Exception:
        return False


# ---------------------------------------------------------
# Territory visualisation
# ---------------------------------------------------------

def _draw_territory_pitch(ax) -> None:
    ax.set_xlim(0, 120)
    ax.set_ylim(80, 0)

    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], linewidth=1.2)
    ax.plot([60, 60], [0, 80], linewidth=0.9)

    ax.plot([40, 40], [0, 80], linewidth=0.8, linestyle="--", alpha=0.6)
    ax.plot([80, 80], [0, 80], linewidth=0.8, linestyle="--", alpha=0.6)

    ax.plot([0, 120], [26.7, 26.7], linewidth=0.8, linestyle=":", alpha=0.6)
    ax.plot([0, 120], [53.3, 53.3], linewidth=0.8, linestyle=":", alpha=0.6)

    ax.plot([0, 18, 18, 0], [18, 18, 62, 62], linewidth=0.9)
    ax.plot([120, 102, 102, 120], [18, 18, 62, 62], linewidth=0.9)

    ax.plot([0, 6, 6, 0], [30, 30, 50, 50], linewidth=0.9)
    ax.plot([120, 114, 114, 120], [30, 30, 50, 50], linewidth=0.9)

    ax.add_patch(plt.Circle((60, 40), 10, fill=False, linewidth=0.9))
    ax.scatter([60, 12, 108], [40, 40, 40], s=8, zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def _territory_zone_bounds(third: str, channel: str):
    third_bounds = {
        "Defensive Third": (0.0, 40.0),
        "Middle Third": (40.0, 80.0),
        "Attacking Third": (80.0, 120.0),
    }

    channel_bounds = {
        "Left": (0.0, 26.7),
        "Centre": (26.7, 53.3),
        "Right": (53.3, 80.0),
    }

    if third not in third_bounds or channel not in channel_bounds:
        return None

    x0, x1 = third_bounds[third]
    y0, y1 = channel_bounds[channel]

    return x0, x1, y0, y1


def _draw_team_territory_panel(
    ax,
    events: pd.DataFrame,
    team_name: str,
    period_label: str,
) -> None:
    """
    Draw one team's event-derived 3x3 territory panel.
    """
    metrics = build_team_territory_metrics(
        events,
        team_name,
        period_label,
    )

    zones = build_zone_distribution(
        events,
        team_name,
        period_label,
    )

    _draw_territory_pitch(
        ax
    )

    avg_x = metrics.get(
        "Average X"
    )
    avg_y = metrics.get(
        "Average Y"
    )

    if not zones.empty:
        shares = pd.to_numeric(
            zones["Zone Share %"],
            errors="coerce",
        ).fillna(0)

        max_share = max(
            float(shares.max()),
            1.0,
        )

        for _, row in zones.iterrows():
            bounds = _territory_zone_bounds(
                str(row["Third"]),
                str(row["Channel"]),
            )

            if bounds is None:
                continue

            x0, x1, y0, y1 = bounds

            count = int(
                row.get(
                    "Event Count",
                    0,
                )
            )

            share = float(
                row.get(
                    "Zone Share %",
                    0.0,
                )
            )

            relative = min(
                max(
                    share / max_share,
                    0.0,
                ),
                1.0,
            )

            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                alpha=0.08 + relative * 0.34,
                zorder=1,
            )

            ax.add_patch(
                rect
            )

            label_x = (
                x0 + x1
            ) / 2

            label_y = (
                y0 + y1
            ) / 2

            # If the average-position marker falls inside this same zone,
            # shift the zone percentage/count away from the marker.
            if (
                pd.notna(avg_x)
                and pd.notna(avg_y)
                and x0 <= float(avg_x) < x1
                and y0 <= float(avg_y) < y1
            ):
                # Move the zone figure toward the side with more room.
                if float(avg_x) <= label_x:
                    label_x = min(
                        x1 - 4.5,
                        label_x + 10.0,
                    )
                else:
                    label_x = max(
                        x0 + 4.5,
                        label_x - 10.0,
                    )

                label_y = min(
                    y1 - 4.0,
                    label_y + 5.0,
                )

            ax.text(
                label_x,
                label_y,
                f"{share:.1f}%\n({count})",
                ha="center",
                va="center",
                fontsize=8.2,
                weight=(
                    "bold"
                    if share >= max_share * 0.75
                    else "normal"
                ),
                zorder=3,
            )

    if (
        pd.notna(avg_x)
        and pd.notna(avg_y)
    ):
        ax.scatter(
            [float(avg_x)],
            [float(avg_y)],
            s=150,
            marker="X",
            zorder=6,
        )

        # Keep the callout above the centre circle and away from
        # both the marker and zone value.
        label_y = max(
            4.0,
            float(avg_y) - 14.0,
        )

        ax.annotate(
            "Avg event position",
            xy=(
                float(avg_x),
                float(avg_y),
            ),
            xytext=(
                float(avg_x),
                label_y,
            ),
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.8,
            },
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "alpha": 0.90,
                "linewidth": 0.6,
            },
            zorder=7,
        )

    ax.set_title(
        f"{team_name} — Territory Map",
        fontsize=15,
        weight="bold",
        pad=18,
    )

    ax.text(
        0.5,
        1.015,
        (
            f"{period_label} | "
            f"Territory Index {metrics['Territory Index']:.1f}/100 | "
            f"Attacking-third share {metrics['Attacking Third Share %']:.1f}%"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.2,
    )


def build_territory_figure(
    events: pd.DataFrame,
    team_names: Optional[List[str]] = None,
    period_label: str = "Full Match",
):
    """
    Build a professional two-team Possession & Territory figure.

    The 3x3 pitch zones show each team's share of its own located events.
    The X marker shows average event position.

    Important:
        This is event-derived territory intelligence, not an
        optical-tracking possession/control heatmap.
    """
    if events is None or events.empty:
        fig, ax = plt.subplots(figsize=(11.5, 6.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No territory data available",
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
        team_names
        or []
    )[:2]

    if len(team_names) < 2:
        fig, ax = plt.subplots(figsize=(11.5, 6.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Two teams are required for territory comparison",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
        )
        return fig

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 7.5),
    )

    for ax, team_name in zip(
        axes,
        team_names,
    ):
        _draw_team_territory_panel(
            ax,
            events,
            team_name,
            period_label,
        )

    summary_rows = [
        build_team_territory_metrics(
            events,
            team_name,
            period_label,
        )
        for team_name in team_names
    ]

    t1 = summary_rows[0]
    t2 = summary_rows[1]

    fig.suptitle(
        "Possession & Territory",
        fontsize=19,
        weight="bold",
        y=0.985,
    )

    # Professional header treatment:
    # keep each crest visually attached to its existing
    # "Team — Territory Map" panel heading without repeating the team name.
    _add_team_badge_to_figure(
        fig,
        team_names[0],
        0.105,
        0.850,
        0.050,
        0.050,
    )

    _add_team_badge_to_figure(
        fig,
        team_names[1],
        0.605,
        0.850,
        0.050,
        0.050,
    )

    fig.text(
        0.5,
        0.035,
        (
            f"{team_names[0]}: Avg X {t1['Average X']:.1f}, "
            f"Final-third events {t1['Final Third Events']}, "
            f"Box-zone events {t1['Box Zone Events']}   |   "
            f"{team_names[1]}: Avg X {t2['Average X']:.1f}, "
            f"Final-third events {t2['Final Third Events']}, "
            f"Box-zone events {t2['Box Zone Events']}"
        ),
        ha="center",
        va="center",
        fontsize=9.5,
    )

    fig.text(
        0.5,
        0.01,
        (
            "Zone percentages are shares of each team's located match events. "
            "This is event-derived territory intelligence, not continuous "
            "optical-tracking possession/control."
        ),
        ha="center",
        va="bottom",
        fontsize=8.3,
    )

    fig.tight_layout(
        rect=[
            0.02,
            0.065,
            0.98,
            0.94,
        ]
    )

    return fig


def build_territory_report_payload(
    events: pd.DataFrame,
    team_names: Optional[List[str]] = None,
    period_label: str = "Full Match",
) -> Dict[str, object]:
    result = analyze_possession_territory(
        events,
        team_names,
        period_label,
    )

    return {
        "period": period_label,
        "summaries": result["summaries"].copy(),
        "zones": result["zones"].copy(),
        "half_comparisons": result["half_comparisons"].copy(),
        "insights": result["insights"].copy(),
    }


# ---------------------------------------------------------
# Public interface
# ---------------------------------------------------------

def analyze_possession_territory(
    events: pd.DataFrame,
    team_names: Optional[List[str]] = None,
    period_label: str = "Full Match",
) -> Dict[str, object]:
    """
    Run possession/territory intelligence for selected teams.

    Returns:
        {
            "summaries": DataFrame,
            "zones": DataFrame,
            "half_comparisons": DataFrame,
            "insights": DataFrame,
        }
    """

    if events is None or events.empty:
        return {
            "summaries":
                pd.DataFrame(),

            "zones":
                pd.DataFrame(),

            "half_comparisons":
                pd.DataFrame(),

            "insights":
                pd.DataFrame(),
        }

    if team_names is None:
        if "team" not in events.columns:
            team_names = []
        else:
            team_names = (
                _normalise_series(
                    events[
                        "team"
                    ]
                )
                .replace(
                    "",
                    np.nan,
                )
                .dropna()
                .unique()
                .tolist()
            )

    summary_rows = []
    zone_frames = []
    half_rows = []
    insight_rows = []

    for team_name in team_names:
        metrics = build_team_territory_metrics(
            events,
            team_name,
            period_label,
        )

        summary_rows.append(
            metrics
        )

        zones = build_zone_distribution(
            events,
            team_name,
            period_label,
        )

        if not zones.empty:
            zone_frames.append(
                zones
            )

        half_comparison = compare_territory_halves(
            events,
            team_name,
        )

        half_rows.append(
            half_comparison
        )

        for message in build_territory_insights(
            metrics
        ):
            insight_rows.append(
                {
                    "Team":
                        team_name,

                    "Type":
                        "Territory",

                    "Message":
                        message,
                }
            )

        for message in build_territory_change_insights(
            half_comparison
        ):
            insight_rows.append(
                {
                    "Team":
                        team_name,

                    "Type":
                        "Territory Change",

                    "Message":
                        message,
                }
            )

    summaries_df = pd.DataFrame(
        summary_rows
    )

    zones_df = (
        pd.concat(
            zone_frames,
            ignore_index=True,
        )
        if zone_frames
        else pd.DataFrame()
    )

    half_df = pd.DataFrame(
        half_rows
    )

    insights_df = pd.DataFrame(
        insight_rows
    )

    return {
        "summaries":
            summaries_df,

        "zones":
            zones_df,

        "half_comparisons":
            half_df,

        "insights":
            insights_df,
    }

