"""
Progressive Actions & Chance Creation Intelligence for LiveMatch Intelligence.

Purpose:
    Derive event-based ball progression and chance-creation indicators from
    StatsBomb match events.

This module provides:
    - Progressive passes
    - Progressive carries
    - Final-third entries
    - Box entries
    - Player contribution rankings
    - Team progression summaries
    - First-half vs second-half progression changes
    - Analyst-style progression insights

Important:
    This module uses transparent event-data heuristics. Progressive actions are
    not imported from a proprietary provider definition and should be described
    as project-defined event-based progression indicators.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.image_fetcher import get_team_badge


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROGRESSIVE_DISTANCE_THRESHOLD = 10.0
FINAL_THIRD_X = 80.0
BOX_X = 102.0

MIN_PROGRESSIVE_PLAYER_ACTIONS = 1


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
    label = str(period_label).strip().lower()

    if label == "full match":
        # Exclude penalty-shootout events (StatsBomb period 5) so
        # progressive-action metrics reflect match play rather than shootout actions.
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


def _completed_pass_mask(
    passes: pd.DataFrame,
) -> pd.Series:
    if "pass_outcome" not in passes.columns:
        return pd.Series(
            True,
            index=passes.index,
        )

    outcomes = _normalise_series(
        passes["pass_outcome"]
    )

    return outcomes.eq("")


def _distance_forward(
    start_x,
    end_x,
):
    if pd.isna(start_x) or pd.isna(end_x):
        return np.nan

    return float(end_x) - float(start_x)


def _is_progressive_action(
    start_x,
    end_x,
):
    """
    Transparent project heuristic:
    an action is progressive when it moves the ball at least 10 StatsBomb
    X-units toward the opponent goal.
    """
    distance = _distance_forward(
        start_x,
        end_x,
    )

    if pd.isna(
        distance
    ):
        return False

    return (
        distance
        >= PROGRESSIVE_DISTANCE_THRESHOLD
    )


# ---------------------------------------------------------
# Action extraction
# ---------------------------------------------------------

def extract_team_progressive_actions(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> pd.DataFrame:
    """
    Extract progressive passes and carries for one team.
    """
    period_events = _period_filter(
        events,
        period_label,
    )

    if (
        "team" not in period_events.columns
        or "type" not in period_events.columns
    ):
        return pd.DataFrame()

    teams = _normalise_series(
        period_events[
            "team"
        ]
    )

    event_types = _normalise_series(
        period_events[
            "type"
        ]
    )

    mask = (
        teams.eq(
            team_name
        )
        & event_types.isin(
            [
                "Pass",
                "Carry",
            ]
        )
    )

    actions = period_events.loc[
        mask
    ].copy()

    if actions.empty:
        return actions

    actions[
        "Action Type"
    ] = _normalise_series(
        actions[
            "type"
        ]
    )

    if "player" in actions.columns:
        actions[
            "Player"
        ] = _normalise_series(
            actions[
                "player"
            ]
        )
    else:
        actions[
            "Player"
        ] = ""

    start_coords = actions[
        "location"
    ].apply(
        _extract_xy
    )

    actions[
        "Start X"
    ] = [
        point[0]
        for point in start_coords
    ]

    actions[
        "Start Y"
    ] = [
        point[1]
        for point in start_coords
    ]

    actions[
        "End X"
    ] = np.nan

    actions[
        "End Y"
    ] = np.nan

    # Pass end locations
    if "pass_end_location" in actions.columns:
        pass_mask = (
            actions[
                "Action Type"
            ]
            == "Pass"
        )

        pass_coords = actions.loc[
            pass_mask,
            "pass_end_location",
        ].apply(
            _extract_xy
        )

        actions.loc[
            pass_mask,
            "End X",
        ] = [
            point[0]
            for point in pass_coords
        ]

        actions.loc[
            pass_mask,
            "End Y",
        ] = [
            point[1]
            for point in pass_coords
        ]

    # Carry end locations
    if "carry_end_location" in actions.columns:
        carry_mask = (
            actions[
                "Action Type"
            ]
            == "Carry"
        )

        carry_coords = actions.loc[
            carry_mask,
            "carry_end_location",
        ].apply(
            _extract_xy
        )

        actions.loc[
            carry_mask,
            "End X",
        ] = [
            point[0]
            for point in carry_coords
        ]

        actions.loc[
            carry_mask,
            "End Y",
        ] = [
            point[1]
            for point in carry_coords
        ]

    # Completed passes only.
    completed_mask = pd.Series(
        True,
        index=actions.index,
    )

    pass_rows = actions[
        actions[
            "Action Type"
        ]
        == "Pass"
    ]

    if not pass_rows.empty:
        pass_completed = _completed_pass_mask(
            pass_rows
        )

        completed_mask.loc[
            pass_rows.index
        ] = pass_completed

    actions[
        "Completed"
    ] = completed_mask

    actions[
        "Forward Distance"
    ] = (
        actions[
            "End X"
        ]
        - actions[
            "Start X"
        ]
    )

    actions[
        "Progressive"
    ] = actions.apply(
        lambda row:
            bool(
                row[
                    "Completed"
                ]
            )
            and _is_progressive_action(
                row[
                    "Start X"
                ],
                row[
                    "End X"
                ],
            ),
        axis=1,
    )

    actions[
        "Final Third Entry"
    ] = (
        actions[
            "Completed"
        ]
        & (
            actions[
                "Start X"
            ]
            < FINAL_THIRD_X
        )
        & (
            actions[
                "End X"
            ]
            >= FINAL_THIRD_X
        )
    )

    actions[
        "Box Entry"
    ] = (
        actions[
            "Completed"
        ]
        & (
            actions[
                "Start X"
            ]
            < BOX_X
        )
        & (
            actions[
                "End X"
            ]
            >= BOX_X
        )
    )

    return actions


# ---------------------------------------------------------
# Player summaries
# ---------------------------------------------------------

def build_player_progression_summary(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> pd.DataFrame:
    actions = extract_team_progressive_actions(
        events,
        team_name,
        period_label,
    )

    if actions.empty:
        return pd.DataFrame(
            columns=[
                "Player",
                "Team",
                "Progressive Passes",
                "Progressive Carries",
                "Progressive Actions",
                "Final Third Entries",
                "Box Entries",
                "Forward Distance",
            ]
        )

    valid = actions[
        actions[
            "Player"
        ]
        != ""
    ].copy()

    rows = []

    for player_name, player_actions in valid.groupby(
        "Player"
    ):
        progressive_passes = int(
            (
                (
                    player_actions[
                        "Action Type"
                    ]
                    == "Pass"
                )
                & player_actions[
                    "Progressive"
                ]
            ).sum()
        )

        progressive_carries = int(
            (
                (
                    player_actions[
                        "Action Type"
                    ]
                    == "Carry"
                )
                & player_actions[
                    "Progressive"
                ]
            ).sum()
        )

        progressive_actions = (
            progressive_passes
            + progressive_carries
        )

        final_third_entries = int(
            player_actions[
                "Final Third Entry"
            ].sum()
        )

        box_entries = int(
            player_actions[
                "Box Entry"
            ].sum()
        )

        forward_distance = float(
            pd.to_numeric(
                player_actions.loc[
                    player_actions[
                        "Progressive"
                    ],
                    "Forward Distance",
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )

        if (
            progressive_actions
            < MIN_PROGRESSIVE_PLAYER_ACTIONS
            and final_third_entries == 0
            and box_entries == 0
        ):
            continue

        rows.append(
            {
                "Player":
                    player_name,

                "Team":
                    team_name,

                "Progressive Passes":
                    progressive_passes,

                "Progressive Carries":
                    progressive_carries,

                "Progressive Actions":
                    progressive_actions,

                "Final Third Entries":
                    final_third_entries,

                "Box Entries":
                    box_entries,

                "Forward Distance":
                    forward_distance,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Player",
                "Team",
                "Progressive Passes",
                "Progressive Carries",
                "Progressive Actions",
                "Final Third Entries",
                "Box Entries",
                "Forward Distance",
            ]
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "Progressive Actions",
                "Final Third Entries",
                "Box Entries",
                "Forward Distance",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------
# Team summaries
# ---------------------------------------------------------

def build_team_progression_summary(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> Dict[str, object]:
    actions = extract_team_progressive_actions(
        events,
        team_name,
        period_label,
    )

    players = build_player_progression_summary(
        events,
        team_name,
        period_label,
    )

    if actions.empty:
        return {
            "Team": team_name,
            "Period": period_label,
            "Progressive Passes": 0,
            "Progressive Carries": 0,
            "Progressive Actions": 0,
            "Final Third Entries": 0,
            "Box Entries": 0,
            "Forward Distance": 0.0,
            "Top Progressor": None,
            "Top Final Third Contributor": None,
            "Top Box Entry Contributor": None,
            "Players": players,
        }

    progressive_passes = int(
        (
            (
                actions[
                    "Action Type"
                ]
                == "Pass"
            )
            & actions[
                "Progressive"
            ]
        ).sum()
    )

    progressive_carries = int(
        (
            (
                actions[
                    "Action Type"
                ]
                == "Carry"
            )
            & actions[
                "Progressive"
            ]
        ).sum()
    )

    progressive_actions = (
        progressive_passes
        + progressive_carries
    )

    final_third_entries = int(
        actions[
            "Final Third Entry"
        ].sum()
    )

    box_entries = int(
        actions[
            "Box Entry"
        ].sum()
    )

    forward_distance = float(
        pd.to_numeric(
            actions.loc[
                actions[
                    "Progressive"
                ],
                "Forward Distance",
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    top_progressor = None
    top_final_third = None
    top_box = None

    if not players.empty:
        top_progressor = (
            players.sort_values(
                [
                    "Progressive Actions",
                    "Forward Distance",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .iloc[
                0
            ][
                "Player"
            ]
        )

        top_final_third = (
            players.sort_values(
                [
                    "Final Third Entries",
                    "Progressive Actions",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .iloc[
                0
            ][
                "Player"
            ]
        )

        top_box = (
            players.sort_values(
                [
                    "Box Entries",
                    "Progressive Actions",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .iloc[
                0
            ][
                "Player"
            ]
        )

    return {
        "Team":
            team_name,

        "Period":
            period_label,

        "Progressive Passes":
            progressive_passes,

        "Progressive Carries":
            progressive_carries,

        "Progressive Actions":
            progressive_actions,

        "Final Third Entries":
            final_third_entries,

        "Box Entries":
            box_entries,

        "Forward Distance":
            forward_distance,

        "Top Progressor":
            top_progressor,

        "Top Final Third Contributor":
            top_final_third,

        "Top Box Entry Contributor":
            top_box,

        "Players":
            players,
    }


# ---------------------------------------------------------
# Half-by-half comparison
# ---------------------------------------------------------

def compare_progression_halves(
    events: pd.DataFrame,
    team_name: str,
) -> Dict[str, object]:
    first = build_team_progression_summary(
        events,
        team_name,
        "First Half",
    )

    second = build_team_progression_summary(
        events,
        team_name,
        "Second Half",
    )

    return {
        "Team":
            team_name,

        "First Half Progressive Passes":
            first[
                "Progressive Passes"
            ],

        "Second Half Progressive Passes":
            second[
                "Progressive Passes"
            ],

        "Progressive Pass Change":
            (
                second[
                    "Progressive Passes"
                ]
                - first[
                    "Progressive Passes"
                ]
            ),

        "First Half Progressive Carries":
            first[
                "Progressive Carries"
            ],

        "Second Half Progressive Carries":
            second[
                "Progressive Carries"
            ],

        "Progressive Carry Change":
            (
                second[
                    "Progressive Carries"
                ]
                - first[
                    "Progressive Carries"
                ]
            ),

        "First Half Progressive Actions":
            first[
                "Progressive Actions"
            ],

        "Second Half Progressive Actions":
            second[
                "Progressive Actions"
            ],

        "Progressive Action Change":
            (
                second[
                    "Progressive Actions"
                ]
                - first[
                    "Progressive Actions"
                ]
            ),

        "First Half Final Third Entries":
            first[
                "Final Third Entries"
            ],

        "Second Half Final Third Entries":
            second[
                "Final Third Entries"
            ],

        "Final Third Entry Change":
            (
                second[
                    "Final Third Entries"
                ]
                - first[
                    "Final Third Entries"
                ]
            ),

        "First Half Box Entries":
            first[
                "Box Entries"
            ],

        "Second Half Box Entries":
            second[
                "Box Entries"
            ],

        "Box Entry Change":
            (
                second[
                    "Box Entries"
                ]
                - first[
                    "Box Entries"
                ]
            ),

        "First Half Forward Distance":
            first[
                "Forward Distance"
            ],

        "Second Half Forward Distance":
            second[
                "Forward Distance"
            ],

        "Forward Distance Change":
            (
                second[
                    "Forward Distance"
                ]
                - first[
                    "Forward Distance"
                ]
            ),

        "First Half Top Progressor":
            first[
                "Top Progressor"
            ],

        "Second Half Top Progressor":
            second[
                "Top Progressor"
            ],
    }


# ---------------------------------------------------------
# Analyst insights
# ---------------------------------------------------------

def build_progression_insights(
    summary: Dict[str, object],
) -> List[str]:
    insights = []

    team = summary[
        "Team"
    ]

    insights.append(
        f"{team} recorded "
        f"{summary['Progressive Actions']} progressive actions: "
        f"{summary['Progressive Passes']} passes and "
        f"{summary['Progressive Carries']} carries."
    )

    insights.append(
        f"{team} made "
        f"{summary['Final Third Entries']} final-third entries and "
        f"{summary['Box Entries']} box entries."
    )

    insights.append(
        f"Progressive actions advanced the ball by approximately "
        f"{summary['Forward Distance']:.1f} StatsBomb X-units in total."
    )

    if summary[
        "Top Progressor"
    ]:
        insights.append(
            f"{summary['Top Progressor']} was the leading progression "
            f"contributor for {team}."
        )

    if summary[
        "Top Final Third Contributor"
    ]:
        insights.append(
            f"{summary['Top Final Third Contributor']} led {team}'s "
            f"final-third entry contribution."
        )

    return insights


def build_progression_change_insights(
    comparison: Dict[str, object],
) -> List[str]:
    insights = []

    team = comparison[
        "Team"
    ]

    action_change = comparison[
        "Progressive Action Change"
    ]

    final_third_change = comparison[
        "Final Third Entry Change"
    ]

    box_change = comparison[
        "Box Entry Change"
    ]

    distance_change = comparison[
        "Forward Distance Change"
    ]

    if action_change != 0:
        direction = (
            "increased"
            if action_change > 0
            else "decreased"
        )

        insights.append(
            f"{team}'s progressive-action volume {direction} by "
            f"{abs(action_change)} in the second half."
        )

    if final_third_change != 0:
        insights.append(
            f"{team}'s final-third entries changed by "
            f"{final_third_change:+d} after half-time."
        )

    if box_change != 0:
        insights.append(
            f"{team}'s box entries changed by "
            f"{box_change:+d} after half-time."
        )

    if abs(
        distance_change
    ) >= 50.0:
        direction = (
            "increased"
            if distance_change > 0
            else "decreased"
        )

        insights.append(
            f"{team}'s total progressive forward distance {direction} by "
            f"{abs(distance_change):.1f} X-units in the second half."
        )

    first_progressor = comparison[
        "First Half Top Progressor"
    ]

    second_progressor = comparison[
        "Second Half Top Progressor"
    ]

    if (
        first_progressor
        and second_progressor
        and first_progressor
        != second_progressor
    ):
        insights.append(
            f"The leading progression contributor changed from "
            f"{first_progressor} in the first half to "
            f"{second_progressor} in the second."
        )

    return insights



# ---------------------------------------------------------
# Progressive-action visualisation
# ---------------------------------------------------------

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
    width: float = 0.045,
    height: float = 0.045,
) -> bool:
    badge_path = _resolve_team_badge_path(team_name)

    if badge_path is None:
        return False

    try:
        from PIL import Image as PILImage

        image = PILImage.open(badge_path).convert("RGBA")
        badge_ax = fig.add_axes([left, bottom, width, height])
        badge_ax.imshow(image)
        badge_ax.axis("off")
        return True

    except Exception:
        return False


def _draw_progression_pitch(ax) -> None:
    ax.set_xlim(0, 120)
    ax.set_ylim(80, 0)

    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], linewidth=1.2)
    ax.plot([60, 60], [0, 80], linewidth=0.9)

    ax.plot([0, 18, 18, 0], [18, 18, 62, 62], linewidth=0.9)
    ax.plot([120, 102, 102, 120], [18, 18, 62, 62], linewidth=0.9)

    ax.plot([0, 6, 6, 0], [30, 30, 50, 50], linewidth=0.9)
    ax.plot([120, 114, 114, 120], [30, 30, 50, 50], linewidth=0.9)

    ax.add_patch(
        plt.Circle(
            (60, 40),
            10,
            fill=False,
            linewidth=0.9,
        )
    )

    ax.scatter(
        [60, 12, 108],
        [40, 40, 40],
        s=8,
        zorder=5,
    )

    # Final-third and box-entry reference lines.
    ax.axvline(
        FINAL_THIRD_X,
        linestyle="--",
        linewidth=0.9,
        alpha=0.55,
    )

    ax.axvline(
        BOX_X,
        linestyle=":",
        linewidth=0.9,
        alpha=0.55,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_team_progression_panel(
    ax,
    events: pd.DataFrame,
    team_name: str,
    period_label: str,
    max_actions: int = 24,
) -> None:
    """
    Draw the strongest progressive passes/carries for one team.

    To keep the pitch readable, the panel ranks progressive actions by
    forward distance and plots only the strongest subset.
    """
    actions = extract_team_progressive_actions(
        events,
        team_name,
        period_label,
    )

    summary = build_team_progression_summary(
        events,
        team_name,
        period_label,
    )

    _draw_progression_pitch(ax)

    if actions.empty:
        ax.text(
            60,
            40,
            "No progressive actions available",
            ha="center",
            va="center",
            fontsize=10,
        )
        return

    progressive = actions.loc[
        actions["Progressive"].fillna(False)
    ].copy()

    progressive["Forward Distance"] = pd.to_numeric(
        progressive["Forward Distance"],
        errors="coerce",
    ).fillna(0.0)

    progressive = (
        progressive
        .sort_values(
            "Forward Distance",
            ascending=False,
        )
        .head(max_actions)
    )

    for _, row in progressive.iterrows():
        start_x = row.get("Start X")
        start_y = row.get("Start Y")
        end_x = row.get("End X")
        end_y = row.get("End Y")

        if any(
            pd.isna(value)
            for value in (
                start_x,
                start_y,
                end_x,
                end_y,
            )
        ):
            continue

        action_type = str(
            row.get(
                "Action Type",
                "",
            )
        )

        if action_type == "Carry":
            linestyle = "--"
            linewidth = 1.55
            mutation_scale = 8
            alpha = 0.68
        else:
            linestyle = "-"
            linewidth = 0.95
            mutation_scale = 7
            alpha = 0.34

        arrow = FancyArrowPatch(
            (float(start_x), float(start_y)),
            (float(end_x), float(end_y)),
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=4,
        )

        ax.add_patch(arrow)

    ax.text(
        0.5,
        1.045,
        f"{team_name} — Progressive Actions",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        weight="bold",
    )

    ax.text(
        0.5,
        1.008,
        (
            f"{period_label} | "
            f"{summary['Progressive Actions']} progressive actions | "
            f"{summary['Final Third Entries']} final-third entries | "
            f"{summary['Box Entries']} box entries"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.8,
    )


def build_progressive_actions_figure(
    events: pd.DataFrame,
    team_names: Optional[List[str]] = None,
    period_label: str = "Full Match",
    max_actions_per_team: int = 24,
):
    """
    Build a professional two-team progressive-actions comparison figure.

    Solid arrows = progressive passes.
    Dashed arrows = progressive carries.

    The pitch visual deliberately plots only the strongest actions by forward
    distance so the figure remains readable. Team totals below are calculated
    from all qualifying actions.
    """
    if events is None or events.empty:
        fig, ax = plt.subplots(figsize=(11.5, 6.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No progression data available",
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
                _normalise_series(events["team"])
                .replace("", np.nan)
                .dropna()
                .unique()
                .tolist()
            )

    team_names = list(team_names or [])[:2]

    if len(team_names) < 2:
        fig, ax = plt.subplots(figsize=(11.5, 6.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Two teams are required for progression comparison",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
        )
        return fig

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 8.2),
    )

    for ax, team_name in zip(axes, team_names):
        _draw_team_progression_panel(
            ax,
            events,
            team_name,
            period_label,
            max_actions=max_actions_per_team,
        )

    summaries = [
        build_team_progression_summary(
            events,
            team_name,
            period_label,
        )
        for team_name in team_names
    ]

    s1 = summaries[0]
    s2 = summaries[1]

    fig.suptitle(
        "Progressive Actions",
        fontsize=19,
        weight="bold",
        y=0.985,
    )

    # Professional transparent crests resolved through the shared image system.
    _add_team_badge_to_figure(
        fig,
        team_names[0],
        0.095,
        0.850,
        0.050,
        0.050,
    )

    _add_team_badge_to_figure(
        fig,
        team_names[1],
        0.595,
        0.850,
        0.050,
        0.050,
    )

    comparison_text = (
        f"{team_names[0]}: "
        f"{s1['Progressive Passes']} passes | "
        f"{s1['Progressive Carries']} carries | "
        f"{s1['Forward Distance']:.1f} forward X-units | "
        f"Top progressor: {s1['Top Progressor'] or 'N/A'}"
        "     ||     "
        f"{team_names[1]}: "
        f"{s2['Progressive Passes']} passes | "
        f"{s2['Progressive Carries']} carries | "
        f"{s2['Forward Distance']:.1f} forward X-units | "
        f"Top progressor: {s2['Top Progressor'] or 'N/A'}"
    )

    fig.text(
        0.5,
        0.060,
        comparison_text,
        ha="center",
        va="center",
        fontsize=8.9,
    )

    fig.text(
        0.5,
        0.032,
        (
            "Solid arrows = progressive passes | Dashed arrows = progressive carries | "
            "Dashed vertical line = final-third boundary | Dotted vertical line = box-entry boundary | "
            "Pitch plots strongest actions by forward distance; summary totals use all qualifying actions."
        ),
        ha="center",
        va="center",
        fontsize=8.2,
    )

    fig.text(
        0.5,
        0.010,
        (
            "Project-defined progression heuristic: completed pass/carry advancing "
            f"at least {PROGRESSIVE_DISTANCE_THRESHOLD:.0f} StatsBomb X-units."
        ),
        ha="center",
        va="bottom",
        fontsize=7.8,
    )

    fig.tight_layout(
        rect=[
            0.02,
            0.085,
            0.98,
            0.94,
        ]
    )

    return fig


# ---------------------------------------------------------
# Public interface
# ---------------------------------------------------------

def analyze_progressive_actions(
    events: pd.DataFrame,
    team_names: Optional[List[str]] = None,
    period_label: str = "Full Match",
) -> Dict[str, object]:
    """
    Run progressive-action analysis for selected teams.

    Returns:
        {
            "summaries": DataFrame,
            "players": DataFrame,
            "half_comparisons": DataFrame,
            "insights": DataFrame,
        }
    """

    if events is None or events.empty:
        return {
            "summaries":
                pd.DataFrame(),

            "players":
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
    player_frames = []
    half_rows = []
    insight_rows = []

    for team_name in team_names:
        summary = build_team_progression_summary(
            events,
            team_name,
            period_label,
        )

        summary_rows.append(
            {
                key: value
                for key, value in summary.items()
                if key != "Players"
            }
        )

        if not summary[
            "Players"
        ].empty:
            players = summary[
                "Players"
            ].copy()

            players[
                "Period"
            ] = period_label

            player_frames.append(
                players
            )

        half_comparison = compare_progression_halves(
            events,
            team_name,
        )

        half_rows.append(
            half_comparison
        )

        for message in build_progression_insights(
            summary
        ):
            insight_rows.append(
                {
                    "Team":
                        team_name,

                    "Type":
                        "Progression",

                    "Message":
                        message,
                }
            )

        for message in build_progression_change_insights(
            half_comparison
        ):
            insight_rows.append(
                {
                    "Team":
                        team_name,

                    "Type":
                        "Progression Change",

                    "Message":
                        message,
                }
            )

    summaries_df = pd.DataFrame(
        summary_rows
    )

    players_df = (
        pd.concat(
            player_frames,
            ignore_index=True,
        )
        if player_frames
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

        "players":
            players_df,

        "half_comparisons":
            half_df,

        "insights":
            insights_df,
    }
