"""
Pass Network & Build-up Analyzer for LiveMatch Intelligence.

Purpose:
    Build event-data-derived passing networks from StatsBomb match events.

The module provides:
    - Player-to-player completed pass connections
    - Average player pass-event positions
    - Pass volume and completion
    - Most influential passers and receivers
    - Strongest passing links
    - Build-up comparisons between first and second half

Important:
    This is based on StatsBomb event data, not optical tracking data.
    Average positions represent event locations, not exact formation positions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MIN_CONNECTION_PASSES = 2
MIN_PLAYER_PASSES = 3


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
    Filter events by period label.

    Accepted:
        Full Match
        First Half
        Second Half
    """
    label = str(period_label).strip().lower()

    if label == "full match":
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
    """
    StatsBomb completed passes generally have a missing pass_outcome.
    """
    if "pass_outcome" not in passes.columns:
        return pd.Series(
            True,
            index=passes.index,
        )

    outcomes = _normalise_series(
        passes["pass_outcome"]
    )

    return outcomes.eq("")


# ---------------------------------------------------------
# Core pass extraction
# ---------------------------------------------------------

def extract_team_passes(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> pd.DataFrame:
    """
    Return one team's pass events for a selected match period.
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

    mask = (
        event_types.eq("Pass")
        & teams.eq(team_name)
    )

    passes = period_events.loc[
        mask
    ].copy()

    if passes.empty:
        return passes

    if "player" in passes.columns:
        passes["Passer"] = _normalise_series(
            passes["player"]
        )
    else:
        passes["Passer"] = ""

    if "pass_recipient" in passes.columns:
        passes["Recipient"] = _normalise_series(
            passes["pass_recipient"]
        )
    else:
        passes["Recipient"] = ""

    passes["Completed"] = _completed_pass_mask(
        passes
    )

    if "location" in passes.columns:
        xy = passes[
            "location"
        ].apply(
            _extract_xy
        )

        passes["X"] = [
            point[0]
            for point in xy
        ]

        passes["Y"] = [
            point[1]
            for point in xy
        ]

    else:
        passes["X"] = np.nan
        passes["Y"] = np.nan

    return passes


# ---------------------------------------------------------
# Player nodes
# ---------------------------------------------------------

def build_pass_network_nodes(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> pd.DataFrame:
    """
    Build player node statistics from pass events.
    """
    passes = extract_team_passes(
        events,
        team_name,
        period_label,
    )

    if passes.empty:
        return pd.DataFrame(
            columns=[
                "Player",
                "Team",
                "Average X",
                "Average Y",
                "Passes Attempted",
                "Passes Completed",
                "Pass Completion %",
                "Passes Received",
                "Network Involvement",
            ]
        )

    passer_stats = (
        passes[
            passes["Passer"] != ""
        ]
        .groupby(
            "Passer",
            as_index=False,
        )
        .agg(
            {
                "X": "mean",
                "Y": "mean",
                "Completed": [
                    "size",
                    "sum",
                ],
            }
        )
    )

    passer_stats.columns = [
        "Player",
        "Average X",
        "Average Y",
        "Passes Attempted",
        "Passes Completed",
    ]

    received = (
        passes[
            passes["Completed"]
            & (
                passes["Recipient"]
                != ""
            )
        ]
        .groupby(
            "Recipient"
        )
        .size()
        .rename(
            "Passes Received"
        )
    )

    passer_stats[
        "Passes Received"
    ] = (
        passer_stats[
            "Player"
        ]
        .map(
            received
        )
        .fillna(0)
        .astype(int)
    )

    passer_stats[
        "Pass Completion %"
    ] = np.where(
        passer_stats[
            "Passes Attempted"
        ] > 0,
        (
            passer_stats[
                "Passes Completed"
            ]
            / passer_stats[
                "Passes Attempted"
            ]
            * 100.0
        ),
        0.0,
    )

    passer_stats[
        "Network Involvement"
    ] = (
        passer_stats[
            "Passes Completed"
        ]
        + passer_stats[
            "Passes Received"
        ]
    )

    passer_stats[
        "Team"
    ] = team_name

    passer_stats = passer_stats[
        passer_stats[
            "Passes Attempted"
        ]
        >= MIN_PLAYER_PASSES
    ].copy()

    return passer_stats[
        [
            "Player",
            "Team",
            "Average X",
            "Average Y",
            "Passes Attempted",
            "Passes Completed",
            "Pass Completion %",
            "Passes Received",
            "Network Involvement",
        ]
    ].sort_values(
        "Network Involvement",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------
# Pass connections
# ---------------------------------------------------------

def build_pass_network_edges(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> pd.DataFrame:
    """
    Build completed player-to-player pass connections.
    """
    passes = extract_team_passes(
        events,
        team_name,
        period_label,
    )

    if passes.empty:
        return pd.DataFrame(
            columns=[
                "Passer",
                "Recipient",
                "Team",
                "Pass Count",
            ]
        )

    completed = passes[
        passes["Completed"]
        & (
            passes["Passer"]
            != ""
        )
        & (
            passes["Recipient"]
            != ""
        )
    ].copy()

    if completed.empty:
        return pd.DataFrame(
            columns=[
                "Passer",
                "Recipient",
                "Team",
                "Pass Count",
            ]
        )

    edges = (
        completed
        .groupby(
            [
                "Passer",
                "Recipient",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "Pass Count",
            }
        )
    )

    edges = edges[
        edges[
            "Pass Count"
        ]
        >= MIN_CONNECTION_PASSES
    ].copy()

    edges[
        "Team"
    ] = team_name

    return edges[
        [
            "Passer",
            "Recipient",
            "Team",
            "Pass Count",
        ]
    ].sort_values(
        "Pass Count",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------
# Network summary
# ---------------------------------------------------------

def build_pass_network_summary(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> Dict[str, object]:
    """
    Summarise a team's passing network for one period.
    """
    nodes = build_pass_network_nodes(
        events,
        team_name,
        period_label,
    )

    edges = build_pass_network_edges(
        events,
        team_name,
        period_label,
    )

    passes = extract_team_passes(
        events,
        team_name,
        period_label,
    )

    if passes.empty:
        return {
            "Team": team_name,
            "Period": period_label,
            "Passes Attempted": 0,
            "Passes Completed": 0,
            "Pass Completion %": 0.0,
            "Most Involved Player": None,
            "Top Passer": None,
            "Top Receiver": None,
            "Strongest Link": None,
            "Strongest Link Passes": 0,
            "Average Network X": np.nan,
            "Nodes": nodes,
            "Edges": edges,
        }

    attempted = int(
        len(
            passes
        )
    )

    completed = int(
        passes[
            "Completed"
        ].sum()
    )

    completion = (
        completed
        / attempted
        * 100.0
        if attempted > 0
        else 0.0
    )

    most_involved = None
    top_passer = None
    top_receiver = None
    avg_network_x = np.nan

    if not nodes.empty:
        most_involved = nodes.iloc[
            0
        ][
            "Player"
        ]

        top_passer = (
            nodes.sort_values(
                "Passes Completed",
                ascending=False,
            )
            .iloc[
                0
            ][
                "Player"
            ]
        )

        top_receiver = (
            nodes.sort_values(
                "Passes Received",
                ascending=False,
            )
            .iloc[
                0
            ][
                "Player"
            ]
        )

        avg_network_x = float(
            nodes[
                "Average X"
            ]
            .dropna()
            .mean()
        )

    strongest_link = None
    strongest_link_passes = 0

    if not edges.empty:
        top_edge = edges.iloc[
            0
        ]

        strongest_link = (
            f"{top_edge['Passer']} → "
            f"{top_edge['Recipient']}"
        )

        strongest_link_passes = int(
            top_edge[
                "Pass Count"
            ]
        )

    return {
        "Team":
            team_name,

        "Period":
            period_label,

        "Passes Attempted":
            attempted,

        "Passes Completed":
            completed,

        "Pass Completion %":
            completion,

        "Most Involved Player":
            most_involved,

        "Top Passer":
            top_passer,

        "Top Receiver":
            top_receiver,

        "Strongest Link":
            strongest_link,

        "Strongest Link Passes":
            strongest_link_passes,

        "Average Network X":
            avg_network_x,

        "Nodes":
            nodes,

        "Edges":
            edges,
    }


# ---------------------------------------------------------
# Half-by-half build-up comparison
# ---------------------------------------------------------

def compare_build_up_halves(
    events: pd.DataFrame,
    team_name: str,
) -> Dict[str, object]:
    """
    Compare first-half and second-half network/build-up behaviour.
    """
    first = build_pass_network_summary(
        events,
        team_name,
        "First Half",
    )

    second = build_pass_network_summary(
        events,
        team_name,
        "Second Half",
    )

    avg_x_change = np.nan

    if (
        pd.notna(
            first[
                "Average Network X"
            ]
        )
        and pd.notna(
            second[
                "Average Network X"
            ]
        )
    ):
        avg_x_change = (
            second[
                "Average Network X"
            ]
            - first[
                "Average Network X"
            ]
        )

    return {
        "Team":
            team_name,

        "First Half Passes":
            first[
                "Passes Attempted"
            ],

        "Second Half Passes":
            second[
                "Passes Attempted"
            ],

        "Pass Volume Change":
            (
                second[
                    "Passes Attempted"
                ]
                - first[
                    "Passes Attempted"
                ]
            ),

        "First Half Completion %":
            first[
                "Pass Completion %"
            ],

        "Second Half Completion %":
            second[
                "Pass Completion %"
            ],

        "Completion Change":
            (
                second[
                    "Pass Completion %"
                ]
                - first[
                    "Pass Completion %"
                ]
            ),

        "First Half Network X":
            first[
                "Average Network X"
            ],

        "Second Half Network X":
            second[
                "Average Network X"
            ],

        "Network X Change":
            avg_x_change,

        "First Half Most Involved":
            first[
                "Most Involved Player"
            ],

        "Second Half Most Involved":
            second[
                "Most Involved Player"
            ],

        "First Half Strongest Link":
            first[
                "Strongest Link"
            ],

        "Second Half Strongest Link":
            second[
                "Strongest Link"
            ],
    }


# ---------------------------------------------------------
# Analyst insight generation
# ---------------------------------------------------------

def build_pass_network_insights(
    summary: Dict[str, object],
) -> List[str]:
    """
    Convert pass-network summary into concise analyst-style insights.
    """
    insights = []

    team = summary[
        "Team"
    ]

    completion = summary[
        "Pass Completion %"
    ]

    top_passer = summary[
        "Top Passer"
    ]

    top_receiver = summary[
        "Top Receiver"
    ]

    most_involved = summary[
        "Most Involved Player"
    ]

    strongest_link = summary[
        "Strongest Link"
    ]

    strongest_link_passes = summary[
        "Strongest Link Passes"
    ]

    avg_x = summary[
        "Average Network X"
    ]

    insights.append(
        f"{team} completed "
        f"{summary['Passes Completed']} of "
        f"{summary['Passes Attempted']} passes "
        f"({completion:.1f}%)."
    )

    if most_involved:
        insights.append(
            f"{most_involved} had the highest overall "
            f"pass-network involvement for {team}."
        )

    if top_passer:
        insights.append(
            f"{top_passer} completed the most passes."
        )

    if top_receiver:
        insights.append(
            f"{top_receiver} received the most completed passes."
        )

    if strongest_link:
        insights.append(
            f"The strongest passing connection was "
            f"{strongest_link} "
            f"({strongest_link_passes} completed passes)."
        )

    if pd.notna(
        avg_x
    ):
        insights.append(
            f"The team's average pass-event position was "
            f"{avg_x:.1f} on the StatsBomb X-axis."
        )

    return insights


def build_build_up_change_insights(
    comparison: Dict[str, object],
) -> List[str]:
    """
    Generate half-by-half build-up change insights.
    """
    insights = []

    team = comparison[
        "Team"
    ]

    pass_change = comparison[
        "Pass Volume Change"
    ]

    completion_change = comparison[
        "Completion Change"
    ]

    network_x_change = comparison[
        "Network X Change"
    ]

    if pass_change > 0:
        insights.append(
            f"{team} attempted {pass_change:+d} more passes "
            f"in the second half."
        )
    elif pass_change < 0:
        insights.append(
            f"{team} attempted {abs(pass_change)} fewer passes "
            f"in the second half."
        )

    if abs(
        completion_change
    ) >= 2.0:
        direction = (
            "improved"
            if completion_change > 0
            else "declined"
        )

        insights.append(
            f"{team}'s pass completion {direction} by "
            f"{abs(completion_change):.1f} percentage points."
        )

    if pd.notna(
        network_x_change
    ) and abs(
        network_x_change
    ) >= 3.0:
        if network_x_change > 0:
            insights.append(
                f"{team}'s average pass-network position moved "
                f"{network_x_change:.1f} X units higher "
                f"in the second half."
            )
        else:
            insights.append(
                f"{team}'s average pass-network position moved "
                f"{abs(network_x_change):.1f} X units deeper "
                f"in the second half."
            )

    first_hub = comparison[
        "First Half Most Involved"
    ]

    second_hub = comparison[
        "Second Half Most Involved"
    ]

    if (
        first_hub
        and second_hub
        and first_hub != second_hub
    ):
        insights.append(
            f"The most involved passing hub changed from "
            f"{first_hub} in the first half to "
            f"{second_hub} in the second."
        )

    return insights



# ---------------------------------------------------------
# Pass-network visualisation
# ---------------------------------------------------------

def _short_player_label(player_name: str) -> str:
    """
    Produce compact, recognisable football labels for network plots.

    Examples:
        Rodrigo Javier De Paul -> De Paul
        Alexis Mac Allister -> Mac Allister
        Nicolás Hernán Otamendi -> Otamendi
        Cristian Gabriel Romero -> Romero
        Lionel Andrés Messi Cuccittini -> Messi
    """
    name = str(player_name or "").strip()

    if not name:
        return ""

    parts = [
        part
        for part in name.split()
        if part
    ]

    if len(parts) == 1:
        return parts[0]

    lower_parts = [
        part.casefold()
        for part in parts
    ]

    # Multi-word surnames frequently seen in football.
    surname_particles = {
        "de",
        "del",
        "da",
        "dos",
        "van",
        "von",
        "di",
        "du",
        "le",
        "la",
    }

    # If a surname particle occurs near the end, preserve it with the next token.
    for idx in range(
        len(parts) - 2,
        0,
        -1,
    ):
        if lower_parts[idx] in surname_particles:
            return " ".join(
                parts[idx:idx + 2]
            )

    # Known structure for long legal names:
    # first + middle + common football surname + legal surname
    if len(parts) >= 4:
        return parts[-2]

    # Three-part names often use the final surname, except when the
    # second token is a surname particle.
    if len(parts) == 3:
        if lower_parts[1] in surname_particles:
            return " ".join(
                parts[1:]
            )
        return parts[-1]

    return parts[-1]


def _draw_statsbomb_pitch(ax) -> None:
    ax.set_xlim(0, 120)
    ax.set_ylim(80, 0)

    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], linewidth=1.4)
    ax.plot([60, 60], [0, 80], linewidth=1.0)

    ax.plot([0, 18, 18, 0], [18, 18, 62, 62], linewidth=1.0)
    ax.plot([120, 102, 102, 120], [18, 18, 62, 62], linewidth=1.0)

    ax.plot([0, 6, 6, 0], [30, 30, 50, 50], linewidth=1.0)
    ax.plot([120, 114, 114, 120], [30, 30, 50, 50], linewidth=1.0)

    ax.add_patch(
        plt.Circle((60, 40), 10, fill=False, linewidth=1.0)
    )

    ax.scatter([60, 12, 108], [40, 40, 40], s=10, zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def build_pass_network_figure(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
    min_connection_passes: int = 4,
    max_connections: int = 24,
    title: Optional[str] = None,
):
    """
    Build a clean analyst-style single-team passing network.

    Visual encoding:
        - node position = average pass-event location
        - node size = network involvement
        - line width = completed pass volume
        - only the strongest connections are displayed

    Important:
        Positions are average event locations, not tracking positions.
    """
    nodes = build_pass_network_nodes(
        events,
        team_name,
        period_label,
    )

    edges = build_pass_network_edges(
        events,
        team_name,
        period_label,
    )

    fig, ax = plt.subplots(
        figsize=(11.5, 7.2)
    )

    _draw_statsbomb_pitch(
        ax
    )

    if nodes.empty:
        ax.text(
            60,
            40,
            "No pass-network data available",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
        )

        ax.set_title(
            title or f"{team_name} — Passing Network",
            fontsize=16,
            weight="bold",
            pad=18,
        )

        return fig

    node_positions = {
        str(row["Player"]): (
            float(row["Average X"]),
            float(row["Average Y"]),
        )
        for _, row in nodes.dropna(
            subset=[
                "Average X",
                "Average Y",
            ]
        ).iterrows()
    }

    visual_edges = edges.copy()

    if not visual_edges.empty:
        visual_edges[
            "Pass Count"
        ] = pd.to_numeric(
            visual_edges[
                "Pass Count"
            ],
            errors="coerce",
        ).fillna(0)

        # Remove low-value links first.
        visual_edges = visual_edges[
            visual_edges[
                "Pass Count"
            ] >= max(
                1,
                int(
                    min_connection_passes
                ),
            )
        ].copy()

        # Keep the strongest links only, avoiding spaghetti-like plots.
        visual_edges = (
            visual_edges
            .sort_values(
                "Pass Count",
                ascending=False,
            )
            .head(
                max(
                    1,
                    int(
                        max_connections
                    ),
                )
            )
            .copy()
        )

    max_edge = (
        float(
            visual_edges[
                "Pass Count"
            ].max()
        )
        if not visual_edges.empty
        else 1.0
    )

    min_edge = (
        float(
            visual_edges[
                "Pass Count"
            ].min()
        )
        if not visual_edges.empty
        else 1.0
    )

    # Draw important connections first.
    for _, edge in visual_edges.iterrows():
        passer = str(
            edge[
                "Passer"
            ]
        )
        recipient = str(
            edge[
                "Recipient"
            ]
        )

        if (
            passer not in node_positions
            or recipient not in node_positions
        ):
            continue

        x1, y1 = node_positions[
            passer
        ]
        x2, y2 = node_positions[
            recipient
        ]

        count = float(
            edge[
                "Pass Count"
            ]
        )

        if max_edge > min_edge:
            relative_strength = (
                count
                - min_edge
            ) / (
                max_edge
                - min_edge
            )
        else:
            relative_strength = 1.0

        linewidth = (
            1.0
            + relative_strength
            * 4.5
        )

        alpha = (
            0.28
            + relative_strength
            * 0.42
        )

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=linewidth,
            alpha=alpha,
            zorder=2,
        )

    involvement = pd.to_numeric(
        nodes[
            "Network Involvement"
        ],
        errors="coerce",
    ).fillna(0)

    max_involvement = max(
        float(
            involvement.max()
        ),
        1.0,
    )

    sizes = (
        260
        + (
            involvement
            / max_involvement
        )
        * 1150
    )

    ax.scatter(
        pd.to_numeric(
            nodes[
                "Average X"
            ],
            errors="coerce",
        ),
        pd.to_numeric(
            nodes[
                "Average Y"
            ],
            errors="coerce",
        ),
        s=sizes,
        alpha=0.95,
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
    )

    # Identify the network hub so its label can be emphasised.
    most_involved_player = None

    if not nodes.empty:
        top_idx = involvement.idxmax()
        most_involved_player = str(
            nodes.loc[
                top_idx,
                "Player",
            ]
        )

    for _, row in nodes.iterrows():
        if (
            pd.isna(
                row[
                    "Average X"
                ]
            )
            or pd.isna(
                row[
                    "Average Y"
                ]
            )
        ):
            continue

        player_name = str(
            row[
                "Player"
            ]
        )

        ax.text(
            float(
                row[
                    "Average X"
                ]
            ),
            float(
                row[
                    "Average Y"
                ]
            ),
            _short_player_label(
                player_name
            ),
            ha="center",
            va="center",
            fontsize=(
                9.2
                if player_name
                == most_involved_player
                else 8.2
            ),
            weight=(
                "bold"
                if player_name
                == most_involved_player
                else "semibold"
            ),
            zorder=5,
        )

    summary = build_pass_network_summary(
        events,
        team_name,
        period_label,
    )

    strongest_link = (
        summary.get(
            "Strongest Link"
        )
        or "Unavailable"
    )

    subtitle = (
        f"{period_label} | "
        f"{summary.get('Passes Completed', 0)} / "
        f"{summary.get('Passes Attempted', 0)} completed passes | "
        f"{summary.get('Pass Completion %', 0.0):.1f}% completion"
    )

    ax.set_title(
        title
        or f"{team_name} — Passing Network",
        fontsize=17,
        weight="bold",
        pad=30,
    )

    ax.text(
        0.5,
        1.018,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax.text(
        0.5,
        0.987,
        f"Strongest link: {strongest_link}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    ax.text(
        0.0,
        -0.047,
        (
            f"Showing up to {max_connections} strongest links "
            f"(minimum {min_connection_passes} completed passes)  •  "
            "Node size = network involvement  •  "
            "Line width = completed pass volume  •  "
            "Positions are average event locations"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
    )

    fig.tight_layout()

    return fig


def build_pass_network_report_payload(
    events: pd.DataFrame,
    team_name: str,
    period_label: str = "Full Match",
) -> Dict[str, object]:
    """
    Package pass-network analysis for dashboard and report generators.
    """
    summary = build_pass_network_summary(
        events,
        team_name,
        period_label,
    )

    half_comparison = compare_build_up_halves(
        events,
        team_name,
    )

    return {
        "team": team_name,
        "period": period_label,
        "summary": {
            key: value
            for key, value in summary.items()
            if key not in {"Nodes", "Edges"}
        },
        "nodes": summary["Nodes"].copy(),
        "edges": summary["Edges"].copy(),
        "insights": build_pass_network_insights(summary),
        "half_comparison": half_comparison,
        "half_insights": build_build_up_change_insights(
            half_comparison
        ),
    }


# ---------------------------------------------------------
# Public interface
# ---------------------------------------------------------

def analyze_pass_networks(
    events: pd.DataFrame,
    team_names: Optional[List[str]] = None,
    period_label: str = "Full Match",
) -> Dict[str, object]:
    """
    Run pass-network analysis for one or more teams.

    Returns:
        {
            "summaries": DataFrame,
            "nodes": DataFrame,
            "edges": DataFrame,
            "half_comparisons": DataFrame,
            "insights": DataFrame,
        }
    """

    if events is None or events.empty:
        return {
            "summaries":
                pd.DataFrame(),

            "nodes":
                pd.DataFrame(),

            "edges":
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
    node_frames = []
    edge_frames = []
    half_rows = []
    insight_rows = []

    for team_name in team_names:
        summary = build_pass_network_summary(
            events,
            team_name,
            period_label,
        )

        summary_rows.append(
            {
                key: value
                for key, value in summary.items()
                if key not in {
                    "Nodes",
                    "Edges",
                }
            }
        )

        if not summary[
            "Nodes"
        ].empty:
            nodes = summary[
                "Nodes"
            ].copy()

            nodes[
                "Period"
            ] = period_label

            node_frames.append(
                nodes
            )

        if not summary[
            "Edges"
        ].empty:
            edges = summary[
                "Edges"
            ].copy()

            edges[
                "Period"
            ] = period_label

            edge_frames.append(
                edges
            )

        half_comparison = compare_build_up_halves(
            events,
            team_name,
        )

        half_rows.append(
            half_comparison
        )

        for text in build_pass_network_insights(
            summary
        ):
            insight_rows.append(
                {
                    "Team":
                        team_name,

                    "Type":
                        "Pass Network",

                    "Message":
                        text,
                }
            )

        for text in build_build_up_change_insights(
            half_comparison
        ):
            insight_rows.append(
                {
                    "Team":
                        team_name,

                    "Type":
                        "Build-up Change",

                    "Message":
                        text,
                }
            )

    summaries_df = pd.DataFrame(
        summary_rows
    )

    nodes_df = (
        pd.concat(
            node_frames,
            ignore_index=True,
        )
        if node_frames
        else pd.DataFrame()
    )

    edges_df = (
        pd.concat(
            edge_frames,
            ignore_index=True,
        )
        if edge_frames
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

        "nodes":
            nodes_df,

        "edges":
            edges_df,

        "half_comparisons":
            half_df,

        "insights":
            insights_df,
    }
