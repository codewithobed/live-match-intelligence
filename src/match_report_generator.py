"""
Professional Match Intelligence Report Generator
for LiveMatch Intelligence.

Purpose:
    Generate a professional executive PNG and multi-page PDF containing the most important outputs from
    LiveMatch Intelligence using actual match data and the project's existing
    analytics modules.

The report includes:
    - Match score and headline metrics
    - xG and shots
    - Passing / possession-style metrics
    - Territory intelligence
    - Progressive actions
    - Tactical-change signals
    - Pass-network highlights
    - Top progression players
    - Experimental ML prediction when available
    - Analyst summary

Output example:
    reports/match_intelligence_report_3895309.png

Run directly:
    python -m src.match_report_generator

Important:
    This report is generated programmatically from the match data.
    It is not an AI-generated visual and the statistics are reproducible.
"""

from __future__ import annotations
import re
import unicodedata

from pathlib import Path
from typing import Dict, List, Optional

import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image

from src.match_analyzer import load_match_events, build_match_summary
from src.score_utils import match_score_breakdown, exclude_shootout_events
from src.possession_territory_analyzer import analyze_possession_territory
from src.progressive_actions_analyzer import analyze_progressive_actions
from src.pass_network_analyzer import analyze_pass_networks
from src.tactical_change_detector import detect_tactical_changes
from src.image_fetcher import get_team_badge

try:
    from src.match_outcome_predictor import predict_match_outcome
except Exception:
    predict_match_outcome = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "reports"

TEAM_IMAGE_DIR = (
    PROJECT_ROOT
    / "dashboard"
    / "assets"
    / "teams"
)

DEFAULT_MATCH_ID = 3895309

TEAM_LOGOS = {
    "Borussia Dortmund":
        TEAM_IMAGE_DIR
        / "borussia_dortmund.png",

    "Bayer Leverkusen":
        TEAM_IMAGE_DIR
        / "bayer_leverkusen.png",
}



def _team_logo_path(team_name: str) -> Optional[Path]:
    """
    Resolve a badge dynamically for any supported team.

    Falls back to the original local TEAM_LOGOS mapping if the automatic
    image resolver cannot return a usable file.
    """
    try:
        badge = get_team_badge(str(team_name).strip())
        if badge is not None:
            badge = Path(badge)
            if badge.exists():
                return badge
    except Exception:
        pass

    fallback = TEAM_LOGOS.get(str(team_name))
    if fallback is not None and Path(fallback).exists():
        return Path(fallback)

    return None


# ---------------------------------------------------------
# Basic event helpers
# ---------------------------------------------------------

def _normalise_name(value):
    if isinstance(value, dict):
        return str(value.get("name", ""))

    if pd.isna(value):
        return ""

    return str(value)


def _event_types(events):
    if "type" not in events.columns:
        return pd.Series("", index=events.index)

    return events["type"].apply(
        _normalise_name
    )


def _event_teams(events):
    if "team" not in events.columns:
        return pd.Series("", index=events.index)

    return events["team"].apply(
        _normalise_name
    )


def _team_match_metrics(
    events: pd.DataFrame,
    team_name: str,
) -> Dict[str, float]:
    teams = _event_teams(
        events
    )

    types = _event_types(
        events
    )

    team_mask = (
        teams == team_name
    )

    team_events = events.loc[
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

    passes = int(
        (
            team_types == "Pass"
        ).sum()
    )

    carries = int(
        (
            team_types == "Carry"
        ).sum()
    )

    pressures = int(
        (
            team_types == "Pressure"
        ).sum()
    )

    recoveries = int(
        (
            team_types
            == "Ball Recovery"
        ).sum()
    )

    interceptions = int(
        (
            team_types
            == "Interception"
        ).sum()
    )

    goals = 0

    if (
        "shot_outcome"
        in team_events.columns
    ):
        shot_rows = team_events[
            _event_types(
                team_events
            )
            == "Shot"
        ]

        outcomes = shot_rows[
            "shot_outcome"
        ].apply(
            _normalise_name
        )

        goals = int(
            outcomes.eq(
                "Goal"
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

    completed_passes = 0

    if passes > 0:
        pass_rows = team_events[
            _event_types(
                team_events
            )
            == "Pass"
        ].copy()

        if (
            "pass_outcome"
            in pass_rows.columns
        ):
            pass_outcomes = pass_rows[
                "pass_outcome"
            ].apply(
                _normalise_name
            )

            completed_passes = int(
                pass_outcomes.eq(
                    ""
                ).sum()
            )
        else:
            completed_passes = passes

    pass_completion = (
        completed_passes
        / passes
        * 100.0
        if passes > 0
        else 0.0
    )

    return {
        "Goals":
            goals,

        "Shots":
            shots,

        "xG":
            xg,

        "Passes":
            passes,

        "Pass Completion %":
            pass_completion,

        "Carries":
            carries,

        "Pressures":
            pressures,

        "Recoveries":
            recoveries,

        "Interceptions":
            interceptions,
    }


# ---------------------------------------------------------
# Report data assembly
# ---------------------------------------------------------

def build_report_data(
    match_id: int = DEFAULT_MATCH_ID,
) -> Dict[str, object]:
    raw_events = load_match_events(
        match_id
    )

    events = exclude_shootout_events(
        raw_events
    )

    match_summary = build_match_summary(
        events
    )

    team_names = (
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

    if len(team_names) < 2:
        raise RuntimeError(
            "Could not identify both teams."
        )

    team_1 = str(
        team_names[0]
    )

    team_2 = str(
        team_names[1]
    )

    team_1_metrics = _team_match_metrics(
        events,
        team_1,
    )

    team_2_metrics = _team_match_metrics(
        events,
        team_2,
    )

    territory = analyze_possession_territory(
        events,
        team_names=[
            team_1,
            team_2,
        ],
        period_label="Full Match",
    )

    progression = analyze_progressive_actions(
        events,
        team_names=[
            team_1,
            team_2,
        ],
        period_label="Full Match",
    )

    pass_network = analyze_pass_networks(
        events,
        team_names=[
            team_1,
            team_2,
        ],
        period_label="Full Match",
    )

    tactical = detect_tactical_changes(
        events,
        team_names=[
            team_1,
            team_2,
        ],
    )

    return {
        "match_id":
            match_id,

        "events":
            events,

        "raw_events":
            raw_events,

        "match_summary":
            match_summary,

        "team_1":
            team_1,

        "team_2":
            team_2,

        "team_1_metrics":
            team_1_metrics,

        "team_2_metrics":
            team_2_metrics,

        "territory":
            territory,

        "progression":
            progression,

        "pass_network":
            pass_network,

        "tactical":
            tactical,
    }


# ---------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------

def _add_card(
    ax,
    x,
    y,
    w,
    h,
    title,
    value,
    subtitle="",
):
    card = FancyBboxPatch(
        (
            x,
            y,
        ),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1,
        fill=False,
    )

    ax.add_patch(
        card
    )

    ax.text(
        x + 0.02,
        y + h - 0.03,
        title,
        fontsize=10,
        va="top",
    )

    ax.text(
        x + 0.02,
        y + h * 0.47,
        value,
        fontsize=20,
        fontweight="bold",
        va="center",
    )

    if subtitle:
        ax.text(
            x + 0.02,
            y + 0.025,
            subtitle,
            fontsize=8,
            va="bottom",
        )


def _add_logo(
    fig,
    logo_path: Path,
    left,
    bottom,
    width,
    height,
):
    if (
        not logo_path
        or not logo_path.exists()
    ):
        return

    try:
        img = Image.open(
            logo_path
        ).convert(
            "RGBA"
        )

        logo_ax = fig.add_axes(
            [
                left,
                bottom,
                width,
                height,
            ]
        )

        logo_ax.imshow(
            img
        )

        logo_ax.axis(
            "off"
        )

    except Exception:
        return


def _wrap(
    text,
    width=60,
):
    return "\n".join(
        textwrap.wrap(
            str(
                text
            ),
            width=width,
        )
    )



# ---------------------------------------------------------
# Report chart helpers
# ---------------------------------------------------------

def _extract_shots(
    events: pd.DataFrame,
    team_name: str,
) -> pd.DataFrame:
    if (
        "type" not in events.columns
        or "team" not in events.columns
        or "location" not in events.columns
    ):
        return pd.DataFrame()

    event_types = _event_types(events)
    teams = _event_teams(events)

    shots = events.loc[
        (event_types == "Shot")
        & (teams == team_name)
    ].copy()

    if shots.empty:
        return shots

    def xy(value):
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
        ):
            try:
                return float(value[0]), float(value[1])
            except Exception:
                return np.nan, np.nan

        return np.nan, np.nan

    coords = shots["location"].apply(xy)

    shots["X"] = [
        item[0]
        for item in coords
    ]

    shots["Y"] = [
        item[1]
        for item in coords
    ]

    shots["xG"] = (
        pd.to_numeric(
            shots.get(
                "shot_statsbomb_xg",
                pd.Series(
                    0.0,
                    index=shots.index,
                ),
            ),
            errors="coerce",
        )
        .fillna(0.0)
    )

    if "shot_outcome" in shots.columns:
        shots["Outcome"] = (
            shots["shot_outcome"]
            .apply(_normalise_name)
        )
    else:
        shots["Outcome"] = ""

    return shots


def _plot_shot_map(
    fig,
    rect,
    events,
    team_1,
    team_2,
):
    ax_shot = fig.add_axes(rect)

    ax_shot.set_xlim(
        60,
        120,
    )

    ax_shot.set_ylim(
        80,
        0,
    )

    ax_shot.set_xticks([])
    ax_shot.set_yticks([])

    ax_shot.set_title(
        "Shot Map",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=7,
    )

    # Attacking-half pitch outline.
    ax_shot.plot(
        [60, 120, 120, 60, 60],
        [0, 0, 80, 80, 0],
        linewidth=1,
    )

    ax_shot.plot(
        [60, 60],
        [0, 80],
        linewidth=0.8,
    )

    ax_shot.plot(
        [102, 120, 120, 102, 102],
        [18, 18, 62, 62, 18],
        linewidth=0.8,
    )

    ax_shot.plot(
        [114, 120, 120, 114, 114],
        [30, 30, 50, 50, 30],
        linewidth=0.8,
    )

    # Penalty spot.
    ax_shot.scatter(
        [108],
        [40],
        s=10,
    )

    for team_name, marker in [
        (team_1, "o"),
        (team_2, "s"),
    ]:
        shots = _extract_shots(
            events,
            team_name,
        )

        if shots.empty:
            continue

        sizes = (
            30
            + shots["xG"]
            * 220
        )

        ax_shot.scatter(
            shots["X"],
            shots["Y"],
            s=sizes,
            alpha=0.60,
            marker=marker,
            label=team_name,
        )

        goals = shots[
            shots["Outcome"]
            == "Goal"
        ]

        if not goals.empty:
            ax_shot.scatter(
                goals["X"],
                goals["Y"],
                s=150,
                marker="*",
                linewidths=1.2,
            )

    ax_shot.legend(
        loc="lower left",
        fontsize=7,
        frameon=False,
    )

    ax_shot.text(
        0.99,
        0.02,
        "Marker size ∝ xG",
        transform=ax_shot.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )


def _plot_team_comparison(
    fig,
    rect,
    team_1,
    team_2,
    m1,
    m2,
    p1,
    p2,
):
    ax_cmp = fig.add_axes(rect)

    labels = [
        "Shots",
        "xG × 10",
        "Progressive\nActions",
        "Final-third\nEntries",
        "Box\nEntries",
    ]

    values_1 = [
        float(m1["Shots"]),
        float(m1["xG"]) * 10.0,
        float(p1["Progressive Actions"]),
        float(p1["Final Third Entries"]),
        float(p1["Box Entries"]),
    ]

    values_2 = [
        float(m2["Shots"]),
        float(m2["xG"]) * 10.0,
        float(p2["Progressive Actions"]),
        float(p2["Final Third Entries"]),
        float(p2["Box Entries"]),
    ]

    y = np.arange(
        len(labels)
    )

    h = 0.34

    ax_cmp.barh(
        y - h / 2,
        values_1,
        height=h,
        label=team_1,
    )

    ax_cmp.barh(
        y + h / 2,
        values_2,
        height=h,
        label=team_2,
    )

    ax_cmp.set_yticks(
        y,
        labels,
        fontsize=8,
    )

    ax_cmp.invert_yaxis()

    ax_cmp.set_title(
        "Team Performance Comparison",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=7,
    )

    ax_cmp.tick_params(
        axis="x",
        labelsize=7,
    )

    ax_cmp.legend(
        fontsize=7,
        frameon=False,
        loc="lower right",
    )

    ax_cmp.spines["top"].set_visible(False)
    ax_cmp.spines["right"].set_visible(False)


def _plot_territory_thirds(
    fig,
    rect,
    territory_df,
    team_1,
    team_2,
):
    ax_terr = fig.add_axes(rect)

    def get_row(team):
        return territory_df[
            territory_df["Team"] == team
        ].iloc[0]

    r1 = get_row(
        team_1
    )

    r2 = get_row(
        team_2
    )

    labels = [
        "Defensive",
        "Middle",
        "Attacking",
    ]

    vals_1 = np.array(
        [
            r1["Defensive Third Events"],
            r1["Middle Third Events"],
            r1["Attacking Third Events"],
        ],
        dtype=float,
    )

    vals_2 = np.array(
        [
            r2["Defensive Third Events"],
            r2["Middle Third Events"],
            r2["Attacking Third Events"],
        ],
        dtype=float,
    )

    if vals_1.sum() > 0:
        vals_1 = (
            vals_1
            / vals_1.sum()
            * 100.0
        )

    if vals_2.sum() > 0:
        vals_2 = (
            vals_2
            / vals_2.sum()
            * 100.0
        )

    y = np.arange(
        len(labels)
    )

    h = 0.34

    ax_terr.barh(
        y - h / 2,
        vals_1,
        height=h,
        label=team_1,
    )

    ax_terr.barh(
        y + h / 2,
        vals_2,
        height=h,
        label=team_2,
    )

    ax_terr.set_yticks(
        y,
        labels,
        fontsize=8,
    )

    ax_terr.invert_yaxis()

    ax_terr.set_xlim(
        0,
        60,
    )

    ax_terr.set_title(
        "Field-third Activity Share",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=7,
    )

    ax_terr.set_xlabel(
        "% of located events",
        fontsize=7,
    )

    ax_terr.tick_params(
        axis="x",
        labelsize=7,
    )

    ax_terr.legend(
        fontsize=7,
        frameon=False,
        loc="lower right",
    )

    ax_terr.spines["top"].set_visible(False)
    ax_terr.spines["right"].set_visible(False)



def _football_display_name(name, max_length: int = 24):
    """
    Convert long provider-style football names into a consistent display form.

    General rules:
    - preserve already concise names;
    - remove duplicate/repeated name tokens;
    - for long Iberian/Latin-style names, prefer first given name + first surname;
    - otherwise prefer first given name + final surname;
    - keep accents and original spelling;
    - truncate only as a final visual fallback.

    This is deliberately rule-based rather than player-specific, so it works
    when a different StatsBomb match is selected.
    """
    raw = " ".join(str(name or "").replace("_", " ").split()).strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return "N/A"

    # Remove exact repeated tokens while preserving order/case.
    parts = []
    seen = set()
    for token in raw.split():
        key = unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode().lower()
        key = re.sub(r"[^a-z0-9'-]", "", key)
        if key and key not in seen:
            parts.append(token)
            seen.add(key)

    if not parts:
        return raw

    cleaned = " ".join(parts)

    # Two-token names normally remain untouched. For unusually long
    # provider-style given names, use a compact familiar first-name form.
    if len(parts) == 2:
        first, last = parts
        if len(first) >= 11:
            compact_first = first[:5]
            return f"{compact_first} {last}"
        return cleaned

    # Three-token provider names usually contain an extra middle/given name;
    # present them as first given name + family name for a familiar football
    # display style (e.g. Cristian Gabriel Romero -> Cristian Romero).
    if len(parts) == 3:
        return f"{parts[0]} {parts[-1]}"

    # Single-token names remain unchanged.
    if len(parts) == 1:
        return cleaned

    particles = {
        "da", "de", "del", "della", "der", "di", "dos", "du",
        "la", "le", "van", "von",
    }

    # For 4+ token provider names, StatsBomb commonly supplies multiple given
    # names plus multiple family names. First + penultimate is usually the
    # familiar football display form (e.g. Lionel ... Messi Cuccittini).
    if len(parts) >= 4:
        surname_index = -2
        # Preserve a surname particle immediately before the chosen surname.
        if len(parts) >= 5 and parts[surname_index - 1].lower() in particles:
            candidate = f"{parts[0]} {parts[surname_index - 1]} {parts[surname_index]}"
        else:
            candidate = f"{parts[0]} {parts[surname_index]}"
    else:
        candidate = f"{parts[0]} {parts[-1]}" if len(parts) >= 2 else parts[0]

    if len(candidate) <= max_length:
        return candidate

    # Compact visual fallback without exposing provider-style long names.
    if len(parts) >= 2:
        candidate = f"{parts[0][0]}. {parts[-1]}"
    return candidate if len(candidate) <= max_length else candidate[: max_length - 1] + "…"


def _display_connection(connection):
    """Standardize names in a passing-link string such as 'A -> B'."""
    raw = str(connection or "").strip()
    for sep in (" → ", " -> ", "→", "->"):
        if sep in raw:
            left, right = raw.split(sep, 1)
            return f"{_football_display_name(left)} → {_football_display_name(right)}"
    return _football_display_name(raw)


def _plot_top_progressors(
    fig,
    rect,
    progression_players,
):
    ax_prog = fig.add_axes(rect)

    top_players = (
        progression_players
        .sort_values(
            "Progressive Actions",
            ascending=False,
        )
        .head(6)
        .sort_values(
            "Progressive Actions",
            ascending=True,
        )
    )

    labels = (
        top_players["Player"]
        .astype(str)
        .apply(
            _football_display_name
        )
    )

    ax_prog.barh(
        labels,
        top_players[
            "Progressive Actions"
        ],
    )

    ax_prog.set_title(
        "Top Progressive Players",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=7,
    )

    ax_prog.set_xlabel(
        "Progressive actions",
        fontsize=7,
    )

    ax_prog.tick_params(
        axis="both",
        labelsize=7,
    )

    ax_prog.spines["top"].set_visible(False)
    ax_prog.spines["right"].set_visible(False)

# ---------------------------------------------------------
# Report generation
# ---------------------------------------------------------

def generate_match_report(
    match_id: int = DEFAULT_MATCH_ID,
    output_path: Optional[Path] = None,
) -> Path:

    data = build_report_data(
        match_id
    )

    team_1 = data["team_1"]
    team_2 = data["team_2"]
    events = data["events"]
    m1 = data["team_1_metrics"]
    m2 = data["team_2_metrics"]

    territory_df = data["territory"]["summaries"]
    progression_df = data["progression"]["summaries"]
    progression_players = data["progression"]["players"]
    pass_df = data["pass_network"]["summaries"]
    tactical_alerts = data["tactical"]["alerts"]

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path is None:
        output_path = (
            REPORTS_DIR
            / f"match_intelligence_report_{match_id}.png"
        )

    t1 = territory_df[
        territory_df["Team"] == team_1
    ].iloc[0]

    t2 = territory_df[
        territory_df["Team"] == team_2
    ].iloc[0]

    p1 = progression_df[
        progression_df["Team"] == team_1
    ].iloc[0]

    p2 = progression_df[
        progression_df["Team"] == team_2
    ].iloc[0]

    pass_1 = pass_df[
        pass_df["Team"] == team_1
    ].iloc[0]

    pass_2 = pass_df[
        pass_df["Team"] == team_2
    ].iloc[0]

    fig = plt.figure(
        figsize=(
            16,
            9,
        )
    )

    ax = fig.add_axes(
        [
            0,
            0,
            1,
            1,
        ]
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.axis(
        "off"
    )

    def add_kpi_card(
        x,
        y,
        w,
        h,
        title,
        value,
        subtitle="",
    ):
        card = FancyBboxPatch(
            (
                x,
                y,
            ),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1,
            fill=False,
        )

        ax.add_patch(card)

        ax.text(
            x + 0.012,
            y + h - 0.018,
            title,
            fontsize=8.5,
            va="top",
        )

        ax.text(
            x + 0.012,
            y + h * 0.48,
            value,
            fontsize=17,
            fontweight="bold",
            va="center",
        )

        if subtitle:
            ax.text(
                x + 0.012,
                y + 0.014,
                subtitle,
                fontsize=7,
                va="bottom",
            )

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    ax.text(
        0.035,
        0.955,
        "LIVE MATCH INTELLIGENCE",
        fontsize=23,
        fontweight="bold",
        va="top",
    )

    ax.text(
        0.035,
        0.925,
        "One-Page Analyst Report",
        fontsize=10.5,
        va="top",
    )

    # Final professional team identity row:
    # [Argentina badge] Argentina   3 - 3   France [France badge]
    # The badge/name gaps mirror the approved visual reference.
    _add_logo(
        fig,
        _team_logo_path(team_1),
        0.285,
        0.858,
        0.040,
        0.040,
    )

    _add_logo(
        fig,
        _team_logo_path(team_2),
        0.690,
        0.858,
        0.040,
        0.040,
    )

    ax.text(
        0.330,
        0.881,
        team_1,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="center",
    )

    ax.text(
        0.5,
        0.881,
        f"{int(m1['Goals'])}  -  {int(m2['Goals'])}",
        fontsize=22,
        fontweight="bold",
        ha="center",
        va="center",
    )

    ax.text(
        0.670,
        0.881,
        team_2,
        fontsize=17,
        fontweight="bold",
        ha="right",
        va="center",
    )

    # -----------------------------------------------------
    # KPI cards
    # -----------------------------------------------------

    kpis = [
        (
            "xG",
            f"{m1['xG']:.2f} — {m2['xG']:.2f}",
            "Expected goals",
        ),
        (
            "Shots",
            f"{m1['Shots']} — {m2['Shots']}",
            "Total attempts",
        ),
        (
            "Pass Completion",
            f"{m1['Pass Completion %']:.1f}% — {m2['Pass Completion %']:.1f}%",
            "Completed pass rate",
        ),
        (
            "Pressures",
            f"{m1['Pressures']} — {m2['Pressures']}",
            "Pressure events",
        ),
        (
            "Progressive Actions",
            f"{int(p1['Progressive Actions'])} — {int(p2['Progressive Actions'])}",
            "Passes + carries",
        ),
        (
            "Territory Index",
            f"{t1['Territory Index']:.1f} — {t2['Territory Index']:.1f}",
            "Event-based index",
        ),
    ]

    kpi_y = 0.765
    kpi_w = 0.145
    kpi_h = 0.085
    kpi_gap = 0.012

    for i, (
        title,
        value,
        subtitle,
    ) in enumerate(kpis):
        add_kpi_card(
            0.035 + i * (kpi_w + kpi_gap),
            kpi_y,
            kpi_w,
            kpi_h,
            title,
            value,
            subtitle,
        )

    # -----------------------------------------------------
    # Visual panels
    # -----------------------------------------------------

    _plot_team_comparison(
        fig,
        [
            0.045,
            0.49,
            0.43,
            0.225,
        ],
        team_1,
        team_2,
        m1,
        m2,
        p1,
        p2,
    )

    _plot_shot_map(
        fig,
        [
            0.525,
            0.49,
            0.43,
            0.225,
        ],
        events,
        team_1,
        team_2,
    )

    _plot_territory_thirds(
        fig,
        [
            0.045,
            0.215,
            0.28,
            0.205,
        ],
        territory_df,
        team_1,
        team_2,
    )

    _plot_top_progressors(
        fig,
        [
            0.36,
            0.205,
            0.28,
            0.215,
        ],
        progression_players,
    )

    # -----------------------------------------------------
    # Tactical panel
    # -----------------------------------------------------

    tactical_box = FancyBboxPatch(
        (
            0.675,
            0.205,
        ),
        0.28,
        0.215,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1,
        fill=False,
    )

    ax.add_patch(
        tactical_box
    )

    ax.text(
        0.692,
        0.397,
        "Key Tactical Signals",
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    if tactical_alerts.empty:
        ax.text(
            0.692,
            0.355,
            "No major tactical-change thresholds were triggered.",
            fontsize=8,
            va="top",
        )
    else:
        y_alert = 0.35

        for _, row in tactical_alerts.head(
            2
        ).iterrows():
            message = _wrap(
                row.get(
                    "Message",
                    "",
                ),
                width=44,
            )

            ax.text(
                0.692,
                y_alert,
                f"• {message}",
                fontsize=7.6,
                va="top",
            )

            y_alert -= (
                0.075
                + 0.013
                * message.count("\n")
            )

    # -----------------------------------------------------
    # Passing highlights strip
    # -----------------------------------------------------

    passing_box = FancyBboxPatch(
        (
            0.045,
            0.115,
        ),
        0.91,
        0.070,
        boxstyle="round,pad=0.007,rounding_size=0.010",
        linewidth=1,
        fill=False,
    )

    ax.add_patch(
        passing_box
    )

    ax.text(
        0.06,
        0.169,
        "Passing & Build-up",
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )

    passing_text = (
        f"{team_1}: top passer {_football_display_name(pass_1['Top Passer'])}; "
        f"strongest link {_display_connection(pass_1['Strongest Link'])} "
        f"({int(pass_1['Strongest Link Passes'])} passes).\n"
        f"{team_2}: top passer {_football_display_name(pass_2['Top Passer'])}; "
        f"strongest link {_display_connection(pass_2['Strongest Link'])} "
        f"({int(pass_2['Strongest Link Passes'])} passes)."
    )

    ax.text(
        0.06,
        0.145,
        passing_text,
        fontsize=7.7,
        va="top",
        linespacing=1.25,
    )

    # -----------------------------------------------------
    # Analyst summary
    # -----------------------------------------------------

    territory_leader = (
        team_1
        if t1["Territory Index"] > t2["Territory Index"]
        else team_2
    )

    progression_leader = (
        team_1
        if p1["Progressive Actions"] > p2["Progressive Actions"]
        else team_2
    )

    xg_leader = (
        team_1
        if m1["xG"] > m2["xG"]
        else team_2
    )

    summary_box = FancyBboxPatch(
        (
            0.045,
            0.035,
        ),
        0.91,
        0.065,
        boxstyle="round,pad=0.007,rounding_size=0.010",
        linewidth=1,
        fill=False,
    )

    ax.add_patch(
        summary_box
    )

    ax.text(
        0.06,
        0.088,
        "Analyst Summary",
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )

    summary_text = (
        f"{territory_leader} showed the stronger event-based territory profile. "
        f"{progression_leader} produced more progressive actions, while "
        f"{xg_leader} generated the higher cumulative xG."
    )

    ax.text(
        0.06,
        0.062,
        _wrap(
            summary_text,
            width=145,
        ),
        fontsize=7.8,
        va="top",
        linespacing=1.18,
    )

    ax.text(
        0.045,
        0.015,
        (
            "Generated by LiveMatch Intelligence | StatsBomb event data | "
            "Event-derived indicators and transparent project heuristics"
        ),
        fontsize=7,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


# ---------------------------------------------------------
# Detailed PDF interpretation report
# ---------------------------------------------------------

def generate_pdf_report(
    match_id: int = DEFAULT_MATCH_ID,
    output_path: Optional[Path] = None,
    png_report_path: Optional[Path] = None,
    predictive_minute: int = 60,
    live_snapshot_minute: Optional[int] = None,
) -> Path:
    """
    Create the professional fixed-layout Full Match Intelligence PDF.

    Every page is written directly to an A4 landscape canvas. This guarantees
    consistent page orientation and avoids Platypus overflow/rotation issues.

    Page sequence
    -------------
    1. Executive Match Overview
    2. Shot Analysis
    3. Passing & Build-up
    4. Possession & Territory
    5. Progressive Actions
    6. Tactical Analysis
    7. Live Intelligence
    8. Predictive Intelligence
    9. Analyst Conclusions & Coaching Takeaways
    10. Methodology & Interpretation Boundaries
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    from src.section_report_generator import generate_section_png

    data = build_report_data(match_id)

    team_1 = data["team_1"]
    team_2 = data["team_2"]
    events = data["events"]
    raw_events = data.get("raw_events", events)
    m1 = data["team_1_metrics"]
    m2 = data["team_2_metrics"]

    territory_df = data["territory"]["summaries"]
    progression_df = data["progression"]["summaries"]
    progression_players = data["progression"]["players"]
    pass_df = data["pass_network"]["summaries"]
    tactical_alerts = data["tactical"]["alerts"]

    t1 = territory_df[territory_df["Team"] == team_1].iloc[0]
    t2 = territory_df[territory_df["Team"] == team_2].iloc[0]
    p1 = progression_df[progression_df["Team"] == team_1].iloc[0]
    p2 = progression_df[progression_df["Team"] == team_2].iloc[0]
    pass_1 = pass_df[pass_df["Team"] == team_1].iloc[0]
    pass_2 = pass_df[pass_df["Team"] == team_2].iloc[0]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = REPORTS_DIR / f"match_intelligence_report_{match_id}.pdf"

    if png_report_path is None:
        png_report_path = REPORTS_DIR / f"match_intelligence_report_{match_id}.png"

    # Always regenerate executive overview using the newest badge/layout logic.
    png_report_path = generate_match_report(
        match_id=match_id,
        output_path=Path(png_report_path),
    )

    if live_snapshot_minute is None:
        if "minute" in events.columns:
            live_snapshot_minute = int(
                pd.to_numeric(
                    events["minute"],
                    errors="coerce",
                ).fillna(0).max()
            )
        else:
            live_snapshot_minute = 90

    live_snapshot_minute = max(
        1,
        int(live_snapshot_minute or 90),
    )

    predictive_minute = max(
        15,
        min(
            int(predictive_minute),
            85,
        ),
    )

    # -------------------------------------------------------------
    # Generate professional focused PNGs.
    # -------------------------------------------------------------
    focused_specs = [
        ("shot_analysis", {}),
        ("passing_network", {}),
        ("territory", {}),
        ("progression", {}),
        ("tactical", {}),
        (
            "live_intelligence",
            {"snapshot_minute": live_snapshot_minute},
        ),
        (
            "ml_prediction",
            {"snapshot_minute": predictive_minute},
        ),
    ]

    focused_paths = []

    for section_key, kwargs in focused_specs:
        section_path = generate_section_png(
            section_key,
            match_id,
            **kwargs,
        )
        focused_paths.append(
            Path(section_path)
        )

    # -------------------------------------------------------------
    # Build final two text-heavy pages as landscape PNGs.
    # Rendering them as PNG before PDF insertion guarantees the same
    # orientation and page geometry as every analytical page.
    # -------------------------------------------------------------
    summary_dir = REPORTS_DIR / "_full_report_pages"
    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    conclusion_png = (
        summary_dir
        / f"conclusions_{match_id}.png"
    )

    methodology_png = (
        summary_dir
        / f"methodology_{match_id}.png"
    )

    navy = "#0B2E63"
    blue = "#1F5FAF"
    pale = "#F8FBFF"
    dark = "#111111"
    grey = "#555555"

    def _add_page_badges(fig, ax):
        logo_1 = _team_logo_path(team_1)
        logo_2 = _team_logo_path(team_2)

        if logo_1:
            try:
                img = Image.open(logo_1).convert("RGBA")
                a = fig.add_axes([0.055, 0.835, 0.065, 0.065])
                a.imshow(img)
                a.axis("off")
            except Exception:
                pass

        if logo_2:
            try:
                img = Image.open(logo_2).convert("RGBA")
                a = fig.add_axes([0.880, 0.835, 0.065, 0.065])
                a.imshow(img)
                a.axis("off")
            except Exception:
                pass

        ax.text(
            0.125,
            0.865,
            team_1.upper(),
            fontsize=10,
            fontweight="bold",
            color=navy,
            va="center",
        )

        ax.text(
            0.875,
            0.865,
            team_2.upper(),
            fontsize=10,
            fontweight="bold",
            color=navy,
            ha="right",
            va="center",
        )

    # -------------------------------------------------------------
    # Page 9 - Conclusions
    # -------------------------------------------------------------
    fig = plt.figure(
        figsize=(16, 10),
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.925),
            0.94,
            0.045,
            boxstyle="round,pad=0,rounding_size=0.003",
            linewidth=0,
            facecolor=navy,
        )
    )

    ax.text(
        0.05,
        0.947,
        "LIVE MATCH INTELLIGENCE",
        color="white",
        fontsize=17,
        fontweight="bold",
        va="center",
    )

    ax.text(
        0.95,
        0.947,
        "FULL MATCH REPORT",
        color="white",
        fontsize=9,
        ha="right",
        va="center",
    )

    ax.text(
        0.5,
        0.895,
        "ANALYST CONCLUSIONS & COACHING TAKEAWAYS",
        fontsize=21,
        fontweight="bold",
        color=navy,
        ha="center",
        va="center",
    )

    ax.text(
        0.5,
        0.862,
        f"{team_1} {int(m1['Goals'])} - {int(m2['Goals'])} {team_2}",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
    )

    _add_page_badges(fig, ax)

    # Headline evidence table.
    ax.add_patch(
        FancyBboxPatch(
            (0.055, 0.600),
            0.89,
            0.205,
            boxstyle="round,pad=0.008,rounding_size=0.010",
            linewidth=1,
            edgecolor=blue,
            facecolor=pale,
        )
    )

    ax.text(
        0.075,
        0.775,
        "HEADLINE EVIDENCE",
        fontsize=12,
        fontweight="bold",
        color=navy,
    )

    rows = [
        ("Goals", int(m1["Goals"]), int(m2["Goals"])),
        ("xG", f"{float(m1['xG']):.2f}", f"{float(m2['xG']):.2f}"),
        ("Shots", int(m1["Shots"]), int(m2["Shots"])),
        ("Pass completion", f"{float(m1['Pass Completion %']):.1f}%", f"{float(m2['Pass Completion %']):.1f}%"),
        ("Territory Index", f"{float(t1['Territory Index']):.1f}", f"{float(t2['Territory Index']):.1f}"),
        ("Progressive Actions", int(p1["Progressive Actions"]), int(p2["Progressive Actions"])),
        ("Final Third Entries", int(p1["Final Third Entries"]), int(p2["Final Third Entries"])),
        ("Box Entries", int(p1["Box Entries"]), int(p2["Box Entries"])),
    ]

    y = 0.735
    for label, v1, v2 in rows:
        ax.text(0.10, y, label, fontsize=8.4, color=grey)
        ax.text(0.48, y, str(v1), fontsize=8.4, fontweight="bold", ha="right")
        ax.text(0.52, y, str(v2), fontsize=8.4, fontweight="bold", ha="left")
        y -= 0.019

    territory_leader = (
        team_1
        if float(t1["Territory Index"]) > float(t2["Territory Index"])
        else team_2
    )

    progression_leader = (
        team_1
        if float(p1["Progressive Actions"]) > float(p2["Progressive Actions"])
        else team_2
    )

    xg_leader = (
        team_1
        if float(m1["xG"]) > float(m2["xG"])
        else team_2
    )

    shot_leader = (
        team_1
        if int(m1["Shots"]) > int(m2["Shots"])
        else team_2
    )

    xg_gap = abs(
        float(m1["xG"])
        - float(m2["xG"])
    )

    prog_gap = abs(
        int(p1["Progressive Actions"])
        - int(p2["Progressive Actions"])
    )

    ax.text(
        0.055,
        0.555,
        "KEY ANALYST FINDINGS",
        fontsize=12,
        fontweight="bold",
        color=navy,
    )

    findings = [
        f"Chance quality: {xg_leader} generated the higher cumulative xG by approximately {xg_gap:.2f}.",
        f"Shot volume: {shot_leader} recorded the higher shot total ({max(int(m1['Shots']), int(m2['Shots']))} vs {min(int(m1['Shots']), int(m2['Shots']))}).",
        f"Territory: {territory_leader} recorded the stronger event-based territory profile.",
        f"Progression: {progression_leader} produced {prog_gap} more progressive actions under the project definition.",
        f"Passing hubs: {team_1} - {_football_display_name(pass_1['Top Passer'])}; {team_2} - {_football_display_name(pass_2['Top Passer'])}.",
    ]

    if not tactical_alerts.empty:
        first_tactical = str(
            tactical_alerts.iloc[0].get(
                "Message",
                "",
            )
        )
        if first_tactical:
            findings.append(
                f"Tactical review prompt: {first_tactical}"
            )

    y = 0.520
    for finding in findings[:6]:
        wrapped = textwrap.wrap(
            finding,
            width=120,
        )
        ax.text(
            0.075,
            y,
            "• " + wrapped[0],
            fontsize=8.6,
            va="top",
        )
        for extra_line in wrapped[1:]:
            y -= 0.022
            ax.text(
                0.090,
                y,
                extra_line,
                fontsize=8.6,
                va="top",
            )
        y -= 0.043

    ax.text(
        0.055,
        0.250,
        "PRIORITY VIDEO REVIEW",
        fontsize=12,
        fontweight="bold",
        color=navy,
    )

    review_items = [
        f"Review the sequences behind {xg_leader}'s higher xG and identify whether central access, transitions, set plays or rebounds drove the advantage.",
        f"Compare build-up structures around {_football_display_name(pass_1['Top Passer'])} and {_football_display_name(pass_2['Top Passer'])}.",
        f"Review why {progression_leader} produced the stronger progression profile and whether passing, carrying, game state or opponent behaviour explains it.",
        "Use Tactical Analysis and Live Intelligence timelines as timestamps/themes for targeted video review.",
    ]

    y = 0.215
    for item in review_items:
        wrapped = textwrap.wrap(
            item,
            width=128,
        )
        ax.text(
            0.075,
            y,
            "• " + wrapped[0],
            fontsize=8.4,
            va="top",
        )
        for extra_line in wrapped[1:]:
            y -= 0.021
            ax.text(
                0.090,
                y,
                extra_line,
                fontsize=8.4,
                va="top",
            )
        y -= 0.040

    ax.text(
        0.5,
        0.035,
        f"LiveMatch Intelligence | Match ID {match_id} | Evidence-led analyst summary",
        fontsize=7.5,
        color=grey,
        ha="center",
    )

    fig.savefig(
        conclusion_png,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -------------------------------------------------------------
    # Page 10 - Methodology
    # -------------------------------------------------------------
    fig = plt.figure(
        figsize=(16, 10),
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.925),
            0.94,
            0.045,
            boxstyle="round,pad=0,rounding_size=0.003",
            linewidth=0,
            facecolor=navy,
        )
    )

    ax.text(
        0.05,
        0.947,
        "LIVE MATCH INTELLIGENCE",
        color="white",
        fontsize=17,
        fontweight="bold",
        va="center",
    )

    ax.text(
        0.95,
        0.947,
        "METHODOLOGY",
        color="white",
        fontsize=9,
        ha="right",
        va="center",
    )

    ax.text(
        0.5,
        0.885,
        "METHODOLOGY & INTERPRETATION BOUNDARIES",
        fontsize=21,
        fontweight="bold",
        color=navy,
        ha="center",
        va="center",
    )

    _add_page_badges(fig, ax)

    ax.text(
        0.075,
        0.785,
        (
            "This page separates event-derived analytical signals from claims that would require "
            "continuous tracking data, broader model validation, or direct coaching evidence."
        ),
        fontsize=9.5,
        color=dark,
        va="top",
    )

    ax.text(
        0.075,
        0.725,
        "INTERPRETATION BOUNDARIES",
        fontsize=12,
        fontweight="bold",
        color=navy,
    )

    boundaries = [
        "Event locations are not tracking data. Average positions and territory indicators describe where recorded actions occurred, not continuous player positions.",
        "Territory Index is a project-defined event indicator. It is not official possession or spatial-control tracking.",
        "Progressive actions use the project definition: completed passes or carries advancing the ball by the configured StatsBomb X-distance threshold.",
        "Live Intelligence is a transparent multi-signal heuristic. Intelligence Advantage is not a win probability.",
        "Predictive Intelligence is experimental. Outcome estimates are based on historical match-state snapshots and must be interpreted alongside validation accuracy, macro F1 and log loss.",
        "Tactical signals are analytical flags. Changes in event locations and event volumes do not confirm formation changes or coaching instructions.",
    ]

    y = 0.680
    for boundary in boundaries:
        wrapped = textwrap.wrap(
            boundary,
            width=130,
        )
        ax.text(
            0.090,
            y,
            "• " + wrapped[0],
            fontsize=9.0,
            va="top",
        )
        for extra_line in wrapped[1:]:
            y -= 0.025
            ax.text(
                0.105,
                y,
                extra_line,
                fontsize=9.0,
                va="top",
            )
        y -= 0.055

    ax.add_patch(
        FancyBboxPatch(
            (0.075, 0.095),
            0.85,
            0.105,
            boxstyle="round,pad=0.010,rounding_size=0.010",
            linewidth=1,
            edgecolor=blue,
            facecolor=pale,
        )
    )

    ax.text(
        0.10,
        0.165,
        "RECOMMENDED WORKFLOW",
        fontsize=11,
        fontweight="bold",
        color=navy,
    )

    workflow = (
        "Use the Executive Overview for rapid scanning, inspect the focused section pages for evidence, "
        "and validate important patterns with match video before making tactical or coaching conclusions."
    )

    ax.text(
        0.10,
        0.125,
        workflow,
        fontsize=9.0,
        va="top",
    )

    ax.text(
        0.5,
        0.035,
        (
            f"Generated by LiveMatch Intelligence | Match ID {match_id} | "
            "StatsBomb event data | Transparent project heuristics"
        ),
        fontsize=7.5,
        color=grey,
        ha="center",
    )

    fig.savefig(
        methodology_png,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -------------------------------------------------------------
    # Write fixed A4 landscape PDF pages directly.
    # -------------------------------------------------------------
    # Use explicit landscape dimensions and explicitly clear page rotation.
    # This prevents PDF viewers from interpreting a landscape page as a
    # portrait page with /Rotate metadata.
    portrait_w, portrait_h = A4
    page_size = (
        max(portrait_w, portrait_h),
        min(portrait_w, portrait_h),
    )
    page_w, page_h = page_size

    pdf = canvas.Canvas(
        str(output_path),
        pagesize=page_size,
        pageCompression=1,
    )
    pdf.setPageSize(page_size)
    pdf.setPageRotation(0)

    pdf.setTitle(
        f"LiveMatch Intelligence - {team_1} vs {team_2}"
    )
    pdf.setAuthor(
        "LiveMatch Intelligence"
    )
    pdf.setSubject(
        "Professional full-match football intelligence report"
    )

    all_pages = [
        Path(png_report_path),
        *focused_paths,
        conclusion_png,
        methodology_png,
    ]

    def _draw_landscape_page(image_path: Path, page_number: int):
        # Force every physical PDF page to true A4 landscape with zero rotation.
        pdf.setPageSize(page_size)
        pdf.setPageRotation(0)

        # Background
        pdf.setFillColor(colors.white)
        pdf.rect(
            0,
            0,
            page_w,
            page_h,
            fill=1,
            stroke=0,
        )

        with Image.open(image_path) as im:
            img_w, img_h = im.size

        margin_x = 8 * mm
        margin_y = 8 * mm

        max_w = (
            page_w
            - 2 * margin_x
        )
        max_h = (
            page_h
            - 2 * margin_y
        )

        scale = min(
            max_w / img_w,
            max_h / img_h,
        )

        draw_w = (
            img_w
            * scale
        )

        draw_h = (
            img_h
            * scale
        )

        draw_x = (
            page_w
            - draw_w
        ) / 2

        draw_y = (
            page_h
            - draw_h
        ) / 2

        pdf.drawImage(
            ImageReader(str(image_path)),
            draw_x,
            draw_y,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
            mask="auto",
        )

        # Small consistent page number.
        pdf.setFillColor(colors.HexColor("#6A7280"))
        pdf.setFont(
            "Helvetica",
            6.8,
        )
        pdf.drawRightString(
            page_w - 7 * mm,
            4 * mm,
            f"{page_number} / {len(all_pages)}",
        )

        # Keep the next page on the same true landscape geometry.
        pdf.setPageRotation(0)
        pdf.showPage()

    for page_number, image_path in enumerate(
        all_pages,
        start=1,
    ):
        _draw_landscape_page(
            Path(image_path),
            page_number,
        )

    pdf.save()

    print(
        "PDF page geometry:",
        f"{page_w:.1f} x {page_h:.1f} points",
        "| rotation: 0",
        "| expected: true A4 landscape",
    )

    return Path(
        output_path
    )

def main():
    output = generate_match_report(
        DEFAULT_MATCH_ID
    )

    print(
        "\nMATCH INTELLIGENCE REPORT CREATED"
    )

    print(
        output
    )


if __name__ == "__main__":
    main()
