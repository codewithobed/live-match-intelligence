"""
Live Intelligence Analyzer
LiveMatch Intelligence

Purpose
-------
Create a reusable match-command-centre layer that combines recent momentum,
score state, attacking threat, territory, progression and tactical signals.

The module is intentionally generic and works with any StatsBomb event match
that supplies the required event fields.

Important
---------
The composite "Intelligence Advantage" is a transparent analytical heuristic,
not a trained prediction model and not a claim about coaching intent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.image_fetcher import get_team_badge

from src.score_utils import exclude_shootout_events
from src.possession_territory_analyzer import analyze_possession_territory
from src.progressive_actions_analyzer import analyze_progressive_actions
from src.tactical_change_detector import detect_tactical_changes


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _normalise_name(value) -> str:
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


def _safe_numeric(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0.0)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(number) or np.isinf(number):
        return default
    return number


def _max_minute(events: pd.DataFrame) -> int:
    if events is None or events.empty or "minute" not in events.columns:
        return 0
    return int(_safe_numeric(events["minute"]).max())


def _team_names(events: pd.DataFrame) -> List[str]:
    return (
        _event_teams(events)
        .replace("", np.nan)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def events_until_minute(events_df: pd.DataFrame, minute: int) -> pd.DataFrame:
    if "minute" not in events_df.columns:
        return events_df.copy()
    minutes = _safe_numeric(events_df["minute"])
    return events_df.loc[minutes <= int(minute)].copy()


def events_in_window(
    events_df: pd.DataFrame,
    end_minute: int,
    window: int = 10,
) -> pd.DataFrame:
    if "minute" not in events_df.columns:
        return events_df.copy()

    minutes = _safe_numeric(events_df["minute"])
    start_minute = max(0, int(end_minute) - int(window))

    return events_df.loc[
        (minutes > start_minute)
        & (minutes <= int(end_minute))
    ].copy()


# -----------------------------------------------------------------------------
# Core live metrics and legacy-compatible momentum
# -----------------------------------------------------------------------------

def team_live_metrics(events_df: pd.DataFrame, team_name: str) -> Dict[str, float]:
    """Calculate core metrics for one team from a subset of events."""
    if events_df is None or events_df.empty:
        return {
            "Goals": 0,
            "Shots": 0,
            "xG": 0.0,
            "Passes": 0,
            "Pass Completion %": 0.0,
            "Pressures": 0,
            "Carries": 0,
            "Recoveries": 0,
            "Interceptions": 0,
        }

    teams = _event_teams(events_df)
    types = _event_types(events_df)
    team_mask = teams.eq(str(team_name))
    team_events = events_df.loc[team_mask].copy()
    team_types = types.loc[team_mask]

    passes = team_events.loc[team_types.eq("Pass")].copy()
    shots = team_events.loc[team_types.eq("Shot")].copy()

    total_xg = 0.0
    goals = 0

    if not shots.empty:
        if "shot_statsbomb_xg" in shots.columns:
            total_xg = float(_safe_numeric(shots["shot_statsbomb_xg"]).sum())

        if "shot_outcome" in shots.columns:
            outcomes = shots["shot_outcome"].apply(_normalise_name)
            periods = _safe_numeric(
                shots.get("period", pd.Series(0, index=shots.index))
            )
            goals = int((outcomes.eq("Goal") & periods.ne(5)).sum())

    completed_passes = 0
    if not passes.empty:
        if "pass_outcome" in passes.columns:
            outcomes = passes["pass_outcome"].apply(_normalise_name)
            completed_passes = int(outcomes.eq("").sum() + passes["pass_outcome"].isna().sum())
            # Avoid double-counting normalised empty values when the raw value is NaN.
            completed_passes = min(completed_passes, len(passes))
        else:
            completed_passes = len(passes)

    pass_completion = (
        completed_passes / len(passes) * 100.0
        if len(passes) > 0
        else 0.0
    )

    return {
        "Goals": int(goals),
        "Shots": int(team_types.eq("Shot").sum()),
        "xG": float(total_xg),
        "Passes": int(team_types.eq("Pass").sum()),
        "Pass Completion %": float(pass_completion),
        "Pressures": int(team_types.eq("Pressure").sum()),
        "Carries": int(team_types.eq("Carry").sum()),
        "Recoveries": int(team_types.eq("Ball Recovery").sum()),
        "Interceptions": int(team_types.eq("Interception").sum()),
    }


def momentum_points(metrics: Dict[str, float]) -> float:
    """
    Transparent MVP momentum heuristic retained from the dashboard logic.
    """
    return (
        _safe_float(metrics.get("Shots")) * 4.0
        + _safe_float(metrics.get("xG")) * 10.0
        + _safe_float(metrics.get("Pressures")) * 0.35
        + _safe_float(metrics.get("Carries")) * 0.08
        + _safe_float(metrics.get("Recoveries")) * 0.20
        + _safe_float(metrics.get("Passes")) * 0.03
    )


def relative_momentum_score(
    team_a_metrics: Dict[str, float],
    team_b_metrics: Dict[str, float],
) -> Tuple[float, float]:
    score_a = momentum_points(team_a_metrics)
    score_b = momentum_points(team_b_metrics)
    total = score_a + score_b

    if total <= 0:
        return 50.0, 50.0

    return (
        round(score_a / total * 100.0, 1),
        round(score_b / total * 100.0, 1),
    )


def build_momentum_timeline(
    events_df: pd.DataFrame,
    team_a_name: str,
    team_b_name: str,
    max_minute: Optional[int] = None,
    window: int = 10,
) -> pd.DataFrame:
    """Build minute-by-minute rolling momentum shares."""
    if max_minute is None:
        max_minute = _max_minute(events_df)

    rows = []

    for minute in range(1, int(max_minute) + 1):
        window_df = events_in_window(
            events_df,
            minute,
            window,
        )

        team_a_metrics = team_live_metrics(window_df, team_a_name)
        team_b_metrics = team_live_metrics(window_df, team_b_name)
        score_a, score_b = relative_momentum_score(
            team_a_metrics,
            team_b_metrics,
        )

        rows.append({
            "Minute": minute,
            team_a_name: score_a,
            team_b_name: score_b,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Higher-level command-centre intelligence
# -----------------------------------------------------------------------------

def _normalised_pair(value_a: float, value_b: float) -> Tuple[float, float]:
    """Convert two non-negative values into comparable 0-100 shares."""
    a = max(0.0, _safe_float(value_a))
    b = max(0.0, _safe_float(value_b))
    total = a + b
    if total <= 0:
        return 50.0, 50.0
    return a / total * 100.0, b / total * 100.0


def _territory_signal(events, teams) -> Dict[str, float]:
    try:
        result = analyze_possession_territory(
            events,
            team_names=list(teams),
            period_label="Full Match",
        )
        summaries = result.get("summaries", pd.DataFrame())
    except Exception:
        summaries = pd.DataFrame()

    output = {team: 0.0 for team in teams}
    if summaries.empty or "Team" not in summaries.columns:
        return output

    for team in teams:
        rows = summaries.loc[summaries["Team"] == team]
        if not rows.empty:
            output[team] = _safe_float(rows.iloc[0].get("Territory Index"))
    return output


def _progression_signal(events, teams) -> Dict[str, float]:
    try:
        result = analyze_progressive_actions(
            events,
            team_names=list(teams),
            period_label="Full Match",
        )
        summaries = result.get("summaries", pd.DataFrame())
    except Exception:
        summaries = pd.DataFrame()

    output = {team: 0.0 for team in teams}
    if summaries.empty or "Team" not in summaries.columns:
        return output

    for team in teams:
        rows = summaries.loc[summaries["Team"] == team]
        if not rows.empty:
            output[team] = _safe_float(rows.iloc[0].get("Progressive Actions"))
    return output


def _tactical_signal(events, teams) -> Dict[str, float]:
    """Positive values represent stronger second-half attacking/tactical movement."""
    output = {team: 0.0 for team in teams}

    try:
        result = detect_tactical_changes(events, team_names=list(teams))
        comparisons = result.get("team_comparisons", pd.DataFrame())
    except Exception:
        comparisons = pd.DataFrame()

    if comparisons.empty or "Team" not in comparisons.columns:
        return output

    for team in teams:
        rows = comparisons.loc[comparisons["Team"] == team]
        if rows.empty:
            continue

        row = rows.iloc[0]
        # Transparent blend of directional second-half shifts.
        value = (
            _safe_float(row.get("Average X Change")) * 0.8
            + _safe_float(row.get("Pressure Change %")) * 12.0
            + _safe_float(row.get("Shot Change")) * 1.5
            + _safe_float(row.get("xG Change")) * 5.0
            + _safe_float(row.get("Attack Index Change %")) * 8.0
        )
        output[team] = value

    return output


def _top_recent_threat(window_events: pd.DataFrame, team_name: str) -> Dict[str, object]:
    if window_events is None or window_events.empty:
        return {"Player": "N/A", "Shots": 0, "xG": 0.0}

    teams = _event_teams(window_events)
    types = _event_types(window_events)
    shots = window_events.loc[
        teams.eq(team_name) & types.eq("Shot")
    ].copy()

    if shots.empty or "player" not in shots.columns:
        return {"Player": "N/A", "Shots": 0, "xG": 0.0}

    shots["Player"] = shots["player"].apply(_normalise_name)
    shots["xG"] = _safe_numeric(
        shots.get("shot_statsbomb_xg", pd.Series(0.0, index=shots.index))
    )

    grouped = (
        shots.groupby("Player", dropna=False)
        .agg(Shots=("Player", "size"), xG=("xG", "sum"))
        .sort_values(["xG", "Shots"], ascending=False)
    )

    if grouped.empty:
        return {"Player": "N/A", "Shots": 0, "xG": 0.0}

    player = str(grouped.index[0]) or "N/A"
    row = grouped.iloc[0]
    return {
        "Player": player,
        "Shots": int(row["Shots"]),
        "xG": float(row["xG"]),
    }


def analyze_live_intelligence(
    events: pd.DataFrame,
    team_names: Optional[List[str]] = None,
    selected_minute: Optional[int] = None,
    rolling_window: int = 10,
) -> Dict[str, object]:
    """
    Build a command-centre snapshot for one match state.

    Returns score-state metrics, recent momentum, full-match contextual signals,
    a transparent composite Intelligence Advantage score, recent threats and alerts.
    """
    if events is None or events.empty:
        raise ValueError("events must contain match event data")

    clean_events = exclude_shootout_events(events)

    if team_names is None:
        team_names = _team_names(clean_events)

    if len(team_names) < 2:
        raise ValueError("Two teams are required for Live Intelligence")

    team_1, team_2 = team_names[:2]

    if selected_minute is None:
        selected_minute = _max_minute(clean_events)

    selected_minute = int(selected_minute)
    rolling_window = max(1, int(rolling_window))

    live_events = events_until_minute(clean_events, selected_minute)
    window_events = events_in_window(
        clean_events,
        selected_minute,
        rolling_window,
    )

    live_1 = team_live_metrics(live_events, team_1)
    live_2 = team_live_metrics(live_events, team_2)
    recent_1 = team_live_metrics(window_events, team_1)
    recent_2 = team_live_metrics(window_events, team_2)

    momentum_1, momentum_2 = relative_momentum_score(recent_1, recent_2)

    territory = _territory_signal(live_events, (team_1, team_2))
    progression = _progression_signal(live_events, (team_1, team_2))
    tactical = _tactical_signal(live_events, (team_1, team_2))

    territory_1, territory_2 = _normalised_pair(
        territory[team_1], territory[team_2]
    )
    progression_1, progression_2 = _normalised_pair(
        progression[team_1], progression[team_2]
    )
    xg_1, xg_2 = _normalised_pair(live_1["xG"], live_2["xG"])
    shots_1, shots_2 = _normalised_pair(live_1["Shots"], live_2["Shots"])

    # Tactical signal may be negative, so shift both values to a non-negative pair.
    tactical_min = min(tactical[team_1], tactical[team_2], 0.0)
    tactical_a = tactical[team_1] - tactical_min + 1.0
    tactical_b = tactical[team_2] - tactical_min + 1.0
    tactical_1, tactical_2 = _normalised_pair(tactical_a, tactical_b)

    # Composite: weighted, transparent, and deliberately not a prediction probability.
    intelligence_1 = (
        momentum_1 * 0.32
        + territory_1 * 0.15
        + progression_1 * 0.15
        + xg_1 * 0.18
        + shots_1 * 0.08
        + tactical_1 * 0.12
    )
    intelligence_2 = (
        momentum_2 * 0.32
        + territory_2 * 0.15
        + progression_2 * 0.15
        + xg_2 * 0.18
        + shots_2 * 0.08
        + tactical_2 * 0.12
    )

    total_intel = intelligence_1 + intelligence_2
    if total_intel <= 0:
        intelligence_1 = intelligence_2 = 50.0
    else:
        intelligence_1 = intelligence_1 / total_intel * 100.0
        intelligence_2 = intelligence_2 / total_intel * 100.0

    threat_1 = _top_recent_threat(window_events, team_1)
    threat_2 = _top_recent_threat(window_events, team_2)

    alerts: List[str] = []

    if abs(momentum_1 - momentum_2) >= 12:
        leader = team_1 if momentum_1 > momentum_2 else team_2
        value = max(momentum_1, momentum_2)
        alerts.append(
            f"Recent momentum currently favours {leader} ({value:.1f}/100 over the last {rolling_window} minutes)."
        )

    if abs(live_1["xG"] - live_2["xG"]) >= 0.50:
        leader = team_1 if live_1["xG"] > live_2["xG"] else team_2
        alerts.append(
            f"Chance-quality advantage: {leader} leads the current xG comparison."
        )

    if abs(territory_1 - territory_2) >= 12:
        leader = team_1 if territory_1 > territory_2 else team_2
        alerts.append(
            f"Territorial activity currently favours {leader}."
        )

    if abs(progression_1 - progression_2) >= 12:
        leader = team_1 if progression_1 > progression_2 else team_2
        alerts.append(
            f"Progressive-action advantage currently favours {leader}."
        )

    if not alerts:
        alerts.append(
            "No single intelligence dimension currently shows a strong enough separation to trigger a high-priority alert."
        )

    timeline = build_momentum_timeline(
        clean_events,
        team_1,
        team_2,
        max_minute=selected_minute,
        window=rolling_window,
    )

    score_text = f"{team_1} {live_1['Goals']} - {live_2['Goals']} {team_2}"

    return {
        "teams": (team_1, team_2),
        "selected_minute": selected_minute,
        "rolling_window": rolling_window,
        "score_text": score_text,
        "live_metrics": {
            team_1: live_1,
            team_2: live_2,
        },
        "recent_metrics": {
            team_1: recent_1,
            team_2: recent_2,
        },
        "momentum": {
            team_1: momentum_1,
            team_2: momentum_2,
        },
        "signals": {
            "Territory": {team_1: territory_1, team_2: territory_2},
            "Progression": {team_1: progression_1, team_2: progression_2},
            "xG Threat": {team_1: xg_1, team_2: xg_2},
            "Shot Volume": {team_1: shots_1, team_2: shots_2},
            "Tactical Shift": {team_1: tactical_1, team_2: tactical_2},
        },
        "intelligence_advantage": {
            team_1: round(intelligence_1, 1),
            team_2: round(intelligence_2, 1),
        },
        "recent_threats": {
            team_1: threat_1,
            team_2: threat_2,
        },
        "alerts": alerts,
        "timeline": timeline,
        "methodology": (
            "Intelligence Advantage is a transparent weighted heuristic combining recent momentum, "
            "territory, progression, xG threat, shot volume and event-derived tactical shifts. "
            "It is not a win probability or a trained forecasting model."
        ),
    }

# -----------------------------------------------------------------------------
# Professional Live Intelligence visual
# -----------------------------------------------------------------------------

def _resolve_team_badge_path(team_name: str):
    try:
        path = get_team_badge(str(team_name).strip())
        if path is not None and Path(path).exists():
            return Path(path)
    except Exception:
        pass
    return None


def _add_team_badge(
    fig,
    team_name: str,
    left: float,
    bottom: float,
    width: float = 0.060,
    height: float = 0.060,
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


def _football_display_name(value: str) -> str:
    name = str(value or "").strip()
    if not name:
        return "N/A"

    parts = [p for p in name.split() if p]
    if len(parts) <= 2:
        return name

    # Provider names often include extra family names; this keeps a familiar
    # football-report display form without changing underlying identity.
    return f"{parts[0]} {parts[-2]}"


def build_live_intelligence_figure(
    events: pd.DataFrame,
    team_names: Optional[List[str]] = None,
    selected_minute: Optional[int] = None,
    rolling_window: int = 10,
):
    """
    Build a professional Live Intelligence Command Centre visual.

    The Intelligence Advantage shown here is a transparent multi-signal heuristic,
    not a win probability and not a trained prediction model.
    """
    result = analyze_live_intelligence(
        events,
        team_names=team_names,
        selected_minute=selected_minute,
        rolling_window=rolling_window,
    )

    team_1, team_2 = result["teams"]
    live = result["live_metrics"]
    recent = result["recent_metrics"]
    momentum = result["momentum"]
    signals = result["signals"]
    advantage = result["intelligence_advantage"]
    threats = result["recent_threats"]
    alerts = result["alerts"]
    timeline = result["timeline"]

    NAVY = "#0B2E63"
    BLUE = "#1F5FAF"
    PALE = "#F8FBFF"
    TEXT = "#111111"
    BORDER = "#C7D4E3"

    fig = plt.figure(figsize=(16, 10.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------------------------------------------------
    # Brand header
    # ---------------------------------------------------------
    header = FancyBboxPatch(
        (0.025, 0.935),
        0.95,
        0.045,
        boxstyle="round,pad=0,rounding_size=0.003",
        linewidth=0,
        facecolor=NAVY,
    )
    ax.add_patch(header)

    ax.text(
        0.045, 0.957,
        "LIVE MATCH INTELLIGENCE",
        fontsize=17,
        fontweight="bold",
        color="white",
        ha="left",
        va="center",
    )

    ax.text(
        0.955, 0.957,
        "COMMAND CENTRE",
        fontsize=9,
        color="white",
        ha="right",
        va="center",
    )

    ax.text(
        0.5, 0.902,
        "LIVE INTELLIGENCE",
        fontsize=22,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="center",
    )

    ax.text(
        0.5, 0.872,
        f"{team_1} {live[team_1]['Goals']}  —  {live[team_2]['Goals']} {team_2}",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        ha="center",
        va="center",
    )

    ax.text(
        0.5, 0.845,
        f"Match state: {result['selected_minute']}'  |  Rolling window: last {result['rolling_window']} minutes",
        fontsize=8.5,
        color="#555555",
        ha="center",
        va="center",
    )

    _add_team_badge(fig, team_1, 0.060, 0.825, 0.065, 0.065)
    _add_team_badge(fig, team_2, 0.875, 0.825, 0.065, 0.065)

    ax.text(
        0.125, 0.850, team_1.upper(),
        fontsize=10.5, fontweight="bold", color=NAVY,
        ha="left", va="center",
    )
    ax.text(
        0.875, 0.850, team_2.upper(),
        fontsize=10.5, fontweight="bold", color=NAVY,
        ha="right", va="center",
    )

    # ---------------------------------------------------------
    # Intelligence Advantage
    # ---------------------------------------------------------
    advantage_bar = FancyBboxPatch(
        (0.055, 0.760),
        0.89,
        0.050,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=BLUE,
        facecolor=PALE,
    )
    ax.add_patch(advantage_bar)

    ax.text(
        0.5, 0.795,
        "INTELLIGENCE ADVANTAGE",
        fontsize=10.5,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="center",
    )

    ax.text(
        0.160, 0.778,
        f"{team_1}: {advantage[team_1]:.1f}",
        fontsize=11.5,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        0.840, 0.778,
        f"{team_2}: {advantage[team_2]:.1f}",
        fontsize=11.5,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Split bar
    left_share = advantage[team_1] / 100.0
    bar_x = 0.285
    bar_w = 0.43
    bar_y = 0.772
    bar_h = 0.012

    ax.add_patch(FancyBboxPatch(
        (bar_x, bar_y), bar_w, bar_h,
        boxstyle="round,pad=0,rounding_size=0.006",
        linewidth=0.8, fill=False, edgecolor=BORDER
    ))

    # Left share = team 1, right share = team 2.
    ax.add_patch(FancyBboxPatch(
        (bar_x, bar_y), bar_w * left_share, bar_h,
        boxstyle="round,pad=0,rounding_size=0.006",
        linewidth=0, facecolor=BLUE
    ))

    ax.add_patch(FancyBboxPatch(
        (bar_x + bar_w * left_share, bar_y),
        bar_w * (1.0 - left_share),
        bar_h,
        boxstyle="round,pad=0,rounding_size=0.006",
        linewidth=0,
        facecolor="#9EB9D8"
    ))

    ax.plot(
        [bar_x + bar_w * left_share, bar_x + bar_w * left_share],
        [bar_y - 0.002, bar_y + bar_h + 0.002],
        linewidth=0.8,
        color="#666666",
    )

    # ---------------------------------------------------------
    # Momentum timeline
    # ---------------------------------------------------------
    section = FancyBboxPatch(
        (0.055, 0.710), 0.89, 0.028,
        boxstyle="round,pad=0.001,rounding_size=0.004",
        linewidth=0, facecolor=NAVY,
    )
    ax.add_patch(section)
    ax.text(
        0.5, 0.724,
        "MATCH MOMENTUM TIMELINE",
        fontsize=10,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )

    chart = fig.add_axes([0.075, 0.505, 0.85, 0.185])

    if timeline is not None and not timeline.empty:
        chart.plot(
            timeline["Minute"],
            timeline[team_1],
            linewidth=2.0,
            label=team_1,
        )
        chart.plot(
            timeline["Minute"],
            timeline[team_2],
            linewidth=2.0,
            label=team_2,
        )
        chart.axhline(50, linewidth=0.8, linestyle="--")
        chart.set_ylim(0, 100)
        chart.set_xlim(
            max(1, int(timeline["Minute"].min())),
            int(timeline["Minute"].max()),
        )

    chart.set_ylabel("Momentum share", fontsize=8)
    chart.set_xlabel("Minute", fontsize=8)
    chart.tick_params(axis="both", labelsize=7)
    chart.spines["top"].set_visible(False)
    chart.spines["right"].set_visible(False)
    chart.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)

    # ---------------------------------------------------------
    # Team signal cards
    # ---------------------------------------------------------
    signal_bar = FancyBboxPatch(
        (0.055, 0.445), 0.89, 0.028,
        boxstyle="round,pad=0.001,rounding_size=0.004",
        linewidth=0, facecolor=NAVY,
    )
    ax.add_patch(signal_bar)
    ax.text(
        0.5, 0.459,
        "CURRENT INTELLIGENCE SIGNALS",
        fontsize=10,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )

    signal_names = [
        ("Recent Momentum", momentum),
        ("Territory", signals["Territory"]),
        ("Progression", signals["Progression"]),
        ("xG Threat", signals["xG Threat"]),
        ("Shot Volume", signals["Shot Volume"]),
    ]

    for team, left in ((team_1, 0.055), (team_2, 0.525)):
        card = FancyBboxPatch(
            (left, 0.240),
            0.420,
            0.180,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.0,
            edgecolor=BLUE,
            facecolor=PALE,
        )
        ax.add_patch(card)

        ax.text(
            left + 0.020, 0.398,
            team.upper(),
            fontsize=10, fontweight="bold",
            color=NAVY,
            ha="left", va="top",
        )

        y = 0.365
        for label, mapping in signal_names:
            value = float(mapping[team])
            ax.text(
                left + 0.020, y,
                label,
                fontsize=7.8,
                color="#555555",
                ha="left", va="center",
            )
            ax.text(
                left + 0.390, y,
                f"{value:.1f}",
                fontsize=8.6,
                fontweight="bold",
                ha="right", va="center",
            )
            # mini bar
            ax.add_patch(FancyBboxPatch(
                (left + 0.145, y - 0.006),
                0.190, 0.010,
                boxstyle="round,pad=0,rounding_size=0.004",
                linewidth=0.5, fill=False, edgecolor=BORDER
            ))
            ax.add_patch(FancyBboxPatch(
                (left + 0.145, y - 0.006),
                0.190 * max(0.0, min(100.0, value)) / 100.0,
                0.010,
                boxstyle="round,pad=0,rounding_size=0.004",
                linewidth=0, facecolor=BLUE
            ))
            y -= 0.030

        threat = threats[team]

        if int(threat["Shots"]) <= 0:
            threat_text = "Recent threat: No shots in recent window"
        else:
            threat_text = (
                f"Recent threat: {_football_display_name(threat['Player'])}  |  "
                f"{int(threat['Shots'])} shots  |  {float(threat['xG']):.2f} xG"
            )

        ax.text(
            left + 0.020, 0.262,
            threat_text,
            fontsize=7.5,
            fontweight="bold",
            ha="left", va="center",
        )

    # ---------------------------------------------------------
    # Alerts
    # ---------------------------------------------------------
    alerts_bar = FancyBboxPatch(
        (0.055, 0.195), 0.89, 0.028,
        boxstyle="round,pad=0.001,rounding_size=0.004",
        linewidth=0, facecolor=NAVY,
    )
    ax.add_patch(alerts_bar)
    ax.text(
        0.5, 0.209,
        "LIVE INTELLIGENCE ALERTS",
        fontsize=10,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )

    alerts_box = FancyBboxPatch(
        (0.055, 0.075),
        0.89,
        0.095,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=BLUE,
        facecolor="#F8FBFF",
    )
    ax.add_patch(alerts_box)

    y = 0.145
    for message in alerts[:3]:
        ax.text(
            0.080, y,
            f"• {message}",
            fontsize=8.0,
            ha="left",
            va="top",
            color=TEXT,
        )
        y -= 0.035

    # ---------------------------------------------------------
    # Footer / methodology
    # ---------------------------------------------------------
    ax.text(
        0.5, 0.050,
        "Intelligence Advantage combines recent momentum, territory, progression, xG threat, shot volume and tactical-shift signals.",
        fontsize=7.3,
        ha="center",
        va="center",
        color="#444444",
    )

    footer = FancyBboxPatch(
        (0.025, 0.018),
        0.95,
        0.028,
        boxstyle="round,pad=0,rounding_size=0.002",
        linewidth=0,
        facecolor=NAVY,
    )
    ax.add_patch(footer)

    ax.text(
        0.5, 0.032,
        "Live Intelligence Command Centre  |  Multi-signal heuristic  |  Not a win probability or trained prediction model",
        fontsize=7.4,
        color="white",
        ha="center",
        va="center",
    )

    return fig
