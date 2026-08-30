
"""
Section Report Export Framework
LiveMatch Intelligence

Generates focused PNG and PDF reports for major analytics sections:
- Team Performance Comparison
- Player Comparison
- Passing Network & Build-up
- Possession & Territory
- Progressive Actions
- Tactical Changes
- Shot Analysis
- Experimental ML Prediction

The framework is intentionally generic:
- PNG = compact visual report
- PDF = visual + interpretation + methodology
"""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
from typing import Dict, Optional, List, Tuple
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from src.match_analyzer import load_match_events
from src.player_analyzer import calculate_player_stats
from src.pass_network_analyzer import (
    analyze_pass_networks,
    build_pass_network_figure,
)
from src.possession_territory_analyzer import (
    analyze_possession_territory,
    _draw_team_territory_panel,
)
from src.progressive_actions_analyzer import (
    analyze_progressive_actions,
    _draw_team_progression_panel,
)
from src.tactical_change_detector import detect_tactical_changes
from src.live_intelligence_analyzer import (
    analyze_live_intelligence,
    build_live_intelligence_figure,
)
from src.score_utils import match_score_breakdown, exclude_shootout_events
from src.image_fetcher import get_player_image, get_team_badge

try:
    from src.match_outcome_predictor import (
        predict_match_outcome,
        build_predictive_intelligence_figure,
    )
except Exception:
    predict_match_outcome = None
    build_predictive_intelligence_figure = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports" / "sections"
DEFAULT_MATCH_ID = 3895309

TEAM_IMAGE_DIR = (
    PROJECT_ROOT
    / "dashboard"
    / "assets"
    / "teams"
)

TEAM_LOGOS = {
    "Borussia Dortmund": TEAM_IMAGE_DIR / "borussia_dortmund.png",
    "Bayer Leverkusen": TEAM_IMAGE_DIR / "bayer_leverkusen.png",
}

PLAYER_IMAGE_DIR = (
    PROJECT_ROOT
    / "dashboard"
    / "assets"
    / "players"
)

PLAYER_IMAGES = {
    "Granit Xhaka": PLAYER_IMAGE_DIR / "granit_xhaka.jpg",
    "Florian Wirtz": PLAYER_IMAGE_DIR / "florian_wirtz.jpg",
}

TEAM_LEAGUES = {
    # German clubs
    "Borussia Dortmund": "Bundesliga",
    "Bayer Leverkusen": "Bundesliga",

    # National teams
    "Argentina": "FIFA World Cup",
    "France": "FIFA World Cup",
}

TEAM_COUNTRIES = {
    # German clubs
    "Borussia Dortmund": "Germany",
    "Bayer Leverkusen": "Germany",

    # National teams
    "Argentina": "Argentina",
    "France": "France",
}

PLAYER_COUNTRIES = {
    "Florian Wirtz": "Germany",
    "Granit Xhaka": "Switzerland",
}


SECTION_LABELS = {
    "team_comparison": "Team Performance Comparison",
    "player_comparison": "Player Comparison",
    "passing_network": "Passing Network & Build-up",
    "territory": "Possession & Territory",
    "progression": "Progressive Actions",
    "tactical": "Tactical Changes",
    "live_intelligence": "Live Intelligence",
    "shot_analysis": "Shot Analysis",
    "ml_prediction": "Predictive Intelligence",
}


# ---------------------------------------------------------
# Generic helpers
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
    return events["type"].apply(_normalise_name)


def _event_teams(events):
    if "team" not in events.columns:
        return pd.Series("", index=events.index)
    return events["team"].apply(_normalise_name)


def _team_names(events) -> List[str]:
    return (
        _event_teams(events)
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )


def _wrap(value, width=60):
    return "\n".join(textwrap.wrap(str(value), width=width))


def _football_display_name(value: str) -> str:
    """
    Convert long provider player names into familiar football-report display names
    without changing the underlying identity used for matching.

    The rule is generic:
    - one/two-part names are preserved
    - for longer names, keep the first given name + the penultimate surname token
      when the final token looks like an additional family name
    - preserve common particles with the surname where possible
    """
    name = str(value or "").strip()

    if not name:
        return "N/A"

    parts = [
        p
        for p in name.split()
        if p
    ]

    if len(parts) <= 2:
        return name

    particles = {
        "da",
        "de",
        "del",
        "di",
        "dos",
        "du",
        "la",
        "le",
        "van",
        "von",
    }

    first = parts[0]

    # For common four-plus token provider names such as
    # "Lionel Andrés Messi Cuccittini" or "Kylian Mbappé Lottin",
    # the penultimate token is usually the familiar football surname.
    surname_index = -2

    # If the token before the chosen surname is a surname particle,
    # keep the particle too.
    if (
        len(parts) >= 4
        and parts[surname_index - 1].casefold() in particles
    ):
        surname = (
            parts[surname_index - 1]
            + " "
            + parts[surname_index]
        )
    else:
        surname = parts[surname_index]

    return f"{first} {surname}"


def _safe_number(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _resolve_team_logo(team_name: str):
    """
    Resolve a team/national-team badge in this order:

    1. Existing project badge.
    2. Known filename fallback.
    3. Automatically fetched/cached badge via src.image_fetcher.
    """
    name = str(team_name).strip()

    direct = TEAM_LOGOS.get(
        name
    )

    if (
        direct
        and direct.exists()
    ):
        return direct

    lowered = name.lower()

    if "dortmund" in lowered:
        path = (
            TEAM_IMAGE_DIR
            / "borussia_dortmund.png"
        )

        if path.exists():
            return path

    if "leverkusen" in lowered:
        path = (
            TEAM_IMAGE_DIR
            / "bayer_leverkusen.png"
        )

        if path.exists():
            return path

    # Automatic online lookup + local cache.
    try:
        fetched = get_team_badge(
            name
        )

        if (
            fetched is not None
            and Path(fetched).exists()
        ):
            return Path(
                fetched
            )

    except Exception:
        pass

    return None


def _resolve_player_image(
    player_name: str,
    represented_team: str | None = None,
    competition: str | None = None,
):
    """
    Resolve a player image in this order:

    1. Existing manually supplied project image.
    2. Filename-based local fallback.
    3. Context-aware automatic fetch/cache using:
       player name + represented team + competition.
    """
    name = str(player_name).strip()

    direct = PLAYER_IMAGES.get(name)

    if (
        direct
        and direct.exists()
    ):
        return direct

    stem = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
    )

    for suffix in (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    ):
        candidate = (
            PLAYER_IMAGE_DIR
            / f"{stem}{suffix}"
        )

        if candidate.exists():
            return candidate

    try:
        fetched = get_player_image(
            player_name=name,
            represented_team=represented_team,
            competition=competition,
        )

        if (
            fetched is not None
            and Path(fetched).exists()
        ):
            return Path(
                fetched
            )

    except Exception:
        pass

    return None


def _add_image_to_figure(
    fig,
    image_path,
    left: float,
    bottom: float,
    width: float,
    height: float,
):
    if image_path is None or not Path(image_path).exists():
        return False

    try:
        from PIL import Image as PILImage
        image = PILImage.open(image_path).convert("RGBA")

        image_ax = fig.add_axes([left, bottom, width, height])
        image_ax.imshow(image)
        image_ax.axis("off")
        return True
    except Exception:
        return False


def _add_team_logo(
    fig,
    team_name: str,
    left: float,
    bottom: float,
    width: float = 0.05,
    height: float = 0.05,
):
    return _add_image_to_figure(
        fig,
        _resolve_team_logo(team_name),
        left,
        bottom,
        width,
        height,
    )


def _add_player_image(
    fig,
    player_name: str,
    left: float,
    bottom: float,
    width: float = 0.075,
    height: float = 0.075,
    represented_team: str | None = None,
    competition: str | None = None,
):
    return _add_image_to_figure(
        fig,
        _resolve_player_image(
            player_name,
            represented_team=represented_team,
            competition=competition,
        ),
        left,
        bottom,
        width,
        height,
    )


def _completed_pass_rate(events: pd.DataFrame, team_name: str) -> float:
    teams = _event_teams(events)
    types = _event_types(events)

    passes = events.loc[
        (teams == team_name)
        & (types == "Pass")
    ].copy()

    if passes.empty:
        return 0.0

    if "pass_outcome" not in passes.columns:
        return 100.0

    outcomes = passes["pass_outcome"].apply(_normalise_name)
    completed = int(outcomes.eq("").sum())

    return completed / len(passes) * 100.0


def _events_until_minute(events_df: pd.DataFrame, minute: int) -> pd.DataFrame:
    if "minute" not in events_df.columns:
        return events_df.copy()

    minutes = pd.to_numeric(
        events_df["minute"],
        errors="coerce",
    ).fillna(0)

    return events_df.loc[
        minutes <= minute
    ].copy()


def _events_in_window(
    events_df: pd.DataFrame,
    end_minute: int,
    window: int = 10,
) -> pd.DataFrame:
    if "minute" not in events_df.columns:
        return events_df.copy()

    minutes = pd.to_numeric(
        events_df["minute"],
        errors="coerce",
    ).fillna(0)

    start_minute = max(
        0,
        end_minute - window,
    )

    return events_df.loc[
        (minutes > start_minute)
        & (minutes <= end_minute)
    ].copy()


def _momentum_points(metrics: Dict[str, float]) -> float:
    return (
        metrics["Shots"] * 4.0
        + metrics["xG"] * 10.0
        + metrics["Pressures"] * 0.35
        + metrics["Carries"] * 0.08
        + metrics["Recoveries"] * 0.20
        + metrics["Passes"] * 0.03
    )


def _relative_momentum_score(a, b):
    score_a = _momentum_points(a)
    score_b = _momentum_points(b)
    total = score_a + score_b

    if total <= 0:
        return 50.0, 50.0

    return (
        round(score_a / total * 100.0, 1),
        round(score_b / total * 100.0, 1),
    )


def _team_basic_metrics(events: pd.DataFrame, team_name: str) -> Dict[str, float]:
    teams = _event_teams(events)
    types = _event_types(events)
    mask = teams == team_name
    team_events = events.loc[mask].copy()
    team_types = types.loc[mask]

    shots = int((team_types == "Shot").sum())
    passes = int((team_types == "Pass").sum())
    pressures = int((team_types == "Pressure").sum())
    carries = int((team_types == "Carry").sum())

    goals = 0
    xg = 0.0

    shot_rows = team_events.loc[
        _event_types(team_events) == "Shot"
    ].copy()

    if not shot_rows.empty:
        if "shot_outcome" in shot_rows.columns:
            outcomes = shot_rows["shot_outcome"].apply(_normalise_name)

            periods = pd.to_numeric(
                shot_rows.get(
                    "period",
                    pd.Series(0, index=shot_rows.index),
                ),
                errors="coerce",
            ).fillna(0)

            goals = int(
                (
                    outcomes.eq("Goal")
                    & periods.ne(5)
                ).sum()
            )

        if "shot_statsbomb_xg" in shot_rows.columns:
            xg = float(
                pd.to_numeric(
                    shot_rows["shot_statsbomb_xg"],
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            )

    return {
        "Goals": goals,
        "Shots": shots,
        "xG": xg,
        "Passes": passes,
        "Pass Completion %": _completed_pass_rate(events, team_name),
        "Pressures": pressures,
        "Carries": carries,
        "Recoveries": int((team_types == "Ball Recovery").sum()),
        "Interceptions": int((team_types == "Interception").sum()),
    }


def _extract_shots(events: pd.DataFrame, team_name: str) -> pd.DataFrame:
    teams = _event_teams(events)
    types = _event_types(events)

    shots = events.loc[
        (teams == team_name)
        & (types == "Shot")
    ].copy()

    if shots.empty:
        return shots

    def xy(value):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except Exception:
                return np.nan, np.nan
        return np.nan, np.nan

    coords = shots["location"].apply(xy)
    shots["X"] = [v[0] for v in coords]
    shots["Y"] = [v[1] for v in coords]

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

    shots["Player"] = (
        shots.get(
            "player",
            pd.Series("", index=shots.index),
        )
        .apply(_normalise_name)
    )

    return shots


# ---------------------------------------------------------
# Section data builders
# ---------------------------------------------------------

def _build_team_comparison(events, teams):
    t1, t2 = teams[:2]

    m1 = _team_basic_metrics(events, t1)
    m2 = _team_basic_metrics(events, t2)

    territory = analyze_possession_territory(
        events,
        [t1, t2],
        "Full Match",
    )["summaries"]

    progression = analyze_progressive_actions(
        events,
        [t1, t2],
        "Full Match",
    )["summaries"]

    tr1 = territory[territory["Team"] == t1].iloc[0]
    tr2 = territory[territory["Team"] == t2].iloc[0]
    pr1 = progression[progression["Team"] == t1].iloc[0]
    pr2 = progression[progression["Team"] == t2].iloc[0]

    table = pd.DataFrame({
        "Metric": [
            "Goals",
            "Shots",
            "xG",
            "Pass Completion %",
            "Pressures",
            "Territory Index",
            "Progressive Actions",
            "Final Third Entries",
            "Box Entries",
        ],
        t1: [
            m1["Goals"],
            m1["Shots"],
            round(m1["xG"], 2),
            round(m1["Pass Completion %"], 1),
            m1["Pressures"],
            round(_safe_number(tr1["Territory Index"]), 1),
            int(pr1["Progressive Actions"]),
            int(pr1["Final Third Entries"]),
            int(pr1["Box Entries"]),
        ],
        t2: [
            m2["Goals"],
            m2["Shots"],
            round(m2["xG"], 2),
            round(m2["Pass Completion %"], 1),
            m2["Pressures"],
            round(_safe_number(tr2["Territory Index"]), 1),
            int(pr2["Progressive Actions"]),
            int(pr2["Final Third Entries"]),
            int(pr2["Box Entries"]),
        ],
    })

    insights = [
        f"{t1} vs {t2}: the comparison combines attacking, passing, pressure, territory and progression indicators.",
        f"{t1} recorded {m1['Shots']} shots and {m1['xG']:.2f} xG; {t2} recorded {m2['Shots']} shots and {m2['xG']:.2f} xG.",
        f"Progressive actions were {int(pr1['Progressive Actions'])} for {t1} and {int(pr2['Progressive Actions'])} for {t2}.",
    ]

    score_info = match_score_breakdown(
        events,
        t1,
        t2,
    )

    if score_info["has_shootout"]:
        shootout_winner = (
            score_info["winner_on_penalties"]
            or "Shootout winner"
        )

        insights.append(
            f"{shootout_winner} won the penalty shootout "
            f"{score_info['team_1_penalties']}–{score_info['team_2_penalties']}; "
            "shootout kicks are excluded from the match-goal totals above."
        )

    return {
        "title": SECTION_LABELS["team_comparison"],
        "subtitle": f"{t1} vs {t2}",
        "table": table,
        "insights": insights,
        "teams": (t1, t2),
        "metrics": (m1, m2, tr1, tr2, pr1, pr2),
    }


def _infer_player_position(
    events: pd.DataFrame,
    player_name: str,
) -> str:
    """
    Infer the player's most frequently recorded StatsBomb position
    in the selected match.

    Returns a clean position label such as:
    - Center Midfield
    - Left Wing
    - Center Forward

    Falls back to 'Position unavailable' if position data is absent.
    """
    if (
        events is None
        or events.empty
        or "player" not in events.columns
        or "position" not in events.columns
    ):
        return "Position unavailable"

    players = events["player"].apply(
        _normalise_name
    )

    player_events = events.loc[
        players == str(
            player_name
        )
    ].copy()

    if player_events.empty:
        return "Position unavailable"

    positions = (
        player_events["position"]
        .apply(
            _normalise_name
        )
        .replace(
            "",
            np.nan,
        )
        .dropna()
    )

    if positions.empty:
        return "Position unavailable"

    return str(
        positions.value_counts().index[0]
    )


def _build_player_comparison(
    events,
    teams,
    player_1: Optional[str],
    player_2: Optional[str],
):
    if not player_1 or not player_2:
        # Select two highly involved players as a fallback.
        players = (
            events.get(
                "player",
                pd.Series("", index=events.index),
            )
            .apply(_normalise_name)
            .replace("", np.nan)
            .dropna()
            .value_counts()
        )
        selected = players.head(2).index.tolist()
        if len(selected) < 2:
            raise RuntimeError("Not enough players were found for a comparison.")
        player_1, player_2 = selected[0], selected[1]

    # calculate_player_stats takes the full events DataFrame and returns
    # one row per player. Select the two requested players afterwards,
    # matching the main dashboard workflow.
    all_player_stats = calculate_player_stats(
        events
    )

    if (
        all_player_stats is None
        or all_player_stats.empty
    ):
        raise RuntimeError(
            "Player statistics could not be calculated."
        )

    player_1_rows = all_player_stats[
        all_player_stats["Player"]
        == player_1
    ]

    player_2_rows = all_player_stats[
        all_player_stats["Player"]
        == player_2
    ]

    if (
        player_1_rows.empty
        or player_2_rows.empty
    ):
        raise RuntimeError(
            "One or both selected players were not found in the calculated player statistics."
        )

    stats_1 = player_1_rows.iloc[0]
    stats_2 = player_2_rows.iloc[0]

    team_1 = str(
        stats_1.get(
            "Team",
            "",
        )
    )

    team_2 = str(
        stats_2.get(
            "Team",
            "",
        )
    )

    metrics = [
        "Passes",
        "Pass Completion %",
        "Carries",
        "Shots",
        "xG",
        "Goals",
        "Pressures",
        "Interceptions",
        "Recoveries",
    ]

    table_rows = []

    for metric in metrics:
        if (
            metric in stats_1.index
            and metric in stats_2.index
        ):
            table_rows.append({
                "Metric": metric,
                player_1: stats_1[metric],
                player_2: stats_2[metric],
            })

    table = pd.DataFrame(
        table_rows
    )

    insights = [
        (
            f"This report compares {player_1} ({team_1}) and "
            f"{player_2} ({team_2}) using their recorded match-event contributions."
        ),
        (
            "The comparison should be interpreted in role context: "
            "a defender, midfielder and forward are expected to contribute differently."
        ),
    ]

    position_1 = _infer_player_position(
        events,
        player_1,
    )

    position_2 = _infer_player_position(
        events,
        player_2,
    )

    competition_1 = TEAM_LEAGUES.get(
        team_1,
        "Competition unavailable",
    )

    competition_2 = TEAM_LEAGUES.get(
        team_2,
        "Competition unavailable",
    )

    return {
        "title":
            SECTION_LABELS[
                "player_comparison"
            ],

        "subtitle":
            f"{player_1} vs {player_2}",

        "table":
            table,

        "insights":
            insights,

        "players":
            (
                player_1,
                player_2,
            ),

        "player_teams":
            (
                team_1,
                team_2,
            ),

        "player_positions":
            (
                position_1,
                position_2,
            ),

        "player_competitions":
            (
                competition_1,
                competition_2,
            ),

        "stats":
            (
                stats_1,
                stats_2,
            ),
    }


def _build_passing_network(events, teams):
    t1, t2 = teams[:2]
    result = analyze_pass_networks(
        events,
        [t1, t2],
        "Full Match",
    )

    summary = result["summaries"].copy()

    columns = [
        col for col in [
            "Team",
            "Passes Attempted",
            "Pass Completion %",
            "Strongest Link Passes",
            "Average Network X",
        ]
        if col in summary.columns
    ]

    table = summary[columns].copy()

    insights = []
    for _, row in summary.iterrows():
        insights.append(
            f"{row['Team']}: top passer {row.get('Top Passer', 'N/A')}; "
            f"most involved {row.get('Most Involved Player', 'N/A')}; "
            f"strongest link {row.get('Strongest Link', 'N/A')}."
        )

    return {
        "title": SECTION_LABELS["passing_network"],
        "subtitle": f"{t1} vs {t2}",
        "table": table,
        "insights": insights,
        "result": result,
        "teams": (t1, t2),
    }


def _build_territory(events, teams):
    t1, t2 = teams[:2]
    result = analyze_possession_territory(
        events,
        [t1, t2],
        "Full Match",
    )

    summary = result["summaries"].copy()

    columns = [
        "Team",
        "Event Share %",
        "Average X",
        "Attacking Third Share %",
        "Final Third Events",
        "Box Zone Events",
        "Territory Index",
    ]
    table = summary[[c for c in columns if c in summary.columns]].copy()

    insights = []
    for _, row in summary.iterrows():
        insights.append(
            f"{row['Team']} recorded a territory index of "
            f"{_safe_number(row.get('Territory Index')):.1f}, with "
            f"{_safe_number(row.get('Attacking Third Share %')):.1f}% "
            f"of located events in the attacking third."
        )

    return {
        "title": SECTION_LABELS["territory"],
        "subtitle": f"{t1} vs {t2}",
        "table": table,
        "insights": insights,
        "result": result,
        "teams": (t1, t2),
    }


def _build_progression(events, teams):
    t1, t2 = teams[:2]
    result = analyze_progressive_actions(
        events,
        [t1, t2],
        "Full Match",
    )

    summary = result["summaries"].copy()
    players = result["players"].copy()

    columns = [
        "Team",
        "Progressive Passes",
        "Progressive Carries",
        "Progressive Actions",
        "Final Third Entries",
        "Box Entries",
        "Forward Distance",
    ]

    table = summary[[c for c in columns if c in summary.columns]].copy()

    insights = []
    for _, row in summary.iterrows():
        insights.append(
            f"{row['Team']} recorded {int(row['Progressive Actions'])} "
            f"progressive actions, including {int(row['Final Third Entries'])} "
            f"final-third entries and {int(row['Box Entries'])} box entries."
        )

    return {
        "title": SECTION_LABELS["progression"],
        "subtitle": f"{t1} vs {t2}",
        "table": table,
        "insights": insights,
        "players": players,
        "result": result,
        "teams": (t1, t2),
    }


def _build_tactical(events, teams):
    t1, t2 = teams[:2]
    result = detect_tactical_changes(
        events,
        team_names=[t1, t2],
    )

    team_shifts = result.get("team_comparisons", pd.DataFrame()).copy()
    alerts = result.get("alerts", pd.DataFrame()).copy()

    if team_shifts.empty:
        table = pd.DataFrame()
    else:
        preferred = [
            "Team",
            "First Half Avg X",
            "Second Half Avg X",
            "Average X Change",
            "Pressure Change %",
            "Shot Change",
            "xG Change",
        ]
        table = team_shifts[
            [c for c in preferred if c in team_shifts.columns]
        ].copy()

    insights = []
    if not alerts.empty and "Message" in alerts.columns:
        insights = alerts["Message"].astype(str).head(8).tolist()

    if not insights:
        insights = [
            "No major tactical-change thresholds were triggered by the event-based detector."
        ]

    return {
        "title": SECTION_LABELS["tactical"],
        "subtitle": f"{t1} vs {t2}",
        "table": table,
        "insights": insights,
        "result": result,
        "teams": (t1, t2),
    }


def _build_live_intelligence(
    events,
    teams,
    snapshot_minute=85,
    rolling_window=10,
):
    t1, t2 = teams[:2]

    result = analyze_live_intelligence(
        events,
        team_names=[t1, t2],
        selected_minute=snapshot_minute,
        rolling_window=rolling_window,
    )

    live = result["live_metrics"]
    recent = result["recent_metrics"]
    momentum = result["momentum"]
    advantage = result["intelligence_advantage"]
    signals = result["signals"]
    threats = result["recent_threats"]

    rows = []
    for team in (t1, t2):
        rows.append({
            "Team": team,
            "Score": int(live[team]["Goals"]),
            "xG": round(float(live[team]["xG"]), 2),
            "Shots": int(live[team]["Shots"]),
            "Recent Momentum": round(float(momentum[team]), 1),
            "Intelligence Advantage": round(float(advantage[team]), 1),
            "Territory Signal": round(float(signals["Territory"][team]), 1),
            "Progression Signal": round(float(signals["Progression"][team]), 1),
            "xG Threat Signal": round(float(signals["xG Threat"][team]), 1),
            "Shot Volume Signal": round(float(signals["Shot Volume"][team]), 1),
        })

    table = pd.DataFrame(rows)

    insights = list(result.get("alerts", []))
    if not insights:
        insights = [
            "No high-priority Live Intelligence alerts were triggered at this match state."
        ]

    return {
        "title": SECTION_LABELS["live_intelligence"],
        "subtitle": (
            f"{t1} vs {t2} — {int(result['selected_minute'])}' match state"
        ),
        "table": table,
        "insights": insights,
        "result": result,
        "events": events,
        "teams": (t1, t2),
        "snapshot_minute": int(result["selected_minute"]),
        "rolling_window": int(result["rolling_window"]),
        "recent_metrics": recent,
        "recent_threats": threats,
    }


def _build_shot_analysis(events, teams):
    t1, t2 = teams[:2]
    s1 = _extract_shots(events, t1)
    s2 = _extract_shots(events, t2)

    rows = []
    for team, shots in [(t1, s1), (t2, s2)]:
        on_target_outcomes = {"Goal", "Saved", "Saved to Post"}
        shots_on_target = (
            int(shots["Outcome"].isin(on_target_outcomes).sum())
            if not shots.empty else 0
        )
        rows.append({
            "Team": team,
            "Shots": len(shots),
            "Shots on Target": shots_on_target,
            "xG": round(float(shots["xG"].sum()) if not shots.empty else 0.0, 2),
            "Goals": int((shots["Outcome"] == "Goal").sum()) if not shots.empty else 0,
            "Average xG / Shot": round(
                float(shots["xG"].mean()) if not shots.empty else 0.0,
                3,
            ),
        })

    table = pd.DataFrame(rows)

    leader = table.sort_values("xG", ascending=False).iloc[0]

    insights = [
        f"{leader['Team']} generated the higher cumulative xG ({leader['xG']:.2f}).",
        "Use the shot map to assess both shot volume and location; larger markers indicate higher xG.",
    ]

    return {
        "title": SECTION_LABELS["shot_analysis"],
        "subtitle": f"{t1} vs {t2}",
        "table": table,
        "insights": insights,
        "shots": (s1, s2),
        "teams": (t1, t2),
    }


def _build_ml_prediction(events, teams, snapshot_minute=85):
    t1, t2 = teams[:2]

    if predict_match_outcome is None:
        raise RuntimeError(
            "The experimental ML predictor is unavailable in the current environment."
        )

    checkpoint_events = _events_until_minute(
        events,
        snapshot_minute,
    )

    recent_events = _events_in_window(
        events,
        snapshot_minute,
        10,
    )

    home = _team_basic_metrics(
        checkpoint_events,
        t1,
    )
    away = _team_basic_metrics(
        checkpoint_events,
        t2,
    )

    home_recent = _team_basic_metrics(
        recent_events,
        t1,
    )
    away_recent = _team_basic_metrics(
        recent_events,
        t2,
    )

    home_momentum, away_momentum = _relative_momentum_score(
        home_recent,
        away_recent,
    )

    base_features = {
        "snapshot_minute": snapshot_minute,
        "home_goals": home["Goals"],
        "away_goals": away["Goals"],
        "goal_difference": home["Goals"] - away["Goals"],
        "home_xg": home["xG"],
        "away_xg": away["xG"],
        "xg_difference": home["xG"] - away["xG"],
        "home_shots": home["Shots"],
        "away_shots": away["Shots"],
        "shot_difference": home["Shots"] - away["Shots"],
        "home_passes": home["Passes"],
        "away_passes": away["Passes"],
        "pass_difference": home["Passes"] - away["Passes"],
        "home_pass_completion": home["Pass Completion %"],
        "away_pass_completion": away["Pass Completion %"],
        "home_pressures": home["Pressures"],
        "away_pressures": away["Pressures"],
        "pressure_difference": home["Pressures"] - away["Pressures"],
        "home_carries": home["Carries"],
        "away_carries": away["Carries"],
        "home_recoveries": home["Recoveries"],
        "away_recoveries": away["Recoveries"],
        "home_interceptions": home["Interceptions"],
        "away_interceptions": away["Interceptions"],
        "home_recent_xg": home_recent["xG"],
        "away_recent_xg": away_recent["xG"],
        "home_recent_shots": home_recent["Shots"],
        "away_recent_shots": away_recent["Shots"],
        "home_recent_pressures": home_recent["Pressures"],
        "away_recent_pressures": away_recent["Pressures"],
        "home_momentum": home_momentum,
        "away_momentum": away_momentum,
        "momentum_difference": home_momentum - away_momentum,
    }

    prediction = predict_match_outcome(
        match_minute=snapshot_minute,
        base_features=base_features,
    )

    table = pd.DataFrame(
        [
            {
                "Outcome": f"{t1} Win",
                "Probability %": round(float(prediction["Home Win"]) * 100.0, 1),
            },
            {
                "Outcome": "Draw",
                "Probability %": round(float(prediction["Draw"]) * 100.0, 1),
            },
            {
                "Outcome": f"{t2} Win",
                "Probability %": round(float(prediction["Away Win"]) * 100.0, 1),
            },
        ]
    )

    insights = [
        (
            f"Experimental model checkpoint: {prediction.get('model_minute', snapshot_minute)}'. "
            f"Calibration variant: {prediction.get('model_variant', 'N/A')}."
        ),
        (
            "These probabilities are research estimates and should not be treated as "
            "production-grade forecasting probabilities without broader validation and calibration."
        ),
    ]

    return {
        "title": SECTION_LABELS["ml_prediction"],
        "subtitle": f"{t1} vs {t2} — {snapshot_minute}' checkpoint",
        "table": table,
        "insights": insights,
        "prediction": prediction,
        "base_features": base_features,
        "teams": (t1, t2),
    }


def build_section_payload(
    section_key: str,
    match_id: int = DEFAULT_MATCH_ID,
    player_1: Optional[str] = None,
    player_2: Optional[str] = None,
    snapshot_minute: int = 85,
) -> Dict[str, object]:
    raw_events = load_match_events(
        match_id
    )

    events = exclude_shootout_events(
        raw_events
    )

    teams = _team_names(
        events
    )

    if len(teams) < 2:
        raise RuntimeError("Both teams could not be identified.")

    builders = {
        "team_comparison": lambda: _build_team_comparison(events, teams),
        "player_comparison": lambda: _build_player_comparison(
            events,
            teams,
            player_1,
            player_2,
        ),
        "passing_network": lambda: _build_passing_network(events, teams),
        "territory": lambda: _build_territory(events, teams),
        "progression": lambda: _build_progression(events, teams),
        "tactical": lambda: _build_tactical(events, teams),
        "live_intelligence": lambda: _build_live_intelligence(
            events,
            teams,
            snapshot_minute,
            10,
        ),
        "shot_analysis": lambda: _build_shot_analysis(events, teams),
        "ml_prediction": lambda: _build_ml_prediction(
            events,
            teams,
            snapshot_minute,
        ),
    }

    if section_key not in builders:
        raise ValueError(
            f"Unsupported section_key: {section_key}"
        )

    payload = builders[section_key]()

    if section_key == "team_comparison":
        score_info = match_score_breakdown(
            raw_events,
            teams[0],
            teams[1],
        )

        if score_info["has_shootout"]:
            shootout_winner = (
                score_info[
                    "winner_on_penalties"
                ]
                or "Shootout winner"
            )

            payload["insights"].append(
                f"{shootout_winner} won the penalty shootout "
                f"{score_info['team_1_penalties']}–"
                f"{score_info['team_2_penalties']}; "
                "shootout kicks are excluded from the match statistics."
            )

    payload["section_key"] = section_key
    payload["match_id"] = match_id
    payload["events"] = events

    return payload



# ---------------------------------------------------------
# Head-to-head comparison layout helpers
# ---------------------------------------------------------

def _draw_identity_card(
    fig,
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle,
    image_path=None,
    badge_team=None,
    align="left",
    secondary_text=None,
):
    """
    Draw a clean borderless sports identity block.
    """
    if align == "left":
        image_x = x + 0.025
        text_x = x + 0.145
        ha = "left"
    else:
        image_x = x + w - 0.115
        text_x = x + w - 0.145
        ha = "right"

    if image_path is not None:
        _add_image_to_figure(
            fig,
            image_path,
            image_x,
            y + 0.038,
            0.09,
            0.12,
        )
    elif badge_team:
        _add_team_logo(
            fig,
            badge_team,
            image_x + 0.008,
            y + 0.05,
            0.075,
            0.075,
        )

    ax.text(
        text_x,
        y + h - 0.04,
        str(title),
        fontsize=16,
        fontweight="bold",
        va="top",
        ha=ha,
    )

    ax.text(
        text_x,
        y + 0.072,
        str(subtitle),
        fontsize=9,
        va="bottom",
        ha=ha,
    )

    if secondary_text:
        ax.text(
            text_x,
            y + 0.043,
            str(secondary_text),
            fontsize=7.8,
            va="bottom",
            ha=ha,
        )

    # Player cards retain a small club badge next to the club text.
    if badge_team and image_path is not None:
        if align == "left":
            badge_x = text_x
            badge_text_x = text_x + 0.035
            badge_ha = "left"
        else:
            badge_x = text_x - 0.028
            badge_text_x = text_x - 0.038
            badge_ha = "right"

        _add_team_logo(
            fig,
            badge_team,
            badge_x,
            y + 0.008,
            0.024,
            0.024,
        )

        ax.text(
            badge_text_x,
            y + 0.02,
            str(badge_team),
            fontsize=7.2,
            va="center",
            ha=badge_ha,
        )


def _draw_vs_badge(ax, x, y):
    circle = plt.Circle(
        (x, y),
        0.038,
        fill=False,
        linewidth=1.4,
    )
    ax.add_patch(circle)

    ax.text(
        x,
        y,
        "VS",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
    )


def _draw_entity_metric_matrix(
    ax,
    x,
    y,
    w,
    h,
    table,
    left_name,
    right_name,
    max_rows=9,
):
    """
    Draw comparison as:
    LEFT ENTITY | METRIC | RIGHT ENTITY
    """
    shown = table.head(max_rows).copy()

    row_count = len(shown) + 1
    row_h = h / max(row_count, 1)

    col_fracs = [0.34, 0.32, 0.34]
    x_positions = [
        x,
        x + w * col_fracs[0],
        x + w * (col_fracs[0] + col_fracs[1]),
        x + w,
    ]

    headers = [
        str(left_name),
        "Metric",
        str(right_name),
    ]

    header_y = y + h - row_h

    for c in range(3):
        rect = FancyBboxPatch(
            (x_positions[c], header_y),
            x_positions[c + 1] - x_positions[c],
            row_h,
            boxstyle="square,pad=0",
            linewidth=0.8,
            facecolor="black",
            edgecolor="black",
        )
        ax.add_patch(rect)

        ax.text(
            (x_positions[c] + x_positions[c + 1]) / 2,
            header_y + row_h / 2,
            headers[c],
            fontsize=8.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )

    for r, (_, row) in enumerate(shown.iterrows()):
        row_y = header_y - (r + 1) * row_h

        values = [
            _format_cell(row.get(left_name, "")),
            str(row.get("Metric", "")),
            _format_cell(row.get(right_name, "")),
        ]

        for c in range(3):
            rect = FancyBboxPatch(
                (x_positions[c], row_y),
                x_positions[c + 1] - x_positions[c],
                row_h,
                boxstyle="square,pad=0",
                linewidth=0.55,
                fill=False,
            )
            ax.add_patch(rect)

            ax.text(
                (x_positions[c] + x_positions[c + 1]) / 2,
                row_y + row_h / 2,
                values[c],
                fontsize=8,
                ha="center",
                va="center",
            )


def _draw_comparison_matrix(
    ax,
    x,
    y,
    w,
    h,
    table,
    left_name,
    right_name,
    max_rows=9,
):
    """
    Draw a three-column comparison:
    Metric | Left entity | Right entity.
    """
    shown = table.head(max_rows).copy()

    col_widths = [
        0.42,
        0.29,
        0.29,
    ]

    row_count = len(shown) + 1
    row_h = h / max(row_count, 1)

    # Header
    header_y = y + h - row_h

    headers = [
        "Metric",
        str(left_name),
        str(right_name),
    ]

    x_positions = [
        x,
        x + w * col_widths[0],
        x + w * (col_widths[0] + col_widths[1]),
        x + w,
    ]

    for c in range(3):
        rect = FancyBboxPatch(
            (
                x_positions[c],
                header_y,
            ),
            x_positions[c + 1] - x_positions[c],
            row_h,
            boxstyle="square,pad=0",
            linewidth=0.8,
            fill=False,
        )
        ax.add_patch(rect)

        ax.text(
            (
                x_positions[c]
                + x_positions[c + 1]
            ) / 2,
            header_y + row_h / 2,
            headers[c],
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Body rows
    for r, (_, row) in enumerate(
        shown.iterrows()
    ):
        row_y = (
            header_y
            - (r + 1) * row_h
        )

        metric = str(
            row.get(
                "Metric",
                "",
            )
        )

        left_value = row.get(
            left_name,
            "",
        )

        right_value = row.get(
            right_name,
            "",
        )

        values = [
            metric,
            _format_cell(left_value),
            _format_cell(right_value),
        ]

        for c in range(3):
            rect = FancyBboxPatch(
                (
                    x_positions[c],
                    row_y,
                ),
                x_positions[c + 1] - x_positions[c],
                row_h,
                boxstyle="square,pad=0",
                linewidth=0.65,
                fill=False,
            )
            ax.add_patch(rect)

            ax.text(
                (
                    x_positions[c]
                    + x_positions[c + 1]
                ) / 2,
                row_y + row_h / 2,
                values[c],
                fontsize=8,
                ha="center",
                va="center",
            )


def _draw_player_stat_cards(
    ax,
    x,
    y,
    w,
    h,
    stats_1,
    stats_2,
    player_1,
    player_2,
):
    metrics = [
        ("Passes", "Passes"),
        ("Pass Completion %", "Pass Completion"),
        ("Carries", "Carries"),
        ("Shots", "Shots"),
        ("xG", "xG"),
        ("Goals", "Goals"),
        ("Pressures", "Pressures"),
        ("Interceptions", "Interceptions"),
        ("Recoveries", "Recoveries"),
    ]

    cols = 3
    rows = 3
    gap_x = 0.012
    gap_y = 0.014

    card_w = (
        w
        - gap_x * (cols - 1)
    ) / cols

    card_h = (
        h
        - gap_y * (rows - 1)
    ) / rows

    for idx, (metric, label) in enumerate(metrics):
        row = idx // cols
        col = idx % cols

        cx = x + col * (card_w + gap_x)
        cy = y + (rows - 1 - row) * (card_h + gap_y)

        patch = FancyBboxPatch(
            (cx, cy),
            card_w,
            card_h,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            linewidth=0.8,
            fill=False,
        )
        ax.add_patch(patch)

        v1 = stats_1.get(metric, 0)
        v2 = stats_2.get(metric, 0)

        if metric == "Pass Completion %":
            value_text = f"{float(v1):.1f}%   |   {float(v2):.1f}%"
        elif metric == "xG":
            value_text = f"{float(v1):.2f}   |   {float(v2):.2f}"
        else:
            value_text = f"{int(v1)}   |   {int(v2)}"

        ax.text(
            cx + card_w / 2,
            cy + card_h - 0.012,
            label,
            fontsize=7.5,
            ha="center",
            va="top",
        )

        ax.text(
            cx + card_w / 2,
            cy + card_h * 0.43,
            value_text,
            fontsize=10.5,
            fontweight="bold",
            ha="center",
            va="center",
        )


def _draw_team_stat_cards(
    ax,
    x,
    y,
    w,
    h,
    table,
    team_1,
    team_2,
):
    wanted = [
        "Goals",
        "Shots",
        "xG",
        "Pass Completion %",
        "Pressures",
        "Territory Index",
        "Progressive Actions",
        "Final Third Entries",
        "Box Entries",
    ]

    cols = 3
    rows = 3
    gap_x = 0.012
    gap_y = 0.014

    card_w = (
        w
        - gap_x * (cols - 1)
    ) / cols

    card_h = (
        h
        - gap_y * (rows - 1)
    ) / rows

    lookup = {
        str(row["Metric"]): row
        for _, row in table.iterrows()
    }

    for idx, metric in enumerate(wanted):
        if metric not in lookup:
            continue

        row_i = idx // cols
        col_i = idx % cols

        cx = x + col_i * (card_w + gap_x)
        cy = y + (rows - 1 - row_i) * (card_h + gap_y)

        patch = FancyBboxPatch(
            (cx, cy),
            card_w,
            card_h,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            linewidth=0.8,
            fill=False,
        )
        ax.add_patch(patch)

        row_data = lookup[metric]
        v1 = row_data[team_1]
        v2 = row_data[team_2]

        if metric == "Pass Completion %":
            value_text = f"{float(v1):.1f}%   |   {float(v2):.1f}%"
        elif metric in ("xG", "Territory Index"):
            value_text = f"{float(v1):.2f}   |   {float(v2):.2f}"
        else:
            value_text = f"{int(float(v1))}   |   {int(float(v2))}"

        ax.text(
            cx + card_w / 2,
            cy + card_h - 0.012,
            metric,
            fontsize=7.5,
            ha="center",
            va="top",
        )

        ax.text(
            cx + card_w / 2,
            cy + card_h * 0.43,
            value_text,
            fontsize=10.5,
            fontweight="bold",
            ha="center",
            va="center",
        )


def _render_player_comparison_png(
    payload,
    match_id,
    output_path,
):
    player_1, player_2 = payload["players"]
    team_1, team_2 = payload["player_teams"]

    position_1, position_2 = payload.get(
        "player_positions",
        (
            "Position unavailable",
            "Position unavailable",
        ),
    )

    competition_1, competition_2 = payload.get(
        "player_competitions",
        (
            TEAM_LEAGUES.get(
                team_1,
                "Competition unavailable",
            ),
            TEAM_LEAGUES.get(
                team_2,
                "Competition unavailable",
            ),
        ),
    )

    fig = plt.figure(
        figsize=(13.5, 7.6)
    )

    ax = fig.add_axes(
        [0, 0, 1, 1]
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------

    ax.text(
        0.035,
        0.955,
        "LIVE MATCH INTELLIGENCE",
        fontsize=21,
        fontweight="bold",
        va="top",
    )

    ax.text(
        0.035,
        0.91,
        "Player Comparison",
        fontsize=18,
        fontweight="bold",
        va="top",
    )

    # -----------------------------------------------------
    # Left player identity
    # -----------------------------------------------------

    _add_player_image(
        fig,
        player_1,
        0.055,
        0.71,
        0.085,
        0.11,
        represented_team=team_1,
        competition=competition_1,
    )

    ax.text(
        0.155,
        0.805,
        player_1,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="center",
    )

    _add_team_logo(
        fig,
        team_1,
        0.155,
        0.755,
        0.026,
        0.026,
    )

    ax.text(
        0.19,
        0.768,
        team_1,
        fontsize=8.5,
        ha="left",
        va="center",
    )

    ax.text(
        0.155,
        0.725,
        position_1,
        fontsize=8.5,
        ha="left",
        va="center",
    )

    ax.text(
        0.155,
        0.69,
        TEAM_LEAGUES.get(
            team_1,
            "Competition unavailable",
        ),
        fontsize=8.5,
        ha="left",
        va="center",
    )

    # -----------------------------------------------------
    # VS
    # -----------------------------------------------------

    ax.text(
        0.50,
        0.765,
        "VS",
        fontsize=15,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # -----------------------------------------------------
    # Right player identity
    # -----------------------------------------------------

    ax.text(
        0.69,
        0.805,
        player_2,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="center",
    )

    _add_team_logo(
        fig,
        team_2,
        0.69,
        0.755,
        0.026,
        0.026,
    )

    ax.text(
        0.725,
        0.768,
        team_2,
        fontsize=8.5,
        ha="left",
        va="center",
    )

    ax.text(
        0.69,
        0.725,
        position_2,
        fontsize=8.5,
        ha="left",
        va="center",
    )

    ax.text(
        0.69,
        0.69,
        TEAM_LEAGUES.get(
            team_2,
            "Competition unavailable",
        ),
        fontsize=8.5,
        ha="left",
        va="center",
    )

    _add_player_image(
        fig,
        player_2,
        0.865,
        0.71,
        0.085,
        0.11,
        represented_team=team_2,
        competition=competition_2,
    )

    # -----------------------------------------------------
    # PLAYER | METRIC | PLAYER
    # Full-width table directly beneath player identities
    # -----------------------------------------------------

    _draw_entity_metric_matrix(
        ax,
        0.055,
        0.34,
        0.89,
        0.30,
        payload["table"],
        player_1,
        player_2,
        max_rows=9,
    )

    # -----------------------------------------------------
    # Analyst interpretation
    # -----------------------------------------------------

    box = FancyBboxPatch(
        (
            0.055,
            0.075,
        ),
        0.89,
        0.20,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1,
        fill=False,
    )

    ax.add_patch(
        box
    )

    ax.text(
        0.075,
        0.242,
        "Analyst Interpretation",
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    y_text = 0.205

    for insight in payload[
        "insights"
    ][:4]:
        wrapped = _wrap(
            insight,
            width=110,
        )

        ax.text(
            0.075,
            y_text,
            f"• {wrapped}",
            fontsize=8,
            va="top",
        )

        y_text -= (
            0.048
            + 0.010
            * wrapped.count(
                "\n"
            )
        )

    ax.text(
        0.055,
        0.022,
        (
            f"Player comparison report | Match ID {match_id} | "
            "Event-derived indicators"
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



def _figure_to_array(source_figure):
    """
    Convert a matplotlib figure into an RGB image array for embedding
    inside another report figure.
    """
    buffer = BytesIO()

    source_figure.savefig(
        buffer,
        format="png",
        dpi=145,
        bbox_inches="tight",
    )

    buffer.seek(0)

    image = plt.imread(
        buffer,
        format="png",
    )

    buffer.close()

    return image



def _render_territory_png(
    payload,
    match_id,
    output_path,
):
    """
    Dedicated focused report for Possession & Territory.

    Page visual:
        - team logos attached to their respective territory panels
        - two event-derived 3x3 territory maps
        - compact comparison metrics
        - analyst interpretation

    This is explicitly event-derived territory intelligence rather than
    optical-tracking possession/control.
    """
    team_1, team_2 = payload["teams"]

    events = payload.get(
        "events"
    )

    if events is None or events.empty:
        raise RuntimeError(
            "Possession & Territory report requires the match event data."
        )

    fig = plt.figure(
        figsize=(16, 9.5)
    )

    report_ax = fig.add_axes(
        [0, 0, 1, 1]
    )

    report_ax.set_xlim(
        0,
        1,
    )
    report_ax.set_ylim(
        0,
        1,
    )
    report_ax.axis(
        "off"
    )

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------
    report_ax.text(
        0.035,
        0.965,
        "LIVE MATCH INTELLIGENCE",
        fontsize=22,
        fontweight="bold",
        va="top",
    )

    report_ax.text(
        0.035,
        0.925,
        "Possession & Territory",
        fontsize=18,
        fontweight="bold",
        va="top",
    )

    report_ax.text(
        0.035,
        0.89,
        f"{team_1} vs {team_2}",
        fontsize=11,
        va="top",
    )

    # -----------------------------------------------------
    # Team identities — logo stays with its own territory map
    # -----------------------------------------------------
    _add_team_logo(
        fig,
        team_1,
        0.055,
        0.842,
        0.036,
        0.036,
    )

    report_ax.text(
        0.095,
        0.860,
        team_1,
        fontsize=9,
        fontweight="bold",
        va="center",
        ha="left",
    )

    _add_team_logo(
        fig,
        team_2,
        0.53,
        0.842,
        0.036,
        0.036,
    )

    report_ax.text(
        0.57,
        0.860,
        team_2,
        fontsize=9,
        fontweight="bold",
        va="center",
        ha="left",
    )

    # -----------------------------------------------------
    # Two territory maps using the same analyzer renderer
    # -----------------------------------------------------
    left_ax = fig.add_axes(
        [0.035, 0.40, 0.455, 0.415]
    )

    _draw_team_territory_panel(
        left_ax,
        events,
        team_1,
        "Full Match",
    )

    right_ax = fig.add_axes(
        [0.51, 0.40, 0.455, 0.415]
    )

    _draw_team_territory_panel(
        right_ax,
        events,
        team_2,
        "Full Match",
    )

    # -----------------------------------------------------
    # Compact territory comparison
    # -----------------------------------------------------
    summary = payload["result"][
        "summaries"
    ].copy()

    metrics = [
        (
            "Territory Index",
            "Territory Index",
        ),
        (
            "Average X",
            "Average X",
        ),
        (
            "Attacking Third Share %",
            "Attacking Third Share %",
        ),
        (
            "Final Third Events",
            "Final Third Events",
        ),
        (
            "Box Zone Events",
            "Box Zone Events",
        ),
    ]

    rows = []

    for display_name, column_name in metrics:
        row_values = {
            "Metric":
                display_name,
        }

        for team in (
            team_1,
            team_2,
        ):
            team_rows = summary[
                summary[
                    "Team"
                ]
                == team
            ]

            value = ""

            if not team_rows.empty:
                value = team_rows.iloc[
                    0
                ].get(
                    column_name,
                    "",
                )

            row_values[
                team
            ] = value

        rows.append(
            row_values
        )

    comparison_table = pd.DataFrame(
        rows
    )

    _draw_entity_metric_matrix(
        report_ax,
        0.08,
        0.235,
        0.84,
        0.13,
        comparison_table,
        team_1,
        team_2,
        max_rows=5,
    )

    # -----------------------------------------------------
    # Analyst interpretation
    # -----------------------------------------------------
    box = FancyBboxPatch(
        (
            0.06,
            0.055,
        ),
        0.88,
        0.145,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1,
        fill=False,
    )

    report_ax.add_patch(
        box
    )

    report_ax.text(
        0.078,
        0.178,
        "Analyst Interpretation",
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    y = 0.148

    for insight in payload.get(
        "insights",
        [],
    )[:4]:
        wrapped = _wrap(
            insight,
            width=118,
        )

        report_ax.text(
            0.078,
            y,
            f"• {wrapped}",
            fontsize=8.2,
            va="top",
        )

        y -= (
            0.028
            + 0.011
            * wrapped.count(
                "\n"
            )
        )

    report_ax.text(
        0.06,
        0.018,
        (
            f"Possession & Territory report | Match ID {match_id} | "
            "Event-derived territory indicators; not continuous optical-tracking possession/control"
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



def _render_progression_png(
    payload,
    match_id,
    output_path,
):
    """
    Dedicated focused report for Progressive Actions.

    Page visual:
        - professional team crests
        - two cleaned progressive-action pitch maps
        - Argentina | Metric | France style comparison matrix
        - analyst interpretation

    The pitch visual shows only the strongest actions by forward distance for
    readability, while all summary totals use every qualifying action.
    """
    team_1, team_2 = payload["teams"]

    events = payload.get("events")

    if events is None or events.empty:
        raise RuntimeError(
            "Progressive Actions report requires the match event data."
        )

    fig = plt.figure(
        figsize=(16, 9.5)
    )

    report_ax = fig.add_axes(
        [0, 0, 1, 1]
    )
    report_ax.set_xlim(0, 1)
    report_ax.set_ylim(0, 1)
    report_ax.axis("off")

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------
    report_ax.text(
        0.035,
        0.965,
        "LIVE MATCH INTELLIGENCE",
        fontsize=22,
        fontweight="bold",
        va="top",
    )

    report_ax.text(
        0.035,
        0.925,
        "Progressive Actions",
        fontsize=18,
        fontweight="bold",
        va="top",
    )

    report_ax.text(
        0.035,
        0.89,
        f"{team_1} vs {team_2}",
        fontsize=11,
        va="top",
    )

    # -----------------------------------------------------
    # Team identities
    # -----------------------------------------------------
    _add_team_logo(
        fig,
        team_1,
        0.055,
        0.842,
        0.036,
        0.036,
    )

    report_ax.text(
        0.095,
        0.860,
        team_1,
        fontsize=9,
        fontweight="bold",
        va="center",
        ha="left",
    )

    _add_team_logo(
        fig,
        team_2,
        0.53,
        0.842,
        0.036,
        0.036,
    )

    report_ax.text(
        0.57,
        0.860,
        team_2,
        fontsize=9,
        fontweight="bold",
        va="center",
        ha="left",
    )

    # -----------------------------------------------------
    # Two cleaned progression maps
    # -----------------------------------------------------
    left_ax = fig.add_axes(
        [0.035, 0.405, 0.455, 0.405]
    )

    _draw_team_progression_panel(
        left_ax,
        events,
        team_1,
        "Full Match",
        max_actions=24,
    )

    right_ax = fig.add_axes(
        [0.51, 0.405, 0.455, 0.405]
    )

    _draw_team_progression_panel(
        right_ax,
        events,
        team_2,
        "Full Match",
        max_actions=24,
    )

    summary = payload["result"]["summaries"].copy()

    # -----------------------------------------------------
    # Compact team comparison
    # -----------------------------------------------------
    metrics = [
        ("Progressive Passes", "Progressive Passes"),
        ("Progressive Carries", "Progressive Carries"),
        ("Progressive Actions", "Progressive Actions"),
        ("Final Third Entries", "Final Third Entries"),
        ("Box Entries", "Box Entries"),
        ("Forward Distance", "Forward Distance"),
    ]

    rows = []

    for display_name, column_name in metrics:
        row_values = {
            "Metric": display_name,
        }

        for team in (
            team_1,
            team_2,
        ):
            team_rows = summary[
                summary["Team"] == team
            ]

            value = ""

            if not team_rows.empty:
                value = team_rows.iloc[0].get(
                    column_name,
                    "",
                )

            row_values[team] = value

        rows.append(row_values)

    comparison_table = pd.DataFrame(rows)

    _draw_entity_metric_matrix(
        report_ax,
        0.08,
        0.225,
        0.84,
        0.145,
        comparison_table,
        team_1,
        team_2,
        max_rows=6,
    )

    # -----------------------------------------------------
    # Analyst interpretation
    # -----------------------------------------------------
    box = FancyBboxPatch(
        (
            0.06,
            0.052,
        ),
        0.88,
        0.14,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1,
        fill=False,
    )

    report_ax.add_patch(box)

    report_ax.text(
        0.078,
        0.173,
        "Analyst Interpretation",
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    y = 0.143

    for insight in payload.get(
        "insights",
        [],
    )[:4]:
        wrapped = _wrap(
            insight,
            width=118,
        )

        report_ax.text(
            0.078,
            y,
            f"• {wrapped}",
            fontsize=8.1,
            va="top",
        )

        y -= (
            0.027
            + 0.010
            * wrapped.count("\n")
        )

    report_ax.text(
        0.06,
        0.018,
        (
            f"Progressive Actions report | Match ID {match_id} | "
            "Solid arrows = progressive passes; dashed arrows = progressive carries; "
            "pitch shows strongest actions only"
        ),
        fontsize=7,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path



def _render_passing_network_png(
    payload,
    match_id,
    output_path,
):
    """
    Dedicated focused report for Passing Network & Build-up.

    The report shows one cleaned event-derived passing network per team,
    followed by concise team summary metrics and analyst interpretation.
    """
    team_1, team_2 = payload["teams"]

    events = payload.get(
        "events"
    )

    if events is None or events.empty:
        raise RuntimeError(
            "Passing-network report requires the match event data."
        )

    # Build the exact same cleaned network visual used by the analyzer.
    network_1 = build_pass_network_figure(
        events,
        team_1,
        "Full Match",
        min_connection_passes=4,
        max_connections=24,
        title=f"{team_1} — Passing Network",
    )

    network_2 = build_pass_network_figure(
        events,
        team_2,
        "Full Match",
        min_connection_passes=4,
        max_connections=24,
        title=f"{team_2} — Passing Network",
    )

    image_1 = _figure_to_array(
        network_1
    )
    image_2 = _figure_to_array(
        network_2
    )

    plt.close(
        network_1
    )
    plt.close(
        network_2
    )

    fig = plt.figure(
        figsize=(16, 9.5)
    )

    report_ax = fig.add_axes(
        [0, 0, 1, 1]
    )
    report_ax.set_xlim(
        0,
        1,
    )
    report_ax.set_ylim(
        0,
        1,
    )
    report_ax.axis(
        "off"
    )

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------
    report_ax.text(
        0.035,
        0.965,
        "LIVE MATCH INTELLIGENCE",
        fontsize=22,
        fontweight="bold",
        va="top",
    )

    report_ax.text(
        0.035,
        0.925,
        "Passing Network & Build-up",
        fontsize=18,
        fontweight="bold",
        va="top",
    )

    report_ax.text(
        0.035,
        0.89,
        f"{team_1} vs {team_2}",
        fontsize=11,
        va="top",
    )

    # Team badges belong to their respective passing-network panels.
    # Keeping them here makes the visual association immediately clear:
    # left badge = left network, right badge = right network.
    _add_team_logo(
        fig,
        team_1,
        0.055,
        0.842,
        0.036,
        0.036,
    )

    report_ax.text(
        0.095,
        0.860,
        team_1,
        fontsize=9,
        fontweight="bold",
        va="center",
        ha="left",
    )

    _add_team_logo(
        fig,
        team_2,
        0.53,
        0.842,
        0.036,
        0.036,
    )

    report_ax.text(
        0.57,
        0.860,
        team_2,
        fontsize=9,
        fontweight="bold",
        va="center",
        ha="left",
    )

    # -----------------------------------------------------
    # Two actual passing-network visuals
    # -----------------------------------------------------
    left_ax = fig.add_axes(
        [0.035, 0.39, 0.455, 0.46]
    )
    left_ax.imshow(
        image_1
    )
    left_ax.axis(
        "off"
    )

    right_ax = fig.add_axes(
        [0.51, 0.39, 0.455, 0.46]
    )
    right_ax.imshow(
        image_2
    )
    right_ax.axis(
        "off"
    )

    # -----------------------------------------------------
    # Compact team comparison table
    # -----------------------------------------------------
    summary = payload["result"][
        "summaries"
    ].copy()

    metrics = [
        (
            "Passes Attempted",
            "Passes Attempted",
        ),
        (
            "Passes Completed",
            "Passes Completed",
        ),
        (
            "Pass Completion %",
            "Pass Completion %",
        ),
        (
            "Strongest Link Passes",
            "Strongest Link Passes",
        ),
        (
            "Average Network X",
            "Average Network X",
        ),
    ]

    rows = []

    for display_name, column_name in metrics:
        values = {
            "Metric":
                display_name,
        }

        for team in (
            team_1,
            team_2,
        ):
            team_rows = summary[
                summary[
                    "Team"
                ]
                == team
            ]

            if team_rows.empty:
                value = ""
            else:
                value = team_rows.iloc[
                    0
                ].get(
                    column_name,
                    "",
                )

            values[
                team
            ] = value

        rows.append(
            values
        )

    comparison_table = pd.DataFrame(
        rows
    )

    _draw_entity_metric_matrix(
        report_ax,
        0.08,
        0.235,
        0.84,
        0.13,
        comparison_table,
        team_1,
        team_2,
        max_rows=5,
    )

    # -----------------------------------------------------
    # Analyst interpretation
    # -----------------------------------------------------
    box = FancyBboxPatch(
        (
            0.06,
            0.055,
        ),
        0.88,
        0.145,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1,
        fill=False,
    )

    report_ax.add_patch(
        box
    )

    report_ax.text(
        0.078,
        0.178,
        "Analyst Interpretation",
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    y = 0.148

    for insight in payload.get(
        "insights",
        [],
    )[:4]:
        wrapped = _wrap(
            insight,
            width=118,
        )

        report_ax.text(
            0.078,
            y,
            f"• {wrapped}",
            fontsize=8.2,
            va="top",
        )

        y -= (
            0.028
            + 0.011
            * wrapped.count(
                "\n"
            )
        )

    report_ax.text(
        0.06,
        0.018,
        (
            f"Passing-network report | Match ID {match_id} | "
            "Node positions are average pass-event locations, not optical-tracking positions"
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


def _render_team_comparison_png(
    payload,
    match_id,
    output_path,
):
    team_1, team_2 = payload["teams"]

    fig = plt.figure(figsize=(13.5, 7.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.035,
        0.955,
        "LIVE MATCH INTELLIGENCE",
        fontsize=21,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.035,
        0.91,
        "Team Performance Comparison",
        fontsize=18,
        fontweight="bold",
        va="top",
    )

    # ---------------------------------------------
    # Team 1 identity
    # ---------------------------------------------
    _add_team_logo(
        fig,
        team_1,
        0.055,
        0.735,
        0.085,
        0.085,
    )

    ax.text(
        0.155,
        0.805,
        team_1,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        0.155,
        0.755,
        TEAM_COUNTRIES.get(team_1, "Country unavailable"),
        fontsize=9,
        ha="left",
        va="center",
    )
    ax.text(
        0.155,
        0.715,
        TEAM_LEAGUES.get(team_1, "Bundesliga"),
        fontsize=9,
        ha="left",
        va="center",
    )

    # ---------------------------------------------
    # VS
    # ---------------------------------------------
    ax.text(
        0.50,
        0.775,
        "VS",
        fontsize=15,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # ---------------------------------------------
    # Team 2 identity
    # ---------------------------------------------
    ax.text(
        0.69,
        0.805,
        team_2,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        0.69,
        0.755,
        TEAM_COUNTRIES.get(team_2, "Country unavailable"),
        fontsize=9,
        ha="left",
        va="center",
    )
    ax.text(
        0.69,
        0.715,
        TEAM_LEAGUES.get(team_2, "Bundesliga"),
        fontsize=9,
        ha="left",
        va="center",
    )

    _add_team_logo(
        fig,
        team_2,
        0.865,
        0.735,
        0.085,
        0.085,
    )

    # ---------------------------------------------
    # TEAM | MATRIX | TEAM
    # ---------------------------------------------
    _draw_entity_metric_matrix(
        ax,
        0.055,
        0.34,
        0.89,
        0.31,
        payload["table"],
        team_1,
        team_2,
        max_rows=9,
    )

    # ---------------------------------------------
    # Analyst interpretation
    # ---------------------------------------------
    box = FancyBboxPatch(
        (0.055, 0.075),
        0.89,
        0.20,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1,
        fill=False,
    )
    ax.add_patch(box)

    ax.text(
        0.075,
        0.242,
        "Analyst Interpretation",
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    y_text = 0.205
    for insight in payload["insights"][:4]:
        wrapped = _wrap(insight, width=110)
        ax.text(
            0.075,
            y_text,
            f"• {wrapped}",
            fontsize=8,
            va="top",
        )
        y_text -= 0.048 + 0.010 * wrapped.count("\n")

    ax.text(
        0.055,
        0.022,
        f"Team comparison report | Match ID {match_id} | Event-derived indicators",
        fontsize=7,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    return output_path



def _render_predictive_intelligence_png(
    payload,
    match_id,
    output_path,
):
    """
    Professional focused Predictive Intelligence report.
    """
    if build_predictive_intelligence_figure is None:
        raise RuntimeError(
            "Predictive Intelligence visual is unavailable. "
            "Update src.match_outcome_predictor.py first."
        )

    team_1, team_2 = payload["teams"]

    fig = build_predictive_intelligence_figure(
        payload["prediction"],
        [team_1, team_2],
        payload["base_features"],
        match_minute=payload["base_features"].get(
            "snapshot_minute",
            payload["prediction"].get("model_minute", 0),
        ),
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    return output_path


def _render_live_intelligence_png(
    payload,
    match_id,
    output_path,
):
    """Render the professional Live Intelligence Command Centre report."""
    team_1, team_2 = payload["teams"]

    fig = build_live_intelligence_figure(
        payload["events"],
        [team_1, team_2],
        selected_minute=payload.get("snapshot_minute", 85),
        rolling_window=payload.get("rolling_window", 10),
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    return output_path


def _render_tactical_png(
    payload,
    match_id,
    output_path,
):
    """
    Professional focused Tactical Analysis report.

    Uses the existing tactical-change detector outputs, but presents them in
    the same polished LiveMatch Intelligence report family as Shot Analysis.
    """
    team_1, team_2 = payload["teams"]
    result = payload.get("result", {})

    team_df = result.get(
        "team_comparisons",
        pd.DataFrame(),
    ).copy()

    player_df = result.get(
        "player_comparisons",
        pd.DataFrame(),
    ).copy()

    alerts_df = result.get(
        "alerts",
        pd.DataFrame(),
    ).copy()

    NAVY = "#0B2E63"
    BLUE = "#1F5FAF"
    PALE = "#F8FBFF"
    TEXT = "#111111"

    def row_for(team):
        rows = team_df.loc[
            team_df["Team"] == team
        ]

        return (
            rows.iloc[0]
            if not rows.empty
            else pd.Series(dtype=object)
        )

    def safe(value):
        return _safe_number(
            value,
            0.0,
        )

    def change_text(
        value,
        percent=False,
    ):
        value = safe(value)

        if abs(value) < 1e-9:
            return "No material change"

        arrow = "↑" if value > 0 else "↓"

        if percent:
            return f"{arrow} {abs(value) * 100:.0f}%"

        return f"{arrow} {abs(value):.2f}"

    def top_shifts(team):
        if (
            player_df is None
            or player_df.empty
            or "Team" not in player_df.columns
        ):
            return pd.DataFrame()

        rows = player_df.loc[
            player_df["Team"] == team
        ].copy()

        if rows.empty:
            return rows

        rows["Abs X Change"] = pd.to_numeric(
            rows.get(
                "X Change",
                pd.Series(
                    0.0,
                    index=rows.index,
                ),
            ),
            errors="coerce",
        ).fillna(0.0).abs()

        return (
            rows
            .sort_values(
                "Abs X Change",
                ascending=False,
            )
            .head(3)
        )

    r1 = row_for(team_1)
    r2 = row_for(team_2)

    fig = plt.figure(
        figsize=(11.7, 15.2)
    )

    ax = fig.add_axes(
        [0, 0, 1, 1]
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # -----------------------------------------------------
    # Brand bar
    # -----------------------------------------------------
    top_bar = FancyBboxPatch(
        (0.02, 0.944),
        0.96,
        0.040,
        boxstyle="round,pad=0,rounding_size=0.002",
        linewidth=0,
        facecolor=NAVY,
    )
    ax.add_patch(top_bar)

    ax.text(
        0.040,
        0.964,
        "LIVE MATCH INTELLIGENCE",
        fontsize=17,
        fontweight="bold",
        color="white",
        va="center",
        ha="left",
    )

    ax.text(
        0.960,
        0.964,
        "FOCUSED SECTION REPORT",
        fontsize=8.5,
        color="white",
        va="center",
        ha="right",
    )

    # -----------------------------------------------------
    # Report identity
    # -----------------------------------------------------
    ax.text(
        0.500,
        0.908,
        "TACTICAL ANALYSIS",
        fontsize=21,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="center",
    )

    ax.text(
        0.500,
        0.882,
        f"{team_1} vs {team_2}",
        fontsize=12.5,
        color="#444444",
        ha="center",
        va="center",
    )

    ax.text(
        0.500,
        0.858,
        "FIRST HALF  →  SECOND HALF",
        fontsize=9.5,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="center",
    )

    _add_team_logo(
        fig,
        team_1,
        0.060,
        0.835,
        0.072,
        0.072,
    )

    _add_team_logo(
        fig,
        team_2,
        0.868,
        0.835,
        0.072,
        0.072,
    )

    ax.text(
        0.096,
        0.827,
        str(team_1).upper(),
        fontsize=10.5,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="top",
    )

    ax.text(
        0.904,
        0.827,
        str(team_2).upper(),
        fontsize=10.5,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="top",
    )

    # -----------------------------------------------------
    # Tactical metric cards
    # -----------------------------------------------------
    metric_defs = [
        (
            "AVG ACTION X",
            "First Half Avg X",
            "Second Half Avg X",
            "Average X Change",
            False,
        ),
        (
            "PRESSURES",
            "First Half Pressures",
            "Second Half Pressures",
            "Pressure Change %",
            True,
        ),
        (
            "SHOTS",
            "First Half Shots",
            "Second Half Shots",
            "Shot Change",
            False,
        ),
        (
            "xG",
            "First Half xG",
            "Second Half xG",
            "xG Change",
            False,
        ),
        (
            "ATTACK INDEX",
            "First Half Attack Index",
            "Second Half Attack Index",
            "Attack Index Change %",
            True,
        ),
    ]

    lefts = [
        0.050,
        0.232,
        0.414,
        0.596,
        0.778,
    ]
    card_w = 0.172

    for left, (
        label,
        first_col,
        second_col,
        change_col,
        pct,
    ) in zip(
        lefts,
        metric_defs,
    ):
        card = FancyBboxPatch(
            (left, 0.704),
            card_w,
            0.098,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.0,
            edgecolor=BLUE,
            facecolor=PALE,
        )
        ax.add_patch(card)

        ax.text(
            left + card_w / 2,
            0.781,
            label,
            fontsize=7.8,
            fontweight="bold",
            color=NAVY,
            ha="center",
            va="center",
        )

        v11 = safe(r1.get(first_col))
        v12 = safe(r1.get(second_col))
        v21 = safe(r2.get(first_col))
        v22 = safe(r2.get(second_col))

        if label in {"PRESSURES", "SHOTS"}:
            t1_value = f"{int(round(v11))} → {int(round(v12))}"
            t2_value = f"{int(round(v21))} → {int(round(v22))}"
        else:
            t1_value = f"{v11:.2f} → {v12:.2f}"
            t2_value = f"{v21:.2f} → {v22:.2f}"

        # Clear team labels inside each card.
        ax.text(
            left + 0.020,
            0.764,
            str(team_1),
            fontsize=5.9,
            color="#555555",
            ha="left",
            va="center",
        )

        ax.text(
            left + card_w - 0.020,
            0.764,
            str(team_2),
            fontsize=5.9,
            color="#555555",
            ha="right",
            va="center",
        )

        # Subtle centre divider to prevent the two sides reading as one value.
        ax.plot(
            [
                left + card_w / 2,
                left + card_w / 2,
            ],
            [
                0.714,
                0.765,
            ],
            linewidth=0.65,
            color="#B9C8D8",
        )

        ax.text(
            left + card_w * 0.25,
            0.744,
            t1_value,
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="center",
        )

        ax.text(
            left + card_w * 0.75,
            0.744,
            t2_value,
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="center",
        )

        ax.text(
            left + card_w * 0.25,
            0.718,
            change_text(
                r1.get(change_col),
                percent=pct,
            ),
            fontsize=6.2,
            ha="center",
            va="center",
        )

        ax.text(
            left + card_w * 0.75,
            0.718,
            change_text(
                r2.get(change_col),
                percent=pct,
            ),
            fontsize=6.2,
            ha="center",
            va="center",
        )

    # -----------------------------------------------------
    # Tactical momentum chart
    # -----------------------------------------------------
    section_bar = FancyBboxPatch(
        (0.055, 0.662),
        0.890,
        0.024,
        boxstyle="round,pad=0.001,rounding_size=0.004",
        linewidth=0,
        facecolor=NAVY,
    )
    ax.add_patch(section_bar)

    ax.text(
        0.500,
        0.674,
        "TACTICAL MOMENTUM",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )

    def momentum_values(row):
        return [
            safe(row.get("Average X Change")),
            safe(row.get("Pressure Change %")) * 10,
            safe(row.get("Shot Change")),
            safe(row.get("xG Change")) * 5,
            safe(row.get("Attack Index Change %")) * 10,
        ]

    labels = [
        "Position",
        "Pressure",
        "Shots",
        "xG",
        "Attack",
    ]

    m1 = momentum_values(r1)
    m2 = momentum_values(r2)

    chart = fig.add_axes(
        [0.085, 0.505, 0.830, 0.135]
    )

    x = np.arange(
        len(labels)
    )
    width = 0.32

    chart.axhline(
        0,
        linewidth=0.8,
        color="#666666",
    )

    chart.bar(
        x - width / 2,
        m1,
        width,
        label=team_1,
        alpha=0.78,
    )

    chart.bar(
        x + width / 2,
        m2,
        width,
        label=team_2,
        alpha=0.78,
    )

    chart.set_xticks(
        x
    )
    chart.set_xticklabels(
        labels,
        fontsize=7.5,
    )
    chart.set_ylabel(
        "Relative change signal",
        fontsize=7,
    )
    chart.tick_params(
        axis="y",
        labelsize=6.5,
    )
    chart.spines["top"].set_visible(False)
    chart.spines["right"].set_visible(False)
    chart.legend(
        loc="upper left",
        frameon=False,
        fontsize=7.2,
        ncol=2,
    )

    # -----------------------------------------------------
    # Key player position shifts
    # -----------------------------------------------------
    player_bar = FancyBboxPatch(
        (0.055, 0.463),
        0.890,
        0.024,
        boxstyle="round,pad=0.001,rounding_size=0.004",
        linewidth=0,
        facecolor=NAVY,
    )
    ax.add_patch(player_bar)

    ax.text(
        0.500,
        0.475,
        "KEY PLAYER POSITION SHIFTS",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )

    for team, left in (
        (team_1, 0.055),
        (team_2, 0.525),
    ):
        shifts = top_shifts(team)

        card = FancyBboxPatch(
            (left, 0.315),
            0.420,
            0.125,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.0,
            edgecolor=BLUE,
            facecolor=PALE,
        )
        ax.add_patch(card)

        ax.text(
            left + 0.018,
            0.418,
            str(team).upper(),
            fontsize=9,
            fontweight="bold",
            color=NAVY,
            ha="left",
            va="top",
        )

        if shifts.empty:
            ax.text(
                left + 0.018,
                0.375,
                "No qualifying positional shifts.",
                fontsize=7.7,
                ha="left",
                va="top",
            )
        else:
            y = 0.380

            for _, row in shifts.iterrows():
                player = _football_display_name(
                    row.get(
                        "Player",
                        "",
                    )
                )

                change = safe(
                    row.get(
                        "X Change",
                        0.0,
                    )
                )

                arrow = "↑" if change > 0 else "↓"
                direction = (
                    "higher"
                    if change > 0
                    else "deeper"
                )

                ax.text(
                    left + 0.018,
                    y,
                    (
                        f"{arrow} {player}: "
                        f"{abs(change):.1f} X units {direction}"
                    ),
                    fontsize=7.3,
                    ha="left",
                    va="top",
                )

                y -= 0.031

    # -----------------------------------------------------
    # Analyst interpretation
    # -----------------------------------------------------
    interpretation_bar = FancyBboxPatch(
        (0.055, 0.274),
        0.890,
        0.024,
        boxstyle="round,pad=0.001,rounding_size=0.004",
        linewidth=0,
        facecolor=NAVY,
    )
    ax.add_patch(interpretation_bar)

    ax.text(
        0.500,
        0.286,
        "ANALYST INTERPRETATION",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )

    interpretation_box = FancyBboxPatch(
        (0.055, 0.105),
        0.890,
        0.145,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=BLUE,
        facecolor="#F8FBFF",
    )
    ax.add_patch(interpretation_box)

    messages = []

    if (
        alerts_df is not None
        and not alerts_df.empty
    ):
        type_series = alerts_df.get(
            "Type",
            pd.Series(
                "",
                index=alerts_df.index,
            ),
        ).astype(str)

        team_alerts = alerts_df.loc[
            type_series.ne(
                "Player Position Shift"
            )
        ]

        messages = (
            team_alerts.get(
                "Message",
                pd.Series(dtype=str),
            )
            .dropna()
            .astype(str)
            .head(4)
            .tolist()
        )

    if not messages:
        for team, row in (
            (team_1, r1),
            (team_2, r2),
        ):
            messages.append(
                (
                    f"{team}: average event position changed "
                    f"{safe(row.get('Average X Change')):+.1f} X units; "
                    f"pressure volume changed "
                    f"{safe(row.get('Pressure Change %')):+.0%}."
                )
            )

    y = 0.224

    for message in messages[:4]:
        wrapped = _wrap(
            message,
            width=115,
        )

        ax.text(
            0.078,
            y,
            f"• {wrapped}",
            fontsize=7.6,
            color=TEXT,
            ha="left",
            va="top",
            linespacing=1.25,
        )

        y -= (
            0.030
            + 0.009
            * wrapped.count("\n")
        )

    # -----------------------------------------------------
    # Methodology strip + footer
    # -----------------------------------------------------
    ax.text(
        0.065,
        0.073,
        "Average X uses StatsBomb event coordinates.",
        fontsize=6.6,
        color="#333333",
        ha="left",
        va="center",
    )

    ax.text(
        0.500,
        0.073,
        "Player shifts compare average first-half vs second-half event locations.",
        fontsize=6.6,
        color="#333333",
        ha="center",
        va="center",
    )

    ax.text(
        0.935,
        0.073,
        "Signals are analytical indicators, not confirmed tactical instructions.",
        fontsize=6.6,
        color="#333333",
        ha="right",
        va="center",
    )

    footer = FancyBboxPatch(
        (0.020, 0.018),
        0.960,
        0.030,
        boxstyle="round,pad=0,rounding_size=0.002",
        linewidth=0,
        facecolor=NAVY,
    )
    ax.add_patch(footer)

    ax.text(
        0.500,
        0.033,
        (
            f"Tactical Analysis report  |  Match ID {match_id}  |  "
            "Event-derived first-half vs second-half intelligence"
        ),
        fontsize=7.3,
        color="white",
        ha="center",
        va="center",
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def _render_shot_analysis_png(
    payload,
    match_id,
    output_path,
):
    """
    Professional focused report for Shot Analysis.

    This layout is intentionally generic and data-driven:
        - one report title and one matchup line
        - large team crests used as identity anchors
        - team names shown under the crests
        - shot-map names used only as section labels
        - dynamic shot/xG metrics and contributors
        - shot-analysis-specific methodology footer
    """
    team_1, team_2 = payload["teams"]
    shots_1, shots_2 = payload["shots"]
    table = payload["table"].copy()

    NAVY = "#0B2E63"
    BLUE = "#1F5FAF"
    LIGHT_BLUE = "#EAF2FB"
    BORDER = "#B9C8D8"
    TEXT = "#111111"

    def row_for(team):
        rows = table.loc[
            table["Team"] == team
        ]

        return (
            rows.iloc[0]
            if not rows.empty
            else pd.Series(dtype=object)
        )

    def contributor_summary(shots):
        if shots is None or shots.empty:
            return {
                "Top Shooter": "N/A",
                "Top Shooter Shots": 0,
                "Top xG Player": "N/A",
                "Top xG": 0.0,
                "High Quality Chances": 0,
            }

        valid = shots.loc[
            shots["Player"].astype(str).str.strip().ne("")
        ].copy()

        if valid.empty:
            return {
                "Top Shooter": "N/A",
                "Top Shooter Shots": 0,
                "Top xG Player": "N/A",
                "Top xG": 0.0,
                "High Quality Chances": int(
                    (shots["xG"] >= 0.20).sum()
                ),
            }

        shot_counts = (
            valid["Player"]
            .value_counts()
        )

        xg_by_player = (
            valid.groupby("Player")["xG"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        return {
            "Top Shooter":
                str(shot_counts.index[0]),

            "Top Shooter Shots":
                int(shot_counts.iloc[0]),

            "Top xG Player":
                str(xg_by_player.index[0]),

            "Top xG":
                float(xg_by_player.iloc[0]),

            "High Quality Chances":
                int(
                    (shots["xG"] >= 0.20).sum()
                ),
        }

    def draw_half_pitch(
        ax,
        shots,
    ):
        ax.set_xlim(
            60,
            120,
        )
        ax.set_ylim(
            80,
            0,
        )
        ax.set_aspect(
            "equal",
            adjustable="box",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_color(
                BORDER
            )
            spine.set_linewidth(
                0.9
            )

        # StatsBomb attacking half.
        ax.plot(
            [60, 120, 120, 60, 60],
            [0, 0, 80, 80, 0],
            linewidth=0.9,
            color="#777777",
        )

        ax.plot(
            [102, 120, 120, 102],
            [18, 18, 62, 62],
            linewidth=0.9,
            color="#777777",
        )

        ax.plot(
            [114, 120, 120, 114],
            [30, 30, 50, 50],
            linewidth=0.9,
            color="#777777",
        )

        theta = np.linspace(
            -np.pi / 2,
            np.pi / 2,
            100,
        )

        ax.plot(
            102
            - 10
            * np.cos(
                theta
            ),
            40
            + 10
            * np.sin(
                theta
            ),
            linewidth=0.9,
            color="#777777",
        )

        if (
            shots is not None
            and not shots.empty
        ):
            normal = shots.loc[
                shots["Outcome"]
                != "Goal"
            ]

            goals = shots.loc[
                shots["Outcome"]
                == "Goal"
            ]

            if not normal.empty:
                ax.scatter(
                    normal["X"],
                    normal["Y"],
                    s=36
                    + normal["xG"]
                    * 320,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.7,
                    color=BLUE,
                    label="Shot",
                    zorder=4,
                )

            if not goals.empty:
                ax.scatter(
                    goals["X"],
                    goals["Y"],
                    s=110
                    + goals["xG"]
                    * 450,
                    marker="*",
                    edgecolors="black",
                    linewidths=0.9,
                    color="#F28C28",
                    label="Goal",
                    zorder=6,
                )

        # Legend remains visible even when a team has no goals.
        shot_proxy = ax.scatter(
            [],
            [],
            s=55,
            color=BLUE,
            edgecolors="white",
            linewidths=0.7,
            label="Shot",
        )

        goal_proxy = ax.scatter(
            [],
            [],
            s=125,
            marker="*",
            color="#F28C28",
            edgecolors="black",
            linewidths=0.9,
            label="Goal",
        )

        ax.legend(
            handles=[
                shot_proxy,
                goal_proxy,
            ],
            loc="upper left",
            frameon=False,
            fontsize=7.3,
            handletextpad=0.5,
        )

        # xG marker-size key.
        key_vals = [
            0.10,
            0.30,
            0.50,
        ]

        key_x = [
            66,
            73,
            81,
        ]

        key_y = 70.5

        for x, value in zip(
            key_x,
            key_vals,
        ):
            ax.scatter(
                [x],
                [key_y],
                s=36
                + value
                * 320,
                color=BLUE,
                alpha=0.55,
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )

            ax.text(
                x,
                65.2,
                f"{value:.2f}",
                fontsize=6.5,
                ha="center",
                va="center",
            )

        ax.text(
            73.2,
            76.0,
            "xG size",
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
        )

    r1 = row_for(
        team_1
    )

    r2 = row_for(
        team_2
    )

    c1 = contributor_summary(
        shots_1
    )

    c2 = contributor_summary(
        shots_2
    )

    # Portrait-oriented report so the PDF can display Page 1 at a useful size.
    fig = plt.figure(
        figsize=(
            11.7,
            15.2,
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

    # -----------------------------------------------------
    # Top brand bar
    # -----------------------------------------------------
    brand_bar = FancyBboxPatch(
        (
            0.02,
            0.944,
        ),
        0.96,
        0.040,
        boxstyle="round,pad=0,rounding_size=0.002",
        linewidth=0,
        facecolor=NAVY,
    )

    ax.add_patch(
        brand_bar
    )

    ax.text(
        0.040,
        0.964,
        "LIVE MATCH INTELLIGENCE",
        fontsize=17,
        fontweight="bold",
        color="white",
        va="center",
        ha="left",
    )

    ax.text(
        0.960,
        0.964,
        "FOCUSED SECTION REPORT",
        fontsize=8.5,
        color="white",
        va="center",
        ha="right",
    )

    # -----------------------------------------------------
    # Main report identity
    # -----------------------------------------------------
    ax.text(
        0.500,
        0.908,
        "SHOT ANALYSIS",
        fontsize=21,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="center",
    )

    ax.text(
        0.500,
        0.882,
        f"{team_1} vs {team_2}",
        fontsize=12.5,
        color="#444444",
        ha="center",
        va="center",
    )

    # Larger crests deliberately separated from the map headings.
    _add_team_logo(
        fig,
        team_1,
        0.060,
        0.845,
        0.075,
        0.075,
    )

    _add_team_logo(
        fig,
        team_2,
        0.865,
        0.845,
        0.075,
        0.075,
    )

    ax.text(
        0.0975,
        0.840,
        str(
            team_1
        ).upper(),
        fontsize=11,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="top",
    )

    ax.text(
        0.9025,
        0.840,
        str(
            team_2
        ).upper(),
        fontsize=11,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="top",
    )

    # -----------------------------------------------------
    # Section bars above each shot map
    # -----------------------------------------------------
    for x, team in (
        (
            0.045,
            team_1,
        ),
        (
            0.515,
            team_2,
        ),
    ):
        bar = FancyBboxPatch(
            (
                x,
                0.806,
            ),
            0.440,
            0.024,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            linewidth=0,
            facecolor=NAVY,
        )

        ax.add_patch(
            bar
        )

        ax.text(
            x
            + 0.220,
            0.818,
            f"{str(team).upper()} - SHOT MAP",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )

    def subtitle(
        row,
    ):
        return (
            f"Full Match | "
            f"{int(_safe_number(row.get('Shots')))} shots | "
            f"{int(_safe_number(row.get('Shots on Target')))} on target | "
            f"{int(_safe_number(row.get('Goals')))} goals | "
            f"{_safe_number(row.get('xG')):.2f} xG"
        )

    ax.text(
        0.265,
        0.788,
        subtitle(
            r1
        ),
        fontsize=7.8,
        ha="center",
        va="center",
        color=TEXT,
    )

    ax.text(
        0.735,
        0.788,
        subtitle(
            r2
        ),
        fontsize=7.8,
        ha="center",
        va="center",
        color=TEXT,
    )

    # -----------------------------------------------------
    # Shot maps
    # -----------------------------------------------------
    left_pitch = fig.add_axes(
        [
            0.055,
            0.455,
            0.420,
            0.320,
        ]
    )

    right_pitch = fig.add_axes(
        [
            0.525,
            0.455,
            0.420,
            0.320,
        ]
    )

    draw_half_pitch(
        left_pitch,
        shots_1,
    )

    draw_half_pitch(
        right_pitch,
        shots_2,
    )

    # -----------------------------------------------------
    # Comparison matrix with dynamic contributors
    # -----------------------------------------------------
    comparison_rows = pd.DataFrame(
        [
            {
                team_1:
                    int(
                        _safe_number(
                            r1.get(
                                "Shots"
                            )
                        )
                    ),
                "Metric":
                    "Shots",
                team_2:
                    int(
                        _safe_number(
                            r2.get(
                                "Shots"
                            )
                        )
                    ),
            },
            {
                team_1:
                    int(
                        _safe_number(
                            r1.get(
                                "Shots on Target"
                            )
                        )
                    ),
                "Metric":
                    "Shots on Target",
                team_2:
                    int(
                        _safe_number(
                            r2.get(
                                "Shots on Target"
                            )
                        )
                    ),
            },
            {
                team_1:
                    int(
                        _safe_number(
                            r1.get(
                                "Goals"
                            )
                        )
                    ),
                "Metric":
                    "Goals",
                team_2:
                    int(
                        _safe_number(
                            r2.get(
                                "Goals"
                            )
                        )
                    ),
            },
            {
                team_1:
                    round(
                        _safe_number(
                            r1.get(
                                "xG"
                            )
                        ),
                        2,
                    ),
                "Metric":
                    "Total xG",
                team_2:
                    round(
                        _safe_number(
                            r2.get(
                                "xG"
                            )
                        ),
                        2,
                    ),
            },
            {
                team_1:
                    round(
                        _safe_number(
                            r1.get(
                                "Average xG / Shot"
                            )
                        ),
                        3,
                    ),
                "Metric":
                    "Average xG / Shot",
                team_2:
                    round(
                        _safe_number(
                            r2.get(
                                "Average xG / Shot"
                            )
                        ),
                        3,
                    ),
            },
            {
                team_1:
                    c1[
                        "High Quality Chances"
                    ],
                "Metric":
                    "High-quality chances (xG >= 0.20)",
                team_2:
                    c2[
                        "High Quality Chances"
                    ],
            },
        ]
    )

    # Dedicated navy section label.
    comparison_bar = FancyBboxPatch(
        (
            0.055,
            0.420,
        ),
        0.890,
        0.024,
        boxstyle="round,pad=0.001,rounding_size=0.004",
        linewidth=0,
        facecolor=NAVY,
    )

    ax.add_patch(
        comparison_bar
    )

    ax.text(
        0.500,
        0.432,
        "SHOT ANALYSIS COMPARISON",
        fontsize=10,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )

    _draw_entity_metric_matrix(
        ax,
        0.055,
        0.305,
        0.890,
        0.115,
        comparison_rows,
        team_1,
        team_2,
        max_rows=6,
    )

    # -----------------------------------------------------
    # Analyst interpretation
    # -----------------------------------------------------
    interpretation_box = FancyBboxPatch(
        (
            0.055,
            0.095,
        ),
        0.890,
        0.185,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.1,
        edgecolor=BLUE,
        facecolor="#F8FBFF",
    )

    ax.add_patch(
        interpretation_box
    )

    ax.text(
        0.085,
        0.258,
        "ANALYST INTERPRETATION",
        fontsize=11.5,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="top",
    )

    shots1 = int(
        _safe_number(
            r1.get(
                "Shots"
            )
        )
    )

    shots2 = int(
        _safe_number(
            r2.get(
                "Shots"
            )
        )
    )

    xg1 = _safe_number(
        r1.get(
            "xG"
        )
    )

    xg2 = _safe_number(
        r2.get(
            "xG"
        )
    )

    avg1 = _safe_number(
        r1.get(
            "Average xG / Shot"
        )
    )

    avg2 = _safe_number(
        r2.get(
            "Average xG / Shot"
        )
    )

    volume_leader = (
        team_1
        if shots1 >= shots2
        else team_2
    )

    xg_leader = (
        team_1
        if xg1 >= xg2
        else team_2
    )

    quality_leader = (
        team_1
        if avg1 >= avg2
        else team_2
    )

    interpretation = [
        (
            f"Volume: {volume_leader} produced the higher shot volume "
            f"({shots1} vs {shots2})."
        ),
        (
            f"Chance creation: {xg_leader} generated the higher total xG "
            f"({max(xg1, xg2):.2f} vs {min(xg1, xg2):.2f})."
        ),
        (
            f"Efficiency: {quality_leader} produced the higher average chance quality "
            f"({max(avg1, avg2):.3f} vs {min(avg1, avg2):.3f} xG per shot)."
        ),
        (
            f"Key threats: {_football_display_name(c1['Top xG Player'])} led {team_1} in xG "
            f"({c1['Top xG']:.2f}); {_football_display_name(c2['Top xG Player'])} led {team_2} "
            f"({c2['Top xG']:.2f})."
        ),
    ]

    y = 0.226

    for item in interpretation:
        wrapped = _wrap(
            item,
            width=118,
        )

        ax.text(
            0.085,
            y,
            f"• {wrapped}",
            fontsize=8.2,
            color=TEXT,
            ha="left",
            va="top",
            linespacing=1.25,
        )

        y -= (
            0.031
            + 0.010
            * wrapped.count(
                "\n"
            )
        )

    # -----------------------------------------------------
    # Methodology strip + footer
    # -----------------------------------------------------
    ax.text(
        0.065,
        0.068,
        "Shot locations use StatsBomb 120 x 80 coordinates.",
        fontsize=7,
        color="#333333",
        ha="left",
        va="center",
    )

    ax.text(
        0.500,
        0.068,
        "Marker size represents StatsBomb xG; star = goal.",
        fontsize=7,
        color="#333333",
        ha="center",
        va="center",
    )

    ax.text(
        0.935,
        0.068,
        "Full Match excludes penalty-shootout period 5.",
        fontsize=7,
        color="#333333",
        ha="right",
        va="center",
    )

    footer = FancyBboxPatch(
        (
            0.020,
            0.018,
        ),
        0.960,
        0.030,
        boxstyle="round,pad=0,rounding_size=0.002",
        linewidth=0,
        facecolor=NAVY,
    )

    ax.add_patch(
        footer
    )

    ax.text(
        0.500,
        0.033,
        (
            f"Shot Analysis report  |  Match ID {match_id}  |  "
            "xG represents estimated chance quality, not goal certainty"
        ),
        fontsize=7.5,
        color="white",
        ha="center",
        va="center",
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
# PNG rendering
# ---------------------------------------------------------

def _format_cell(value):
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _draw_generic_table(ax, table: pd.DataFrame, max_rows=10):
    ax.axis("off")

    if table is None or table.empty:
        ax.text(
            0.5,
            0.5,
            "No table data available.",
            ha="center",
            va="center",
            fontsize=10,
        )
        return

    shown = table.head(max_rows).copy()

    cell_text = [
        [_format_cell(v) for v in row]
        for row in shown.to_numpy()
    ]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=list(shown.columns),
        cellLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.35)


def _draw_section_chart(fig, payload):
    section_key = payload["section_key"]
    table = payload["table"]

    chart_ax = fig.add_axes([0.07, 0.34, 0.42, 0.42])

    if section_key == "shot_analysis":
        t1, t2 = payload["teams"]
        s1, s2 = payload["shots"]

        chart_ax.set_xlim(60, 120)
        chart_ax.set_ylim(80, 0)
        chart_ax.set_xticks([])
        chart_ax.set_yticks([])

        chart_ax.plot(
            [60, 120, 120, 60, 60],
            [0, 0, 80, 80, 0],
            linewidth=1,
        )
        chart_ax.plot(
            [102, 120, 120, 102, 102],
            [18, 18, 62, 62, 18],
            linewidth=0.8,
        )

        for team, shots, marker in [
            (t1, s1, "o"),
            (t2, s2, "s"),
        ]:
            if shots.empty:
                continue

            chart_ax.scatter(
                shots["X"],
                shots["Y"],
                s=30 + shots["xG"] * 220,
                alpha=0.65,
                marker=marker,
                label=team,
            )

            goals = shots[shots["Outcome"] == "Goal"]
            if not goals.empty:
                chart_ax.scatter(
                    goals["X"],
                    goals["Y"],
                    s=145,
                    marker="*",
                )

        chart_ax.legend(
            fontsize=7,
            frameon=False,
            loc="lower left",
        )
        chart_ax.set_title(
            "Shot locations and chance quality",
            fontsize=11,
            fontweight="bold",
        )

    elif section_key == "player_comparison":
        p1, p2 = payload["players"]
        numeric = table.copy()

        numeric[p1] = pd.to_numeric(
            numeric[p1],
            errors="coerce",
        )
        numeric[p2] = pd.to_numeric(
            numeric[p2],
            errors="coerce",
        )
        numeric = numeric.dropna(
            subset=[p1, p2],
            how="all",
        ).head(8)

        y = np.arange(len(numeric))
        h = 0.35

        chart_ax.barh(
            y - h / 2,
            numeric[p1].fillna(0),
            height=h,
            label=p1,
        )
        chart_ax.barh(
            y + h / 2,
            numeric[p2].fillna(0),
            height=h,
            label=p2,
        )
        chart_ax.set_yticks(y, numeric["Metric"], fontsize=8)
        chart_ax.invert_yaxis()
        chart_ax.legend(fontsize=7, frameon=False)
        chart_ax.set_title(
            "Player metric comparison",
            fontsize=11,
            fontweight="bold",
        )

    elif section_key == "progression" and "players" in payload:
        players = payload["players"].copy()

        top = (
            players.sort_values(
                "Progressive Actions",
                ascending=False,
            )
            .head(8)
            .sort_values(
                "Progressive Actions",
                ascending=True,
            )
        )

        chart_ax.barh(
            top["Player"],
            top["Progressive Actions"],
        )
        chart_ax.tick_params(axis="both", labelsize=7)
        chart_ax.set_title(
            "Top progressive players",
            fontsize=11,
            fontweight="bold",
        )

    elif section_key == "passing_network":
        summary = payload["result"]["summaries"].copy()
        teams = payload["teams"]

        metric_names = [
            "Passes Attempted",
            "Pass Completion %",
            "Strongest Link Passes",
            "Average Network X",
        ]

        available = [
            m for m in metric_names
            if m in summary.columns
        ]

        y = np.arange(len(available))
        h = 0.35

        values = {}
        for team in teams:
            row = summary[summary["Team"] == team].iloc[0]
            values[team] = [
                _safe_number(row.get(metric))
                for metric in available
            ]

        # Normalise each metric for fair visual comparison because scales differ.
        norm_1 = []
        norm_2 = []
        for i, metric in enumerate(available):
            v1 = values[teams[0]][i]
            v2 = values[teams[1]][i]
            max_v = max(v1, v2, 1.0)
            norm_1.append(v1 / max_v * 100.0)
            norm_2.append(v2 / max_v * 100.0)

        chart_ax.barh(
            y - h / 2,
            norm_1,
            height=h,
            label=teams[0],
        )
        chart_ax.barh(
            y + h / 2,
            norm_2,
            height=h,
            label=teams[1],
        )
        chart_ax.set_yticks(y, available, fontsize=8)
        chart_ax.invert_yaxis()
        chart_ax.set_xlim(0, 110)
        chart_ax.set_xlabel("Relative comparison within metric", fontsize=7)
        chart_ax.legend(fontsize=7, frameon=False)
        chart_ax.set_title(
            "Passing & build-up comparison",
            fontsize=11,
            fontweight="bold",
        )

    elif section_key == "tactical":
        table = payload["table"].copy()
        teams = payload["teams"]

        metrics = [
            m for m in [
                "Average X Change",
                "Pressure Change %",
                "Shot Change",
                "xG Change",
            ]
            if m in table.columns
        ]

        y = np.arange(len(metrics))
        h = 0.35

        row_1 = table[table["Team"] == teams[0]].iloc[0]
        row_2 = table[table["Team"] == teams[1]].iloc[0]

        v1 = [_safe_number(row_1.get(m)) for m in metrics]
        v2 = [_safe_number(row_2.get(m)) for m in metrics]

        chart_ax.barh(
            y - h / 2,
            v1,
            height=h,
            label=teams[0],
        )
        chart_ax.barh(
            y + h / 2,
            v2,
            height=h,
            label=teams[1],
        )
        chart_ax.axvline(0, linewidth=0.8)
        chart_ax.set_yticks(y, metrics, fontsize=8)
        chart_ax.invert_yaxis()
        chart_ax.legend(fontsize=7, frameon=False)
        chart_ax.set_title(
            "First-half vs second-half changes",
            fontsize=11,
            fontweight="bold",
        )

    elif section_key == "territory":
        teams = payload["teams"]
        summary = payload["result"]["summaries"]

        categories = [
            "Territory Index",
            "Attacking Third Share %",
        ]

        values_1 = []
        values_2 = []

        for team in teams:
            row = summary[
                summary["Team"] == team
            ].iloc[0]

            if team == teams[0]:
                values_1 = [
                    _safe_number(row.get(c))
                    for c in categories
                ]
            else:
                values_2 = [
                    _safe_number(row.get(c))
                    for c in categories
                ]

        y = np.arange(len(categories))
        h = 0.35

        chart_ax.barh(
            y - h / 2,
            values_1,
            height=h,
            label=teams[0],
        )
        chart_ax.barh(
            y + h / 2,
            values_2,
            height=h,
            label=teams[1],
        )
        chart_ax.set_yticks(y, categories, fontsize=8)
        chart_ax.invert_yaxis()
        chart_ax.legend(fontsize=7, frameon=False)
        chart_ax.set_title(
            "Territorial comparison",
            fontsize=11,
            fontweight="bold",
        )

    elif section_key == "ml_prediction":
        chart_ax.barh(
            table["Outcome"],
            table["Probability %"],
        )
        chart_ax.set_xlim(0, 100)
        chart_ax.set_xlabel("Probability (%)", fontsize=8)
        chart_ax.set_title(
            "Experimental outcome probabilities",
            fontsize=11,
            fontweight="bold",
        )

    else:
        # Generic comparison chart.
        numeric_cols = [
            c for c in table.columns
            if c != "Metric"
            and pd.api.types.is_numeric_dtype(
                pd.to_numeric(
                    table[c],
                    errors="coerce",
                )
            )
        ]

        if (
            "Metric" in table.columns
            and len(numeric_cols) >= 2
        ):
            shown = table.head(8).copy()
            col1, col2 = numeric_cols[:2]

            v1 = pd.to_numeric(
                shown[col1],
                errors="coerce",
            ).fillna(0)

            v2 = pd.to_numeric(
                shown[col2],
                errors="coerce",
            ).fillna(0)

            y = np.arange(len(shown))
            h = 0.35

            chart_ax.barh(
                y - h / 2,
                v1,
                height=h,
                label=col1,
            )
            chart_ax.barh(
                y + h / 2,
                v2,
                height=h,
                label=col2,
            )

            chart_ax.set_yticks(
                y,
                shown["Metric"],
                fontsize=7,
            )
            chart_ax.invert_yaxis()
            chart_ax.legend(fontsize=7, frameon=False)
            chart_ax.set_title(
                "Comparison overview",
                fontsize=11,
                fontweight="bold",
            )
        else:
            chart_ax.axis("off")
            chart_ax.text(
                0.5,
                0.5,
                "Focused chart not available for this section.\nSee the comparison table.",
                ha="center",
                va="center",
                fontsize=10,
            )

    chart_ax.spines["top"].set_visible(False)
    chart_ax.spines["right"].set_visible(False)


def generate_section_png(
    section_key: str,
    match_id: int = DEFAULT_MATCH_ID,
    player_1: Optional[str] = None,
    player_2: Optional[str] = None,
    snapshot_minute: int = 85,
    output_path: Optional[Path] = None,
) -> Path:
    payload = build_section_payload(
        section_key,
        match_id,
        player_1,
        player_2,
        snapshot_minute,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path is None:
        output_path = (
            REPORTS_DIR
            / f"{section_key}_{match_id}.png"
        )

    # Dedicated sports-style layouts for comparison reports.
    if section_key == "player_comparison":
        return _render_player_comparison_png(
            payload,
            match_id,
            output_path,
        )

    if section_key == "team_comparison":
        return _render_team_comparison_png(
            payload,
            match_id,
            output_path,
        )

    if section_key == "passing_network":
        return _render_passing_network_png(
            payload,
            match_id,
            output_path,
        )

    if section_key == "territory":
        return _render_territory_png(
            payload,
            match_id,
            output_path,
        )

    if section_key == "progression":
        return _render_progression_png(
            payload,
            match_id,
            output_path,
        )

    if section_key == "tactical":
        return _render_tactical_png(
            payload,
            match_id,
            output_path,
        )

    if section_key == "live_intelligence":
        return _render_live_intelligence_png(
            payload,
            match_id,
            output_path,
        )

    if section_key == "shot_analysis":
        return _render_shot_analysis_png(
            payload,
            match_id,
            output_path,
        )

    if section_key == "ml_prediction":
        return _render_predictive_intelligence_png(
            payload,
            match_id,
            output_path,
        )

    fig = plt.figure(
        figsize=(13.5, 7.6)
    )

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.94,
        "LIVE MATCH INTELLIGENCE",
        fontsize=21,
        fontweight="bold",
        va="top",
    )

    ax.text(
        0.05,
        0.895,
        payload["title"],
        fontsize=18,
        fontweight="bold",
        va="top",
    )

    ax.text(
        0.05,
        0.855,
        payload["subtitle"],
        fontsize=11,
        va="top",
    )

    # -----------------------------------------------------
    # Team identity / player imagery
    # -----------------------------------------------------

    if section_key == "player_comparison":
        player_names = payload.get("players", ("", ""))
        player_teams = payload.get("player_teams", ("", ""))

        # Player 1 identity
        _add_player_image(
            fig,
            player_names[0],
            0.69,
            0.835,
            0.065,
            0.075,
        )
        _add_team_logo(
            fig,
            player_teams[0],
            0.765,
            0.855,
            0.035,
            0.035,
        )
        ax.text(
            0.81,
            0.882,
            str(player_names[0]),
            fontsize=9,
            fontweight="bold",
            va="center",
        )
        ax.text(
            0.81,
            0.855,
            str(player_teams[0]),
            fontsize=7.5,
            va="center",
        )

        # Player 2 identity
        _add_player_image(
            fig,
            player_names[1],
            0.69,
            0.745,
            0.065,
            0.075,
        )
        _add_team_logo(
            fig,
            player_teams[1],
            0.765,
            0.765,
            0.035,
            0.035,
        )
        ax.text(
            0.81,
            0.792,
            str(player_names[1]),
            fontsize=9,
            fontweight="bold",
            va="center",
        )
        ax.text(
            0.81,
            0.765,
            str(player_teams[1]),
            fontsize=7.5,
            va="center",
        )

    else:
        report_teams = payload.get("teams", ("", ""))

        if len(report_teams) >= 1:
            _add_team_logo(
                fig,
                report_teams[0],
                0.70,
                0.855,
                0.045,
                0.045,
            )
            ax.text(
                0.755,
                0.877,
                str(report_teams[0]),
                fontsize=9,
                fontweight="bold",
                va="center",
            )

        if len(report_teams) >= 2:
            _add_team_logo(
                fig,
                report_teams[1],
                0.70,
                0.795,
                0.045,
                0.045,
            )
            ax.text(
                0.755,
                0.817,
                str(report_teams[1]),
                fontsize=9,
                fontweight="bold",
                va="center",
            )

    _draw_section_chart(
        fig,
        payload,
    )

    table_ax = fig.add_axes(
        [0.54, 0.34, 0.41, 0.42]
    )

    _draw_generic_table(
        table_ax,
        payload["table"],
        max_rows=10,
    )

    insight_box = FancyBboxPatch(
        (0.05, 0.08),
        0.90,
        0.20,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1,
        fill=False,
    )
    ax.add_patch(insight_box)

    ax.text(
        0.07,
        0.245,
        "Analyst Interpretation",
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    y = 0.21
    for item in payload["insights"][:5]:
        wrapped = _wrap(item, width=105)
        ax.text(
            0.07,
            y,
            f"• {wrapped}",
            fontsize=8.5,
            va="top",
        )
        y -= 0.042 + 0.012 * wrapped.count("\n")

    ax.text(
        0.05,
        0.025,
        (
            f"Section report | Match ID {match_id} | "
            "Event-derived indicators and transparent project heuristics"
        ),
        fontsize=7,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


# ---------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------


def _generate_predictive_intelligence_pdf_two_page(
    payload,
    match_id,
    png_path,
    output_path,
):
    """
    Deterministic two-page Predictive Intelligence PDF.

    Page 1: full professional Predictive Intelligence PNG.
    Page 2: structured evidence, validation context and interpretation boundaries.

    This dedicated writer avoids any conditional Platypus-flow ambiguity for the
    Predictive Intelligence section and guarantees exactly two pages.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.platypus import (
        Table,
        TableStyle,
        Paragraph,
    )
    from reportlab.lib.utils import ImageReader
    from PIL import Image as PILImage

    navy = colors.HexColor("#0B2E63")
    blue = colors.HexColor("#1F5FAF")
    pale = colors.HexColor("#F8FBFF")
    grey = colors.HexColor("#555555")

    page_w, page_h = A4
    c = canvas.Canvas(
        str(output_path),
        pagesize=A4,
    )
    c.setTitle(
        f"LiveMatch Intelligence - {payload['title']}"
    )
    c.setAuthor(
        "LiveMatch Intelligence"
    )

    # ------------------------------------------------------------------
    # Page 1 - full visual
    # ------------------------------------------------------------------
    with PILImage.open(png_path) as im:
        img_w, img_h = im.size

    margin_x = 10 * mm
    margin_y = 10 * mm
    max_w = page_w - 2 * margin_x
    max_h = page_h - 2 * margin_y

    scale = min(
        max_w / img_w,
        max_h / img_h,
    )

    draw_w = img_w * scale
    draw_h = img_h * scale
    draw_x = (page_w - draw_w) / 2
    draw_y = (page_h - draw_h) / 2

    c.drawImage(
        ImageReader(str(png_path)),
        draw_x,
        draw_y,
        width=draw_w,
        height=draw_h,
        preserveAspectRatio=True,
        mask="auto",
    )

    c.showPage()

    # ------------------------------------------------------------------
    # Page 2 - detailed evidence
    # ------------------------------------------------------------------
    prediction = payload["prediction"]
    base = payload["base_features"]
    team_1, team_2 = payload["teams"]

    left = 15 * mm
    usable_w = page_w - 30 * mm
    y = page_h - 18 * mm

    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(
        left,
        y,
        "PREDICTIVE INTELLIGENCE EVIDENCE",
    )
    y -= 12 * mm

    # Shared table style helpers.
    header_style = ParagraphStyle(
        "PredHeader",
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=8.5,
        textColor=colors.white,
        alignment=1,
    )
    body_style = ParagraphStyle(
        "PredBody",
        fontName="Helvetica",
        fontSize=7.4,
        leading=9,
        textColor=colors.HexColor("#222222"),
        alignment=1,
    )
    body_left = ParagraphStyle(
        "PredBodyLeft",
        parent=body_style,
        alignment=0,
    )

    def draw_table(data, widths, current_y):
        table = Table(
            data,
            colWidths=widths,
            repeatRows=1,
        )
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), navy),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#C8D3DF")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        tw, th = table.wrap(
            usable_w,
            page_h,
        )
        table.drawOn(
            c,
            left,
            current_y - th,
        )
        return current_y - th

    def p(text_value, header=False, left_align=False):
        style = (
            header_style
            if header
            else (
                body_left
                if left_align
                else body_style
            )
        )
        return Paragraph(
            str(text_value),
            style,
        )

    # Estimated probabilities
    probability_data = [
        [
            p("Outcome", header=True),
            p("Estimated Probability", header=True),
        ],
        [p(f"{team_1} Win", left_align=True), p(f"{float(prediction['Home Win']) * 100:.1f}%")],
        [p("Draw", left_align=True), p(f"{float(prediction['Draw']) * 100:.1f}%")],
        [p(f"{team_2} Win", left_align=True), p(f"{float(prediction['Away Win']) * 100:.1f}%")],
    ]
    y = draw_table(
        probability_data,
        [105 * mm, 65 * mm],
        y,
    )
    y -= 9 * mm

    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(
        left,
        y,
        "MATCH-STATE INPUTS",
    )
    y -= 7 * mm

    input_data = [
        [
            p("Metric", header=True),
            p(team_1, header=True),
            p(team_2, header=True),
            p("Difference", header=True),
        ],
        [p("Score", left_align=True), p(int(float(base.get("home_goals", 0)))), p(int(float(base.get("away_goals", 0)))), p(f"{float(base.get('goal_difference', 0)):+.0f}")],
        [p("xG", left_align=True), p(f"{float(base.get('home_xg', 0)):.2f}"), p(f"{float(base.get('away_xg', 0)):.2f}"), p(f"{float(base.get('xg_difference', 0)):+.2f}")],
        [p("Shots", left_align=True), p(int(float(base.get("home_shots", 0)))), p(int(float(base.get("away_shots", 0)))), p(f"{float(base.get('shot_difference', 0)):+.0f}")],
        [p("Pressures", left_align=True), p(int(float(base.get("home_pressures", 0)))), p(int(float(base.get("away_pressures", 0)))), p(f"{float(base.get('pressure_difference', 0)):+.0f}")],
        [p("Recent xG", left_align=True), p(f"{float(base.get('home_recent_xg', 0)):.2f}"), p(f"{float(base.get('away_recent_xg', 0)):.2f}"), p(f"{float(base.get('home_recent_xg', 0)) - float(base.get('away_recent_xg', 0)):+.2f}")],
        [p("Recent Shots", left_align=True), p(int(float(base.get("home_recent_shots", 0)))), p(int(float(base.get("away_recent_shots", 0)))), p(f"{float(base.get('home_recent_shots', 0)) - float(base.get('away_recent_shots', 0)):+.0f}")],
        [p("Momentum", left_align=True), p(f"{float(base.get('home_momentum', 0)):.1f}"), p(f"{float(base.get('away_momentum', 0)):.1f}"), p(f"{float(base.get('momentum_difference', 0)):+.1f}")],
    ]
    y = draw_table(
        input_data,
        [55 * mm, 39 * mm, 39 * mm, 37 * mm],
        y,
    )
    y -= 9 * mm

    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(
        left,
        y,
        "MODEL VALIDATION CONTEXT",
    )
    y -= 7 * mm

    def validation_value(value, percent=False):
        if value is None:
            return "N/A"
        try:
            number = float(value)
            return (
                f"{number * 100:.1f}%"
                if percent
                else f"{number:.3f}"
            )
        except Exception:
            return str(value)

    validation_data = [
        [
            p("Model Detail", header=True),
            p("Value", header=True),
        ],
        [p("Checkpoint", left_align=True), p(f"{prediction.get('model_minute', 'N/A')}'")],
        [p("Calibration Variant", left_align=True), p(prediction.get("model_variant", "Unknown"))],
        [p("Validation Accuracy", left_align=True), p(validation_value(prediction.get("validation_accuracy"), True))],
        [p("Validation Macro F1", left_align=True), p(validation_value(prediction.get("validation_macro_f1")))],
        [p("Validation Log Loss", left_align=True), p(validation_value(prediction.get("validation_log_loss")))],
    ]
    y = draw_table(
        validation_data,
        [90 * mm, 80 * mm],
        y,
    )
    y -= 9 * mm

    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(
        left,
        y,
        "INTERPRETATION BOUNDARIES",
    )
    y -= 6 * mm

    boundaries = [
        "Outcome values are experimental research estimates generated from historical StatsBomb match-state snapshots.",
        "The displayed values are not guaranteed outcomes and should not be described as fully calibrated production probabilities.",
        "Model quality should be interpreted alongside validation accuracy, macro F1 and log loss rather than from a single match prediction.",
        "The selected time-aware model uses the latest available checkpoint that does not exceed the current replay minute.",
    ]

    boundary_style = ParagraphStyle(
        "PredBoundary",
        fontName="Helvetica",
        fontSize=8.2,
        leading=11,
        textColor=colors.HexColor("#222222"),
    )

    boundary_rows = [
        [
            Paragraph(
                "•",
                ParagraphStyle(
                    "PredBullet",
                    fontName="Helvetica-Bold",
                    fontSize=10,
                    textColor=blue,
                    alignment=1,
                ),
            ),
            Paragraph(
                item,
                boundary_style,
            ),
        ]
        for item in boundaries
    ]

    boundary_table = Table(
        boundary_rows,
        colWidths=[8 * mm, 165 * mm],
    )
    boundary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), pale),
            ("BOX", (0, 0), (-1, -1), 0.8, blue),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    bw, bh = boundary_table.wrap(
        usable_w,
        page_h,
    )
    boundary_table.drawOn(
        c,
        left,
        y - bh,
    )
    y = y - bh - 7 * mm

    c.setFillColor(grey)
    c.setFont("Helvetica", 7.5)
    c.drawString(
        left,
        max(10 * mm, y),
        (
            f"Generated by LiveMatch Intelligence | Match ID {match_id} | "
            f"{int(float(base.get('snapshot_minute', prediction.get('model_minute', 0))))}' predictive checkpoint"
        ),
    )

    c.save()

    return output_path


def generate_section_pdf(
    section_key: str,
    match_id: int = DEFAULT_MATCH_ID,
    player_1: Optional[str] = None,
    player_2: Optional[str] = None,
    snapshot_minute: int = 85,
    output_path: Optional[Path] = None,
) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image,
        Table,
        TableStyle,
        PageBreak,
    )

    payload = build_section_payload(
        section_key,
        match_id,
        player_1,
        player_2,
        snapshot_minute,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path is None:
        output_path = (
            REPORTS_DIR
            / f"{section_key}_{match_id}.pdf"
        )

    png_path = generate_section_png(
        section_key,
        match_id,
        player_1,
        player_2,
        snapshot_minute,
    )

    if section_key == "ml_prediction":
        return _generate_predictive_intelligence_pdf_two_page(
            payload,
            match_id,
            png_path,
            output_path,
        )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        spaceAfter=6,
    )

    h1 = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        spaceBefore=5,
        spaceAfter=7,
    )

    body = ParagraphStyle(
        "SectionBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=7,
    )

    small = ParagraphStyle(
        "SectionSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#555555"),
    )

    story = []

    # The dedicated Progressive Actions PNG already contains the complete
    # report title, matchup and team identities. Avoid repeating those
    # elements above the embedded report on PDF page 1.
    if section_key not in {"progression", "shot_analysis", "tactical", "live_intelligence", "ml_prediction"}:
        story.append(
            Paragraph(
                payload["title"],
                title_style,
            )
        )

        story.append(
            Paragraph(
                payload["subtitle"],
                body,
            )
        )

        story.append(
            Spacer(1, 3 * mm)
        )

    img = Image(
        str(png_path)
    )

    max_w = 180 * mm

    if section_key in {"shot_analysis", "tactical", "live_intelligence", "ml_prediction"}:
        max_h = 252 * mm
    else:
        max_h = 102 * mm

    scale = min(
        max_w / img.imageWidth,
        max_h / img.imageHeight,
    )

    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale

    story.append(img)

    # Passing Network & Build-up already contains a dedicated
    # Analyst Interpretation panel inside its visual PNG.
    # Do not repeat the same interpretation beneath the image,
    # otherwise it spills onto a mostly empty extra page.
    if section_key not in {"passing_network", "territory", "progression", "shot_analysis", "tactical", "live_intelligence", "ml_prediction"}:
        story.append(
            Spacer(1, 5 * mm)
        )

        story.append(
            Paragraph(
                "Interpretation",
                h1,
            )
        )

        for insight in payload["insights"]:
            story.append(
                Paragraph(
                    f"• {insight}",
                    body,
                )
            )

    story.append(
        PageBreak()
    )

    table = payload["table"].copy()

    if section_key == "ml_prediction":
        navy = colors.HexColor("#0B2E63")
        blue = colors.HexColor("#1F5FAF")
        pale = colors.HexColor("#F8FBFF")

        prediction = payload["prediction"]
        base = payload["base_features"]
        team_1, team_2 = payload["teams"]

        story.append(
            Paragraph(
                "PREDICTIVE INTELLIGENCE EVIDENCE",
                ParagraphStyle(
                    "PredictiveEvidenceTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=17,
                    leading=21,
                    textColor=navy,
                    spaceAfter=10,
                ),
            )
        )

        probability_rows = [
            ["Outcome", "Estimated Probability"],
            [f"{team_1} Win", f"{float(prediction['Home Win']) * 100:.1f}%"],
            ["Draw", f"{float(prediction['Draw']) * 100:.1f}%"],
            [f"{team_2} Win", f"{float(prediction['Away Win']) * 100:.1f}%"],
        ]

        probability_table = Table(
            probability_rows,
            colWidths=[105 * mm, 65 * mm],
            repeatRows=1,
        )
        probability_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), navy),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#C8D3DF")),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ])
        )
        story.append(probability_table)

        story.append(Spacer(1, 9 * mm))

        story.append(
            Paragraph(
                "MATCH-STATE INPUTS",
                ParagraphStyle(
                    "PredictiveInputsTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=15,
                    leading=19,
                    textColor=navy,
                    spaceAfter=8,
                ),
            )
        )

        input_rows = [
            ["Metric", team_1, team_2, "Difference"],
            [
                "Score",
                str(int(float(base.get("home_goals", 0)))),
                str(int(float(base.get("away_goals", 0)))),
                f"{float(base.get('goal_difference', 0)):+.0f}",
            ],
            [
                "xG",
                f"{float(base.get('home_xg', 0)):.2f}",
                f"{float(base.get('away_xg', 0)):.2f}",
                f"{float(base.get('xg_difference', 0)):+.2f}",
            ],
            [
                "Shots",
                str(int(float(base.get("home_shots", 0)))),
                str(int(float(base.get("away_shots", 0)))),
                f"{float(base.get('shot_difference', 0)):+.0f}",
            ],
            [
                "Pressures",
                str(int(float(base.get("home_pressures", 0)))),
                str(int(float(base.get("away_pressures", 0)))),
                f"{float(base.get('pressure_difference', 0)):+.0f}",
            ],
            [
                "Recent xG",
                f"{float(base.get('home_recent_xg', 0)):.2f}",
                f"{float(base.get('away_recent_xg', 0)):.2f}",
                f"{float(base.get('home_recent_xg', 0)) - float(base.get('away_recent_xg', 0)):+.2f}",
            ],
            [
                "Recent Shots",
                str(int(float(base.get("home_recent_shots", 0)))),
                str(int(float(base.get("away_recent_shots", 0)))),
                f"{float(base.get('home_recent_shots', 0)) - float(base.get('away_recent_shots', 0)):+.0f}",
            ],
            [
                "Momentum",
                f"{float(base.get('home_momentum', 0)):.1f}",
                f"{float(base.get('away_momentum', 0)):.1f}",
                f"{float(base.get('momentum_difference', 0)):+.1f}",
            ],
        ]

        input_table = Table(
            input_rows,
            colWidths=[55 * mm, 39 * mm, 39 * mm, 37 * mm],
            repeatRows=1,
        )
        input_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), navy),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.8),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#C8D3DF")),
                ("TOPPADDING", (0, 0), (-1, -1), 5.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
            ])
        )
        story.append(input_table)

        story.append(Spacer(1, 9 * mm))

        story.append(
            Paragraph(
                "MODEL VALIDATION CONTEXT",
                ParagraphStyle(
                    "PredictiveValidationTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=15,
                    leading=19,
                    textColor=navy,
                    spaceAfter=8,
                ),
            )
        )

        def _validation_value(value, percent=False):
            if value is None:
                return "N/A"
            try:
                number = float(value)
                return f"{number * 100:.1f}%" if percent else f"{number:.3f}"
            except Exception:
                return str(value)

        validation_rows = [
            ["Model Detail", "Value"],
            ["Checkpoint", f"{prediction.get('model_minute', 'N/A')}'"],
            ["Calibration Variant", str(prediction.get("model_variant", "Unknown"))],
            ["Validation Accuracy", _validation_value(prediction.get("validation_accuracy"), True)],
            ["Validation Macro F1", _validation_value(prediction.get("validation_macro_f1"))],
            ["Validation Log Loss", _validation_value(prediction.get("validation_log_loss"))],
        ]

        validation_table = Table(
            validation_rows,
            colWidths=[90 * mm, 80 * mm],
            repeatRows=1,
        )
        validation_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), navy),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.0),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#C8D3DF")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(validation_table)

        story.append(Spacer(1, 9 * mm))

        story.append(
            Paragraph(
                "INTERPRETATION BOUNDARIES",
                ParagraphStyle(
                    "PredictiveBoundaryTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=15,
                    leading=19,
                    textColor=navy,
                    spaceAfter=8,
                ),
            )
        )

        boundary_style = ParagraphStyle(
            "PredictiveBoundaryBody",
            parent=body,
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#222222"),
        )

        bullet_style = ParagraphStyle(
            "PredictiveBoundaryBullet",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=blue,
            alignment=1,
        )

        boundaries = [
            "Outcome values are experimental research estimates generated from historical StatsBomb match-state snapshots.",
            "The displayed values are not guaranteed outcomes and should not be described as fully calibrated production probabilities.",
            "Model quality should be interpreted alongside validation accuracy, macro F1 and log loss rather than from a single match prediction.",
            "The selected time-aware model uses the latest available checkpoint that does not exceed the current replay minute.",
        ]

        boundary_rows = [
            [Paragraph("•", bullet_style), Paragraph(item, boundary_style)]
            for item in boundaries
        ]

        boundary_table = Table(
            boundary_rows,
            colWidths=[8 * mm, 165 * mm],
        )
        boundary_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), pale),
                ("BOX", (0, 0), (-1, -1), 0.8, blue),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ])
        )
        story.append(boundary_table)

        story.append(Spacer(1, 7 * mm))
        story.append(
            Paragraph(
                (
                    f"Generated by LiveMatch Intelligence | Match ID {match_id} | "
                    f"{int(float(base.get('snapshot_minute', prediction.get('model_minute', 0))))}' predictive checkpoint"
                ),
                small,
            )
        )

    elif section_key == "live_intelligence":
        navy = colors.HexColor("#0B2E63")
        blue = colors.HexColor("#1F5FAF")
        pale = colors.HexColor("#F8FBFF")

        story.append(
            Paragraph(
                "LIVE INTELLIGENCE EVIDENCE",
                ParagraphStyle(
                    "LiveIntelEvidenceTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=17,
                    leading=21,
                    textColor=navy,
                    spaceAfter=10,
                ),
            )
        )

        evidence = payload["table"].copy()
        if evidence.empty:
            story.append(
                Paragraph(
                    "No structured Live Intelligence evidence was available.",
                    body,
                )
            )
        else:
            shown_columns = [
                "Team",
                "Score",
                "xG",
                "Shots",
                "Recent Momentum",
                "Intelligence Advantage",
                "Territory Signal",
                "Progression Signal",
            ]
            shown = evidence[
                [c for c in shown_columns if c in evidence.columns]
            ].copy()

            header_style = ParagraphStyle(
                "LiveIntelEvidenceHeader",
                fontName="Helvetica-Bold",
                fontSize=6.8,
                leading=8.2,
                textColor=colors.white,
                alignment=1,
            )

            body_style_small = ParagraphStyle(
                "LiveIntelEvidenceBody",
                fontName="Helvetica",
                fontSize=7.0,
                leading=8.4,
                textColor=colors.HexColor("#222222"),
                alignment=1,
            )

            rows = [[
                Paragraph(
                    str(column).replace(" ", "<br/>"),
                    header_style,
                )
                for column in shown.columns
            ]]

            for _, row in shown.iterrows():
                rows.append([
                    Paragraph(
                        _format_cell(value),
                        body_style_small,
                    )
                    for value in row.tolist()
                ])

            # Give more room to long analytical headings and less to short numeric columns.
            width_map = {
                "Team": 24 * mm,
                "Score": 14 * mm,
                "xG": 14 * mm,
                "Shots": 15 * mm,
                "Recent Momentum": 28 * mm,
                "Intelligence Advantage": 31 * mm,
                "Territory Signal": 26 * mm,
                "Progression Signal": 28 * mm,
            }

            col_widths = [
                width_map.get(column, 22 * mm)
                for column in shown.columns
            ]

            evidence_table = Table(
                rows,
                colWidths=col_widths,
                repeatRows=1,
            )
            evidence_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#C8D3DF")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ])
            )
            story.append(evidence_table)

        story.append(Spacer(1, 10 * mm))

        story.append(
            Paragraph(
                "RECENT WINDOW ACTIVITY",
                ParagraphStyle(
                    "LiveIntelRecentTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=15,
                    leading=19,
                    textColor=navy,
                    spaceAfter=8,
                ),
            )
        )

        team_1, team_2 = payload["teams"]
        recent = payload.get("recent_metrics", {})
        threats = payload.get("recent_threats", {})

        recent_rows = [[
            "Team", "Shots", "xG", "Pressures", "Carries", "Recoveries", "Recent Threat"
        ]]

        for team in (team_1, team_2):
            metrics = recent.get(team, {})
            threat = threats.get(team, {})
            if int(threat.get("Shots", 0)) > 0:
                threat_text = (
                    f"{_football_display_name(threat.get('Player', ''))} | "
                    f"{int(threat.get('Shots', 0))} shots | "
                    f"{float(threat.get('xG', 0.0)):.2f} xG"
                )
            else:
                threat_text = "No shots in recent window"

            recent_rows.append([
                team,
                int(metrics.get("Shots", 0)),
                f"{float(metrics.get('xG', 0.0)):.2f}",
                int(metrics.get("Pressures", 0)),
                int(metrics.get("Carries", 0)),
                int(metrics.get("Recoveries", 0)),
                threat_text,
            ])

        recent_table = Table(
            recent_rows,
            colWidths=[
                27 * mm,
                16 * mm,
                16 * mm,
                22 * mm,
                20 * mm,
                23 * mm,
                54 * mm,
            ],
            repeatRows=1,
        )
        recent_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), navy),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.4),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#C8D3DF")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(recent_table)

        story.append(Spacer(1, 10 * mm))

        story.append(
            Paragraph(
                "LIVE INTELLIGENCE ALERTS",
                ParagraphStyle(
                    "LiveIntelAlertsTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=15,
                    leading=19,
                    textColor=navy,
                    spaceAfter=8,
                ),
            )
        )

        alert_style = ParagraphStyle(
            "LiveIntelAlertBody",
            parent=body,
            fontName="Helvetica",
            fontSize=8.6,
            leading=12,
            textColor=colors.HexColor("#222222"),
        )
        bullet_style = ParagraphStyle(
            "LiveIntelBullet",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=blue,
            alignment=1,
        )

        alert_rows = [
            [Paragraph("•", bullet_style), Paragraph(str(item), alert_style)]
            for item in payload.get("insights", [])[:4]
        ]
        if alert_rows:
            alerts_table = Table(alert_rows, colWidths=[8 * mm, 165 * mm])
            alerts_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), pale),
                    ("BOX", (0, 0), (-1, -1), 0.8, blue),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ])
            )
            story.append(alerts_table)

        story.append(Spacer(1, 10 * mm))

        story.append(
            Paragraph(
                "INTERPRETATION BOUNDARIES",
                ParagraphStyle(
                    "LiveIntelBoundaryTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=15,
                    leading=19,
                    textColor=navy,
                    spaceAfter=8,
                ),
            )
        )

        boundaries = [
            "Intelligence Advantage is a transparent weighted multi-signal heuristic; it is not a win probability or a trained forecasting model.",
            "Recent Momentum uses only the selected rolling window and can differ materially from cumulative full-match indicators such as xG or shot volume.",
            "Territory and progression are event-derived measures and do not represent continuous optical-tracking possession or control.",
            "Tactical-shift signals describe changes in recorded event behaviour and should not be treated as confirmed coaching instructions or formation changes.",
        ]

        boundary_rows = [
            [Paragraph("•", bullet_style), Paragraph(item, alert_style)]
            for item in boundaries
        ]
        boundary_table = Table(boundary_rows, colWidths=[8 * mm, 165 * mm])
        boundary_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), pale),
                ("BOX", (0, 0), (-1, -1), 0.8, blue),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ])
        )
        story.append(boundary_table)

        story.append(Spacer(1, 7 * mm))
        story.append(
            Paragraph(
                (
                    f"Generated by LiveMatch Intelligence | Match ID {match_id} | "
                    f"{payload.get('snapshot_minute', 85)}' command-centre snapshot"
                ),
                small,
            )
        )

    elif section_key == "tactical":
        navy = colors.HexColor("#0B2E63")
        blue = colors.HexColor("#1F5FAF")
        pale = colors.HexColor("#F8FBFF")

        story.append(
            Paragraph(
                "TACTICAL EVIDENCE TABLE",
                ParagraphStyle(
                    "TacticalEvidenceTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=17,
                    leading=21,
                    textColor=navy,
                    spaceAfter=10,
                ),
            )
        )

        team_evidence = payload.get(
            "result",
            {},
        ).get(
            "team_comparisons",
            pd.DataFrame(),
        ).copy()

        if team_evidence.empty:
            story.append(
                Paragraph(
                    "No structured tactical evidence was available.",
                    body,
                )
            )
        else:
            wanted = [
                "Team",
                "First Half Avg X",
                "Second Half Avg X",
                "Pressure Change %",
                "Shot Change",
                "xG Change",
                "Attack Index Change %",
            ]

            shown = team_evidence[
                [
                    column
                    for column in wanted
                    if column in team_evidence.columns
                ]
            ].copy()

            for column in [
                "Pressure Change %",
                "Attack Index Change %",
            ]:
                if column in shown.columns:
                    shown[column] = (
                        pd.to_numeric(
                            shown[column],
                            errors="coerce",
                        )
                        .fillna(0.0)
                        * 100.0
                    ).round(1)

            for column in [
                "First Half Avg X",
                "Second Half Avg X",
                "xG Change",
            ]:
                if column in shown.columns:
                    shown[column] = pd.to_numeric(
                        shown[column],
                        errors="coerce",
                    ).round(2)

            rows = [
                list(shown.columns)
            ]

            for _, row in shown.iterrows():
                rows.append(
                    [
                        _format_cell(
                            value
                        )
                        for value in row.tolist()
                    ]
                )

            col_count = len(rows[0])
            total_width = 180 * mm
            col_width = total_width / max(
                col_count,
                1,
            )

            evidence_table = Table(
                rows,
                colWidths=[
                    col_width
                ] * col_count,
                repeatRows=1,
            )

            evidence_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        navy,
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTNAME",
                        (0, 1),
                        (0, -1),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7.5,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.45,
                        colors.HexColor("#C8D3DF"),
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                ])
            )

            story.append(
                evidence_table
            )

        story.append(
            Spacer(
                1,
                10 * mm,
            )
        )

        story.append(
            Paragraph(
                "KEY PLAYER POSITION SHIFTS",
                ParagraphStyle(
                    "TacticalPlayerTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=15,
                    leading=19,
                    textColor=navy,
                    spaceAfter=8,
                ),
            )
        )

        player_evidence = payload.get(
            "result",
            {},
        ).get(
            "player_comparisons",
            pd.DataFrame(),
        ).copy()

        player_rows = [[
            "Team",
            "Player",
            "First Half X",
            "Second Half X",
            "X Change",
        ]]

        if not player_evidence.empty:
            player_evidence["Abs X Change"] = pd.to_numeric(
                player_evidence.get(
                    "X Change",
                    pd.Series(
                        0.0,
                        index=player_evidence.index,
                    ),
                ),
                errors="coerce",
            ).fillna(0.0).abs()

            for team in payload["teams"]:
                top_rows = (
                    player_evidence.loc[
                        player_evidence["Team"] == team
                    ]
                    .sort_values(
                        "Abs X Change",
                        ascending=False,
                    )
                    .head(3)
                )

                for _, row in top_rows.iterrows():
                    player_rows.append([
                        _format_cell(
                            row.get(
                                "Team",
                                "",
                            )
                        ),
                        _football_display_name(
                            row.get(
                                "Player",
                                "",
                            )
                        ),
                        f"{_safe_number(row.get('First Half X')):.2f}",
                        f"{_safe_number(row.get('Second Half X')):.2f}",
                        f"{_safe_number(row.get('X Change')):+.2f}",
                    ])

        player_table = Table(
            player_rows,
            colWidths=[
                35 * mm,
                58 * mm,
                28 * mm,
                28 * mm,
                28 * mm,
            ],
            repeatRows=1,
        )

        player_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    navy,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7.5,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.45,
                    colors.HexColor("#C8D3DF"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ])
        )

        story.append(
            player_table
        )

        story.append(
            Spacer(
                1,
                10 * mm,
            )
        )

        story.append(
            Paragraph(
                "INTERPRETATION BOUNDARIES",
                ParagraphStyle(
                    "TacticalBoundaryTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=15,
                    leading=19,
                    textColor=navy,
                    spaceAfter=8,
                ),
            )
        )

        boundary_style = ParagraphStyle(
            "TacticalBoundaryBody",
            parent=body,
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#222222"),
        )

        bullet_style = ParagraphStyle(
            "TacticalBoundaryBullet",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=blue,
            alignment=1,
        )

        boundaries = [
            (
                "Tactical-change signals compare recorded first-half and second-half "
                "event behaviour; they do not prove a manager changed formation or issued "
                "a specific tactical instruction."
            ),
            (
                "Average X and player-position shifts use average StatsBomb event locations, "
                "not continuous optical-tracking positions."
            ),
            (
                "Pressure, shots, xG and Attack Index changes should be interpreted with "
                "game state, substitutions, opponent behaviour and match context."
            ),
            (
                "Large player shifts can reflect role changes, substitutions or different "
                "event involvement and should be treated as analyst flags rather than exact "
                "positional coordinates."
            ),
        ]

        boundary_rows = [
            [
                Paragraph(
                    "•",
                    bullet_style,
                ),
                Paragraph(
                    item,
                    boundary_style,
                ),
            ]
            for item in boundaries
        ]

        boundary_table = Table(
            boundary_rows,
            colWidths=[
                8 * mm,
                165 * mm,
            ],
        )

        boundary_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    pale,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    blue,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ])
        )

        story.append(
            boundary_table
        )

        story.append(
            Spacer(
                1,
                8 * mm,
            )
        )

        story.append(
            Paragraph(
                (
                    f"Generated by LiveMatch Intelligence  |  Match ID {match_id}  |  "
                    "Focused tactical report"
                ),
                small,
            )
        )

    elif section_key == "shot_analysis":
        navy = colors.HexColor("#0B2E63")
        blue = colors.HexColor("#1F5FAF")
        pale = colors.HexColor("#F8FBFF")

        story.append(
            Paragraph(
                "EVIDENCE TABLE",
                ParagraphStyle(
                    "ShotEvidenceTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=17,
                    leading=21,
                    textColor=navy,
                    spaceAfter=10,
                ),
            )
        )

        if table.empty:
            story.append(
                Paragraph(
                    "No structured shot-analysis evidence was available.",
                    body,
                )
            )
        else:
            shown = table.head(15).copy()

            rows = [
                list(
                    shown.columns
                )
            ]

            for _, row in shown.iterrows():
                rows.append(
                    [
                        _format_cell(
                            value
                        )
                        for value in row.tolist()
                    ]
                )

            col_count = len(
                rows[0]
            )

            total_width = 180 * mm
            col_width = (
                total_width
                / max(
                    col_count,
                    1,
                )
            )

            report_table = Table(
                rows,
                colWidths=[
                    col_width
                ]
                * col_count,
                repeatRows=1,
            )

            report_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        navy,
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTNAME",
                        (0, 1),
                        (0, -1),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8.1,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.45,
                        colors.HexColor("#C8D3DF"),
                    ),
                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, -1),
                        colors.white,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                ])
            )

            story.append(
                report_table
            )

        story.append(
            Spacer(
                1,
                15 * mm,
            )
        )

        story.append(
            Paragraph(
                "INTERPRETATION BOUNDARIES",
                ParagraphStyle(
                    "ShotBoundariesTitle",
                    parent=h1,
                    fontName="Helvetica-Bold",
                    fontSize=17,
                    leading=21,
                    textColor=navy,
                    spaceAfter=10,
                ),
            )
        )

        boundary_text_style = ParagraphStyle(
            "ShotBoundaryBody",
            parent=body,
            fontName="Helvetica",
            fontSize=8.6,
            leading=12,
            textColor=colors.HexColor("#222222"),
        )

        boundary_bullet_style = ParagraphStyle(
            "ShotBoundaryBullet",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=blue,
            alignment=1,
        )

        boundary_data = [
            [
                Paragraph("•", boundary_bullet_style),
                Paragraph(
                    "Shot locations describe recorded StatsBomb shot events "
                    "rather than continuous optical-tracking positions or attacking control.",
                    boundary_text_style,
                ),
            ],
            [
                Paragraph("•", boundary_bullet_style),
                Paragraph(
                    "StatsBomb xG estimates the probability-quality of a recorded chance; "
                    "it is an analytical estimate rather than a guarantee that a shot should score.",
                    boundary_text_style,
                ),
            ],
            [
                Paragraph("•", boundary_bullet_style),
                Paragraph(
                    "Marker size represents xG and star markers identify goals. "
                    "The Full Match view excludes penalty-shootout period 5.",
                    boundary_text_style,
                ),
            ],
            [
                Paragraph("•", boundary_bullet_style),
                Paragraph(
                    "Shot-volume and xG comparisons should be interpreted with game state, "
                    "tactical role, opponent behaviour and match context.",
                    boundary_text_style,
                ),
            ],
        ]

        boundary_table = Table(
            boundary_data,
            colWidths=[
                8 * mm,
                165 * mm,
            ],
        )

        boundary_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    pale,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (0, -1),
                    blue,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (1, 0),
                    (1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9.0,
                ),
                (
                    "LEADING",
                    (0, 0),
                    (-1, -1),
                    13,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    blue,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0,
                    colors.white,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ])
        )

        story.append(
            boundary_table
        )

        story.append(
            Spacer(
                1,
                12 * mm,
            )
        )

        story.append(
            Paragraph(
                (
                    f"Generated by LiveMatch Intelligence  |  Match ID {match_id}  |  "
                    "Focused section report"
                ),
                small,
            )
        )

    else:
        story.append(
            Paragraph(
                "Evidence Table",
                h1,
            )
        )

    if section_key not in {"shot_analysis", "tactical", "live_intelligence", "ml_prediction"} and table.empty:
        story.append(
            Paragraph(
                "No structured comparison table was available.",
                body,
            )
        )
    elif section_key not in {"shot_analysis", "tactical", "live_intelligence", "ml_prediction"}:
        shown = table.head(15).copy()

        rows = [
            list(shown.columns)
        ]

        for _, row in shown.iterrows():
            rows.append(
                [
                    _format_cell(value)
                    for value in row.tolist()
                ]
            )

        col_count = len(rows[0])
        total_width = 180 * mm
        col_width = total_width / max(col_count, 1)

        report_table = Table(
            rows,
            colWidths=[col_width] * col_count,
            repeatRows=1,
        )

        report_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#EFEFEF"),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7.5,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#CCCCCC"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
            ])
        )

        story.append(report_table)

    if section_key == "progression":
        summary = payload.get(
            "result",
            {},
        ).get(
            "summaries",
            pd.DataFrame(),
        )

        if not summary.empty:
            story.append(
                Spacer(1, 4 * mm)
            )

            story.append(
                Paragraph(
                    "Leading Contributors",
                    h1,
                )
            )

            contributor_rows = [[
                "Team",
                "Top Progressor",
                "Top Final Third Contributor",
                "Top Box Entry Contributor",
            ]]

            for _, row in summary.iterrows():
                contributor_rows.append([
                    _format_cell(row.get("Team", "")),
                    _format_cell(row.get("Top Progressor", "")),
                    _format_cell(row.get("Top Final Third Contributor", "")),
                    _format_cell(row.get("Top Box Entry Contributor", "")),
                ])

            contributor_table = Table(
                contributor_rows,
                colWidths=[
                    32 * mm,
                    48 * mm,
                    50 * mm,
                    50 * mm,
                ],
                repeatRows=1,
            )

            contributor_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#EFEFEF"),
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7.2,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.HexColor("#CCCCCC"),
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                ])
            )

            story.append(contributor_table)

    if section_key not in {"shot_analysis", "tactical", "live_intelligence", "ml_prediction"}:
        story.append(
            Spacer(1, 5 * mm)
        )

        story.append(
            Paragraph(
                "Interpretation Boundaries",
                h1,
            )
        )

    boundaries = [
        "Event data describes recorded actions rather than continuous optical-tracking positions.",
        "Project-defined indicators such as territory and progressive actions should be interpreted according to their stated methodology.",
        "Comparison results should be interpreted in tactical and positional context rather than as absolute player or team quality rankings.",
    ]

    if section_key == "ml_prediction":
        boundaries.append(
            "Experimental ML probabilities are research estimates and are not production-grade forecasting probabilities."
        )

    if section_key == "tactical":
        boundaries.append(
            "Tactical-change signals are analyst flags, not confirmed coaching instructions or exact formation changes."
        )

    if section_key == "passing_network":
        boundaries.append(
            "Passing-network nodes use average pass-event locations; node size represents network involvement, line width represents completed-pass volume, and only stronger passing links are displayed for readability."
        )

    if section_key == "territory":
        boundaries.append(
            "Territory-map zone percentages are shares of each team's located match events; the average-position marker is the mean recorded event location, not continuous spatial control or player-tracking occupancy."
        )

    if section_key == "progression":
        boundaries.append(
            "Progressive actions use the project-defined heuristic of a completed pass or carry advancing at least 10 StatsBomb X-units toward the opponent goal. The pitch visual shows only the strongest actions by forward distance for readability; summary totals use all qualifying actions."
        )

    if section_key not in {"shot_analysis", "tactical", "live_intelligence", "ml_prediction"}:
        for item in boundaries:
            story.append(
                Paragraph(
                    f"• {item}",
                    body,
                )
            )

    if section_key not in {"shot_analysis", "tactical", "live_intelligence", "ml_prediction"}:
        story.append(
            Spacer(1, 4 * mm)
        )

        story.append(
            Paragraph(
                (
                    f"Generated by LiveMatch Intelligence | Match ID {match_id} | "
                    "Focused section report"
                ),
                small,
            )
        )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=f"LiveMatch Intelligence - {payload['title']}",
        author="LiveMatch Intelligence",
    )

    doc.build(story)

    return output_path


def available_section_reports() -> List[Tuple[str, str]]:
    return list(SECTION_LABELS.items())


if __name__ == "__main__":
    print(
        "Available section reports:"
    )

    for key, label in available_section_reports():
        print(
            f"- {key}: {label}"
        )
