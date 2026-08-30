from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# ---------------------------------------------------------
# Allow dashboard to import modules from the src folder
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.match_analyzer import load_match_events, build_match_summary
from src.player_analyzer import calculate_player_stats
from src.match_outcome_predictor import predict_match_outcome
from src.tactical_change_detector import detect_tactical_changes
from src.pass_network_analyzer import analyze_pass_networks
from src.possession_territory_analyzer import analyze_possession_territory
from src.progressive_actions_analyzer import analyze_progressive_actions
from src.match_report_generator import generate_match_report, generate_pdf_report
from src.section_report_generator import (
    generate_section_png,
    generate_section_pdf,
    available_section_reports,
)
from src.match_catalog import (
    load_open_competitions,
    load_open_matches,
)
from src.score_utils import match_score_breakdown, exclude_shootout_events
# ---------------------------------------------------------
# Local image assets
# ---------------------------------------------------------

ASSET_DIR = Path(__file__).resolve().parent / "assets"
PLAYER_IMAGE_DIR = ASSET_DIR / "players"
TEAM_IMAGE_DIR = ASSET_DIR / "teams"

PLAYER_IMAGES = {
    "Granit Xhaka": PLAYER_IMAGE_DIR / "granit_xhaka.jpg",
    "Florian Wirtz": PLAYER_IMAGE_DIR / "florian_wirtz.jpg",
}

TEAM_IMAGES = {
    "Borussia Dortmund": TEAM_IMAGE_DIR / "borussia_dortmund.png",
    "Bayer Leverkusen": TEAM_IMAGE_DIR / "bayer_leverkusen.png",
}


def show_player_image(player_name, width=180):
    """Display a local player image safely when available."""

    image_path = PLAYER_IMAGES.get(player_name)

    if not image_path:
        st.caption("Player image not added yet.")
        return False

    if not image_path.exists():
        st.caption("Player image not added yet.")
        return False

    try:
        st.image(str(image_path), width=width)
        return True

    except Exception:
        st.warning(
            f"Image for {player_name} could not be displayed. "
            "Please replace it with a valid JPG or PNG file."
        )
        return False


def show_team_image(team_name, width=110):
    """Display a local club badge when one has been added."""
    image_path = TEAM_IMAGES.get(team_name)

    if image_path and image_path.exists():
        st.image(str(image_path), width=width)
        return True

    return False


# ---------------------------------------------------------
# Live intelligence helpers
# ---------------------------------------------------------

def safe_numeric(series):
    """Convert a pandas Series to numeric values safely."""
    return pd.to_numeric(series, errors="coerce").fillna(0)


def events_until_minute(events_df, minute):
    """Return only events that had occurred by the selected match minute."""
    if "minute" not in events_df.columns:
        return events_df.copy()

    return events_df[
        safe_numeric(events_df["minute"]) <= minute
    ].copy()


def events_in_window(events_df, end_minute, window=10):
    """Return events in a rolling match-minute window."""
    if "minute" not in events_df.columns:
        return events_df.copy()

    minutes = safe_numeric(events_df["minute"])
    start_minute = max(0, end_minute - window)

    return events_df[
        (minutes > start_minute) & (minutes <= end_minute)
    ].copy()


def team_live_metrics(events_df, team_name):
    """Calculate core live metrics for one team from a subset of events."""
    team_events = events_df[
        events_df["team"] == team_name
    ].copy()

    passes = team_events[team_events["type"] == "Pass"]
    shots = team_events[team_events["type"] == "Shot"]
    pressures = team_events[team_events["type"] == "Pressure"]
    carries = team_events[team_events["type"] == "Carry"]
    recoveries = team_events[team_events["type"] == "Ball Recovery"]
    interceptions = team_events[team_events["type"] == "Interception"]

    total_xg = 0.0
    goals = 0

    if not shots.empty:
        if "shot_statsbomb_xg" in shots.columns:
            total_xg = safe_numeric(shots["shot_statsbomb_xg"]).sum()

        if "shot_outcome" in shots.columns:
            goals = shots["shot_outcome"].eq("Goal").sum()

    completed_passes = 0

    if not passes.empty and "pass_outcome" in passes.columns:
        completed_passes = passes["pass_outcome"].isna().sum()

    pass_completion = (
        completed_passes / len(passes) * 100
        if len(passes) > 0
        else 0.0
    )

    return {
        "Goals": int(goals),
        "Shots": int(len(shots)),
        "xG": float(total_xg),
        "Passes": int(len(passes)),
        "Pass Completion %": float(pass_completion),
        "Pressures": int(len(pressures)),
        "Carries": int(len(carries)),
        "Recoveries": int(len(recoveries)),
        "Interceptions": int(len(interceptions)),
    }


def momentum_points(metrics):
    """
    Prototype momentum score.

    The score deliberately combines attacking threat and activity.
    It is a transparent heuristic for the MVP, not a trained model.
    """
    return (
        metrics["Shots"] * 4.0
        + metrics["xG"] * 10.0
        + metrics["Pressures"] * 0.35
        + metrics["Carries"] * 0.08
        + metrics["Recoveries"] * 0.20
        + metrics["Passes"] * 0.03
    )


def relative_momentum_score(team_a_metrics, team_b_metrics):
    """Convert two raw momentum values into comparable 0-100 shares."""
    score_a = momentum_points(team_a_metrics)
    score_b = momentum_points(team_b_metrics)
    total = score_a + score_b

    if total <= 0:
        return 50.0, 50.0

    return (
        round(score_a / total * 100, 1),
        round(score_b / total * 100, 1),
    )


def build_momentum_timeline(
    events_df,
    team_a_name,
    team_b_name,
    max_minute,
    window=10,
):
    """Build minute-by-minute rolling momentum shares for both teams."""
    rows = []

    for minute in range(1, max_minute + 1):
        window_df = events_in_window(
            events_df,
            minute,
            window,
        )

        team_a_metrics = team_live_metrics(
            window_df,
            team_a_name,
        )

        team_b_metrics = team_live_metrics(
            window_df,
            team_b_name,
        )

        score_a, score_b = relative_momentum_score(
            team_a_metrics,
            team_b_metrics,
        )

        rows.append(
            {
                "Minute": minute,
                team_a_name: score_a,
                team_b_name: score_b,
            }
        )

    return pd.DataFrame(rows)


def get_goal_events(events_df):
    """Return goal events with minute, team and scorer where available."""
    required = {"type", "minute", "shot_outcome"}

    if not required.issubset(events_df.columns):
        return pd.DataFrame(
            columns=["Minute", "Team", "Scorer", "Label"]
        )

    goals = events_df[
        (events_df["type"] == "Shot")
        & (events_df["shot_outcome"] == "Goal")
    ].copy()

    if goals.empty:
        return pd.DataFrame(
            columns=["Minute", "Team", "Scorer", "Label"]
        )

    goals["Minute"] = safe_numeric(goals["minute"]).astype(int)

    if "team" in goals.columns:
        goals["Team"] = goals["team"].astype(str)
    else:
        goals["Team"] = "Unknown team"

    if "player" in goals.columns:
        goals["Scorer"] = goals["player"].fillna("Unknown scorer").astype(str)
    else:
        goals["Scorer"] = "Unknown scorer"

    goals["Label"] = (
        goals["Minute"].astype(str)
        + "' "
        + goals["Scorer"]
    )

    return goals[
        ["Minute", "Team", "Scorer", "Label"]
    ].sort_values("Minute")


def filter_match_period(events_df, period_label):
    """Filter events for full match, first half or second half."""
    if period_label == "Full Match":
        return events_df.copy()

    if "period" in events_df.columns:
        period_values = safe_numeric(events_df["period"])

        if period_label == "First Half":
            return events_df[period_values == 1].copy()

        if period_label == "Second Half":
            return events_df[period_values == 2].copy()

    if "minute" in events_df.columns:
        minute_values = safe_numeric(events_df["minute"])

        if period_label == "First Half":
            return events_df[minute_values <= 45].copy()

        if period_label == "Second Half":
            return events_df[minute_values > 45].copy()

    return events_df.copy()


def build_average_positions(events_df, team_name):
    """
    Estimate average player positions using recorded event locations.

    This is not optical tracking data; it represents the average location
    of a player's recorded actions.
    """
    required_columns = {"team", "player", "location"}

    if not required_columns.issubset(events_df.columns):
        return pd.DataFrame()

    team_events = events_df[
        (events_df["team"] == team_name)
        & events_df["player"].notna()
        & events_df["location"].notna()
    ].copy()

    if team_events.empty:
        return pd.DataFrame()

    def extract_x(location):
        if isinstance(location, (list, tuple)) and len(location) >= 2:
            return float(location[0])
        return None

    def extract_y(location):
        if isinstance(location, (list, tuple)) and len(location) >= 2:
            return float(location[1])
        return None

    team_events["X"] = team_events["location"].apply(extract_x)
    team_events["Y"] = team_events["location"].apply(extract_y)

    team_events = team_events.dropna(subset=["X", "Y"]).copy()

    if team_events.empty:
        return pd.DataFrame()

    positions = (
        team_events
        .groupby("player", as_index=False)
        .agg(
            X=("X", "mean"),
            Y=("Y", "mean"),
            Involvements=("X", "size"),
        )
        .rename(columns={"player": "Player"})
    )

    positions["Team"] = team_name

    positions["Zone"] = positions["X"].apply(
        lambda x: (
            "Defensive Third"
            if x < 40
            else "Middle Third"
            if x < 80
            else "Attacking Third"
        )
    )

    positions["Channel"] = positions["Y"].apply(
        lambda y: (
            "Left"
            if y < 26.7
            else "Centre"
            if y < 53.3
            else "Right"
        )
    )

    return positions.sort_values("Involvements", ascending=False)


def tactical_team_summary(position_df):
    """Build compact tactical summary values from average-position data."""
    if position_df.empty:
        return {
            "Players": 0,
            "Avg X": 0.0,
            "Most Advanced": "N/A",
            "Deepest": "N/A",
            "Most Involved": "N/A",
        }

    most_advanced = position_df.loc[position_df["X"].idxmax()]
    deepest = position_df.loc[position_df["X"].idxmin()]
    most_involved = position_df.loc[position_df["Involvements"].idxmax()]

    return {
        "Players": int(len(position_df)),
        "Avg X": float(position_df["X"].mean()),
        "Most Advanced": str(most_advanced["Player"]),
        "Deepest": str(deepest["Player"]),
        "Most Involved": str(most_involved["Player"]),
    }


def tactical_zone_counts(position_df):
    """Count players by average field third."""
    if position_df.empty:
        return {
            "Defensive Third": 0,
            "Middle Third": 0,
            "Attacking Third": 0,
        }

    counts = position_df["Zone"].value_counts()

    return {
        "Defensive Third": int(counts.get("Defensive Third", 0)),
        "Middle Third": int(counts.get("Middle Third", 0)),
        "Attacking Third": int(counts.get("Attacking Third", 0)),
    }


def build_shot_map_data(events_df):
    """Prepare shot locations and metadata from StatsBomb event data."""
    if "type" not in events_df.columns:
        return pd.DataFrame()

    shots = events_df[
        events_df["type"] == "Shot"
    ].copy()

    if shots.empty or "location" not in shots.columns:
        return pd.DataFrame()

    def extract_x(location):
        if isinstance(location, (list, tuple)) and len(location) >= 2:
            return float(location[0])
        return None

    def extract_y(location):
        if isinstance(location, (list, tuple)) and len(location) >= 2:
            return float(location[1])
        return None

    shots["X"] = shots["location"].apply(extract_x)
    shots["Y"] = shots["location"].apply(extract_y)

    shots = shots.dropna(
        subset=["X", "Y"]
    ).copy()

    if "player" in shots.columns:
        shots["Player"] = shots["player"].fillna("Unknown player").astype(str)
    else:
        shots["Player"] = "Unknown player"

    if "team" in shots.columns:
        shots["Team"] = shots["team"].fillna("Unknown team").astype(str)
    else:
        shots["Team"] = "Unknown team"

    if "minute" in shots.columns:
        shots["Minute"] = safe_numeric(shots["minute"]).astype(int)
    else:
        shots["Minute"] = 0

    if "shot_statsbomb_xg" in shots.columns:
        shots["xG"] = safe_numeric(shots["shot_statsbomb_xg"])
    else:
        shots["xG"] = 0.0

    if "shot_outcome" in shots.columns:
        shots["Outcome"] = shots["shot_outcome"].fillna("Unknown").astype(str)
    else:
        shots["Outcome"] = "Unknown"

    shots["Is Goal"] = shots["Outcome"].eq("Goal")
    shots["Marker"] = shots["Is Goal"].map(
        {
            True: "Goal",
            False: "Shot",
        }
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
            "Marker",
        ]
    ]


def shot_summary_for_team(shot_df, team_name):
    """Return compact shot-map summary metrics for one team."""
    team_shots = shot_df[
        shot_df["Team"] == team_name
    ]

    if team_shots.empty:
        return {
            "Shots": 0,
            "Goals": 0,
            "xG": 0.0,
            "Avg xG": 0.0,
        }

    return {
        "Shots": int(len(team_shots)),
        "Goals": int(team_shots["Is Goal"].sum()),
        "xG": float(team_shots["xG"].sum()),
        "Avg xG": float(team_shots["xG"].mean()),
    }


def baseline_match_probabilities(
    minute,
    max_minute,
    team_a_live,
    team_b_live,
    team_a_momentum,
    team_b_momentum,
):
    """
    Transparent prototype match-state probability estimator.

    IMPORTANT:
    This is not a trained machine-learning model. It combines scoreline,
    cumulative xG, shots, rolling momentum and time remaining into a
    softmax-style baseline so the MVP can demonstrate live predictive UX.
    """

    goal_diff = team_a_live["Goals"] - team_b_live["Goals"]
    xg_diff = team_a_live["xG"] - team_b_live["xG"]
    shot_diff = team_a_live["Shots"] - team_b_live["Shots"]
    momentum_diff = (team_a_momentum - team_b_momentum) / 100.0

    match_progress = min(
        max(float(minute) / max(float(max_minute), 1.0), 0.0),
        1.0,
    )

    # Scoreline matters increasingly as the match approaches full time.
    score_weight = 1.25 + (2.75 * match_progress)

    # Non-score performance indicators matter throughout the match.
    performance_signal = (
        1.35 * xg_diff
        + 0.08 * shot_diff
        + 0.90 * momentum_diff
    )

    team_a_strength = (
        score_weight * goal_diff
        + performance_signal
    )

    team_b_strength = -team_a_strength

    # Draw likelihood is strongest when score and performance are balanced,
    # but it gradually falls as a decisive scoreline persists late in a match.
    balance_penalty = (
        1.60 * abs(goal_diff)
        + 0.55 * abs(xg_diff)
        + 0.035 * abs(shot_diff)
        + 0.60 * abs(momentum_diff)
    )

    draw_strength = (
        1.15
        - balance_penalty
        + 0.35 * match_progress
    )

    # At full time, make the observed result overwhelmingly dominant.
    if minute >= max_minute:
        if goal_diff > 0:
            return 99.0, 0.5, 0.5
        if goal_diff < 0:
            return 0.5, 0.5, 99.0
        return 0.5, 99.0, 0.5

    logits = [
        team_a_strength,
        draw_strength,
        team_b_strength,
    ]

    max_logit = max(logits)
    exp_values = [
        math.exp(value - max_logit)
        for value in logits
    ]
    denominator = sum(exp_values)

    probabilities = [
        value / denominator * 100.0
        for value in exp_values
    ]

    return tuple(
        round(value, 1)
        for value in probabilities
    )


def prediction_driver_text(
    team_a_name,
    team_b_name,
    team_a_live,
    team_b_live,
    team_a_momentum,
    team_b_momentum,
):
    """Explain the strongest observable drivers behind the baseline estimate."""
    drivers = []

    goal_diff = team_a_live["Goals"] - team_b_live["Goals"]
    xg_diff = team_a_live["xG"] - team_b_live["xG"]
    shot_diff = team_a_live["Shots"] - team_b_live["Shots"]
    momentum_diff = team_a_momentum - team_b_momentum

    if goal_diff != 0:
        leader = team_a_name if goal_diff > 0 else team_b_name
        drivers.append(
            f"**Scoreline:** {leader} are ahead."
        )
    else:
        drivers.append(
            "**Scoreline:** the match is currently level."
        )

    if abs(xg_diff) >= 0.20:
        leader = team_a_name if xg_diff > 0 else team_b_name
        drivers.append(
            f"**Chance quality:** {leader} lead cumulative xG."
        )

    if abs(shot_diff) >= 3:
        leader = team_a_name if shot_diff > 0 else team_b_name
        drivers.append(
            f"**Shot volume:** {leader} have created more attempts."
        )

    if abs(momentum_diff) >= 8:
        leader = team_a_name if momentum_diff > 0 else team_b_name
        drivers.append(
            f"**Recent momentum:** {leader} are stronger in the selected rolling window."
        )

    if len(drivers) == 1:
        drivers.append(
            "**Performance balance:** the other live indicators are relatively close."
        )

    return drivers


def build_live_alerts(
    team_a_name,
    team_b_name,
    team_a_metrics,
    team_b_metrics,
):
    """Generate simple data-driven live analyst alerts."""
    alerts = []

    xg_gap = team_a_metrics["xG"] - team_b_metrics["xG"]
    shot_gap = team_a_metrics["Shots"] - team_b_metrics["Shots"]
    pressure_gap = team_a_metrics["Pressures"] - team_b_metrics["Pressures"]

    if abs(xg_gap) >= 0.30:
        stronger = team_a_name if xg_gap > 0 else team_b_name
        alerts.append(
            f"Chance quality swing: **{stronger}** created the "
            f"stronger chances in the current rolling window."
        )

    if abs(shot_gap) >= 3:
        stronger = team_a_name if shot_gap > 0 else team_b_name
        alerts.append(
            f"Shot-volume pressure: **{stronger}** recorded at least "
            f"three more attempts in the current rolling window."
        )

    if abs(pressure_gap) >= 10:
        stronger = team_a_name if pressure_gap > 0 else team_b_name
        alerts.append(
            f"Pressing intensity: **{stronger}** registered substantially "
            f"more pressure actions in the current rolling window."
        )

    return alerts

# ---------------------------------------------------------
# Experimental ML checkpoint helpers
# ---------------------------------------------------------

ML_SNAPSHOT_MINUTES = [15, 30, 45, 60, 75, 85]


def ml_checkpoint_for_minute(match_minute):
    """
    Return the latest trained ML snapshot at or before the replay minute.

    The historical ML pipeline was trained at fixed match-state checkpoints,
    so inference should use the same checkpoint structure.
    """
    eligible = [
        minute
        for minute in ML_SNAPSHOT_MINUTES
        if minute <= int(match_minute)
    ]

    if not eligible:
        return None

    return max(eligible)


# ---------------------------------------------------------
# Pass-network display helpers
# ---------------------------------------------------------

def short_player_name(player_name):
    """
    Compact player labels for dense pitch visualisations.
    Keeps the final surname token, plus a preceding token when needed.
    """
    name = str(player_name).strip()

    if not name:
        return ""

    parts = [
        part
        for part in name.split()
        if part
    ]

    if len(parts) == 1:
        return parts[0]

    common_particles = {
        "de",
        "da",
        "do",
        "dos",
        "del",
        "van",
        "von",
        "di",
        "la",
        "le",
    }

    if (
        len(parts) >= 3
        and parts[-2].lower() in common_particles
    ):
        return " ".join(
            parts[-2:]
        )

    return parts[-1]


# ---------------------------------------------------------
# App configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="LiveMatch Intelligence",
    page_icon="⚽",
    layout="wide"
)

# ---------------------------------------------------------
# Match selection
# ---------------------------------------------------------

@st.cache_data(
    show_spinner=False
)
def get_competition_catalog():
    return load_open_competitions()


@st.cache_data(
    show_spinner=False
)
def get_match_catalog(
    competition_id,
    season_id,
):
    return load_open_matches(
        competition_id,
        season_id,
    )


competition_catalog = (
    get_competition_catalog()
)

if competition_catalog.empty:
    st.error(
        "No StatsBomb Open Data competitions were available."
    )
    st.stop()


st.sidebar.header(
    "🏆 Match Selector"
)

competition_labels = []

for _, competition_row in (
    competition_catalog.iterrows()
):
    country = str(
        competition_row.get(
            "country_name",
            "",
        )
    ).strip()

    competition_name = str(
        competition_row[
            "competition_name"
        ]
    )

    season_name = str(
        competition_row[
            "season_name"
        ]
    )

    if country:
        label = (
            f"{country} | "
            f"{competition_name} — "
            f"{season_name}"
        )
    else:
        label = (
            f"{competition_name} — "
            f"{season_name}"
        )

    competition_labels.append(
        label
    )


# Prefer the Bundesliga season containing the current MVP match
# when it exists in StatsBomb Open Data.
default_competition_index = 0

for idx, (_, row) in enumerate(
    competition_catalog.iterrows()
):
    competition_text = str(
        row.get(
            "competition_name",
            "",
        )
    ).lower()

    season_text = str(
        row.get(
            "season_name",
            "",
        )
    )

    if (
        "bundesliga"
        in competition_text
        and season_text
        in {
            "2023/2024",
            "2023-2024",
        }
    ):
        default_competition_index = idx
        break


selected_competition_label = (
    st.sidebar.selectbox(
        "Competition & Season",
        options=competition_labels,
        index=default_competition_index,
        key="competition_season_selector",
    )
)

selected_competition_index = (
    competition_labels.index(
        selected_competition_label
    )
)

selected_competition = (
    competition_catalog.iloc[
        selected_competition_index
    ]
)

selected_competition_id = int(
    selected_competition[
        "competition_id"
    ]
)

selected_season_id = int(
    selected_competition[
        "season_id"
    ]
)

match_catalog = get_match_catalog(
    selected_competition_id,
    selected_season_id,
)

if match_catalog.empty:
    st.sidebar.error(
        "No matches were available for this competition and season."
    )
    st.stop()


match_labels = (
    match_catalog[
        "match_label"
    ]
    .astype(str)
    .tolist()
)

default_match_index = 0

# Preserve the original MVP demonstration as the default
# when it exists in the selected competition.
current_mvp_rows = match_catalog[
    match_catalog[
        "match_id"
    ].astype(int)
    == 3895309
]

if not current_mvp_rows.empty:
    current_mvp_match_id = int(
        current_mvp_rows.iloc[0][
            "match_id"
        ]
    )

    for idx, row in (
        match_catalog.iterrows()
    ):
        if int(
            row[
                "match_id"
            ]
        ) == current_mvp_match_id:
            default_match_index = idx
            break


selected_match_label = (
    st.sidebar.selectbox(
        "Match",
        options=match_labels,
        index=default_match_index,
        key="match_selector",
    )
)

selected_match_index = (
    match_labels.index(
        selected_match_label
    )
)

selected_match = (
    match_catalog.iloc[
        selected_match_index
    ]
)

MATCH_ID = int(
    selected_match[
        "match_id"
    ]
)

st.sidebar.caption(
    f"StatsBomb Match ID: {MATCH_ID}"
)

st.sidebar.divider()

st.sidebar.header(
    "🧭 Navigation"
)

NAV_PAGE = st.sidebar.radio(
    "Workspace",
    [
        "🏠 Match Overview",
        "📊 Team Intelligence",
        "👤 Player Intelligence",
        "🧠 Tactical Intelligence",
        "🔗 Passing & Build-up",
        "🗺️ Possession & Territory",
        "🚀 Progressive Actions",
        "🎯 Shot Analysis",
        "📈 Live Intelligence",
        "📄 Reports & Export",
    ],
    key="main_navigation",
    label_visibility="collapsed",
)

if (
    st.session_state.get(
        "_active_match_id"
    )
    != MATCH_ID
):
    # Prevent a report preview from the previous selected
    # match being shown after the user changes matches.
    for state_key in [
        "generated_match_report_path",
        "generated_match_pdf_path",
        "focused_section_png_path",
        "focused_section_pdf_path",
        "focused_section_last_label",
    ]:
        st.session_state.pop(
            state_key,
            None,
        )

    st.session_state[
        "_active_match_id"
    ] = MATCH_ID


# ---------------------------------------------------------
# Load match data
# ---------------------------------------------------------

@st.cache_data
def get_match_data(match_id):
    raw_events = load_match_events(match_id)

    events = exclude_shootout_events(
        raw_events
    )

    summary = build_match_summary(
        events
    )

    return (
        raw_events,
        events,
        summary,
    )


raw_events, events, summary = get_match_data(
    MATCH_ID
)

player_stats = calculate_player_stats(
    events
)


# ---------------------------------------------------------
# Extract team information
# ---------------------------------------------------------

team_1 = summary.iloc[0]
team_2 = summary.iloc[1]

team_1_name = team_1["Team"]
team_2_name = team_2["Team"]

score_breakdown = match_score_breakdown(
    raw_events,
    team_1_name,
    team_2_name,
)

team_1_goals = int(
    score_breakdown[
        "team_1_goals"
    ]
)

team_2_goals = int(
    score_breakdown[
        "team_2_goals"
    ]
)

# Correct the shared summary so downstream dashboard comparisons
# use match goals excluding penalty-shootout kicks.
summary.loc[
    summary["Team"] == team_1_name,
    "Goals",
] = team_1_goals

summary.loc[
    summary["Team"] == team_2_name,
    "Goals",
] = team_2_goals

team_1 = summary.iloc[0]
team_2 = summary.iloc[1]


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("⚽ LiveMatch Intelligence")

st.subheader(
    "Real-Time Football Analytics & Decision Support Platform"
)

st.write(
    """
    LiveMatch Intelligence transforms football event data into
    actionable match, team and player insights for analysts,
    coaches and performance staff.
    """
)

st.divider()

st.caption(f"Current workspace: {NAV_PAGE}  •  Match ID {MATCH_ID}")


if NAV_PAGE == "🏠 Match Overview":
    # ---------------------------------------------------------
    # Match Centre
    # ---------------------------------------------------------

    st.subheader("🏟️ Match Centre")

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        show_team_image(team_1_name)
        st.markdown(f"### {team_1_name}")
        st.metric("Goals", team_1_goals)

    with col2:
        st.markdown("### FT")
        st.markdown(
            f"## {team_1_goals} — {team_2_goals}"
        )

        if score_breakdown["has_shootout"]:
            shootout_winner = (
                score_breakdown[
                    "winner_on_penalties"
                ]
                or "Shootout winner"
            )

            st.caption(
                f"{shootout_winner} won "
                f"{score_breakdown['team_1_penalties']}–"
                f"{score_breakdown['team_2_penalties']} on penalties"
            )

    with col3:
        show_team_image(team_2_name)
        st.markdown(f"### {team_2_name}")
        st.metric("Goals", team_2_goals)

    st.divider()


    # ---------------------------------------------------------
    # Match metrics
    # ---------------------------------------------------------

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            f"{team_1_name} xG",
            f"{team_1['xG']:.2f}"
        )

    with metric2:
        st.metric(
            f"{team_2_name} xG",
            f"{team_2['xG']:.2f}"
        )

    with metric3:
        total_shots = int(
            team_1["Shots"] + team_2["Shots"]
        )
        st.metric("Total Shots", total_shots)

    with metric4:
        st.metric(
            "Match Events",
            f"{len(events):,}"
        )

    st.divider()

if NAV_PAGE == "📊 Team Intelligence":
    # ---------------------------------------------------------
    # Team Performance Comparison
    # ---------------------------------------------------------

    st.subheader("📊 Team Performance Comparison")

    display_columns = [
        "Team",
        "Goals",
        "Shots",
        "xG",
        "Passes",
        "Completed Passes",
        "Pass Completion %",
        "Carries",
        "Pressures",
        "Interceptions",
    ]

    st.dataframe(
        summary[display_columns],
        width="stretch",
        hide_index=True
    )

    st.divider()


    # ---------------------------------------------------------
    # Team Intelligence
    # ---------------------------------------------------------

    st.subheader("🧠 Team Intelligence")

    team_insights = []

    if team_1["xG"] > team_2["xG"]:
        team_insights.append(
            f"**{team_1_name}** generated the stronger chance quality based on xG."
        )
    elif team_2["xG"] > team_1["xG"]:
        team_insights.append(
            f"**{team_2_name}** generated the stronger chance quality based on xG."
        )

    if team_1["Shots"] > team_2["Shots"]:
        team_insights.append(
            f"**{team_1_name}** recorded more shots."
        )
    elif team_2["Shots"] > team_1["Shots"]:
        team_insights.append(
            f"**{team_2_name}** recorded more shots."
        )

    if team_1["Pass Completion %"] > team_2["Pass Completion %"]:
        team_insights.append(
            f"**{team_1_name}** had the better pass completion rate."
        )
    elif team_2["Pass Completion %"] > team_1["Pass Completion %"]:
        team_insights.append(
            f"**{team_2_name}** had the better pass completion rate."
        )

    if team_1["Pressures"] > team_2["Pressures"]:
        team_insights.append(
            f"**{team_1_name}** applied more pressure actions."
        )
    elif team_2["Pressures"] > team_1["Pressures"]:
        team_insights.append(
            f"**{team_2_name}** applied more pressure actions."
        )

    for insight in team_insights:
        st.info(insight)

    st.divider()

if NAV_PAGE == "👤 Player Intelligence":
    # ---------------------------------------------------------
    # Player Intelligence
    # ---------------------------------------------------------

    st.subheader("👤 Player Intelligence")

    st.write(
        "Select a player to explore individual match performance."
    )

    player_names = sorted(
        player_stats["Player"].dropna().unique()
    )

    selected_player = st.selectbox(
        "Select Player",
        player_names,
        key="single_player"
    )

    selected_stats = player_stats[
        player_stats["Player"] == selected_player
    ].iloc[0]

    player_profile_image, player_profile_text = st.columns([1, 4])

    with player_profile_image:
        show_player_image(selected_player, width=180)

    with player_profile_text:
        st.markdown(f"### {selected_player}")
        st.write(f"**Team:** {selected_stats['Team']}")
        st.caption(
            "The image appears automatically when a matching local file "
            "is available in dashboard/assets/players."
        )

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric("Passes", int(selected_stats["Passes"]))

    with p2:
        st.metric(
            "Pass Completion %",
            f"{selected_stats['Pass Completion %']:.1f}%"
        )

    with p3:
        st.metric("Carries", int(selected_stats["Carries"]))

    with p4:
        st.metric("Shots", int(selected_stats["Shots"]))

    p5, p6, p7, p8 = st.columns(4)

    with p5:
        st.metric("xG", f"{selected_stats['xG']:.2f}")

    with p6:
        st.metric("Goals", int(selected_stats["Goals"]))

    with p7:
        st.metric("Pressures", int(selected_stats["Pressures"]))

    with p8:
        st.metric(
            "Interceptions",
            int(selected_stats["Interceptions"])
        )

    st.divider()


    # ---------------------------------------------------------
    # Player Comparison
    # ---------------------------------------------------------



    comparison_mode = st.radio(
        "Comparison Mode",
        [
            "Opposition Comparison",
            "Same-Team Comparison"
        ],
        horizontal=True
    )

    teams = sorted(
        player_stats["Team"].dropna().unique()
    )


    # ---------------------------------------------------------
    # Opposition Comparison
    # ---------------------------------------------------------

    if comparison_mode == "Opposition Comparison":

        st.write(
            "Compare one player from each team."
        )

        opposition_col1, opposition_col2 = st.columns(2)

        with opposition_col1:

            team_a = st.selectbox(
                "Player 1 Team",
                teams,
                index=0,
                key="opposition_team_a"
            )

            team_a_players = sorted(
                player_stats[
                    player_stats["Team"] == team_a
                ]["Player"]
                .dropna()
                .unique()
            )

            player_1 = st.selectbox(
                "Select Player 1",
                team_a_players,
                key="opposition_player_1"
            )

        with opposition_col2:

            opponent_teams = [
                team
                for team in teams
                if team != team_a
            ]

            team_b = st.selectbox(
                "Player 2 Team",
                opponent_teams,
                index=0,
                key="opposition_team_b"
            )

            team_b_players = sorted(
                player_stats[
                    player_stats["Team"] == team_b
                ]["Player"]
                .dropna()
                .unique()
            )

            player_2 = st.selectbox(
                "Select Player 2",
                team_b_players,
                key="opposition_player_2"
            )


    # ---------------------------------------------------------
    # Same-Team Comparison
    # ---------------------------------------------------------

    else:

        st.write(
            "Compare two players from the same team."
        )

        same_team = st.selectbox(
            "Select Team",
            teams,
            key="same_team_selection"
        )

        same_team_players = sorted(
            player_stats[
                player_stats["Team"] == same_team
            ]["Player"]
            .dropna()
            .unique()
        )

        same_col1, same_col2 = st.columns(2)

        with same_col1:
            player_1 = st.selectbox(
                "Select Player 1",
                same_team_players,
                index=0,
                key="same_player_1"
            )

        with same_col2:
            default_second = (
                1 if len(same_team_players) > 1 else 0
            )

            player_2 = st.selectbox(
                "Select Player 2",
                same_team_players,
                index=default_second,
                key="same_player_2"
            )


    # ---------------------------------------------------------
    # Get player comparison data
    # ---------------------------------------------------------

    player_1_stats = player_stats[
        player_stats["Player"] == player_1
    ].iloc[0]

    player_2_stats = player_stats[
        player_stats["Player"] == player_2
    ].iloc[0]

    # ---------------------------------------------------------
    # Player comparison profile cards
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Player comparison profile cards
    # ---------------------------------------------------------

    comparison_profile_1, comparison_profile_2 = st.columns(2)

    with comparison_profile_1:
        st.markdown(f"### {player_1}")
        st.caption(str(player_1_stats["Team"]))

        show_player_image(player_1, width=180)

        st.markdown("#### Match Statistics")

        stat_a1, stat_a2 = st.columns(2)

        with stat_a1:
            st.metric("Passes", int(player_1_stats["Passes"]))
            st.metric("Carries", int(player_1_stats["Carries"]))
            st.metric("Shots", int(player_1_stats["Shots"]))
            st.metric("Goals", int(player_1_stats["Goals"]))

        with stat_a2:
            st.metric(
                "Pass Completion",
                f"{player_1_stats['Pass Completion %']:.1f}%"
            )
            st.metric(
                "xG",
                f"{player_1_stats['xG']:.2f}"
            )
            st.metric(
                "Pressures",
                int(player_1_stats["Pressures"])
            )
            st.metric(
                "Interceptions",
                int(player_1_stats["Interceptions"])
            )

        st.metric(
            "Recoveries",
            int(player_1_stats["Recoveries"])
        )


    with comparison_profile_2:
        st.markdown(f"### {player_2}")
        st.caption(str(player_2_stats["Team"]))

        show_player_image(player_2, width=180)

        st.markdown("#### Match Statistics")

        stat_b1, stat_b2 = st.columns(2)

        with stat_b1:
            st.metric("Passes", int(player_2_stats["Passes"]))
            st.metric("Carries", int(player_2_stats["Carries"]))
            st.metric("Shots", int(player_2_stats["Shots"]))
            st.metric("Goals", int(player_2_stats["Goals"]))

        with stat_b2:
            st.metric(
                "Pass Completion",
                f"{player_2_stats['Pass Completion %']:.1f}%"
            )
            st.metric(
                "xG",
                f"{player_2_stats['xG']:.2f}"
            )
            st.metric(
                "Pressures",
                int(player_2_stats["Pressures"])
            )
            st.metric(
                "Interceptions",
                int(player_2_stats["Interceptions"])
            )

        st.metric(
            "Recoveries",
            int(player_2_stats["Recoveries"])
        )

    # ---------------------------------------------------------
    # Player comparison intelligence
    # ---------------------------------------------------------

    st.markdown("### 🧠 Comparison Insight")

    insights = []

    if player_1_stats["Passes"] > player_2_stats["Passes"]:
        insights.append(
            f"**{player_1}** had greater passing involvement."
        )
    elif player_2_stats["Passes"] > player_1_stats["Passes"]:
        insights.append(
            f"**{player_2}** had greater passing involvement."
        )

    if player_1_stats["Pass Completion %"] > player_2_stats["Pass Completion %"]:
        insights.append(
            f"**{player_1}** recorded the higher pass completion rate."
        )
    elif player_2_stats["Pass Completion %"] > player_1_stats["Pass Completion %"]:
        insights.append(
            f"**{player_2}** recorded the higher pass completion rate."
        )

    if player_1_stats["Pressures"] > player_2_stats["Pressures"]:
        insights.append(
            f"**{player_1}** recorded more pressing actions."
        )
    elif player_2_stats["Pressures"] > player_1_stats["Pressures"]:
        insights.append(
            f"**{player_2}** recorded more pressing actions."
        )

    if player_1_stats["xG"] > player_2_stats["xG"]:
        insights.append(
            f"**{player_1}** produced greater goal threat based on xG."
        )
    elif player_2_stats["xG"] > player_1_stats["xG"]:
        insights.append(
            f"**{player_2}** produced greater goal threat based on xG."
        )

    if player_1_stats["Carries"] > player_2_stats["Carries"]:
        insights.append(
            f"**{player_1}** was more involved in ball carrying."
        )
    elif player_2_stats["Carries"] > player_1_stats["Carries"]:
        insights.append(
            f"**{player_2}** was more involved in ball carrying."
        )

    if player_1_stats["Recoveries"] > player_2_stats["Recoveries"]:
        insights.append(
            f"**{player_1}** recorded more ball recoveries."
        )
    elif player_2_stats["Recoveries"] > player_1_stats["Recoveries"]:
        insights.append(
            f"**{player_2}** recorded more ball recoveries."
        )

    if insights:
        for insight in insights:
            st.info(insight)
    else:
        st.info(
            "The selected players recorded similar values "
            "across the current comparison metrics."
        )

    st.divider()

    st.caption(
        "Images are loaded locally. Add more files to dashboard/assets/players "
        "and dashboard/assets/teams, then extend PLAYER_IMAGES or TEAM_IMAGES "
        "near the top of app.py."
    )

if NAV_PAGE == "🧠 Tactical Intelligence":
    # ---------------------------------------------------------
    # Tactical & Positional Analysis
    # ---------------------------------------------------------

    st.subheader("🧠 Tactical & Positional Analysis")

    st.write(
        """
        View estimated average player positions from recorded event locations.
        Use the period selector to compare how each team's shape changed during
        the match.
        """
    )

    tactical_period = st.radio(
        "Tactical View",
        ["Full Match", "First Half", "Second Half"],
        horizontal=True,
        key="tactical_period",
    )

    tactical_events = filter_match_period(events, tactical_period)

    team_1_positions = build_average_positions(
        tactical_events,
        team_1_name,
    )

    team_2_positions = build_average_positions(
        tactical_events,
        team_2_name,
    )

    if not team_1_positions.empty or not team_2_positions.empty:

        tactical_team = st.radio(
            "Team Shape",
            [team_1_name, team_2_name, "Both Teams"],
            horizontal=True,
            key="tactical_team",
        )

        if tactical_team == team_1_name:
            tactical_positions = team_1_positions.copy()
        elif tactical_team == team_2_name:
            tactical_positions = team_2_positions.copy()
        else:
            tactical_positions = pd.concat(
                [team_1_positions, team_2_positions],
                ignore_index=True,
            )

        tactical_pitch_lines = pd.DataFrame(
            [
                {"x": 0, "y": 0, "x2": 120, "y2": 0},
                {"x": 120, "y": 0, "x2": 120, "y2": 80},
                {"x": 120, "y": 80, "x2": 0, "y2": 80},
                {"x": 0, "y": 80, "x2": 0, "y2": 0},
                {"x": 60, "y": 0, "x2": 60, "y2": 80},
                {"x": 0, "y": 18, "x2": 18, "y2": 18},
                {"x": 18, "y": 18, "x2": 18, "y2": 62},
                {"x": 18, "y": 62, "x2": 0, "y2": 62},
                {"x": 102, "y": 18, "x2": 120, "y2": 18},
                {"x": 102, "y": 18, "x2": 102, "y2": 62},
                {"x": 102, "y": 62, "x2": 120, "y2": 62},
                {"x": 0, "y": 30, "x2": 6, "y2": 30},
                {"x": 6, "y": 30, "x2": 6, "y2": 50},
                {"x": 6, "y": 50, "x2": 0, "y2": 50},
                {"x": 114, "y": 30, "x2": 120, "y2": 30},
                {"x": 114, "y": 30, "x2": 114, "y2": 50},
                {"x": 114, "y": 50, "x2": 120, "y2": 50},
            ]
        )

        tactical_pitch = alt.Chart(
            tactical_pitch_lines
        ).mark_rule(
            stroke="#7f7f7f",
            strokeWidth=1.3,
        ).encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-3, 123]), axis=None),
            x2="x2:Q",
            y=alt.Y("y:Q", scale=alt.Scale(domain=[82, -2]), axis=None),
            y2="y2:Q",
        )

        tactical_centre_circle_data = pd.DataFrame(
            {
                "x": [
                    60 + 10 * math.cos(math.radians(angle))
                    for angle in range(0, 361, 5)
                ],
                "y": [
                    40 + 10 * math.sin(math.radians(angle))
                    for angle in range(0, 361, 5)
                ],
                "order": list(range(len(range(0, 361, 5)))),
            }
        )

        tactical_centre_circle = alt.Chart(
            tactical_centre_circle_data
        ).mark_line(
            stroke="#7f7f7f",
            strokeWidth=1.2,
        ).encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-3, 123]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[82, -2]), axis=None),
            order="order:Q",
        )

        tactical_points = alt.Chart(
            tactical_positions
        ).mark_circle(
            opacity=0.88,
            stroke="white",
            strokeWidth=1.2,
        ).encode(
            x=alt.X("X:Q", scale=alt.Scale(domain=[-3, 123]), axis=None),
            y=alt.Y("Y:Q", scale=alt.Scale(domain=[82, -2]), axis=None),
            color=alt.Color("Team:N", title="Team"),
            size=alt.Size(
                "Involvements:Q",
                title="Recorded involvements",
                scale=alt.Scale(range=[160, 1000]),
            ),
            tooltip=[
                alt.Tooltip("Player:N", title="Player"),
                alt.Tooltip("Team:N", title="Team"),
                alt.Tooltip("X:Q", title="Average X", format=".1f"),
                alt.Tooltip("Y:Q", title="Average Y", format=".1f"),
                alt.Tooltip(
                    "Involvements:Q",
                    title="Recorded involvements",
                    format=".0f",
                ),
                alt.Tooltip("Zone:N", title="Field third"),
                alt.Tooltip("Channel:N", title="Channel"),
            ],
        )

        tactical_labels = alt.Chart(
            tactical_positions
        ).mark_text(
            dy=-15,
            fontSize=10,
        ).encode(
            x=alt.X("X:Q", scale=alt.Scale(domain=[-3, 123]), axis=None),
            y=alt.Y("Y:Q", scale=alt.Scale(domain=[82, -2]), axis=None),
            text="Player:N",
            color=alt.Color("Team:N", legend=None),
        )

        tactical_chart = alt.layer(
            tactical_pitch,
            tactical_centre_circle,
            tactical_points,
            tactical_labels,
        ).properties(
            height=560
        ).configure_view(
            strokeWidth=0
        )

        st.altair_chart(tactical_chart, width="stretch")

        st.caption(
            "Average positions are calculated from recorded event locations, "
            "not player-tracking data. Marker size represents the number of "
            "recorded involvements in the selected period."
        )

        st.markdown("### 📍 Team Shape Summary")

        team_1_tactical_summary = tactical_team_summary(team_1_positions)
        team_2_tactical_summary = tactical_team_summary(team_2_positions)

        tactical_summary_col1, tactical_summary_col2 = st.columns(2)

        with tactical_summary_col1:
            show_team_image(team_1_name, width=70)
            st.markdown(f"#### {team_1_name}")
            st.metric(
                "Average Action Position",
                f"{team_1_tactical_summary['Avg X']:.1f} / 120",
            )
            st.write(
                f"**Most advanced:** {team_1_tactical_summary['Most Advanced']}  \n"
                f"**Deepest:** {team_1_tactical_summary['Deepest']}  \n"
                f"**Most involved:** {team_1_tactical_summary['Most Involved']}"
            )

        with tactical_summary_col2:
            show_team_image(team_2_name, width=70)
            st.markdown(f"#### {team_2_name}")
            st.metric(
                "Average Action Position",
                f"{team_2_tactical_summary['Avg X']:.1f} / 120",
            )
            st.write(
                f"**Most advanced:** {team_2_tactical_summary['Most Advanced']}  \n"
                f"**Deepest:** {team_2_tactical_summary['Deepest']}  \n"
                f"**Most involved:** {team_2_tactical_summary['Most Involved']}"
            )

        st.markdown("### 🧩 Tactical Zone Insight")

        team_1_zones = tactical_zone_counts(team_1_positions)
        team_2_zones = tactical_zone_counts(team_2_positions)

        if team_1_tactical_summary["Avg X"] > team_2_tactical_summary["Avg X"]:
            st.info(
                f"**{team_1_name}** recorded a slightly higher average "
                f"action position during the {tactical_period.lower()}."
            )
        elif team_2_tactical_summary["Avg X"] > team_1_tactical_summary["Avg X"]:
            st.info(
                f"**{team_2_name}** recorded a slightly higher average "
                f"action position during the {tactical_period.lower()}."
            )
        else:
            st.info("Both teams recorded the same average action position.")

        st.write(
            f"**{team_1_name}:** "
            f"{team_1_zones['Defensive Third']} players averaged in the defensive third, "
            f"{team_1_zones['Middle Third']} in the middle third, and "
            f"{team_1_zones['Attacking Third']} in the attacking third."
        )

        st.write(
            f"**{team_2_name}:** "
            f"{team_2_zones['Defensive Third']} players averaged in the defensive third, "
            f"{team_2_zones['Middle Third']} in the middle third, and "
            f"{team_2_zones['Attacking Third']} in the attacking third."
        )

    else:
        st.warning(
            "Average-position data could not be calculated for this match."
        )

    st.divider()


    # ---------------------------------------------------------
    # Tactical Change Intelligence
    # ---------------------------------------------------------

    st.subheader("🔄 Tactical Change Intelligence")

    st.write(
        """
        Compare first-half and second-half event behaviour to surface
        meaningful tactical and positional shifts. These are data-derived
        signals from StatsBomb event locations and event counts rather than
        confirmed coaching instructions or optical-tracking formations.
        """
    )

    try:
        tactical_change_result = detect_tactical_changes(
            events,
            team_names=[
                team_1_name,
                team_2_name,
            ],
        )

        tactical_team_changes = tactical_change_result[
            "team_comparisons"
        ]

        tactical_player_changes = tactical_change_result[
            "player_comparisons"
        ]

        tactical_alerts = tactical_change_result[
            "alerts"
        ]

        # -----------------------------------------------------
        # Team tactical shifts
        # -----------------------------------------------------

        st.markdown("### 🧭 Team Tactical Shifts")

        if tactical_team_changes.empty:
            st.info(
                "No half-by-half team comparison could be calculated "
                "from the available event data."
            )

        else:
            tactical_team_display = tactical_team_changes.copy()

            team_display_columns = [
                "Team",
                "First Half Avg X",
                "Second Half Avg X",
                "Average X Change",
                "First Half Pressures",
                "Second Half Pressures",
                "Pressure Change %",
                "First Half Shots",
                "Second Half Shots",
                "Shot Change",
                "First Half xG",
                "Second Half xG",
                "xG Change",
                "Attack Index Change %",
            ]

            team_display_columns = [
                column
                for column in team_display_columns
                if column in tactical_team_display.columns
            ]

            tactical_team_display = tactical_team_display[
                team_display_columns
            ].copy()

            for column in [
                "First Half Avg X",
                "Second Half Avg X",
                "Average X Change",
                "First Half xG",
                "Second Half xG",
                "xG Change",
            ]:
                if column in tactical_team_display.columns:
                    tactical_team_display[column] = (
                        pd.to_numeric(
                            tactical_team_display[column],
                            errors="coerce",
                        )
                        .round(2)
                    )

            for column in [
                "Pressure Change %",
                "Attack Index Change %",
            ]:
                if column in tactical_team_display.columns:
                    tactical_team_display[column] = (
                        pd.to_numeric(
                            tactical_team_display[column],
                            errors="coerce",
                        )
                        * 100.0
                    ).round(1)

            st.dataframe(
                tactical_team_display,
                width="stretch",
                hide_index=True,
            )

            team_shift_col1, team_shift_col2 = st.columns(2)

            for team_index, (
                column_container,
                team_name,
            ) in enumerate(
                [
                    (
                        team_shift_col1,
                        team_1_name,
                    ),
                    (
                        team_shift_col2,
                        team_2_name,
                    ),
                ]
            ):
                with column_container:
                    team_row = tactical_team_changes[
                        tactical_team_changes["Team"]
                        == team_name
                    ]

                    if not team_row.empty:
                        team_row = team_row.iloc[0]

                        show_team_image(
                            team_name,
                            width=65,
                        )

                        st.markdown(
                            f"#### {team_name}"
                        )

                        avg_x_change = team_row.get(
                            "Average X Change",
                            float("nan"),
                        )

                        pressure_change = team_row.get(
                            "Pressure Change %",
                            0.0,
                        )

                        shot_change = team_row.get(
                            "Shot Change",
                            0,
                        )

                        xg_change = team_row.get(
                            "xG Change",
                            0.0,
                        )

                        if pd.notna(
                            avg_x_change
                        ):
                            direction = (
                                "higher"
                                if avg_x_change > 0
                                else "deeper"
                            )

                            st.metric(
                                "Average Action-Position Shift",
                                f"{avg_x_change:+.1f}",
                                help=(
                                    "Positive means event locations "
                                    "were higher up the pitch in the "
                                    "second half."
                                ),
                            )

                            st.caption(
                                f"Second-half event locations were "
                                f"generally {direction}."
                            )

                        st.metric(
                            "Pressure Change",
                            f"{pressure_change * 100:+.0f}%",
                        )

                        st.metric(
                            "Shot Change",
                            f"{int(shot_change):+d}",
                        )

                        st.metric(
                            "xG Change",
                            f"{xg_change:+.2f}",
                        )

        # -----------------------------------------------------
        # Player position shifts
        # -----------------------------------------------------

        st.markdown("### 🧍 Player Position Shifts")

        if tactical_player_changes.empty:
            st.info(
                "No player half-by-half positional comparison "
                "could be calculated."
            )

        else:
            tactical_player_display = (
                tactical_player_changes.copy()
            )

            tactical_player_display[
                "Absolute X Change"
            ] = (
                tactical_player_display[
                    "X Change"
                ]
                .abs()
            )

            tactical_player_display = (
                tactical_player_display
                .sort_values(
                    "Absolute X Change",
                    ascending=False,
                )
                .head(12)
            )

            player_display_columns = [
                "Player",
                "Team",
                "First Half X",
                "Second Half X",
                "X Change",
                "First Half Involvements",
                "Second Half Involvements",
            ]

            for column in [
                "First Half X",
                "Second Half X",
                "X Change",
            ]:
                tactical_player_display[column] = (
                    pd.to_numeric(
                        tactical_player_display[column],
                        errors="coerce",
                    )
                    .round(1)
                )

            st.dataframe(
                tactical_player_display[
                    player_display_columns
                ],
                width="stretch",
                hide_index=True,
            )

            player_shift_chart_data = (
                tactical_player_display[
                    [
                        "Player",
                        "Team",
                        "X Change",
                    ]
                ]
                .copy()
            )

            player_shift_chart = (
                alt.Chart(
                    player_shift_chart_data
                )
                .mark_bar()
                .encode(
                    x=alt.X(
                        "X Change:Q",
                        title=(
                            "Second-half average action-position "
                            "change (X units)"
                        ),
                    ),
                    y=alt.Y(
                        "Player:N",
                        title=None,
                        sort="-x",
                    ),
                    color=alt.Color(
                        "Team:N",
                        title="Team",
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Player:N",
                            title="Player",
                        ),
                        alt.Tooltip(
                            "Team:N",
                            title="Team",
                        ),
                        alt.Tooltip(
                            "X Change:Q",
                            title="X change",
                            format=".1f",
                        ),
                    ],
                )
            )

            st.altair_chart(
                player_shift_chart,
                width="stretch",
            )

            st.caption(
                "Positive X change means the player's recorded second-half "
                "event locations were higher up the pitch; negative means deeper."
            )

        # -----------------------------------------------------
        # Automated tactical alerts
        # -----------------------------------------------------

        st.markdown("### 🚨 Automated Tactical Alerts")

        if tactical_alerts.empty:
            st.success(
                "No tactical-change threshold was triggered "
                "for this match."
            )

        else:
            st.caption(
                f"{len(tactical_alerts)} data-derived tactical signal(s) "
                "were detected."
            )

            alert_type_order = [
                "Team Shape",
                "Pressure Shift",
                "Attacking Shift",
                "Activity Shift",
                "Player Position Shift",
            ]

            for alert_type in alert_type_order:
                alert_subset = tactical_alerts[
                    tactical_alerts[
                        "Type"
                    ]
                    == alert_type
                ]

                if alert_subset.empty:
                    continue

                with st.expander(
                    f"{alert_type} "
                    f"({len(alert_subset)})",
                    expanded=(
                        alert_type
                        in [
                            "Team Shape",
                            "Attacking Shift",
                        ]
                    ),
                ):
                    for _, alert_row in alert_subset.iterrows():
                        st.info(
                            alert_row.get(
                                "Message",
                                "",
                            )
                        )

            other_alerts = tactical_alerts[
                ~tactical_alerts[
                    "Type"
                ].isin(
                    alert_type_order
                )
            ]

            if not other_alerts.empty:
                with st.expander(
                    f"Other Signals "
                    f"({len(other_alerts)})"
                ):
                    for _, alert_row in other_alerts.iterrows():
                        st.info(
                            alert_row.get(
                                "Message",
                                "",
                            )
                        )

        st.warning(
            "Interpretation note: event-location changes can reflect "
            "game state, substitutions, role changes, possession patterns "
            "or tactical decisions. They should be treated as analyst signals "
            "rather than confirmed coaching instructions."
        )

    except Exception as tactical_change_error:
        st.error(
            "Tactical Change Intelligence could not be generated. "
            f"Details: {tactical_change_error}"
        )


    st.divider()

if NAV_PAGE == "🔗 Passing & Build-up":
    # ---------------------------------------------------------
    # Pass Network & Build-up Intelligence
    # ---------------------------------------------------------

    st.subheader("🔗 Pass Network & Build-up Intelligence")

    st.write(
        """
        Explore completed player-to-player passing connections, network hubs,
        average pass-event positions and first-half versus second-half build-up
        changes. Positions are derived from StatsBomb event locations rather
        than optical tracking data.
        """
    )

    pass_network_control_1, pass_network_control_2 = st.columns(2)

    with pass_network_control_1:
        pass_network_team = st.radio(
            "Pass Network Team",
            options=[
                team_1_name,
                team_2_name,
            ],
            horizontal=True,
            key="pass_network_team",
        )

    with pass_network_control_2:
        pass_network_period = st.radio(
            "Pass Network Period",
            options=[
                "Full Match",
                "First Half",
                "Second Half",
            ],
            horizontal=True,
            key="pass_network_period",
        )

    try:
        pass_network_result = analyze_pass_networks(
            events,
            team_names=[
                team_1_name,
                team_2_name,
            ],
            period_label=pass_network_period,
        )

        pass_summaries = pass_network_result[
            "summaries"
        ]

        pass_nodes = pass_network_result[
            "nodes"
        ]

        pass_edges = pass_network_result[
            "edges"
        ]

        pass_half_comparisons = pass_network_result[
            "half_comparisons"
        ]

        pass_insights = pass_network_result[
            "insights"
        ]

        selected_pass_summary = pass_summaries[
            pass_summaries[
                "Team"
            ]
            == pass_network_team
        ].copy()

        selected_pass_nodes = pass_nodes[
            pass_nodes[
                "Team"
            ]
            == pass_network_team
        ].copy()

        selected_pass_edges = pass_edges[
            pass_edges[
                "Team"
            ]
            == pass_network_team
        ].copy()

        if selected_pass_summary.empty:
            st.info(
                "No pass-network summary is available for the selected team "
                "and period."
            )

        else:
            selected_summary_row = selected_pass_summary.iloc[
                0
            ]

            # -------------------------------------------------
            # Headline metrics
            # -------------------------------------------------

            pass_metric_1, pass_metric_2, pass_metric_3, pass_metric_4 = st.columns(4)

            with pass_metric_1:
                st.metric(
                    "Passes Attempted",
                    int(
                        selected_summary_row[
                            "Passes Attempted"
                        ]
                    ),
                )

            with pass_metric_2:
                st.metric(
                    "Pass Completion",
                    f"{selected_summary_row['Pass Completion %']:.1f}%",
                )

            with pass_metric_3:
                strongest_link_value = (
                    selected_summary_row[
                        "Strongest Link"
                    ]
                    if pd.notna(
                        selected_summary_row[
                            "Strongest Link"
                        ]
                    )
                    else "—"
                )

                st.metric(
                    "Strongest Link Passes",
                    int(
                        selected_summary_row[
                            "Strongest Link Passes"
                        ]
                    ),
                    help=str(
                        strongest_link_value
                    ),
                )

            with pass_metric_4:
                avg_network_x = selected_summary_row[
                    "Average Network X"
                ]

                st.metric(
                    "Average Network X",
                    (
                        f"{avg_network_x:.1f}"
                        if pd.notna(
                            avg_network_x
                        )
                        else "—"
                    ),
                    help=(
                        "Average player pass-event position on the "
                        "StatsBomb 0-120 X-axis."
                    ),
                )

            team_badge_col, team_text_col = st.columns(
                [
                    1,
                    8,
                ]
            )

            with team_badge_col:
                show_team_image(
                    pass_network_team,
                    width=70,
                )

            with team_text_col:
                st.markdown(
                    f"### {pass_network_team} — {pass_network_period}"
                )

                st.caption(
                    f"Most involved: "
                    f"{selected_summary_row['Most Involved Player']} | "
                    f"Top passer: "
                    f"{selected_summary_row['Top Passer']} | "
                    f"Top receiver: "
                    f"{selected_summary_row['Top Receiver']}"
                )

                strongest_link = selected_summary_row[
                    "Strongest Link"
                ]

                if pd.notna(
                    strongest_link
                ):
                    st.caption(
                        f"Strongest completed connection: "
                        f"{strongest_link} "
                        f"({int(selected_summary_row['Strongest Link Passes'])} passes)"
                    )

            # -------------------------------------------------
            # Network chart
            # -------------------------------------------------

            st.markdown("### 🕸️ Passing Network")

            if selected_pass_nodes.empty:
                st.info(
                    "No player network nodes met the pass-volume threshold."
                )

            else:
                # Visual-cleanup thresholds.
                # Full pass data remains available in the tables below; these
                # thresholds affect only the network chart.
                if pass_network_period == "Full Match":
                    min_visual_passes = 5
                else:
                    min_visual_passes = 3

                top_network_nodes = (
                    selected_pass_nodes
                    .sort_values(
                        "Network Involvement",
                        ascending=False,
                    )
                    .head(14)
                    .copy()
                )

                top_network_nodes[
                    "Display Name"
                ] = (
                    top_network_nodes[
                        "Player"
                    ]
                    .apply(
                        short_player_name
                    )
                )

                # Use a few controlled node tiers instead of a continuous
                # Altair size scale. This prevents the striped/overlapping
                # rendering seen when many large circles sit close together.
                top_network_nodes = top_network_nodes.reset_index(
                    drop=True
                )

                top_network_nodes[
                    "Node Tier"
                ] = "Regular"

                if len(top_network_nodes) >= 1:
                    top_network_nodes.loc[
                        0,
                        "Node Tier",
                    ] = "Primary Hub"

                if len(top_network_nodes) >= 3:
                    top_network_nodes.loc[
                        1:2,
                        "Node Tier",
                    ] = "Secondary Hub"

                visible_players = set(
                    top_network_nodes[
                        "Player"
                    ].tolist()
                )

                network_edges_visible = (
                    selected_pass_edges[
                        (
                            selected_pass_edges[
                                "Pass Count"
                            ]
                            >= min_visual_passes
                        )
                        & (
                            selected_pass_edges[
                                "Passer"
                            ].isin(
                                visible_players
                            )
                        )
                        & (
                            selected_pass_edges[
                                "Recipient"
                            ].isin(
                                visible_players
                            )
                        )
                    ]
                    .copy()
                )

                # Keep the most meaningful connections only.
                network_edges_visible = (
                    network_edges_visible
                    .sort_values(
                        "Pass Count",
                        ascending=False,
                    )
                    .head(28)
                )

                # Attach passer/recipient average coordinates to each edge.
                node_lookup = top_network_nodes[
                    [
                        "Player",
                        "Average X",
                        "Average Y",
                    ]
                ].copy()

                passer_lookup = node_lookup.rename(
                    columns={
                        "Player":
                            "Passer",
                        "Average X":
                            "X1",
                        "Average Y":
                            "Y1",
                    }
                )

                recipient_lookup = node_lookup.rename(
                    columns={
                        "Player":
                            "Recipient",
                        "Average X":
                            "X2",
                        "Average Y":
                            "Y2",
                    }
                )

                network_edges_plot = (
                    network_edges_visible
                    .merge(
                        passer_lookup,
                        on="Passer",
                        how="inner",
                    )
                    .merge(
                        recipient_lookup,
                        on="Recipient",
                        how="inner",
                    )
                )

                # Reliable pitch made from rule segments, matching the existing
                # shot-map / tactical-map rendering approach.
                pitch_segments = pd.DataFrame(
                    [
                        # Outer boundaries
                        {
                            "x": 0,
                            "y": 0,
                            "x2": 120,
                            "y2": 0,
                        },
                        {
                            "x": 0,
                            "y": 80,
                            "x2": 120,
                            "y2": 80,
                        },
                        {
                            "x": 0,
                            "y": 0,
                            "x2": 0,
                            "y2": 80,
                        },
                        {
                            "x": 120,
                            "y": 0,
                            "x2": 120,
                            "y2": 80,
                        },

                        # Halfway line
                        {
                            "x": 60,
                            "y": 0,
                            "x2": 60,
                            "y2": 80,
                        },

                        # Left penalty box
                        {
                            "x": 0,
                            "y": 18,
                            "x2": 18,
                            "y2": 18,
                        },
                        {
                            "x": 18,
                            "y": 18,
                            "x2": 18,
                            "y2": 62,
                        },
                        {
                            "x": 18,
                            "y": 62,
                            "x2": 0,
                            "y2": 62,
                        },

                        # Right penalty box
                        {
                            "x": 102,
                            "y": 18,
                            "x2": 120,
                            "y2": 18,
                        },
                        {
                            "x": 102,
                            "y": 18,
                            "x2": 102,
                            "y2": 62,
                        },
                        {
                            "x": 102,
                            "y": 62,
                            "x2": 120,
                            "y2": 62,
                        },

                        # Left six-yard box
                        {
                            "x": 0,
                            "y": 30,
                            "x2": 6,
                            "y2": 30,
                        },
                        {
                            "x": 6,
                            "y": 30,
                            "x2": 6,
                            "y2": 50,
                        },
                        {
                            "x": 6,
                            "y": 50,
                            "x2": 0,
                            "y2": 50,
                        },

                        # Right six-yard box
                        {
                            "x": 114,
                            "y": 30,
                            "x2": 120,
                            "y2": 30,
                        },
                        {
                            "x": 114,
                            "y": 30,
                            "x2": 114,
                            "y2": 50,
                        },
                        {
                            "x": 114,
                            "y": 50,
                            "x2": 120,
                            "y2": 50,
                        },
                    ]
                )

                pitch_layer = (
                    alt.Chart(
                        pitch_segments
                    )
                    .mark_rule()
                    .encode(
                        x=alt.X(
                            "x:Q",
                            scale=alt.Scale(
                                domain=[
                                    0,
                                    120,
                                ]
                            ),
                            axis=None,
                        ),
                        y=alt.Y(
                            "y:Q",
                            scale=alt.Scale(
                                domain=[
                                    80,
                                    0,
                                ]
                            ),
                            axis=None,
                        ),
                        x2="x2:Q",
                        y2="y2:Q",
                    )
                )

                circle_angles = np.linspace(
                    0,
                    2 * np.pi,
                    80,
                )

                centre_circle = pd.DataFrame(
                    {
                        "Point Order":
                            np.arange(
                                len(
                                    circle_angles
                                )
                            ),

                        "x":
                            60
                            + 9.15
                            * np.cos(
                                circle_angles
                            ),

                        "y":
                            40
                            + 9.15
                            * np.sin(
                                circle_angles
                            ),
                    }
                )

                centre_circle_layer = (
                    alt.Chart(
                        centre_circle
                    )
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "x:Q",
                            scale=alt.Scale(
                                domain=[
                                    0,
                                    120,
                                ]
                            ),
                            axis=None,
                        ),
                        y=alt.Y(
                            "y:Q",
                            scale=alt.Scale(
                                domain=[
                                    80,
                                    0,
                                ]
                            ),
                            axis=None,
                        ),
                        order=alt.Order(
                            "Point Order:Q"
                        ),
                    )
                )

                centre_spot = pd.DataFrame(
                    {
                        "x": [
                            60
                        ],
                        "y": [
                            40
                        ],
                    }
                )

                centre_spot_layer = (
                    alt.Chart(
                        centre_spot
                    )
                    .mark_circle(
                        size=28
                    )
                    .encode(
                        x=alt.X(
                            "x:Q",
                            scale=alt.Scale(
                                domain=[
                                    0,
                                    120,
                                ]
                            ),
                            axis=None,
                        ),
                        y=alt.Y(
                            "y:Q",
                            scale=alt.Scale(
                                domain=[
                                    80,
                                    0,
                                ]
                            ),
                            axis=None,
                        ),
                    )
                )

                if network_edges_plot.empty:
                    edge_layer = None

                else:
                    # Convert each connection into a two-point path so Altair
                    # draws a simple line instead of a rule with an x2/y2 pair.
                    # This avoids the striped rendering artefact seen when many
                    # dense rule segments overlap.
                    edge_path_rows = []

                    for edge_id, edge_row in network_edges_plot.reset_index(
                        drop=True
                    ).iterrows():
                        edge_path_rows.append(
                            {
                                "Edge ID": edge_id,
                                "Point Order": 0,
                                "X": edge_row["X1"],
                                "Y": edge_row["Y1"],
                                "Passer": edge_row["Passer"],
                                "Recipient": edge_row["Recipient"],
                                "Pass Count": edge_row["Pass Count"],
                            }
                        )

                        edge_path_rows.append(
                            {
                                "Edge ID": edge_id,
                                "Point Order": 1,
                                "X": edge_row["X2"],
                                "Y": edge_row["Y2"],
                                "Passer": edge_row["Passer"],
                                "Recipient": edge_row["Recipient"],
                                "Pass Count": edge_row["Pass Count"],
                            }
                        )

                    edge_path_df = pd.DataFrame(
                        edge_path_rows
                    )

                    edge_layer = (
                        alt.Chart(
                            edge_path_df
                        )
                        .mark_line(
                            opacity=0.30
                        )
                        .encode(
                            x=alt.X(
                                "X:Q",
                                scale=alt.Scale(
                                    domain=[
                                        0,
                                        120,
                                    ]
                                ),
                                axis=None,
                            ),
                            y=alt.Y(
                                "Y:Q",
                                scale=alt.Scale(
                                    domain=[
                                        80,
                                        0,
                                    ]
                                ),
                                axis=None,
                            ),
                            detail="Edge ID:N",
                            order=alt.Order(
                                "Point Order:Q"
                            ),
                            size=alt.Size(
                                "Pass Count:Q",
                                scale=alt.Scale(
                                    range=[
                                        0.6,
                                        3.4,
                                    ]
                                ),
                                legend=None,
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "Passer:N",
                                    title="Passer",
                                ),
                                alt.Tooltip(
                                    "Recipient:N",
                                    title="Recipient",
                                ),
                                alt.Tooltip(
                                    "Pass Count:Q",
                                    title="Completed passes",
                                ),
                            ],
                        )
                    )

                # Clean fixed-size player nodes.
                # Primary and secondary hubs receive a subtle outer ring
                # rather than oversized filled circles.
                node_layer = (
                    alt.Chart(
                        top_network_nodes
                    )
                    .mark_circle(
                        size=230,
                        opacity=0.95,
                        stroke="white",
                        strokeWidth=1.4,
                    )
                    .encode(
                        x=alt.X(
                            "Average X:Q",
                            scale=alt.Scale(
                                domain=[
                                    0,
                                    120,
                                ]
                            ),
                            axis=None,
                        ),
                        y=alt.Y(
                            "Average Y:Q",
                            scale=alt.Scale(
                                domain=[
                                    80,
                                    0,
                                ]
                            ),
                            axis=None,
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Player:N",
                                title="Player",
                            ),
                            alt.Tooltip(
                                "Passes Attempted:Q",
                                title="Passes attempted",
                            ),
                            alt.Tooltip(
                                "Passes Completed:Q",
                                title="Passes completed",
                            ),
                            alt.Tooltip(
                                "Pass Completion %:Q",
                                title="Completion %",
                                format=".1f",
                            ),
                            alt.Tooltip(
                                "Passes Received:Q",
                                title="Passes received",
                            ),
                            alt.Tooltip(
                                "Network Involvement:Q",
                                title="Network involvement",
                            ),
                        ],
                    )
                )

                label_layer = (
                    alt.Chart(
                        top_network_nodes
                    )
                    .mark_text(
                        dy=-12,
                        fontSize=10,
                        baseline="bottom",
                    )
                    .encode(
                        x=alt.X(
                            "Average X:Q",
                            scale=alt.Scale(
                                domain=[
                                    0,
                                    120,
                                ]
                            ),
                            axis=None,
                        ),
                        y=alt.Y(
                            "Average Y:Q",
                            scale=alt.Scale(
                                domain=[
                                    80,
                                    0,
                                ]
                            ),
                            axis=None,
                        ),
                        text="Display Name:N",
                        tooltip=[
                            alt.Tooltip(
                                "Player:N",
                                title="Player",
                            )
                        ],
                    )
                )

                pass_network_chart = (
                    pitch_layer
                    + centre_circle_layer
                    + centre_spot_layer
                )

                if edge_layer is not None:
                    pass_network_chart = (
                        pass_network_chart
                        + edge_layer
                    )

                pass_network_chart = (
                    pass_network_chart
                    + node_layer
                    + label_layer
                ).properties(
                    height=560
                )

                st.altair_chart(
                    pass_network_chart,
                    width="stretch",
                )

                st.caption(
                    f"Visual cleanup: showing the top {len(top_network_nodes)} "
                    f"network players and connections with at least "
                    f"{min_visual_passes} completed passes. Player circles use "
                    "controlled fixed sizes, and each passing connection is drawn "
                    "as a simple two-point line; line thickness represents "
                    "completed-pass volume. Full pass data remains available below."
                )

            # -------------------------------------------------
            # Player network table
            # -------------------------------------------------

            st.markdown("### 👥 Key Network Players")

            if not selected_pass_nodes.empty:
                network_player_table = (
                    selected_pass_nodes[
                        [
                            "Player",
                            "Passes Attempted",
                            "Passes Completed",
                            "Pass Completion %",
                            "Passes Received",
                            "Network Involvement",
                            "Average X",
                            "Average Y",
                        ]
                    ]
                    .copy()
                    .head(12)
                )

                for column in [
                    "Pass Completion %",
                    "Average X",
                    "Average Y",
                ]:
                    network_player_table[
                        column
                    ] = (
                        pd.to_numeric(
                            network_player_table[
                                column
                            ],
                            errors="coerce",
                        )
                        .round(1)
                    )

                st.dataframe(
                    network_player_table,
                    width="stretch",
                    hide_index=True,
                )

            # -------------------------------------------------
            # Strongest connections
            # -------------------------------------------------

            st.markdown("### 🔁 Strongest Passing Connections")

            if selected_pass_edges.empty:
                st.info(
                    "No completed pass connections met the minimum "
                    "connection threshold."
                )

            else:
                strongest_connections = (
                    selected_pass_edges[
                        [
                            "Passer",
                            "Recipient",
                            "Pass Count",
                        ]
                    ]
                    .head(12)
                    .copy()
                )

                st.dataframe(
                    strongest_connections,
                    width="stretch",
                    hide_index=True,
                )

            # -------------------------------------------------
            # Half-by-half build-up
            # -------------------------------------------------

            st.markdown("### 🔀 First-Half vs Second-Half Build-up")

            selected_half_comparison = pass_half_comparisons[
                pass_half_comparisons[
                    "Team"
                ]
                == pass_network_team
            ].copy()

            if selected_half_comparison.empty:
                st.info(
                    "No half-by-half build-up comparison is available."
                )

            else:
                half_row = selected_half_comparison.iloc[
                    0
                ]

                build_col1, build_col2, build_col3 = st.columns(3)

                with build_col1:
                    st.metric(
                        "Pass Volume Change",
                        f"{int(half_row['Pass Volume Change']):+d}",
                    )

                with build_col2:
                    st.metric(
                        "Completion Change",
                        f"{half_row['Completion Change']:+.1f} pp",
                    )

                with build_col3:
                    network_x_change = half_row[
                        "Network X Change"
                    ]

                    st.metric(
                        "Network Position Change",
                        (
                            f"{network_x_change:+.1f}"
                            if pd.notna(
                                network_x_change
                            )
                            else "—"
                        ),
                    )

                build_up_table = pd.DataFrame(
                    {
                        "Metric": [
                            "Passes",
                            "Pass Completion %",
                            "Average Network X",
                            "Most Involved Player",
                            "Strongest Link",
                        ],
                        "First Half": [
                            half_row[
                                "First Half Passes"
                            ],
                            f"{half_row['First Half Completion %']:.1f}%",
                            (
                                f"{half_row['First Half Network X']:.1f}"
                                if pd.notna(
                                    half_row[
                                        "First Half Network X"
                                    ]
                                )
                                else "—"
                            ),
                            half_row[
                                "First Half Most Involved"
                            ],
                            half_row[
                                "First Half Strongest Link"
                            ],
                        ],
                        "Second Half": [
                            half_row[
                                "Second Half Passes"
                            ],
                            f"{half_row['Second Half Completion %']:.1f}%",
                            (
                                f"{half_row['Second Half Network X']:.1f}"
                                if pd.notna(
                                    half_row[
                                        "Second Half Network X"
                                    ]
                                )
                                else "—"
                            ),
                            half_row[
                                "Second Half Most Involved"
                            ],
                            half_row[
                                "Second Half Strongest Link"
                            ],
                        ],
                    }
                )

                build_up_table["First Half"] = (
                    build_up_table["First Half"].astype(str)
                )
                build_up_table["Second Half"] = (
                    build_up_table["Second Half"].astype(str)
                )

                st.dataframe(
                    build_up_table,
                    width="stretch",
                    hide_index=True,
                )

            # -------------------------------------------------
            # Analyst interpretation
            # -------------------------------------------------

            st.markdown("### 🧠 Build-up Analyst Insights")

            selected_pass_insights = pass_insights[
                pass_insights[
                    "Team"
                ]
                == pass_network_team
            ].copy()

            if selected_pass_insights.empty:
                st.info(
                    "No build-up insights were generated for the selected team."
                )

            else:
                network_insight_rows = selected_pass_insights[
                    selected_pass_insights[
                        "Type"
                    ]
                    == "Pass Network"
                ]

                build_change_rows = selected_pass_insights[
                    selected_pass_insights[
                        "Type"
                    ]
                    == "Build-up Change"
                ]

                if not network_insight_rows.empty:
                    with st.expander(
                        "Pass Network Insights",
                        expanded=True,
                    ):
                        for _, insight_row in network_insight_rows.iterrows():
                            st.info(
                                insight_row[
                                    "Message"
                                ]
                            )

                if not build_change_rows.empty:
                    with st.expander(
                        "Build-up Change Insights",
                        expanded=True,
                    ):
                        for _, insight_row in build_change_rows.iterrows():
                            st.info(
                                insight_row[
                                    "Message"
                                ]
                            )

            st.warning(
                "Interpretation note: pass networks describe event-based "
                "connections and average pass locations. They do not represent "
                "continuous off-ball movement or exact formation positions."
            )

    except Exception as pass_network_error:
        st.error(
            "Pass Network & Build-up Intelligence could not be generated. "
            f"Details: {pass_network_error}"
        )


    st.divider()

if NAV_PAGE == "🗺️ Possession & Territory":
    # ---------------------------------------------------------
    # Possession & Territory Intelligence
    # ---------------------------------------------------------

    st.subheader("🗺️ Possession & Territory Intelligence")

    st.write(
        """
        Explore event-based territorial behaviour, field-third activity,
        attacking-third presence and first-half versus second-half spatial shifts.
        These indicators are derived from StatsBomb event locations and should not
        be interpreted as optical-tracking possession or continuous spatial control.
        """
    )

    territory_control_1, territory_control_2 = st.columns(2)

    with territory_control_1:
        territory_team = st.radio(
            "Territory Team",
            options=[
                team_1_name,
                team_2_name,
            ],
            horizontal=True,
            key="territory_team",
        )

    with territory_control_2:
        territory_period = st.radio(
            "Territory Period",
            options=[
                "Full Match",
                "First Half",
                "Second Half",
            ],
            horizontal=True,
            key="territory_period",
        )

    try:
        territory_result = analyze_possession_territory(
            events,
            team_names=[
                team_1_name,
                team_2_name,
            ],
            period_label=territory_period,
        )

        territory_summaries = territory_result[
            "summaries"
        ]

        territory_zones = territory_result[
            "zones"
        ]

        territory_half_comparisons = territory_result[
            "half_comparisons"
        ]

        territory_insights = territory_result[
            "insights"
        ]

        selected_territory_summary = territory_summaries[
            territory_summaries[
                "Team"
            ]
            == territory_team
        ].copy()

        selected_territory_zones = territory_zones[
            territory_zones[
                "Team"
            ]
            == territory_team
        ].copy()

        if selected_territory_summary.empty:
            st.info(
                "No possession/territory summary is available for the selected "
                "team and period."
            )

        else:
            territory_row = selected_territory_summary.iloc[
                0
            ]

            # -------------------------------------------------
            # Headline territory metrics
            # -------------------------------------------------

            territory_metric_1, territory_metric_2, territory_metric_3, territory_metric_4 = st.columns(4)

            with territory_metric_1:
                st.metric(
                    "Event Share",
                    f"{territory_row['Event Share %']:.1f}%",
                    help=(
                        "Share of recorded match events attributed to this team "
                        "during the selected period."
                    ),
                )

            with territory_metric_2:
                st.metric(
                    "Average Event X",
                    (
                        f"{territory_row['Average X']:.1f}"
                        if pd.notna(
                            territory_row[
                                "Average X"
                            ]
                        )
                        else "—"
                    ),
                    help=(
                        "Average event location on the StatsBomb 0-120 X-axis."
                    ),
                )

            with territory_metric_3:
                st.metric(
                    "Attacking Third Share",
                    f"{territory_row['Attacking Third Share %']:.1f}%",
                )

            with territory_metric_4:
                st.metric(
                    "Territory Index",
                    f"{territory_row['Territory Index']:.1f} / 100",
                    help=(
                        "Transparent event-based index combining average X position "
                        "and attacking-third event share."
                    ),
                )

            territory_badge_col, territory_title_col = st.columns(
                [
                    1,
                    8,
                ]
            )

            with territory_badge_col:
                show_team_image(
                    territory_team,
                    width=70,
                )

            with territory_title_col:
                st.markdown(
                    f"### {territory_team} — {territory_period}"
                )

                st.caption(
                    f"Final-third events: "
                    f"{int(territory_row['Final Third Events'])} | "
                    f"Box-zone events: "
                    f"{int(territory_row['Box Zone Events'])} | "
                    f"Located events: "
                    f"{int(territory_row['Located Events'])}"
                )

            # -------------------------------------------------
            # Field-third distribution
            # -------------------------------------------------

            st.markdown("### 🧱 Activity by Field Third")

            third_distribution = pd.DataFrame(
                {
                    "Field Third": [
                        "Defensive Third",
                        "Middle Third",
                        "Attacking Third",
                    ],
                    "Event Count": [
                        int(
                            territory_row[
                                "Defensive Third Events"
                            ]
                        ),
                        int(
                            territory_row[
                                "Middle Third Events"
                            ]
                        ),
                        int(
                            territory_row[
                                "Attacking Third Events"
                            ]
                        ),
                    ],
                }
            )

            total_third_events = third_distribution[
                "Event Count"
            ].sum()

            third_distribution[
                "Share %"
            ] = np.where(
                total_third_events > 0,
                third_distribution[
                    "Event Count"
                ]
                / total_third_events
                * 100.0,
                0.0,
            )

            third_chart = (
                alt.Chart(
                    third_distribution
                )
                .mark_bar(
                    cornerRadiusTopRight=4,
                    cornerRadiusBottomRight=4,
                )
                .encode(
                    x=alt.X(
                        "Share %:Q",
                        title="Share of located events (%)",
                        scale=alt.Scale(
                            domain=[
                                0,
                                100,
                            ]
                        ),
                    ),
                    y=alt.Y(
                        "Field Third:N",
                        title=None,
                        sort=[
                            "Attacking Third",
                            "Middle Third",
                            "Defensive Third",
                        ],
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Field Third:N",
                            title="Field third",
                        ),
                        alt.Tooltip(
                            "Event Count:Q",
                            title="Events",
                        ),
                        alt.Tooltip(
                            "Share %:Q",
                            title="Share",
                            format=".1f",
                        ),
                    ],
                )
            )

            st.altair_chart(
                third_chart,
                width="stretch",
            )

            third_table = third_distribution.copy()

            third_table[
                "Share %"
            ] = third_table[
                "Share %"
            ].round(1)

            st.dataframe(
                third_table,
                width="stretch",
                hide_index=True,
            )

            # -------------------------------------------------
            # Channel distribution
            # -------------------------------------------------

            st.markdown("### ↔️ Activity by Channel")

            channel_distribution = pd.DataFrame(
                {
                    "Channel": [
                        "Left",
                        "Centre",
                        "Right",
                    ],
                    "Event Count": [
                        int(
                            territory_row[
                                "Left Channel Events"
                            ]
                        ),
                        int(
                            territory_row[
                                "Centre Channel Events"
                            ]
                        ),
                        int(
                            territory_row[
                                "Right Channel Events"
                            ]
                        ),
                    ],
                }
            )

            total_channel_events = channel_distribution[
                "Event Count"
            ].sum()

            channel_distribution[
                "Share %"
            ] = np.where(
                total_channel_events > 0,
                channel_distribution[
                    "Event Count"
                ]
                / total_channel_events
                * 100.0,
                0.0,
            )

            channel_chart = (
                alt.Chart(
                    channel_distribution
                )
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Channel:N",
                        title=None,
                        sort=[
                            "Left",
                            "Centre",
                            "Right",
                        ],
                    ),
                    y=alt.Y(
                        "Share %:Q",
                        title="Share of located events (%)",
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Channel:N",
                            title="Channel",
                        ),
                        alt.Tooltip(
                            "Event Count:Q",
                            title="Events",
                        ),
                        alt.Tooltip(
                            "Share %:Q",
                            title="Share",
                            format=".1f",
                        ),
                    ],
                )
            )

            st.altair_chart(
                channel_chart,
                width="stretch",
            )

            # -------------------------------------------------
            # Team comparison
            # -------------------------------------------------

            st.markdown("### ⚖️ Team Territory Comparison")

            territory_comparison_table = territory_summaries[
                [
                    "Team",
                    "Event Share %",
                    "Average X",
                    "Attacking Third Share %",
                    "Final Third Events",
                    "Box Zone Events",
                    "Territory Index",
                ]
            ].copy()

            for column in [
                "Event Share %",
                "Average X",
                "Attacking Third Share %",
                "Territory Index",
            ]:
                territory_comparison_table[
                    column
                ] = (
                    pd.to_numeric(
                        territory_comparison_table[
                            column
                        ],
                        errors="coerce",
                    )
                    .round(1)
                )

            st.dataframe(
                territory_comparison_table,
                width="stretch",
                hide_index=True,
            )

            territory_compare_chart = (
                alt.Chart(
                    territory_comparison_table
                )
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Territory Index:Q",
                        title="Event-based Territory Index",
                        scale=alt.Scale(
                            domain=[
                                0,
                                100,
                            ]
                        ),
                    ),
                    y=alt.Y(
                        "Team:N",
                        title=None,
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Team:N",
                            title="Team",
                        ),
                        alt.Tooltip(
                            "Territory Index:Q",
                            title="Territory Index",
                            format=".1f",
                        ),
                        alt.Tooltip(
                            "Average X:Q",
                            title="Average X",
                            format=".1f",
                        ),
                        alt.Tooltip(
                            "Attacking Third Share %:Q",
                            title="Attacking third share",
                            format=".1f",
                        ),
                    ],
                )
            )

            st.altair_chart(
                territory_compare_chart,
                width="stretch",
            )

            # -------------------------------------------------
            # Half-by-half territory changes
            # -------------------------------------------------

            st.markdown("### 🔀 First-Half vs Second-Half Territory")

            selected_territory_half = territory_half_comparisons[
                territory_half_comparisons[
                    "Team"
                ]
                == territory_team
            ].copy()

            if selected_territory_half.empty:
                st.info(
                    "No first-half vs second-half territory comparison "
                    "is available."
                )

            else:
                half_row = selected_territory_half.iloc[
                    0
                ]

                half_metric_1, half_metric_2, half_metric_3, half_metric_4 = st.columns(4)

                with half_metric_1:
                    average_x_change = half_row[
                        "Average X Change"
                    ]

                    st.metric(
                        "Average X Change",
                        (
                            f"{average_x_change:+.1f}"
                            if pd.notna(
                                average_x_change
                            )
                            else "—"
                        ),
                    )

                with half_metric_2:
                    st.metric(
                        "Attacking Third Share Change",
                        f"{half_row['Attacking Third Share Change']:+.1f} pp",
                    )

                with half_metric_3:
                    st.metric(
                        "Territory Index Change",
                        f"{half_row['Territory Index Change']:+.1f}",
                    )

                with half_metric_4:
                    st.metric(
                        "Final Third Event Change",
                        f"{int(half_row['Final Third Event Change']):+d}",
                    )

                half_territory_table = pd.DataFrame(
                    {
                        "Metric": [
                            "Average X",
                            "Attacking Third Share %",
                            "Territory Index",
                            "Final Third Events",
                            "Box Zone Events",
                        ],
                        "First Half": [
                            (
                                f"{half_row['First Half Average X']:.1f}"
                                if pd.notna(
                                    half_row[
                                        "First Half Average X"
                                    ]
                                )
                                else "—"
                            ),
                            f"{half_row['First Half Attacking Third Share %']:.1f}%",
                            f"{half_row['First Half Territory Index']:.1f}",
                            int(
                                half_row[
                                    "First Half Final Third Events"
                                ]
                            ),
                            int(
                                half_row[
                                    "First Half Box Zone Events"
                                ]
                            ),
                        ],
                        "Second Half": [
                            (
                                f"{half_row['Second Half Average X']:.1f}"
                                if pd.notna(
                                    half_row[
                                        "Second Half Average X"
                                    ]
                                )
                                else "—"
                            ),
                            f"{half_row['Second Half Attacking Third Share %']:.1f}%",
                            f"{half_row['Second Half Territory Index']:.1f}",
                            int(
                                half_row[
                                    "Second Half Final Third Events"
                                ]
                            ),
                            int(
                                half_row[
                                    "Second Half Box Zone Events"
                                ]
                            ),
                        ],
                    }
                )

                half_territory_table["First Half"] = (
                    half_territory_table["First Half"].astype(str)
                )
                half_territory_table["Second Half"] = (
                    half_territory_table["Second Half"].astype(str)
                )

                st.dataframe(
                    half_territory_table,
                    width="stretch",
                    hide_index=True,
                )

            # -------------------------------------------------
            # Zone heat table
            # -------------------------------------------------

            st.markdown("### 🧩 Territory Zone Detail")

            if selected_territory_zones.empty:
                st.info(
                    "No zone-level territory data is available."
                )

            else:
                zone_pivot = (
                    selected_territory_zones
                    .pivot_table(
                        index="Third",
                        columns="Channel",
                        values="Event Count",
                        aggfunc="sum",
                        fill_value=0,
                    )
                    .reset_index()
                )

                preferred_columns = [
                    "Third",
                    "Left",
                    "Centre",
                    "Right",
                ]

                zone_pivot = zone_pivot[
                    [
                        column
                        for column in preferred_columns
                        if column in zone_pivot.columns
                    ]
                ]

                st.dataframe(
                    zone_pivot,
                    width="stretch",
                    hide_index=True,
                )

            # -------------------------------------------------
            # Analyst insights
            # -------------------------------------------------

            st.markdown("### 🧠 Territory Analyst Insights")

            selected_territory_insights = territory_insights[
                territory_insights[
                    "Team"
                ]
                == territory_team
            ].copy()

            if selected_territory_insights.empty:
                st.info(
                    "No territory insights were generated for the selected team."
                )

            else:
                territory_static_insights = selected_territory_insights[
                    selected_territory_insights[
                        "Type"
                    ]
                    == "Territory"
                ]

                territory_change_insights = selected_territory_insights[
                    selected_territory_insights[
                        "Type"
                    ]
                    == "Territory Change"
                ]

                if not territory_static_insights.empty:
                    with st.expander(
                        "Territory Overview",
                        expanded=True,
                    ):
                        for _, insight_row in territory_static_insights.iterrows():
                            st.info(
                                insight_row[
                                    "Message"
                                ]
                            )

                if not territory_change_insights.empty:
                    with st.expander(
                        "Territory Change Insights",
                        expanded=True,
                    ):
                        for _, insight_row in territory_change_insights.iterrows():
                            st.info(
                                insight_row[
                                    "Message"
                                ]
                            )

            st.warning(
                "Interpretation note: these metrics describe recorded event "
                "locations and event shares. They do not measure continuous "
                "possession, off-ball occupancy or exact spatial control."
            )

    except Exception as territory_error:
        st.error(
            "Possession & Territory Intelligence could not be generated. "
            f"Details: {territory_error}"
        )


    st.divider()

if NAV_PAGE == "🚀 Progressive Actions":
    # ---------------------------------------------------------
    # Progressive Actions & Chance Creation Intelligence
    # ---------------------------------------------------------

    st.subheader("🚀 Progressive Actions & Chance Creation Intelligence")

    st.write(
        """
        Identify who moves the ball forward, how teams enter advanced areas,
        and how progression changes between halves. Progressive actions use a
        transparent project definition: a completed pass or carry that advances
        the ball by at least 10 StatsBomb X-units toward the opponent goal.
        """
    )

    progressive_control_1, progressive_control_2 = st.columns(2)

    with progressive_control_1:
        progressive_team = st.radio(
            "Progression Team",
            options=[
                team_1_name,
                team_2_name,
            ],
            horizontal=True,
            key="progressive_team",
        )

    with progressive_control_2:
        progressive_period = st.radio(
            "Progression Period",
            options=[
                "Full Match",
                "First Half",
                "Second Half",
            ],
            horizontal=True,
            key="progressive_period",
        )

    try:
        progressive_result = analyze_progressive_actions(
            events,
            team_names=[
                team_1_name,
                team_2_name,
            ],
            period_label=progressive_period,
        )

        progressive_summaries = progressive_result[
            "summaries"
        ]

        progressive_players = progressive_result[
            "players"
        ]

        progressive_half_comparisons = progressive_result[
            "half_comparisons"
        ]

        progressive_insights = progressive_result[
            "insights"
        ]

        selected_progressive_summary = progressive_summaries[
            progressive_summaries[
                "Team"
            ]
            == progressive_team
        ].copy()

        selected_progressive_players = progressive_players[
            progressive_players[
                "Team"
            ]
            == progressive_team
        ].copy()

        if selected_progressive_summary.empty:
            st.info(
                "No progressive-action summary is available for the selected "
                "team and period."
            )

        else:
            progressive_row = selected_progressive_summary.iloc[
                0
            ]

            # -------------------------------------------------
            # Headline progression metrics
            # -------------------------------------------------

            prog_metric_1, prog_metric_2, prog_metric_3, prog_metric_4 = st.columns(4)

            with prog_metric_1:
                st.metric(
                    "Progressive Actions",
                    int(
                        progressive_row[
                            "Progressive Actions"
                        ]
                    ),
                )

            with prog_metric_2:
                st.metric(
                    "Progressive Passes",
                    int(
                        progressive_row[
                            "Progressive Passes"
                        ]
                    ),
                )

            with prog_metric_3:
                st.metric(
                    "Progressive Carries",
                    int(
                        progressive_row[
                            "Progressive Carries"
                        ]
                    ),
                )

            with prog_metric_4:
                st.metric(
                    "Forward Distance",
                    f"{progressive_row['Forward Distance']:.1f}",
                    help=(
                        "Total StatsBomb X-units gained by progressive "
                        "passes and carries."
                    ),
                )

            prog_metric_5, prog_metric_6 = st.columns(2)

            with prog_metric_5:
                st.metric(
                    "Final-Third Entries",
                    int(
                        progressive_row[
                            "Final Third Entries"
                        ]
                    ),
                )

            with prog_metric_6:
                st.metric(
                    "Box Entries",
                    int(
                        progressive_row[
                            "Box Entries"
                        ]
                    ),
                )

            prog_badge_col, prog_title_col = st.columns(
                [
                    1,
                    8,
                ]
            )

            with prog_badge_col:
                show_team_image(
                    progressive_team,
                    width=70,
                )

            with prog_title_col:
                st.markdown(
                    f"### {progressive_team} — {progressive_period}"
                )

                st.caption(
                    f"Top progressor: "
                    f"{progressive_row['Top Progressor']} | "
                    f"Top final-third contributor: "
                    f"{progressive_row['Top Final Third Contributor']} | "
                    f"Top box-entry contributor: "
                    f"{progressive_row['Top Box Entry Contributor']}"
                )

            # -------------------------------------------------
            # Team comparison
            # -------------------------------------------------

            st.markdown("### ⚖️ Team Progression Comparison")

            progressive_compare_table = progressive_summaries[
                [
                    "Team",
                    "Progressive Passes",
                    "Progressive Carries",
                    "Progressive Actions",
                    "Final Third Entries",
                    "Box Entries",
                    "Forward Distance",
                ]
            ].copy()

            progressive_compare_table[
                "Forward Distance"
            ] = (
                pd.to_numeric(
                    progressive_compare_table[
                        "Forward Distance"
                    ],
                    errors="coerce",
                )
                .round(1)
            )

            st.dataframe(
                progressive_compare_table,
                width="stretch",
                hide_index=True,
            )

            progressive_compare_long = progressive_compare_table.melt(
                id_vars=[
                    "Team",
                ],
                value_vars=[
                    "Progressive Passes",
                    "Progressive Carries",
                    "Final Third Entries",
                    "Box Entries",
                ],
                var_name="Metric",
                value_name="Count",
            )

            progressive_compare_chart = (
                alt.Chart(
                    progressive_compare_long
                )
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Count:Q",
                        title="Actions / Entries",
                    ),
                    y=alt.Y(
                        "Metric:N",
                        title=None,
                    ),
                    color=alt.Color(
                        "Team:N",
                        title="Team",
                    ),
                    xOffset="Team:N",
                    tooltip=[
                        alt.Tooltip(
                            "Team:N",
                            title="Team",
                        ),
                        alt.Tooltip(
                            "Metric:N",
                            title="Metric",
                        ),
                        alt.Tooltip(
                            "Count:Q",
                            title="Count",
                        ),
                    ],
                )
            )

            st.altair_chart(
                progressive_compare_chart,
                width="stretch",
            )

            # -------------------------------------------------
            # Player progression leaders
            # -------------------------------------------------

            st.markdown("### 🏃 Player Progression Leaders")

            if selected_progressive_players.empty:
                st.info(
                    "No player met the progression contribution threshold."
                )

            else:
                player_progression_table = (
                    selected_progressive_players[
                        [
                            "Player",
                            "Progressive Passes",
                            "Progressive Carries",
                            "Progressive Actions",
                            "Final Third Entries",
                            "Box Entries",
                            "Forward Distance",
                        ]
                    ]
                    .copy()
                    .head(15)
                )

                player_progression_table[
                    "Forward Distance"
                ] = (
                    pd.to_numeric(
                        player_progression_table[
                            "Forward Distance"
                        ],
                        errors="coerce",
                    )
                    .round(1)
                )

                st.dataframe(
                    player_progression_table,
                    width="stretch",
                    hide_index=True,
                )

                top_progression_chart_data = (
                    selected_progressive_players[
                        [
                            "Player",
                            "Progressive Passes",
                            "Progressive Carries",
                            "Progressive Actions",
                        ]
                    ]
                    .head(12)
                    .copy()
                )

                top_progression_chart_long = top_progression_chart_data.melt(
                    id_vars=[
                        "Player",
                        "Progressive Actions",
                    ],
                    value_vars=[
                        "Progressive Passes",
                        "Progressive Carries",
                    ],
                    var_name="Action Type",
                    value_name="Count",
                )

                player_progression_chart = (
                    alt.Chart(
                        top_progression_chart_long
                    )
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "Count:Q",
                            title="Progressive actions",
                        ),
                        y=alt.Y(
                            "Player:N",
                            title=None,
                            sort="-x",
                        ),
                        color=alt.Color(
                            "Action Type:N",
                            title="Action type",
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Player:N",
                                title="Player",
                            ),
                            alt.Tooltip(
                                "Action Type:N",
                                title="Action",
                            ),
                            alt.Tooltip(
                                "Count:Q",
                                title="Count",
                            ),
                        ],
                    )
                )

                st.altair_chart(
                    player_progression_chart,
                    width="stretch",
                )

            # -------------------------------------------------
            # Advanced-area contributors
            # -------------------------------------------------

            st.markdown("### 🎯 Advanced-Area Contributors")

            if not selected_progressive_players.empty:
                advanced_area_table = (
                    selected_progressive_players[
                        [
                            "Player",
                            "Final Third Entries",
                            "Box Entries",
                            "Progressive Actions",
                        ]
                    ]
                    .sort_values(
                        [
                            "Final Third Entries",
                            "Box Entries",
                            "Progressive Actions",
                        ],
                        ascending=[
                            False,
                            False,
                            False,
                        ],
                    )
                    .head(12)
                )

                st.dataframe(
                    advanced_area_table,
                    width="stretch",
                    hide_index=True,
                )

            # -------------------------------------------------
            # Half-by-half progression
            # -------------------------------------------------

            st.markdown("### 🔀 First-Half vs Second-Half Progression")

            selected_progressive_half = progressive_half_comparisons[
                progressive_half_comparisons[
                    "Team"
                ]
                == progressive_team
            ].copy()

            if selected_progressive_half.empty:
                st.info(
                    "No first-half vs second-half progression comparison "
                    "is available."
                )

            else:
                prog_half_row = selected_progressive_half.iloc[
                    0
                ]

                prog_half_metric_1, prog_half_metric_2, prog_half_metric_3, prog_half_metric_4 = st.columns(4)

                with prog_half_metric_1:
                    st.metric(
                        "Progressive Action Change",
                        f"{int(prog_half_row['Progressive Action Change']):+d}",
                    )

                with prog_half_metric_2:
                    st.metric(
                        "Progressive Pass Change",
                        f"{int(prog_half_row['Progressive Pass Change']):+d}",
                    )

                with prog_half_metric_3:
                    st.metric(
                        "Final-Third Entry Change",
                        f"{int(prog_half_row['Final Third Entry Change']):+d}",
                    )

                with prog_half_metric_4:
                    st.metric(
                        "Box Entry Change",
                        f"{int(prog_half_row['Box Entry Change']):+d}",
                    )

                progression_half_table = pd.DataFrame(
                    {
                        "Metric": [
                            "Progressive Passes",
                            "Progressive Carries",
                            "Progressive Actions",
                            "Final Third Entries",
                            "Box Entries",
                            "Forward Distance",
                            "Top Progressor",
                        ],
                        "First Half": [
                            int(
                                prog_half_row[
                                    "First Half Progressive Passes"
                                ]
                            ),
                            int(
                                prog_half_row[
                                    "First Half Progressive Carries"
                                ]
                            ),
                            int(
                                prog_half_row[
                                    "First Half Progressive Actions"
                                ]
                            ),
                            int(
                                prog_half_row[
                                    "First Half Final Third Entries"
                                ]
                            ),
                            int(
                                prog_half_row[
                                    "First Half Box Entries"
                                ]
                            ),
                            f"{prog_half_row['First Half Forward Distance']:.1f}",
                            prog_half_row[
                                "First Half Top Progressor"
                            ],
                        ],
                        "Second Half": [
                            int(
                                prog_half_row[
                                    "Second Half Progressive Passes"
                                ]
                            ),
                            int(
                                prog_half_row[
                                    "Second Half Progressive Carries"
                                ]
                            ),
                            int(
                                prog_half_row[
                                    "Second Half Progressive Actions"
                                ]
                            ),
                            int(
                                prog_half_row[
                                    "Second Half Final Third Entries"
                                ]
                            ),
                            int(
                                prog_half_row[
                                    "Second Half Box Entries"
                                ]
                            ),
                            f"{prog_half_row['Second Half Forward Distance']:.1f}",
                            prog_half_row[
                                "Second Half Top Progressor"
                            ],
                        ],
                    }
                )

                # Prevent mixed numeric/string PyArrow conversion issues.
                progression_half_table[
                    "First Half"
                ] = (
                    progression_half_table[
                        "First Half"
                    ]
                    .astype(str)
                )

                progression_half_table[
                    "Second Half"
                ] = (
                    progression_half_table[
                        "Second Half"
                    ]
                    .astype(str)
                )

                st.dataframe(
                    progression_half_table,
                    width="stretch",
                    hide_index=True,
                )

            # -------------------------------------------------
            # Analyst insights
            # -------------------------------------------------

            st.markdown("### 🧠 Progression Analyst Insights")

            selected_progressive_insights = progressive_insights[
                progressive_insights[
                    "Team"
                ]
                == progressive_team
            ].copy()

            if selected_progressive_insights.empty:
                st.info(
                    "No progression insights were generated for the selected team."
                )

            else:
                progression_overview = selected_progressive_insights[
                    selected_progressive_insights[
                        "Type"
                    ]
                    == "Progression"
                ]

                progression_change = selected_progressive_insights[
                    selected_progressive_insights[
                        "Type"
                    ]
                    == "Progression Change"
                ]

                if not progression_overview.empty:
                    with st.expander(
                        "Progression Overview",
                        expanded=True,
                    ):
                        for _, insight_row in progression_overview.iterrows():
                            st.info(
                                insight_row[
                                    "Message"
                                ]
                            )

                if not progression_change.empty:
                    with st.expander(
                        "Progression Change Insights",
                        expanded=True,
                    ):
                        for _, insight_row in progression_change.iterrows():
                            st.info(
                                insight_row[
                                    "Message"
                                ]
                            )

            st.warning(
                "Methodology note: progressive actions use the project-defined "
                "10-X-unit forward-movement threshold. These indicators are "
                "transparent event-based heuristics rather than proprietary "
                "provider metrics."
            )

    except Exception as progressive_error:
        st.error(
            "Progressive Actions & Chance Creation Intelligence could not "
            f"be generated. Details: {progressive_error}"
        )


    st.divider()

if NAV_PAGE == "🎯 Shot Analysis":
    # ---------------------------------------------------------
    # Shot Map
    # ---------------------------------------------------------

    st.subheader("🎯 Shot Map")

    st.write(
        """
        Explore where each shot was taken from. Larger markers indicate
        higher-xG chances, while goal attempts are highlighted separately.
        """
    )

    shot_map_data = build_shot_map_data(
        events
    )

    if not shot_map_data.empty:

        shot_filter = st.radio(
            "Shot Map View",
            [
                "All Shots",
                team_1_name,
                team_2_name,
            ],
            horizontal=True,
            key="shot_map_filter",
        )

        if shot_filter == "All Shots":
            filtered_shots = shot_map_data.copy()
        else:
            filtered_shots = shot_map_data[
                shot_map_data["Team"] == shot_filter
            ].copy()

        team_1_shot_summary = shot_summary_for_team(
            shot_map_data,
            team_1_name,
        )

        team_2_shot_summary = shot_summary_for_team(
            shot_map_data,
            team_2_name,
        )

        shot_metric_1, shot_metric_2, shot_metric_3, shot_metric_4 = st.columns(4)

        with shot_metric_1:
            st.metric(
                f"{team_1_name} Shots",
                team_1_shot_summary["Shots"],
            )

        with shot_metric_2:
            st.metric(
                f"{team_1_name} xG",
                f"{team_1_shot_summary['xG']:.2f}",
            )

        with shot_metric_3:
            st.metric(
                f"{team_2_name} Shots",
                team_2_shot_summary["Shots"],
            )

        with shot_metric_4:
            st.metric(
                f"{team_2_name} xG",
                f"{team_2_shot_summary['xG']:.2f}",
            )

        # -----------------------------------------------------
        # Reliable StatsBomb-style pitch rendering
        # -----------------------------------------------------

        pitch_lines = pd.DataFrame(
            [
                # Outer boundary
                {"x": 0, "y": 0, "x2": 120, "y2": 0},
                {"x": 120, "y": 0, "x2": 120, "y2": 80},
                {"x": 120, "y": 80, "x2": 0, "y2": 80},
                {"x": 0, "y": 80, "x2": 0, "y2": 0},

                # Halfway line
                {"x": 60, "y": 0, "x2": 60, "y2": 80},

                # Left penalty area
                {"x": 0, "y": 18, "x2": 18, "y2": 18},
                {"x": 18, "y": 18, "x2": 18, "y2": 62},
                {"x": 18, "y": 62, "x2": 0, "y2": 62},

                # Right penalty area
                {"x": 102, "y": 18, "x2": 120, "y2": 18},
                {"x": 102, "y": 18, "x2": 102, "y2": 62},
                {"x": 102, "y": 62, "x2": 120, "y2": 62},

                # Left six-yard box
                {"x": 0, "y": 30, "x2": 6, "y2": 30},
                {"x": 6, "y": 30, "x2": 6, "y2": 50},
                {"x": 6, "y": 50, "x2": 0, "y2": 50},

                # Right six-yard box
                {"x": 114, "y": 30, "x2": 120, "y2": 30},
                {"x": 114, "y": 30, "x2": 114, "y2": 50},
                {"x": 114, "y": 50, "x2": 120, "y2": 50},

                # Goal lines
                {"x": 0, "y": 36, "x2": -2, "y2": 36},
                {"x": -2, "y": 36, "x2": -2, "y2": 44},
                {"x": -2, "y": 44, "x2": 0, "y2": 44},

                {"x": 120, "y": 36, "x2": 122, "y2": 36},
                {"x": 122, "y": 36, "x2": 122, "y2": 44},
                {"x": 122, "y": 44, "x2": 120, "y2": 44},
            ]
        )

        pitch_chart = alt.Chart(
            pitch_lines
        ).mark_rule(
            stroke="#7f7f7f",
            strokeWidth=1.4,
        ).encode(
            x=alt.X(
                "x:Q",
                scale=alt.Scale(domain=[-3, 123]),
                axis=None,
            ),
            x2="x2:Q",
            y=alt.Y(
                "y:Q",
                scale=alt.Scale(domain=[82, -2]),
                axis=None,
            ),
            y2="y2:Q",
        )

        # Centre spot
        centre_spot = alt.Chart(
            pd.DataFrame(
                {
                    "x": [60],
                    "y": [40],
                }
            )
        ).mark_circle(
            size=45,
            filled=True,
            color="#7f7f7f",
        ).encode(
            x=alt.X(
                "x:Q",
                scale=alt.Scale(domain=[-3, 123]),
                axis=None,
            ),
            y=alt.Y(
                "y:Q",
                scale=alt.Scale(domain=[82, -2]),
                axis=None,
            ),
        )

        # Penalty spots
        penalty_spots = alt.Chart(
            pd.DataFrame(
                {
                    "x": [12, 108],
                    "y": [40, 40],
                }
            )
        ).mark_circle(
            size=35,
            filled=True,
            color="#7f7f7f",
        ).encode(
            x=alt.X(
                "x:Q",
                scale=alt.Scale(domain=[-3, 123]),
                axis=None,
            ),
            y=alt.Y(
                "y:Q",
                scale=alt.Scale(domain=[82, -2]),
                axis=None,
            ),
        )

        # Approximate centre circle using sampled points

        centre_circle_data = pd.DataFrame(
            {
                "x": [
                    60 + 10 * math.cos(math.radians(angle))
                    for angle in range(0, 361, 5)
                ],
                "y": [
                    40 + 10 * math.sin(math.radians(angle))
                    for angle in range(0, 361, 5)
                ],
                "order": list(range(len(range(0, 361, 5)))),
            }
        )

        centre_circle = alt.Chart(
            centre_circle_data
        ).mark_line(
            stroke="#7f7f7f",
            strokeWidth=1.2,
        ).encode(
            x=alt.X(
                "x:Q",
                scale=alt.Scale(domain=[-3, 123]),
                axis=None,
            ),
            y=alt.Y(
                "y:Q",
                scale=alt.Scale(domain=[82, -2]),
                axis=None,
            ),
            order="order:Q",
        )

        shots_chart = alt.Chart(
            filtered_shots
        ).mark_point(
            filled=True,
            opacity=0.85,
            stroke="white",
            strokeWidth=1.2,
        ).encode(
            x=alt.X(
                "X:Q",
                scale=alt.Scale(domain=[-3, 123]),
                axis=None,
            ),
            y=alt.Y(
                "Y:Q",
                scale=alt.Scale(domain=[82, -2]),
                axis=None,
            ),
            color=alt.Color(
                "Team:N",
                title="Team",
            ),
            size=alt.Size(
                "xG:Q",
                title="xG",
                scale=alt.Scale(range=[90, 900]),
            ),
            shape=alt.Shape(
                "Marker:N",
                title="Outcome Type",
                scale=alt.Scale(
                    domain=["Shot", "Goal"],
                    range=["circle", "diamond"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Player:N", title="Player"),
                alt.Tooltip("Team:N", title="Team"),
                alt.Tooltip("Minute:Q", title="Minute", format=".0f"),
                alt.Tooltip("xG:Q", title="xG", format=".2f"),
                alt.Tooltip("Outcome:N", title="Outcome"),
            ],
        )

        shot_map_chart = alt.layer(
            pitch_chart,
            centre_circle,
            centre_spot,
            penalty_spots,
            shots_chart,
        ).properties(
            height=520
        ).configure_view(
            strokeWidth=0
        )

        st.altair_chart(
            shot_map_chart,
            width="stretch",
        )

        st.caption(
            "StatsBomb pitch coordinates use a 120 × 80 grid. "
            "Marker size represents shot xG; diamond markers represent goals."
        )

        st.markdown("### 🧠 Shot Map Insight")

        if team_1_shot_summary["xG"] > team_2_shot_summary["xG"]:
            st.info(
                f"**{team_1_name}** generated the higher total shot quality "
                f"({team_1_shot_summary['xG']:.2f} xG vs "
                f"{team_2_shot_summary['xG']:.2f})."
            )
        elif team_2_shot_summary["xG"] > team_1_shot_summary["xG"]:
            st.info(
                f"**{team_2_name}** generated the higher total shot quality "
                f"({team_2_shot_summary['xG']:.2f} xG vs "
                f"{team_1_shot_summary['xG']:.2f})."
            )
        else:
            st.info(
                "Both teams generated the same total xG."
            )

        if team_1_shot_summary["Shots"] > team_2_shot_summary["Shots"]:
            st.info(
                f"**{team_1_name}** attempted more shots "
                f"({team_1_shot_summary['Shots']} vs "
                f"{team_2_shot_summary['Shots']})."
            )
        elif team_2_shot_summary["Shots"] > team_1_shot_summary["Shots"]:
            st.info(
                f"**{team_2_name}** attempted more shots "
                f"({team_2_shot_summary['Shots']} vs "
                f"{team_1_shot_summary['Shots']})."
            )

    else:
        st.warning(
            "Shot location data is not available for this match."
        )

    st.divider()

if NAV_PAGE == "📈 Live Intelligence":
    # ---------------------------------------------------------
    # Live Intelligence Replay
    # ---------------------------------------------------------

    st.subheader("📈 Live Intelligence Replay")

    st.write(
        """
        Replay the match minute by minute and view only the information
        that would have been available at that point in the game.
        """
    )

    max_minute = 90

    if "minute" in events.columns:
        available_minutes = safe_numeric(events["minute"])
        if not available_minutes.empty:
            max_minute = max(1, min(120, int(available_minutes.max())))

    selected_minute = st.slider(
        "Match Minute",
        min_value=1,
        max_value=max_minute,
        value=min(60, max_minute),
        step=1,
    )

    rolling_window = st.selectbox(
        "Momentum Window",
        options=[5, 10, 15],
        index=1,
        format_func=lambda value: f"Last {value} minutes",
    )

    live_events = events_until_minute(events, selected_minute)
    window_events = events_in_window(events, selected_minute, rolling_window)

    team_1_live = team_live_metrics(live_events, team_1_name)
    team_2_live = team_live_metrics(live_events, team_2_name)

    team_1_window = team_live_metrics(window_events, team_1_name)
    team_2_window = team_live_metrics(window_events, team_2_name)

    momentum_1, momentum_2 = relative_momentum_score(
        team_1_window,
        team_2_window,
    )

    st.markdown(
        f"### {team_1_name} {team_1_live['Goals']} — "
        f"{team_2_live['Goals']} {team_2_name}"
    )

    st.caption(
        f"Match state at minute {selected_minute}. "
        f"Momentum uses the previous {rolling_window} minutes only."
    )

    live_score_col1, live_score_col2, live_score_col3, live_score_col4 = st.columns(4)

    with live_score_col1:
        st.metric(f"{team_1_name} xG", f"{team_1_live['xG']:.2f}")

    with live_score_col2:
        st.metric(f"{team_2_name} xG", f"{team_2_live['xG']:.2f}")

    with live_score_col3:
        st.metric(f"{team_1_name} Shots", team_1_live["Shots"])

    with live_score_col4:
        st.metric(f"{team_2_name} Shots", team_2_live["Shots"])

    st.markdown(f"#### Momentum — last {rolling_window} minutes")

    momentum_col1, momentum_col2 = st.columns(2)

    with momentum_col1:
        st.metric(team_1_name, f"{momentum_1:.1f} / 100")
        st.progress(int(round(momentum_1)))

    with momentum_col2:
        st.metric(team_2_name, f"{momentum_2:.1f} / 100")
        st.progress(int(round(momentum_2)))

    window_metric_col1, window_metric_col2 = st.columns(2)

    with window_metric_col1:
        st.markdown(f"#### {team_1_name} — recent activity")
        st.write(
            f"**Shots:** {team_1_window['Shots']}  \n"
            f"**xG:** {team_1_window['xG']:.2f}  \n"
            f"**Pressures:** {team_1_window['Pressures']}  \n"
            f"**Carries:** {team_1_window['Carries']}  \n"
            f"**Recoveries:** {team_1_window['Recoveries']}"
        )

    with window_metric_col2:
        st.markdown(f"#### {team_2_name} — recent activity")
        st.write(
            f"**Shots:** {team_2_window['Shots']}  \n"
            f"**xG:** {team_2_window['xG']:.2f}  \n"
            f"**Pressures:** {team_2_window['Pressures']}  \n"
            f"**Carries:** {team_2_window['Carries']}  \n"
            f"**Recoveries:** {team_2_window['Recoveries']}"
        )

    # ---------------------------------------------------------
    # Full-match momentum timeline
    # ---------------------------------------------------------

    st.markdown("### 📉 Momentum Timeline")

    st.write(
        "The chart shows how rolling match momentum changed across the full match."
    )

    timeline_df = build_momentum_timeline(
        events,
        team_1_name,
        team_2_name,
        max_minute,
        rolling_window,
    )

    timeline_long = timeline_df.melt(
        id_vars="Minute",
        value_vars=[team_1_name, team_2_name],
        var_name="Team",
        value_name="Momentum",
    )

    base_chart = alt.Chart(
        timeline_long
    ).encode(
        x=alt.X(
            "Minute:Q",
            title="Match Minute",
            scale=alt.Scale(
                domain=[1, max_minute]
            ),
        ),
        y=alt.Y(
            "Momentum:Q",
            title="Momentum Share",
            scale=alt.Scale(
                domain=[0, 100]
            ),
        ),
        color=alt.Color(
            "Team:N",
            title="Team",
        ),
        tooltip=[
            alt.Tooltip(
                "Minute:Q",
                title="Minute",
                format=".0f",
            ),
            alt.Tooltip(
                "Team:N",
                title="Team",
            ),
            alt.Tooltip(
                "Momentum:Q",
                title="Momentum",
                format=".1f",
            ),
        ],
    )

    momentum_line = base_chart.mark_line(
        strokeWidth=3
    )

    selected_minute_df = pd.DataFrame(
        {
            "Minute": [selected_minute]
        }
    )

    selected_rule = alt.Chart(
        selected_minute_df
    ).mark_rule(
        strokeDash=[6, 4],
    ).encode(
        x="Minute:Q",
    )

    goal_events = get_goal_events(events)

    chart_layers = [
        momentum_line,
        selected_rule,
    ]

    if not goal_events.empty:
        goal_rules = alt.Chart(
            goal_events
        ).mark_rule(
            strokeDash=[2, 2],
        ).encode(
            x="Minute:Q",
            tooltip=[
                alt.Tooltip(
                    "Minute:Q",
                    title="Goal minute",
                    format=".0f",
                ),
                alt.Tooltip(
                    "Team:N",
                    title="Team",
                ),
                alt.Tooltip(
                    "Scorer:N",
                    title="Scorer",
                ),
            ],
        )

        goal_labels = alt.Chart(
            goal_events
        ).mark_text(
            angle=270,
            align="left",
            baseline="middle",
            dx=5,
            dy=-5,
            fontSize=11,
        ).encode(
            x="Minute:Q",
            y=alt.value(10),
            text="Label:N",
            tooltip=[
                "Team:N",
                "Scorer:N",
            ],
        )

        chart_layers.extend(
            [
                goal_rules,
                goal_labels,
            ]
        )

    momentum_chart = alt.layer(
        *chart_layers
    ).properties(
        height=420
    ).interactive()

    st.altair_chart(
        momentum_chart,
        width="stretch",
    )

    if not goal_events.empty:
        goal_text = " • ".join(
            [
                f"{int(row['Minute'])}' — {row['Scorer']} ({row['Team']})"
                for _, row in goal_events.iterrows()
            ]
        )
        st.caption(
            f"⚽ Goal markers: {goal_text}"
        )
    else:
        st.caption(
            "No goal events were found in the current event data."
        )

    current_leader = (
        team_1_name
        if momentum_1 > momentum_2
        else team_2_name
        if momentum_2 > momentum_1
        else None
    )

    if current_leader:
        st.info(
            f"At minute {selected_minute}, **{current_leader}** "
            f"has the stronger rolling {rolling_window}-minute momentum."
        )
    else:
        st.info(
            f"At minute {selected_minute}, rolling momentum is balanced."
        )

    st.markdown("### 🚨 Live Analyst Alerts")

    live_alerts = build_live_alerts(
        team_1_name,
        team_2_name,
        team_1_window,
        team_2_window,
    )

    if live_alerts:
        for alert in live_alerts:
            st.warning(alert)
    else:
        st.success(
            "No major rolling-window alert threshold has been triggered "
            "at this match minute."
        )

    # ---------------------------------------------------------
    # Predictive Intelligence Prototype
    # ---------------------------------------------------------

    st.markdown("### 🔮 Predictive Intelligence")

    st.write(
        """
        Compare the transparent rule-based baseline with the experimental
        machine-learning model trained on historical match-state snapshots.
        """
    )

    st.markdown("#### 🧮 Baseline Prediction")

    team_1_win_prob, draw_prob, team_2_win_prob = baseline_match_probabilities(
        selected_minute,
        max_minute,
        team_1_live,
        team_2_live,
        momentum_1,
        momentum_2,
    )

    prediction_col1, prediction_col2, prediction_col3 = st.columns(3)

    with prediction_col1:
        st.metric(
            f"{team_1_name} Win",
            f"{team_1_win_prob:.1f}%",
        )
        st.progress(
            min(100, max(0, int(round(team_1_win_prob))))
        )

    with prediction_col2:
        st.metric(
            "Draw",
            f"{draw_prob:.1f}%",
        )
        st.progress(
            min(100, max(0, int(round(draw_prob))))
        )

    with prediction_col3:
        st.metric(
            f"{team_2_name} Win",
            f"{team_2_win_prob:.1f}%",
        )
        st.progress(
            min(100, max(0, int(round(team_2_win_prob))))
        )

    probability_rows = pd.DataFrame(
        {
            "Outcome": [
                f"{team_1_name} Win",
                "Draw",
                f"{team_2_name} Win",
            ],
            "Probability": [
                team_1_win_prob,
                draw_prob,
                team_2_win_prob,
            ],
        }
    )

    probability_chart = alt.Chart(
        probability_rows
    ).mark_bar(
        cornerRadiusTopRight=5,
        cornerRadiusBottomRight=5,
    ).encode(
        x=alt.X(
            "Probability:Q",
            title="Estimated Probability (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        y=alt.Y(
            "Outcome:N",
            title=None,
            sort="-x",
        ),
        tooltip=[
            alt.Tooltip("Outcome:N", title="Outcome"),
            alt.Tooltip(
                "Probability:Q",
                title="Estimated probability",
                format=".1f",
            ),
        ],
    )

    st.altair_chart(
        probability_chart,
        width="stretch",
    )

    st.markdown("#### 🧠 Prediction Drivers")

    prediction_drivers = prediction_driver_text(
        team_1_name,
        team_2_name,
        team_1_live,
        team_2_live,
        momentum_1,
        momentum_2,
    )

    for driver in prediction_drivers:
        st.info(driver)

    top_probability = max(
        team_1_win_prob,
        draw_prob,
        team_2_win_prob,
    )

    if top_probability == team_1_win_prob:
        predicted_outcome = f"{team_1_name} win"
    elif top_probability == team_2_win_prob:
        predicted_outcome = f"{team_2_name} win"
    else:
        predicted_outcome = "draw"

    st.success(
        f"Current baseline estimate at minute {selected_minute}: "
        f"**{predicted_outcome}** is the most likely outcome "
        f"({top_probability:.1f}%)."
    )

    st.caption(
        "Baseline inputs: scoreline, elapsed match time, cumulative xG, "
        "shot difference and rolling momentum. This baseline is rule-based "
        "and is not a trained probability model."
    )

    # ---------------------------------------------------------
    # Experimental Machine-Learning Prediction
    # ---------------------------------------------------------

    st.markdown("#### 🤖 Experimental ML Prediction")

    ml_checkpoint = ml_checkpoint_for_minute(
        selected_minute
    )

    if ml_checkpoint is None:
        st.info(
            "The experimental ML model becomes available from the first "
            "trained checkpoint at 15 minutes."
        )

    else:
        # The ML system was trained at fixed historical checkpoints.
        # Reconstruct the match state at exactly that checkpoint rather than
        # feeding an unseen intermediate minute into a checkpoint-specific model.
        ml_events = events_until_minute(
            events,
            ml_checkpoint,
        )

        ml_window_events = events_in_window(
            events,
            ml_checkpoint,
            10,
        )

        ml_team_1 = team_live_metrics(
            ml_events,
            team_1_name,
        )

        ml_team_2 = team_live_metrics(
            ml_events,
            team_2_name,
        )

        ml_team_1_recent = team_live_metrics(
            ml_window_events,
            team_1_name,
        )

        ml_team_2_recent = team_live_metrics(
            ml_window_events,
            team_2_name,
        )

        ml_momentum_1, ml_momentum_2 = relative_momentum_score(
            ml_team_1_recent,
            ml_team_2_recent,
        )

        ml_base_features = {
            "snapshot_minute": ml_checkpoint,

            "home_goals": ml_team_1["Goals"],
            "away_goals": ml_team_2["Goals"],
            "goal_difference": (
                ml_team_1["Goals"]
                - ml_team_2["Goals"]
            ),

            "home_xg": ml_team_1["xG"],
            "away_xg": ml_team_2["xG"],
            "xg_difference": (
                ml_team_1["xG"]
                - ml_team_2["xG"]
            ),

            "home_shots": ml_team_1["Shots"],
            "away_shots": ml_team_2["Shots"],
            "shot_difference": (
                ml_team_1["Shots"]
                - ml_team_2["Shots"]
            ),

            "home_passes": ml_team_1["Passes"],
            "away_passes": ml_team_2["Passes"],
            "pass_difference": (
                ml_team_1["Passes"]
                - ml_team_2["Passes"]
            ),

            "home_pass_completion": (
                ml_team_1["Pass Completion %"]
            ),
            "away_pass_completion": (
                ml_team_2["Pass Completion %"]
            ),

            "home_pressures": ml_team_1["Pressures"],
            "away_pressures": ml_team_2["Pressures"],
            "pressure_difference": (
                ml_team_1["Pressures"]
                - ml_team_2["Pressures"]
            ),

            "home_carries": ml_team_1["Carries"],
            "away_carries": ml_team_2["Carries"],

            "home_recoveries": ml_team_1["Recoveries"],
            "away_recoveries": ml_team_2["Recoveries"],

            "home_interceptions": ml_team_1["Interceptions"],
            "away_interceptions": ml_team_2["Interceptions"],

            "home_recent_xg": ml_team_1_recent["xG"],
            "away_recent_xg": ml_team_2_recent["xG"],

            "home_recent_shots": ml_team_1_recent["Shots"],
            "away_recent_shots": ml_team_2_recent["Shots"],

            "home_recent_pressures": (
                ml_team_1_recent["Pressures"]
            ),
            "away_recent_pressures": (
                ml_team_2_recent["Pressures"]
            ),

            "home_momentum": ml_momentum_1,
            "away_momentum": ml_momentum_2,
            "momentum_difference": (
                ml_momentum_1
                - ml_momentum_2
            ),
        }

        try:
            ml_result = predict_match_outcome(
                match_minute=ml_checkpoint,
                base_features=ml_base_features,
            )

            ml_home_pct = (
                ml_result["Home Win"] * 100.0
            )

            ml_draw_pct = (
                ml_result["Draw"] * 100.0
            )

            ml_away_pct = (
                ml_result["Away Win"] * 100.0
            )

            ml_col1, ml_col2, ml_col3 = st.columns(3)

            with ml_col1:
                st.metric(
                    f"{team_1_name} Win",
                    f"{ml_home_pct:.1f}%",
                )
                st.progress(
                    min(
                        100,
                        max(
                            0,
                            int(round(ml_home_pct)),
                        ),
                    )
                )

            with ml_col2:
                st.metric(
                    "Draw",
                    f"{ml_draw_pct:.1f}%",
                )
                st.progress(
                    min(
                        100,
                        max(
                            0,
                            int(round(ml_draw_pct)),
                        ),
                    )
                )

            with ml_col3:
                st.metric(
                    f"{team_2_name} Win",
                    f"{ml_away_pct:.1f}%",
                )
                st.progress(
                    min(
                        100,
                        max(
                            0,
                            int(round(ml_away_pct)),
                        ),
                    )
                )

            ml_probability_rows = pd.DataFrame(
                {
                    "Outcome": [
                        f"{team_1_name} Win",
                        "Draw",
                        f"{team_2_name} Win",
                    ],
                    "Probability": [
                        ml_home_pct,
                        ml_draw_pct,
                        ml_away_pct,
                    ],
                }
            )

            ml_probability_chart = alt.Chart(
                ml_probability_rows
            ).mark_bar(
                cornerRadiusTopRight=5,
                cornerRadiusBottomRight=5,
            ).encode(
                x=alt.X(
                    "Probability:Q",
                    title="Experimental ML Estimate (%)",
                    scale=alt.Scale(
                        domain=[0, 100]
                    ),
                ),
                y=alt.Y(
                    "Outcome:N",
                    title=None,
                    sort="-x",
                ),
                tooltip=[
                    alt.Tooltip(
                        "Outcome:N",
                        title="Outcome",
                    ),
                    alt.Tooltip(
                        "Probability:Q",
                        title="ML estimate",
                        format=".1f",
                    ),
                ],
            )

            st.altair_chart(
                ml_probability_chart,
                width="stretch",
            )

            ml_top = max(
                ml_home_pct,
                ml_draw_pct,
                ml_away_pct,
            )

            if ml_top == ml_home_pct:
                ml_predicted_outcome = (
                    f"{team_1_name} win"
                )
            elif ml_top == ml_away_pct:
                ml_predicted_outcome = (
                    f"{team_2_name} win"
                )
            else:
                ml_predicted_outcome = "draw"

            st.success(
                f"Experimental ML estimate using the **{ml_checkpoint}' "
                f"checkpoint**: **{ml_predicted_outcome}** is currently "
                f"the highest-probability outcome ({ml_top:.1f}%)."
            )

            validation_accuracy = ml_result.get(
                "validation_accuracy"
            )
            validation_macro_f1 = ml_result.get(
                "validation_macro_f1"
            )
            validation_log_loss = ml_result.get(
                "validation_log_loss"
            )
            model_variant = ml_result.get(
                "model_variant",
                "Unknown",
            )

            validation_parts = [
                f"Checkpoint: **{ml_checkpoint}'**",
                f"Calibration: **{model_variant}**",
            ]

            if validation_accuracy is not None:
                validation_parts.append(
                    "Validation accuracy: "
                    f"**{validation_accuracy * 100:.0f}%**"
                )

            if validation_macro_f1 is not None:
                validation_parts.append(
                    f"Macro F1: **{validation_macro_f1:.3f}**"
                )

            if validation_log_loss is not None:
                validation_parts.append(
                    f"Log Loss: **{validation_log_loss:.3f}**"
                )

            st.info(
                " | ".join(
                    validation_parts
                )
            )

            if selected_minute != ml_checkpoint:
                st.caption(
                    f"The replay is at minute {selected_minute}, but the ML "
                    f"estimate uses the latest trained historical checkpoint "
                    f"at {ml_checkpoint}'. It updates again at the next trained "
                    f"checkpoint."
                )

            st.warning(
                "Experimental research estimate — the calibrated model improved "
                "classification performance, but probability calibration remains "
                "imperfect. Treat these percentages as model estimates rather "
                "than production-grade probabilities."
            )

        except Exception as ml_error:
            st.error(
                "The experimental ML prediction could not be generated. "
                f"Details: {ml_error}"
            )

    st.caption(
        "Momentum remains a transparent prototype heuristic. The ML outcome "
        "layer is trained separately on historical StatsBomb match-state data."
    )

    st.divider()


    # ---------------------------------------------------------

    st.divider()

if NAV_PAGE == "📄 Reports & Export":
    # ---------------------------------------------------------
    # Match Intelligence Reports
    # ---------------------------------------------------------

    st.subheader("📄 Match Intelligence Reports")

    st.write(
        "Generate a visual one-page PNG for rapid review and a detailed PDF that "
        "interprets the match indicators for coaches and analysts."
    )

    st.caption(
        "Both reports are generated from the same reproducible match data. The PDF "
        "explains the evidence, tactical signals, player progression and coaching review areas."
    )

    png_col, pdf_col = st.columns(2)

    with png_col:
        generate_report_clicked = st.button(
            "📊 Generate PNG Report",
            type="primary",
            key="generate_match_report_button",
            width="stretch",
        )

    with pdf_col:
        generate_pdf_clicked = st.button(
            "📄 Generate Detailed PDF Report",
            key="generate_match_pdf_button",
            width="stretch",
        )

    if generate_report_clicked:
        try:
            with st.spinner("Generating one-page match intelligence report..."):
                generated_report_path = generate_match_report(match_id=MATCH_ID)
                st.session_state["generated_match_report_path"] = str(generated_report_path)
            st.success("PNG match intelligence report generated successfully.")
        except Exception as report_error:
            st.error(f"The PNG report could not be generated. Details: {report_error}")

    if generate_pdf_clicked:
        try:
            with st.spinner("Generating detailed coach interpretation PDF..."):
                png_path = generate_match_report(match_id=MATCH_ID)
                generated_pdf_path = generate_pdf_report(
                    match_id=MATCH_ID,
                    png_report_path=png_path,
                )
                st.session_state["generated_match_report_path"] = str(png_path)
                st.session_state["generated_match_pdf_path"] = str(generated_pdf_path)
            st.success("Detailed PDF interpretation report generated successfully.")
        except Exception as pdf_error:
            st.error(f"The PDF report could not be generated. Details: {pdf_error}")

    saved_report_path = st.session_state.get("generated_match_report_path")
    saved_pdf_path = st.session_state.get("generated_match_pdf_path")

    if saved_report_path:
        report_path_obj = Path(saved_report_path)
        if report_path_obj.exists():
            st.markdown("### 🖼️ Visual Report Preview")
            st.image(str(report_path_obj), width="stretch")

            with open(report_path_obj, "rb") as report_file:
                report_bytes = report_file.read()

            st.download_button(
                label="⬇️ Download PNG Report",
                data=report_bytes,
                file_name=report_path_obj.name,
                mime="image/png",
                key="download_match_report_png",
            )

    if saved_pdf_path:
        pdf_path_obj = Path(saved_pdf_path)
        if pdf_path_obj.exists():
            with open(pdf_path_obj, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            st.download_button(
                label="⬇️ Download Detailed PDF Report",
                data=pdf_bytes,
                file_name=pdf_path_obj.name,
                mime="application/pdf",
                key="download_match_report_pdf",
            )

            st.info(
                "The PDF contains the visual dashboard plus match interpretation, "
                "tactical and player intelligence, coaching takeaways and methodology notes."
            )


    # ---------------------------------------------------------
    # Focused Section Reports
    # ---------------------------------------------------------

    st.divider()

    st.subheader("🧾 Focused Section Reports")

    st.write(
        """
        Export a focused PNG or PDF for a specific intelligence module.
        This is especially useful for comparison-heavy sections when a coach,
        analyst or recruiter needs one topic without the full-match report.
        """
    )

    section_options = dict(
        available_section_reports()
    )

    section_label_to_key = {
        label: key
        for key, label in section_options.items()
    }

    selected_section_label = st.selectbox(
        "Report Section",
        options=list(
            section_label_to_key.keys()
        ),
        key="focused_report_section",
    )

    selected_section_key = section_label_to_key[
        selected_section_label
    ]

    focused_player_1 = None
    focused_player_2 = None

    if selected_section_key == "player_comparison":
        report_player_names = (
            events.get(
                "player",
                pd.Series(
                    "",
                    index=events.index,
                ),
            )
            .apply(
                lambda value:
                    value.get(
                        "name",
                        "",
                    )
                    if isinstance(
                        value,
                        dict,
                    )
                    else (
                        ""
                        if pd.isna(
                            value
                        )
                        else str(
                            value
                        )
                    )
            )
            .replace(
                "",
                np.nan,
            )
            .dropna()
            .unique()
            .tolist()
        )

        report_player_names = sorted(
            report_player_names
        )

        fp_col_1, fp_col_2 = st.columns(2)

        with fp_col_1:
            focused_player_1 = st.selectbox(
                "First Player",
                options=report_player_names,
                key="focused_report_player_1",
            )

        with fp_col_2:
            default_second_index = (
                1
                if len(
                    report_player_names
                )
                > 1
                else 0
            )

            focused_player_2 = st.selectbox(
                "Second Player",
                options=report_player_names,
                index=default_second_index,
                key="focused_report_player_2",
            )

    focused_snapshot_minute = 85

    if selected_section_key == "ml_prediction":
        focused_snapshot_minute = st.select_slider(
            "ML Snapshot Minute",
            options=[
                15,
                30,
                45,
                60,
                75,
                85,
            ],
            value=85,
            key="focused_report_ml_minute",
        )

    focused_png_col, focused_pdf_col = st.columns(2)

    with focused_png_col:
        focused_png_clicked = st.button(
            "🖼️ Generate Section PNG",
            key="generate_focused_section_png",
            width="stretch",
        )

    with focused_pdf_col:
        focused_pdf_clicked = st.button(
            "📄 Generate Section PDF",
            key="generate_focused_section_pdf",
            width="stretch",
        )

    if focused_png_clicked:
        try:
            with st.spinner(
                f"Generating {selected_section_label} PNG..."
            ):
                focused_png_path = generate_section_png(
                    section_key=selected_section_key,
                    match_id=MATCH_ID,
                    player_1=focused_player_1,
                    player_2=focused_player_2,
                    snapshot_minute=focused_snapshot_minute,
                )

                st.session_state[
                    "focused_section_png_path"
                ] = str(
                    focused_png_path
                )

                st.session_state[
                    "focused_section_last_label"
                ] = selected_section_label

            st.success(
                f"{selected_section_label} PNG generated successfully."
            )

        except Exception as focused_png_error:
            st.error(
                f"The section PNG could not be generated. Details: {focused_png_error}"
            )

    if focused_pdf_clicked:
        try:
            with st.spinner(
                f"Generating {selected_section_label} PDF..."
            ):
                focused_pdf_path = generate_section_pdf(
                    section_key=selected_section_key,
                    match_id=MATCH_ID,
                    player_1=focused_player_1,
                    player_2=focused_player_2,
                    snapshot_minute=focused_snapshot_minute,
                )

                st.session_state[
                    "focused_section_pdf_path"
                ] = str(
                    focused_pdf_path
                )

                st.session_state[
                    "focused_section_last_label"
                ] = selected_section_label

            st.success(
                f"{selected_section_label} PDF generated successfully."
            )

        except Exception as focused_pdf_error:
            st.error(
                f"The section PDF could not be generated. Details: {focused_pdf_error}"
            )

    focused_png_saved = st.session_state.get(
        "focused_section_png_path"
    )

    focused_pdf_saved = st.session_state.get(
        "focused_section_pdf_path"
    )

    focused_last_label = st.session_state.get(
        "focused_section_last_label",
        selected_section_label,
    )

    if focused_png_saved:
        focused_png_obj = Path(
            focused_png_saved
        )

        if focused_png_obj.exists():
            st.markdown(
                f"### 🖼️ {focused_last_label} Preview"
            )

            st.image(
                str(
                    focused_png_obj
                ),
                width="stretch",
            )

            with open(
                focused_png_obj,
                "rb",
            ) as focused_png_file:
                focused_png_bytes = focused_png_file.read()

            st.download_button(
                label="⬇️ Download Section PNG",
                data=focused_png_bytes,
                file_name=focused_png_obj.name,
                mime="image/png",
                key="download_focused_section_png",
            )

    if focused_pdf_saved:
        focused_pdf_obj = Path(
            focused_pdf_saved
        )

        if focused_pdf_obj.exists():
            with open(
                focused_pdf_obj,
                "rb",
            ) as focused_pdf_file:
                focused_pdf_bytes = focused_pdf_file.read()

            st.download_button(
                label="⬇️ Download Section PDF",
                data=focused_pdf_bytes,
                file_name=focused_pdf_obj.name,
                mime="application/pdf",
                key="download_focused_section_pdf",
            )

            st.info(
                "The focused PDF includes the section visual, interpretation, "
                "evidence table and methodology/interpretation boundaries."
            )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "LiveMatch Intelligence | Football Analytics & Decision Support"
)
