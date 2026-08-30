# LiveMatch Intelligence

**LiveMatch Intelligence** is a football analytics and decision-support platform built with Python and Streamlit. It transforms StatsBomb event data into interactive match intelligence, tactical comparisons, player and team analysis, live-style momentum signals, experimental match-outcome predictions, and professional PDF/PNG reports.

> **Project status:** Portfolio / research project  
> **Data source:** StatsBomb Open Data  
> **Important:** The current implementation uses historical event data to simulate live-match analysis. It is not connected to a real-time commercial event feed.

---

## Overview

LiveMatch Intelligence was designed to answer a practical question:

**How can event-level football data be converted into analyst-friendly intelligence that supports match review, tactical interpretation, and decision-making?**

The platform combines traditional event analysis with custom intelligence layers, including:

- Match overview and team comparison
- Player intelligence and player comparison
- Shot and xG analysis
- Passing networks and build-up structure
- Possession and territory analysis
- Progressive actions and field progression
- First-half vs second-half tactical change detection
- Live-style momentum and intelligence signals
- Experimental time-aware match outcome prediction
- Professional PNG and PDF report generation

---

## Key Features

### Match Intelligence
- Multi-match selection from StatsBomb Open Data
- Match score reconstruction
- Regulation/extra-time analysis
- Penalty shootout events excluded from normal match analytics
- Team-level match summaries
- xG, shots, passing, pressure, territory, and progression indicators

### Team Intelligence
- Side-by-side team comparison
- Pass completion
- Pressure-event analysis
- Territory indicators
- Progressive actions
- Final-third and box entries

### Player Intelligence
- Player-level event statistics
- Player comparison
- Player position/context information
- Player images where available
- Professional football-name formatting for reports

### Shot Analysis
- Shot maps
- Goal markers
- xG-scaled shot markers
- Shot volume and on-target comparison
- High-quality chance comparison
- Analyst interpretation

### Passing & Build-up
- Completed-pass networks
- Average event locations
- Strongest passing connections
- Top passer and most-involved player
- Passing volume and completion comparison
- Analyst interpretation

### Possession & Territory
- Event-derived territory maps
- 3 × 3 pitch-zone activity distribution
- Territory Index
- Average event X position
- Attacking-third share
- Final-third and box-zone event counts

> Territory Index is a project-defined event-based indicator and is **not** the same as official possession or optical-tracking spatial control.

### Progressive Actions
- Progressive passes
- Progressive carries
- Final-third entries
- Box entries
- Forward-distance indicators
- Player progression leaders

### Tactical Analysis
- First-half vs second-half comparison
- Average action-position change
- Pressure change
- Shot and xG change
- Attack Index change
- Player position-shift signals
- Tactical momentum visualisation

> Tactical signals describe event-derived behavioural changes. They do not prove formation changes or coaching instructions.

### Live Intelligence
- Rolling match-momentum timeline
- Recent momentum
- Territory signal
- Progression signal
- xG threat
- Shot-volume signal
- Multi-signal **Intelligence Advantage**
- Live-style alerts and analyst prompts

> Intelligence Advantage is a transparent project heuristic. It is **not** a win probability.

### Predictive Intelligence
- Time-aware match-outcome prediction
- Home/Draw/Away outcome estimates
- Match-state features including:
  - score
  - xG
  - shots
  - pressures
  - recent xG
- Multiple match-minute model checkpoints
- Validation context displayed with each prediction
- Experimental probability disclaimer

The current model is a research prototype and should not be interpreted as a production betting or forecasting system.

---

## Professional Reporting

LiveMatch Intelligence generates both focused section reports and a complete multi-page match intelligence report.

### Focused reports

Available report sections include:

- Team Comparison
- Player Comparison
- Shot Analysis
- Passing Network
- Possession & Territory
- Progressive Actions
- Tactical Analysis
- Live Intelligence
- Predictive Intelligence

Reports can be exported as **PNG** and **PDF**.

### Full Match Intelligence Report

The current professional report contains:

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

Example report:

`reports/LiveMatch_Intelligence_Argentina_vs_France_Final.pdf`

---

## Example Analysis

A key showcase match in the project is the **Argentina vs France 2022 FIFA World Cup Final**.

The full-match report includes:

- 3–3 match score before the penalty shootout
- event-derived xG comparison
- shot comparison
- passing-network analysis
- territory comparison
- progressive-action analysis
- tactical change signals
- live-style momentum intelligence
- 60-minute predictive model output
- final analyst conclusions and methodology notes

Penalty-shootout period events are excluded from normal match-performance metrics.

---

## Project Architecture

```text
live-match-intelligence/
│
├── dashboard/
│   ├── app.py
│   └── assets/
│       ├── players/
│       └── teams/
│
├── src/
│   ├── image_fetcher.py
│   ├── live_intelligence_analyzer.py
│   ├── match_analyzer.py
│   ├── match_catalog.py
│   ├── match_outcome_predictor.py
│   ├── match_report_generator.py
│   ├── pass_network_analyzer.py
│   ├── player_analyzer.py
│   ├── player_comparison.py
│   ├── possession_territory_analyzer.py
│   ├── progressive_actions_analyzer.py
│   ├── score_utils.py
│   ├── section_report_generator.py
│   ├── shot_analysis_analyzer.py
│   ├── statsbomb_explorer.py
│   └── tactical_change_detector.py
│
├── models/
│   ├── enhanced_time_aware/
│   └── time_aware/
│
├── reports/
│   ├── sections/
│   └── _full_report_pages/
│
├── tests/
├── notebooks/
└── README.md
```

The repository also contains training, calibration, evaluation, and dataset-validation scripts for the predictive model.

---

## Core Modules

| Module | Purpose |
|---|---|
| `match_analyzer.py` | Match loading and match-level summaries |
| `player_analyzer.py` | Player statistics |
| `player_comparison.py` | Player comparison analysis |
| `shot_analysis_analyzer.py` | Shot and xG analysis |
| `pass_network_analyzer.py` | Passing-network and build-up analysis |
| `possession_territory_analyzer.py` | Territory and zone analysis |
| `progressive_actions_analyzer.py` | Progressive pass/carry analysis |
| `tactical_change_detector.py` | Half-to-half tactical signal detection |
| `live_intelligence_analyzer.py` | Rolling momentum and live-style intelligence |
| `match_outcome_predictor.py` | Experimental time-aware prediction |
| `section_report_generator.py` | Focused PNG/PDF reports |
| `match_report_generator.py` | Full professional match report |
| `match_catalog.py` | StatsBomb open competition/match selection |
| `image_fetcher.py` | Team/player image resolution and caching |
| `score_utils.py` | Match score and shootout handling |

---

## Technology Stack

- **Python**
- **Streamlit**
- **pandas**
- **NumPy**
- **Matplotlib**
- **Altair**
- **scikit-learn**
- **joblib**
- **ReportLab**
- **Pillow**
- **statsbombpy**

---

## Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd live-match-intelligence
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Running the Dashboard

From the project root:

```powershell
streamlit run dashboard/app.py
```

Streamlit will normally open the application at:

```text
http://localhost:8501
```

---

## Data

The current application uses **StatsBomb Open Data** through `statsbombpy`.

The open-data integration allows the project to:

- discover available competitions
- load open matches
- retrieve event data
- analyse historical matches without commercial credentials

You may see a StatsBomb `NoAuthWarning`. This is expected when using open-data access without commercial credentials.

---

## Predictive Model

The repository includes time-aware match-outcome models at different match checkpoints, including approximately:

- 15 minutes
- 30 minutes
- 45 minutes
- 60 minutes
- 75 minutes
- 85 minutes

Model artefacts and metadata are stored under the project model directories.

The predictive layer is intended to demonstrate:

- feature engineering
- match-state modelling
- time-aware model training
- validation
- probability-style output
- integration of ML into an analyst-facing product

It is an **experimental research estimate**, not a guaranteed forecast.

---

## Interpretation Boundaries

This project deliberately separates analytical signals from stronger claims that the available data cannot support.

- Event locations are **not** continuous player-tracking data.
- Territory Index is a custom event-derived metric.
- Progressive actions follow the project's defined forward-distance logic.
- Tactical shifts are analytical signals, not confirmed coaching instructions.
- Intelligence Advantage is a heuristic, not win probability.
- Predictive probabilities are experimental and should be interpreted with validation metrics.
- Historical event data currently simulates a live workflow; a production live version would require a real-time event provider.

---

## Future Development

Planned or possible extensions include:

- Real-time commercial event-feed integration
- Optical/player-tracking data
- Expected-threat (xT) modelling
- Possession-chain modelling
- Pressing intensity metrics
- Automated match-event alerts
- Model recalibration with a larger historical dataset
- Team/player similarity analysis
- Cloud deployment
- Authentication and saved analyst workspaces
- Automated post-match report delivery

---

## Portfolio Value

This project demonstrates practical experience in:

- football analytics
- data engineering
- exploratory data analysis
- feature engineering
- machine learning
- model evaluation
- interactive dashboard development
- analytical visualisation
- report automation
- modular Python application design
- translating technical outputs into decision-support insights

---

## Repository Cleanliness

Generated development artefacts should not be committed unnecessarily.

Recommended exclusions include:

```text
.venv/
__pycache__/
*.pyc
data/image_cache/
project_structure.txt
full_report_functions.txt
*_test.png
*_cleaned.png
*_refined.png
```

Keep only selected final screenshots and reports that strengthen the portfolio.

---

## Disclaimer

LiveMatch Intelligence is an independent educational and portfolio project.

Team badges, player images, football data, and third-party resources remain the property of their respective owners/providers. The project is not affiliated with FIFA, StatsBomb, any football association, club, player, or competition.

---

## Author

Developed as a football data science and analytics portfolio project.

