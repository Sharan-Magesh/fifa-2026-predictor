"""
Team Router
===========
GET /api/team/{team_name}

Returns everything the Team page needs:
  1. Tournament path probabilities (from monte_carlo_results.csv)
  2. Group opponents
  3. Top-5 key players by composite score (from player_features.py)
  4. Full squad list (from wc2026_squads.csv)
  5. Elo trajectory — last N data points (from elo_ratings.csv if present)

Design note: all heavy data is loaded from disk CSVs, not recomputed at
request time. The pipeline pre-computes everything. The API is read-only.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

PROJECT_ROOT      = Path(__file__).resolve().parents[3]
MONTE_CARLO_PATH  = PROJECT_ROOT / "data" / "processed" / "monte_carlo_results.csv"
SQUADS_PATH       = PROJECT_ROOT / "data" / "processed" / "wc2026_squads.csv"
ELO_PATH          = PROJECT_ROOT / "data" / "processed" / "elo_ratings.csv"
PLAYER_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "player_scores.csv"


def _load_csv_safe(path: Path, label: str) -> Optional[pd.DataFrame]:
    """Load CSV, return None (not 500) if missing — callers decide how to handle."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def _normalise_team_name(name: str) -> str:
    """URL path segments come in as-is. Handle spaces encoded as %20 or +."""
    return name.strip()


# ---------------------------------------------------------------------------
# GET /api/team/{team_name}
# ---------------------------------------------------------------------------
@router.get("/{team_name}")
def get_team(
    team_name: str,
    elo_points: int = Query(20, description="How many Elo history points to return"),
):
    """
    Full team profile for the Team page.

    Response shape:
    {
        "team": "France",
        "group": "B",
        "tournament_probabilities": {
            "p_win_tournament": 0.152,
            "p_final": 0.281,
            "p_advance_sf": 0.451,
            "p_advance_qf": 0.621
        },
        "group_opponents": ["Morocco", "Croatia", "Canada"],
        "key_players": [
            {"player": "Kylian Mbappe", "position": "FW", "composite_score": 0.91},
            ...
        ],
        "squad": [
            {"player": "Hugo Lloris", "position": "GK", "caps": 145},
            ...
        ],
        "elo_trajectory": [
            {"date": "2024-11-15", "elo": 2041},
            ...
        ]
    }
    """
    team_name = _normalise_team_name(team_name)

    # --- 1. Tournament probabilities ---
    mc_df = _load_csv_safe(MONTE_CARLO_PATH, "monte_carlo_results")
    if mc_df is None:
        raise HTTPException(status_code=500, detail="Monte Carlo results not found.")

    team_mc = mc_df[mc_df["team"].str.strip().str.lower() == team_name.lower()]
    if team_mc.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Team '{team_name}' not found. Check /api/match/teams for valid names.",
        )
    team_row = team_mc.iloc[0]

    prob_cols = ["p_win_tournament", "p_final", "p_advance_sf", "p_advance_qf"]
    tournament_probs = {
        col: round(float(team_row[col]), 3)
        for col in prob_cols
        if col in team_row.index and pd.notna(team_row[col])
    }

    group_name = str(team_row["group"]) if "group" in team_row.index else None

    # --- 2. Group opponents ---
    group_opponents = []
    if group_name:
        group_df = mc_df[
            (mc_df["group"] == group_name) &
            (mc_df["team"].str.strip().str.lower() != team_name.lower())
        ]
        group_opponents = group_df["team"].tolist()

    # --- 3. Squad ---
    squad_df = _load_csv_safe(SQUADS_PATH, "wc2026_squads")
    squad_records = []
    if squad_df is not None:
        team_squad = squad_df[squad_df["team"].str.strip().str.lower() == team_name.lower()]
        # Return all available columns — position, caps, club, etc.
        team_squad = team_squad.where(pd.notnull(team_squad), None)
        squad_records = team_squad.drop(columns=["team"], errors="ignore").to_dict(orient="records")

    # --- 4. Key players (top 5 by composite_score) ---
    key_players = []
    player_scores_df = _load_csv_safe(PLAYER_SCORES_PATH, "player_scores")
    if player_scores_df is not None:
        team_players = player_scores_df[
            player_scores_df["team"].str.strip().str.lower() == team_name.lower()
        ]
        if not team_players.empty and "composite_score" in team_players.columns:
            top5 = team_players.nlargest(5, "composite_score")
            top5 = top5.where(pd.notnull(top5), None)
            key_players = top5[["player", "position", "composite_score"]].to_dict(orient="records")
    else:
        # Fall back to squad file if player_scores.csv doesn't exist yet
        # (can happen if player_features.py hasn't been run)
        if squad_records:
            key_players = squad_records[:5]

    # --- 5. Elo trajectory ---
    elo_trajectory = []
    elo_df = _load_csv_safe(ELO_PATH, "elo_ratings")
    if elo_df is not None:
        if "date" in elo_df.columns and "elo" in elo_df.columns:
            # Long format: one row per (team, date)
            team_elo = elo_df[elo_df["team"].str.strip().str.lower() == team_name.lower()]
            if not team_elo.empty:
                team_elo_sorted = team_elo.sort_values("date").tail(elo_points)
                elo_trajectory = team_elo_sorted[["date", "elo"]].to_dict(orient="records")
        elif "elo" in elo_df.columns:
            # Flat snapshot format: one row per team, single current Elo value
            team_elo = elo_df[elo_df["team"].str.strip().str.lower() == team_name.lower()]
            if not team_elo.empty:
                elo_val = team_elo.iloc[0]["elo"]
                if pd.notna(elo_val):
                    elo_trajectory = [{"date": "current", "elo": float(elo_val)}]

    return {
        "team":                    team_row["team"],
        "group":                   group_name,
        "tournament_probabilities": tournament_probs,
        "group_opponents":         group_opponents,
        "key_players":             key_players,
        "squad":                   squad_records,
        "elo_trajectory":          elo_trajectory,
    }
