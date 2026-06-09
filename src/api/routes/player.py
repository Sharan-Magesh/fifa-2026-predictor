"""
Player Router
=============
GET /api/player/compare?player_a=Kylian+Mbappe&player_b=Vinicius+Junior

Returns a head-to-head player comparison for the Player page:
  - Composite score breakdown for each player (intl_xg, club_form, experience)
  - Radar chart data (6 normalised dimensions)
  - Match impact score — player's team-relative contribution

All data comes from player_features.py outputs. No live API calls at request time.
"""

from pathlib import Path
from typing import Optional
import sys

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

PROJECT_ROOT       = Path(__file__).resolve().parents[3]
SQUADS_PATH        = PROJECT_ROOT / "data" / "processed" / "wc2026_squads.csv"
PLAYER_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "player_scores.csv"


def _load_player_scores() -> pd.DataFrame:
    if not PLAYER_SCORES_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="player_scores.csv not found. Run player_features.py first.",
        )
    return pd.read_csv(PLAYER_SCORES_PATH)


def _find_player(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Case-insensitive player name lookup with partial-match fallback.
    Name normalisation is a known pain point — this handles most mismatches.
    """
    # Exact match first (case-insensitive)
    exact = df[df["player"].str.strip().str.lower() == name.strip().lower()]
    if not exact.empty:
        return exact.iloc[0]

    # Partial match fallback — handles "Mbappe" matching "Kylian Mbappe-Lottin"
    partial = df[df["player"].str.lower().str.contains(name.strip().lower(), regex=False)]
    if not partial.empty:
        return partial.iloc[0]

    raise HTTPException(
        status_code=404,
        detail=f"Player '{name}' not found. Use /api/player/search?q={name} to check spelling.",
    )


def _build_radar(row: pd.Series) -> dict:
    """
    Build 6-dimension radar chart data from available score columns.
    All values normalised to [0, 1] — already done in player_features.py.
    Radar dimensions chosen to map to visible skills on the Player page.
    """
    return {
        "international_xg":  round(float(row.get("intl_xg_norm",     0)), 3),
        "club_form":         round(float(row.get("club_form_norm",    0)), 3),
        "experience":        round(float(row.get("experience_norm",   0)), 3),
        "composite_score":   round(float(row.get("composite_score",   0)), 3),
        # If deeper breakdown columns exist from StatsBomb, include them
        "shot_quality":      round(float(row.get("shot_quality_norm", row.get("intl_xg_norm",  0))), 3),
        "involvement":       round(float(row.get("involvement_norm",  row.get("club_form_norm", 0))), 3),
    }


# ---------------------------------------------------------------------------
# GET /api/player/compare
# ---------------------------------------------------------------------------
@router.get("/compare")
def compare_players(
    player_a: str = Query(..., description="First player name, e.g. 'Kylian Mbappe'"),
    player_b: str = Query(..., description="Second player name, e.g. 'Vinicius Junior'"),
):
    """
    Head-to-head player comparison for the Player page radar chart.

    Response shape:
    {
        "player_a": {
            "player": "Kylian Mbappe",
            "team": "France",
            "position": "FW",
            "composite_score": 0.91,
            "breakdown": {
                "intl_xg_contribution": 0.88,
                "club_form_contribution": 0.92,
                "experience_contribution": 0.79
            },
            "radar": {
                "international_xg": 0.88,
                "club_form": 0.92,
                "experience": 0.79,
                "composite_score": 0.91,
                "shot_quality": 0.87,
                "involvement": 0.85
            }
        },
        "player_b": { ... },
        "advantage": "player_a",   # which player has higher composite_score
        "score_delta": 0.07
    }
    """
    df = _load_player_scores()

    row_a = _find_player(df, player_a)
    row_b = _find_player(df, player_b)

    def _build_profile(row: pd.Series) -> dict:
        composite = float(row.get("composite_score", 0))
        return {
            "player":   str(row.get("player",   "Unknown")),
            "team":     str(row.get("team",     "Unknown")),
            "position": str(row.get("position", "Unknown")),
            "composite_score": round(composite, 3),
            "breakdown": {
                "intl_xg_contribution":      round(float(row.get("intl_xg_norm",   0)) * 0.45, 3),
                "club_form_contribution":    round(float(row.get("club_form_norm",  0)) * 0.35, 3),
                "experience_contribution":   round(float(row.get("experience_norm", 0)) * 0.20, 3),
            },
            "radar": _build_radar(row),
        }

    profile_a = _build_profile(row_a)
    profile_b = _build_profile(row_b)

    score_a = profile_a["composite_score"]
    score_b = profile_b["composite_score"]

    advantage   = "player_a" if score_a >= score_b else "player_b"
    score_delta = round(abs(score_a - score_b), 3)

    return {
        "player_a":    profile_a,
        "player_b":    profile_b,
        "advantage":   advantage,
        "score_delta": score_delta,
    }


# ---------------------------------------------------------------------------
# GET /api/player/team/{team_name}
# ---------------------------------------------------------------------------
@router.get("/team/{team_name}")
def get_team_players(team_name: str):
    """
    All players for a given team with their composite scores.
    Used by the Team page squad depth section and the Player page's team selector.

    Response shape:
    {
        "team": "France",
        "players": [
            {"player": "Kylian Mbappe", "position": "FW", "composite_score": 0.91},
            ...
        ]
    }
    """
    df = _load_player_scores()

    team_df = df[df["team"].str.strip().str.lower() == team_name.strip().lower()]
    if team_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No players found for team '{team_name}'.",
        )

    team_df_sorted = team_df.sort_values("composite_score", ascending=False)
    team_df_sorted = team_df_sorted.where(pd.notnull(team_df_sorted), None)

    cols = ["player", "position", "composite_score"]
    available_cols = [c for c in cols if c in team_df_sorted.columns]

    return {
        "team":    team_name,
        "players": team_df_sorted[available_cols].to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# GET /api/player/search
# ---------------------------------------------------------------------------
@router.get("/search")
def search_player(q: str = Query(..., description="Partial player name")):
    """
    Fuzzy player search — returns up to 10 matches.
    Useful for the Player page autocomplete / name-mismatch debugging.

    Response shape:
    {
        "query": "mbappe",
        "results": [
            {"player": "Kylian Mbappe-Lottin", "team": "France", "position": "FW"},
            ...
        ]
    }
    """
    df = _load_player_scores()

    matches = df[df["player"].str.lower().str.contains(q.strip().lower(), regex=False)]
    matches = matches.head(10).where(pd.notnull(matches), None)

    cols = ["player", "team", "position"]
    available_cols = [c for c in cols if c in matches.columns]

    return {
        "query":   q,
        "results": matches[available_cols].to_dict(orient="records"),
    }
