"""
Player Router
=============
GET /api/player/compare?player_a=Kylian+Mbappe&player_b=Vinicius+Junior

Returns a head-to-head player comparison for the Player page:
  - Position-aware composite score breakdown (xG / club form / value / experience)
  - Radar chart data (6 genuinely independent dimensions)
  - Cross-position caveat flag when comparing different role groups

All data comes from player_features.py outputs. No live API calls at request time.
"""

from pathlib import Path
from typing import Optional
import sys
import unicodedata

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


def _deaccent(s: str) -> str:
    """Strip diacritics: 'Mbappé' -> 'Mbappe'."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def _find_player(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Case-insensitive, accent-insensitive player name lookup with partial-match fallback.
    Handles "Mbappe" matching "Mbappé", partial names, hyphenated names, etc.
    """
    query_norm = _deaccent(name.strip()).lower()

    # Build accent-stripped version of stored names once
    stored_norm = df["player"].apply(lambda x: _deaccent(str(x).strip()).lower())

    # Exact match (accent-insensitive)
    exact = df[stored_norm == query_norm]
    if not exact.empty:
        return exact.iloc[0]

    # Partial match fallback
    partial = df[stored_norm.str.contains(query_norm, regex=False)]
    if not partial.empty:
        return partial.iloc[0]

    raise HTTPException(
        status_code=404,
        detail=f"Player '{name}' not found. Use /api/player/search?q={name} to check spelling.",
    )


def _safe(row, col, default=0.0) -> float:
    try:
        v = float(row.get(col, default))
        return default if pd.isna(v) else v
    except (TypeError, ValueError):
        return default


def _build_radar(row: pd.Series) -> dict:
    """
    Build 6-dimension radar chart data — six GENUINELY independent axes
    (the old version padded the radar by duplicating xG and club form
    under different labels, which made every comparison look more
    symmetric than it really was).

      finishing     : recency-weighted international xG/shot (StatsBomb)
      club_form     : club npxG/90 (Understat/FBref, with fallback)
      experience    : international caps
      market_value  : log-scaled Transfermarkt valuation
      productivity  : international goals per cap
      peak_age      : closeness to the empirical 24-29 performance peak
    """
    age  = _safe(row, "age", 27)
    caps = max(_safe(row, "caps", 0), 1)
    goals_per_cap = _safe(row, "goals", 0) / caps

    if 24 <= age <= 29:
        peak = 1.0
    else:
        peak = max(0.0, 1.0 - 0.09 * (24 - age if age < 24 else age - 29))

    return {
        "finishing":     round(_safe(row, "intl_xg_norm"), 3),
        "club_form":     round(_safe(row, "club_form_norm"), 3),
        "experience":    round(_safe(row, "experience_norm"), 3),
        "market_value":  round(_safe(row, "value_norm"), 3),
        "productivity":  round(min(goals_per_cap / 0.8, 1.0), 3),  # 0.8 g/cap ~ all-time elite
        "peak_age":      round(peak, 3),
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
    Composite scores are position-aware (see player_features.POSITION_WEIGHTS),
    breakdown bars sum exactly to the composite, and cross-position
    comparisons are flagged.
    """
    df = _load_player_scores()

    row_a = _find_player(df, player_a)
    row_b = _find_player(df, player_b)

    from src.features.player_features import POSITION_WEIGHTS, DEFAULT_WEIGHTS

    def _build_profile(row: pd.Series) -> dict:
        composite = _safe(row, "composite_score")
        position = str(row.get("position", "Unknown"))
        w_xg, w_form, w_val, w_exp = POSITION_WEIGHTS.get(position, DEFAULT_WEIGHTS)
        return {
            "player":   str(row.get("player",   "Unknown")),
            "team":     str(row.get("team",     "Unknown")),
            "position": position,
            "age":      int(_safe(row, "age", 0)),
            "caps":     int(_safe(row, "caps", 0)),
            "goals":    int(_safe(row, "goals", 0)),
            "club":     str(row.get("club", "")),
            "market_value_m": round(_safe(row, "market_value_m"), 1),
            "composite_score": round(composite, 3),
            # Position-aware weighted contributions — these now sum to the
            # composite score exactly, so the bars in the UI are honest.
            "breakdown": {
                "intl_xg_contribution":      round(_safe(row, "intl_xg_norm")    * w_xg,   3),
                "club_form_contribution":    round(_safe(row, "club_form_norm")  * w_form, 3),
                "market_value_contribution": round(_safe(row, "value_norm")      * w_val,  3),
                "experience_contribution":   round(_safe(row, "experience_norm") * w_exp,  3),
            },
            "weights": {
                "intl_xg": w_xg, "club_form": w_form,
                "market_value": w_val, "experience": w_exp,
            },
            "radar": _build_radar(row),
        }

    profile_a = _build_profile(row_a)
    profile_b = _build_profile(row_b)

    score_a = profile_a["composite_score"]
    score_b = profile_b["composite_score"]

    advantage   = "player_a" if score_a >= score_b else "player_b"
    score_delta = round(abs(score_a - score_b), 3)

    # Comparing across position groups is fine, but the verdict deserves
    # a caveat — an elite keeper and an elite winger aren't on one axis.
    ATTACKING = {"Forward", "Midfielder"}
    cross_position = (
        (profile_a["position"] in ATTACKING) != (profile_b["position"] in ATTACKING)
    )

    return {
        "player_a":    profile_a,
        "player_b":    profile_b,
        "advantage":   advantage,
        "score_delta": score_delta,
        "cross_position": cross_position,
    }


# ---------------------------------------------------------------------------
# GET /api/player/team/{team_name}
# ---------------------------------------------------------------------------
@router.get("/team/{team_name}")
def get_team_players(team_name: str):
    """
    All players for a given team with their composite scores.
    Used by the Team page squad depth section and the Player page's team selector.
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
    """
    df = _load_player_scores()

    q_norm = _deaccent(q.strip()).lower()
    matches = df[df["player"].apply(lambda x: _deaccent(str(x)).lower()).str.contains(q_norm, regex=False)]
    matches = matches.head(10).where(pd.notnull(matches), None)

    cols = ["player", "team", "position"]
    available_cols = [c for c in cols if c in matches.columns]

    return {
        "query":   q,
        "results": matches[available_cols].to_dict(orient="records"),
    }
