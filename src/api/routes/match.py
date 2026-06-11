"""
Match Router
============
GET /api/match/predict?team_a=Brazil&team_b=France

Calls match_outcome.py to return:
  - W/D/L probabilities (XGBoost)
  - Expected goals for each team (Bivariate Poisson)
  - Most likely scoreline and its probability
  - Penalty probability (derived from draw probability × knockout-stage flag)

The caller can pass ?knockout=true to signal this is a knockout match
(which affects how "draw" is presented — in knockout rounds there is no
draw; we redistribute that probability into a "goes to extra time / penalties"
bucket instead of suppressing it).
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SQUADS_PATH  = PROJECT_ROOT / "data" / "processed" / "wc2026_squads.csv"


def _get_valid_teams() -> set:
    """
    Return the set of 48 canonical team names for input validation.

    wc2026_squads.csv spells one team "Ivory Coast", but the canonical
    spelling used everywhere else (wc2026_groups.csv, wc2026_fixtures.csv,
    GROUPS, FIFA_RANKING_POINTS) is "Côte d'Ivoire" — see
    src/features/team_features.py SQUAD_NAME_ALIASES. Without remapping,
    the canonical name the frontend sends is rejected with a 400 here.
    """
    path = SQUADS_PATH
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    raw_teams = set(df["team"].unique())

    from src.features.team_features import SQUAD_NAME_ALIASES
    squad_to_canonical = {v: k for k, v in SQUAD_NAME_ALIASES.items()}

    return {squad_to_canonical.get(t, t) for t in raw_teams}


def _lazy_import_predict():
    """
    Lazily import predict and predict_score from match_outcome.py.
    Lazy import means the model is only loaded when the endpoint is hit,
    not at server startup — keeps startup time fast.
    """
    import sys
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.models.match_outcome import predict, predict_score
    return predict, predict_score


# ---------------------------------------------------------------------------
# GET /api/match/predict
# ---------------------------------------------------------------------------
@router.get("/predict")
def predict_match(
    team_a:   str           = Query(..., description="Home / first team name, e.g. 'Brazil'"),
    team_b:   str           = Query(..., description="Away / second team name, e.g. 'France'"),
    knockout: Optional[bool] = Query(False, description="Set true for knockout-stage matches"),
):
    """
    Predict the outcome of a match between team_a and team_b.

    Response shape:
    {
        "team_a": "Brazil",
        "team_b": "France",
        "outcome_probabilities": {
            "team_a_win": 0.38,
            "draw": 0.26,          # null if knockout=true
            "team_b_win": 0.36,
            "extra_time_or_pens":  # only present if knockout=true
        },
        "expected_goals": {
            "team_a_xg": 1.42,
            "team_b_xg": 1.31
        },
        "most_likely_scoreline": {
            "team_a_goals": 1,
            "team_b_goals": 1,
            "probability": 0.112
        },
        "penalty_probability": 0.18   # only meaningful if knockout=true
    }
    """
    valid_teams = _get_valid_teams()

    # Input validation — give the frontend a useful error message
    if valid_teams:
        if team_a not in valid_teams:
            raise HTTPException(
                status_code=400,
                detail=f"'{team_a}' not found. Check /api/match/teams for valid names.",
            )
        if team_b not in valid_teams:
            raise HTTPException(
                status_code=400,
                detail=f"'{team_b}' not found. Check /api/match/teams for valid names.",
            )

    if team_a == team_b:
        raise HTTPException(status_code=400, detail="team_a and team_b must be different.")

    try:
        predict, predict_score = _lazy_import_predict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # --- Outcome probabilities (XGBoost) ---
    try:
        outcome = predict(team_a, team_b)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict() failed: {e}")

    p_win  = round(float(outcome.get("win",  0)), 3)
    p_draw = round(float(outcome.get("draw", 0)), 3)
    p_loss = round(float(outcome.get("loss", 0)), 3)

    # --- Scoreline and xG (Bivariate Poisson) ---
    try:
        score_result = predict_score(team_a, team_b)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict_score() failed: {e}")

    xg_a       = round(float(score_result.get("xg_a", 0)), 2)
    xg_b       = round(float(score_result.get("xg_b", 0)), 2)
    # predict_score() returns the Poisson-derived likely scoreline under
    # likely_score_a/likely_score_b — read those instead of falling back
    # to a naive round(xg) every time.
    score_a    = int(score_result.get("likely_score_a", round(xg_a)))
    score_b    = int(score_result.get("likely_score_b", round(xg_b)))
    score_prob = round(float(score_result.get("score_prob", 0)), 3)

    # --- Knockout-mode draw redistribution ---
    # In knockout rounds there is no draw result. We present the draw probability
    # as "extra time / penalties" — i.e. the match is still level after 90 mins.
    # Penalty probability is estimated as 40% of that bucket (rough empirical base rate
    # for knockout matches that go to ET: ~40% end in pens rather than ET goals).
    PENALTY_BASE_RATE = 0.40

    if knockout:
        outcome_block = {
            "team_a_win":          p_win,
            "team_b_win":          p_loss,
            "extra_time_or_pens":  round(p_draw, 3),
            "draw":                None,
        }
        penalty_probability = round(p_draw * PENALTY_BASE_RATE, 3)
    else:
        outcome_block = {
            "team_a_win":         p_win,
            "draw":               p_draw,
            "team_b_win":         p_loss,
            "extra_time_or_pens": None,
        }
        penalty_probability = None

    return {
        "team_a": team_a,
        "team_b": team_b,
        "outcome_probabilities": outcome_block,
        "expected_goals": {
            "team_a_xg": xg_a,
            "team_b_xg": xg_b,
        },
        "most_likely_scoreline": {
            "team_a_goals": score_a,
            "team_b_goals": score_b,
            "probability":  score_prob,
        },
        "penalty_probability": penalty_probability,
    }


# ---------------------------------------------------------------------------
# GET /api/match/teams
# ---------------------------------------------------------------------------
@router.get("/teams")
def get_valid_teams():
    """
    Returns the list of valid team names. Used by the frontend for autocomplete.
    """
    valid_teams = sorted(_get_valid_teams())
    if not valid_teams:
        raise HTTPException(status_code=500, detail="Squad file not found.")
    return {"teams": valid_teams}
