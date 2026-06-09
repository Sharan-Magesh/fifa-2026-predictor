"""
Homepage Router
===============
GET /api/homepage/teams

Returns all 48 WC 2026 teams with their pre-computed Monte Carlo probabilities.
Data source: data/processed/monte_carlo_results.csv
Sorted by p_win_tournament descending so the frontend can render a ranked list immediately.

Also exposes the group breakdown so the frontend can optionally show a group view.
"""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()

# ---------------------------------------------------------------------------
# Path resolution — works regardless of where uvicorn is launched from
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # api/routes -> api -> src -> project root
MONTE_CARLO_PATH = PROJECT_ROOT / "data" / "processed" / "monte_carlo_results.csv"
SQUADS_PATH      = PROJECT_ROOT / "data" / "processed" / "wc2026_squads.csv"


def _load_monte_carlo() -> pd.DataFrame:
    if not MONTE_CARLO_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Monte Carlo results not found at {MONTE_CARLO_PATH}. Run run_pipeline.py first.",
        )
    return pd.read_csv(MONTE_CARLO_PATH)


def _load_squads() -> pd.DataFrame:
    if not SQUADS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Squad file not found at {SQUADS_PATH}.",
        )
    return pd.read_csv(SQUADS_PATH)


# ---------------------------------------------------------------------------
# GET /api/homepage/teams
# ---------------------------------------------------------------------------
@router.get("/teams")
def get_all_teams():
    """
    Returns a ranked list of all 48 WC 2026 teams with tournament probabilities.

    Response shape:
    {
        "teams": [
            {
                "team": "Brazil",
                "group": "A",
                "p_win_tournament": 0.142,
                "p_final": 0.261,
                "p_advance_sf": 0.421,
                "p_advance_qf": 0.603,
                "squad_size": 26
            },
            ...
        ]
    }
    """
    mc_df     = _load_monte_carlo()
    squad_df  = _load_squads()

    # Count squad size per team
    squad_counts = (
        squad_df.groupby("team")["player_name"].count().reset_index()
        .rename(columns={"player_name": "squad_size"})
    )

    # Merge squad size into MC results
    merged = mc_df.merge(squad_counts, on="team", how="left")

    # Round probabilities to 3 decimal places — enough precision, clean JSON
    prob_cols = ["p_win_tournament", "p_final", "p_advance_sf", "p_advance_qf"]
    for col in prob_cols:
        if col in merged.columns:
            merged[col] = merged[col].round(3)

    # Sort by win probability descending
    merged = merged.sort_values("p_win_tournament", ascending=False)

    # Replace NaN with None so JSON serialisation doesn't blow up
    merged = merged.where(pd.notnull(merged), None)

    return {"teams": merged.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# GET /api/homepage/groups
# ---------------------------------------------------------------------------
@router.get("/groups")
def get_groups():
    """
    Returns teams organised by group — for the group-stage view on the homepage.

    Response shape:
    {
        "groups": {
            "A": [{"team": "...", "p_win_tournament": 0.12, ...}, ...],
            "B": [...],
            ...
        }
    }
    """
    mc_df = _load_monte_carlo()

    if "group" not in mc_df.columns:
        raise HTTPException(
            status_code=500,
            detail="'group' column missing from monte_carlo_results.csv. Check bracket.py output.",
        )

    prob_cols = ["p_win_tournament", "p_final", "p_advance_sf", "p_advance_qf"]
    for col in prob_cols:
        if col in mc_df.columns:
            mc_df[col] = mc_df[col].round(3)

    mc_df = mc_df.where(pd.notnull(mc_df), None)

    groups: dict = {}
    for group_name, group_df in mc_df.groupby("group"):
        group_df_sorted = group_df.sort_values("p_win_tournament", ascending=False)
        groups[str(group_name)] = group_df_sorted.to_dict(orient="records")

    return {"groups": groups}
