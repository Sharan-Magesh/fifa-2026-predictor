"""
Simulation Router
==================
GET /api/simulation/run

Runs ONE live tournament simulation (group stage -> R32 -> R16 -> QF -> SF -> Final)
and returns the full bracket trace so the frontend can animate it round-by-round.

This is intentionally separate from the offline 100k-run Monte Carlo (which produces
the aggregate win-probability table on the homepage). This endpoint produces a single,
concrete "what-if" bracket — a different random outcome every time it's called.

Performance note: building the prediction cache (~1,128 unique matchups, each requiring
a model inference) is the expensive part. We build it ONCE on first request and cache it
in memory for the lifetime of the server process — subsequent calls just sample a fresh
random outcome from the cached probabilities, which is near-instant.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

router = APIRouter()

# Module-level cache — loaded from disk (or built lazily) on first request,
# reused after that for the lifetime of the server process.
_PREDICTION_CACHE = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDICTION_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "prediction_cache.pkl"


def _get_cache():
    global _PREDICTION_CACHE
    if _PREDICTION_CACHE is None:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        # Fast path: a pre-built cache (~1,128 matchups) is committed to
        # data/processed/prediction_cache.pkl so the first live simulation
        # request doesn't have to wait ~40s for model inference.
        if PREDICTION_CACHE_PATH.exists():
            import pickle
            with open(PREDICTION_CACHE_PATH, "rb") as f:
                _PREDICTION_CACHE = pickle.load(f)
        else:
            from src.models.match_outcome import predict
            from src.simulation.monte_carlo import _build_prediction_cache
            from src.simulation.bracket import GROUPS

            _PREDICTION_CACHE = _build_prediction_cache(GROUPS, predict)

    return _PREDICTION_CACHE


# ---------------------------------------------------------------------------
# GET /api/simulation/run
# ---------------------------------------------------------------------------
@router.get("/run")
def run_bracket_simulation():
    """
    Run one full live simulation of the WC 2026 bracket.

    Response shape:
    {
        "groups": {
            "A": {
                "standings": [
                    {"team": "...", "pts": 9, "gd": 5, "gf": 7, "ga": 2,
                     "w": 3, "d": 0, "l": 0, "position": 1, "group": "A"},
                    ...
                ]
            },
            ...
        },
        "best_thirds": ["Japan", "Croatia", ...],
        "rounds": [
            {
                "name": "Round of 32",
                "key": "round_of_32",
                "matches": [
                    {"team_a": "Spain", "team_b": "Japan", "winner": "Spain"},
                    ...
                ]
            },
            ...
        ],
        "champion": "Brazil"
    }
    """
    try:
        from src.simulation.monte_carlo import run_traced_simulation
        from src.simulation.bracket import GROUPS

        cache = _get_cache()
        trace = run_traced_simulation(GROUPS, cache)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")

    groups_out = {}
    for group_name, rows in trace["standings"].items():
        groups_out[group_name] = {"standings": rows}

    ROUND_LABELS = {
        "round_of_32":  "Round of 32",
        "round_of_16":  "Round of 16",
        "quarterfinal": "Quarterfinals",
        "semifinal":    "Semifinals",
        "final":        "Final",
    }
    ROUND_ORDER = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]

    rounds_out = []
    for key in ROUND_ORDER:
        round_data = trace["rounds"][key]
        matchups = round_data["matchups"]
        winners = round_data["winners"]
        matches = [
            {"team_a": a, "team_b": b, "winner": w}
            for (a, b), w in zip(matchups, winners)
        ]
        rounds_out.append({
            "name": ROUND_LABELS[key],
            "key": key,
            "matches": matches,
        })

    return {
        "groups": groups_out,
        "group_winners": trace["group_winners"],
        "group_runners": trace["group_runners"],
        "best_thirds": trace["best_thirds"],
        "rounds": rounds_out,
        "champion": trace["champion"],
    }


# ---------------------------------------------------------------------------
# GET /api/simulation/status
# ---------------------------------------------------------------------------
@router.get("/status")
def simulation_status():
    """Lets the frontend check whether the prediction cache is already warm,
    so it can show a more accurate 'first run may take longer' message."""
    return {"cache_ready": _PREDICTION_CACHE is not None}
