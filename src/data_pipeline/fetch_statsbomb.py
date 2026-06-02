"""
fetch_statsbomb.py

Fetches xG and shot event data from StatsBomb open data via statsbombpy.

Why StatsBomb:
    Event-level shot data gives us xG per team per match — a far better
    signal than goals scored. A team creating 3.2 xG but scoring 1 goal
    is playing better than the scoreline shows. Our Poisson model uses
    xG as its input, not raw goals.

StatsBomb open data covers:
    - FIFA World Cup (all tournaments)
    - UEFA Euro
    - Copa América
    - La Liga, Champions League (club level — useful for player xG)

Output files:
    data/raw/statsbomb_match_xg.csv   — xG per team per match
    data/raw/statsbomb_player_xg.csv  — xG per player across matches
"""

import pandas as pd
from pathlib import Path
from loguru import logger
from statsbombpy import sb

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# StatsBomb competition IDs we care about
# Find all competitions via sb.competitions()
# We focus on World Cup (competition_id=43) and Euro (competition_id=55)
COMPETITIONS = [
    {"competition_id": 43,  "season_id": 106, "name": "World Cup 2022"},
    {"competition_id": 43,  "season_id": 3,   "name": "World Cup 2018"},
    {"competition_id": 55,  "season_id": 282, "name": "Euro 2024"},
    {"competition_id": 55,  "season_id": 43,  "name": "Euro 2020"},
    {"competition_id": 223, "season_id": 282, "name": "Copa America 2024"},
]


def get_available_competitions() -> pd.DataFrame:
    """
    Returns all competitions available in StatsBomb open data.
    Call this to discover competition_id and season_id values.
    """
    comps = sb.competitions()
    return comps


def fetch_shots_for_match(match_id: int) -> pd.DataFrame:
    """
    Fetches all shot events for a single match.

    Args:
        match_id: StatsBomb match ID

    Returns:
        DataFrame of shot events with xG values, or empty DataFrame if failed.

    StatsBomb shot events include:
        - location (x, y coordinates)
        - shot.statsbomb_xg (their xG model output)
        - shot.outcome.name (Goal, Saved, Off T, etc.)
        - player.name
        - team.name
        - minute
    """
    try:
        events = sb.events(match_id=match_id)
        shots = events[events["type"] == "Shot"].copy()

        if shots.empty:
            return pd.DataFrame()

        # Extract the columns we need
        # statsbombpy returns nested dicts as separate columns
        keep_cols = [
            "match_id", "team", "player", "minute",
            "shot_statsbomb_xg", "shot_outcome", "location",
        ]

        # Only keep columns that exist — statsbombpy column names vary slightly
        available = [c for c in keep_cols if c in shots.columns]
        shots = shots[available].copy()
        shots["match_id"] = match_id

        return shots

    except Exception as e:
        logger.warning(f"Failed to fetch shots for match {match_id}: {e}")
        return pd.DataFrame()


def fetch_match_xg(competition_id: int, season_id: int) -> pd.DataFrame:
    """
    Fetches aggregated xG per team per match for one competition/season.

    Args:
        competition_id: StatsBomb competition ID
        season_id: StatsBomb season ID

    Returns:
        DataFrame with columns [match_id, team, xg, goals, shots]
    """
    try:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
    except Exception as e:
        logger.error(f"Failed to fetch matches for competition {competition_id} season {season_id}: {e}")
        return pd.DataFrame()

    logger.info(f"Processing {len(matches)} matches...")

    all_shots = []
    for match_id in matches["match_id"]:
        shots = fetch_shots_for_match(match_id)
        if not shots.empty:
            all_shots.append(shots)

    if not all_shots:
        return pd.DataFrame()

    shots_df = pd.concat(all_shots, ignore_index=True)

    # Aggregate xG per team per match
    # shot_statsbomb_xg is StatsBomb's xG value per shot (0 to 1)
    agg = shots_df.groupby(["match_id", "team"]).agg(
        xg=("shot_statsbomb_xg", "sum"),
        shots=("shot_statsbomb_xg", "count"),
        goals=("shot_outcome", lambda x: (x == "Goal").sum()),
    ).reset_index()

    return agg


def fetch_all_statsbomb_xg() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches xG data across all configured competitions.

    Returns:
        match_xg_df: xG per team per match
        player_xg_df: xG per player aggregated across all matches
    """
    # First check what's actually available
    logger.info("Checking available StatsBomb competitions...")
    available = get_available_competitions()
    wc_available = available[available["competition_id"] == 43]
    logger.info(f"Available World Cup seasons:\n{wc_available[['competition_id','season_id','season_name']].to_string()}")

    all_match_xg = []
    all_shot_events = []

    for comp in COMPETITIONS:
        cid = comp["competition_id"]
        sid = comp["season_id"]
        name = comp["name"]

        # Verify this season exists in open data
        exists = ((available["competition_id"] == cid) &
                  (available["season_id"] == sid)).any()

        if not exists:
            logger.warning(f"{name} (competition={cid}, season={sid}) not in open data — skipping")
            continue

        logger.info(f"Fetching {name}...")
        match_xg = fetch_match_xg(cid, sid)

        if not match_xg.empty:
            match_xg["competition"] = name
            all_match_xg.append(match_xg)
            logger.info(f"  {name}: {len(match_xg)} team-match records")

    if not all_match_xg:
        logger.error("No xG data retrieved from any competition")
        return pd.DataFrame(), pd.DataFrame()

    match_xg_df = pd.concat(all_match_xg, ignore_index=True)

    # Save
    match_xg_path = RAW_DIR / "statsbomb_match_xg.csv"
    match_xg_df.to_csv(match_xg_path, index=False)
    logger.info(f"Saved match xG to {match_xg_path}")

    return match_xg_df


if __name__ == "__main__":
    match_xg = fetch_all_statsbomb_xg()
    if not match_xg.empty:
        print(f"\nMatch xG records: {len(match_xg)}")
        print(match_xg.head(10).to_string(index=False))
        print(f"\nAverage xG per team per match: {match_xg['xg'].mean():.3f}")