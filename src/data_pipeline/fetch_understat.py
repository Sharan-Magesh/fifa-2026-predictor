# src/data_pipeline/fetch_understat.py

import asyncio
import os
import pandas as pd
import aiohttp
from understat import Understat

# Understat uses lowercase league names as URL slugs
# epl, la_liga, bundesliga, serie_a, ligue_1, rfpl
LEAGUES = [
    "epl",
    "la_liga",
    "bundesliga",
    "serie_a",
    "ligue_1",
    "rfpl",
]

# Year the season STARTED — 2024 = 2024/25 season
SEASON = 2024

OUTPUT_PATH = os.path.join("data", "processed", "understat_players.csv")


async def fetch_league_players(session: aiohttp.ClientSession, league: str, season: int) -> list:
    """
    Fetch all player stats for one league/season.

    Understat() takes the aiohttp session as its only argument.
    It is NOT an async context manager — do not use 'async with Understat()'.
    Just instantiate it and call methods directly.
    """
    understat = Understat(session)
    players = await understat.get_league_players(league, season)
    for p in players:
        p["league"] = league
        p["season"] = season
    return players


async def fetch_all_leagues() -> pd.DataFrame:
    """
    Fire all 6 league requests concurrently inside a single aiohttp session.
    aiohttp.ClientSession IS an async context manager — that's correct usage.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_league_players(session, league, SEASON)
            for league in LEAGUES
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_players = []
    for league, result in zip(LEAGUES, results):
        if isinstance(result, Exception):
            print(f"[fetch_understat] WARNING: Failed to fetch {league}: {result}")
            continue
        all_players.extend(result)

    return pd.DataFrame(all_players)


def clean_understat_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename, cast, and derive per-90 metrics.
    Field names confirmed from the understat package docs.
    """
    keep_cols = {
        "player_name": "player_name",
        "team_title": "club",
        "league": "league",
        "season": "season",
        "games": "games_played",
        "time": "minutes_played",
        "goals": "goals",
        "xG": "xg",
        "assists": "assists",
        "xA": "xa",
        "shots": "shots",
        "key_passes": "key_passes",
        "yellow_cards": "yellow_cards",
        "red_cards": "red_cards",
        "npg": "non_penalty_goals",
        "npxG": "npxg",
    }

    existing = {k: v for k, v in keep_cols.items() if k in df.columns}
    df = df[list(existing.keys())].rename(columns=existing)

    numeric_cols = [
        "games_played", "minutes_played", "goals", "xg",
        "assists", "xa", "shots", "key_passes",
        "yellow_cards", "red_cards", "non_penalty_goals", "npxg"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["minutes_played"] > 0].reset_index(drop=True)

    df["xg_per90"] = (df["xg"] / df["minutes_played"] * 90).round(3)
    df["xa_per90"] = (df["xa"] / df["minutes_played"] * 90).round(3)
    df["npxg_per90"] = (df["npxg"] / df["minutes_played"] * 90).round(3)

    return df


def run():
    print("[fetch_understat] Starting player xG fetch...")

    df = asyncio.run(fetch_all_leagues())

    if df.empty:
        print("[fetch_understat] ERROR: No data returned.")
        return

    print(f"[fetch_understat] Raw rows fetched: {len(df)}")

    df = clean_understat_df(df)

    # Keep highest-minutes record for players appearing in multiple leagues (loans)
    df = df.sort_values("minutes_played", ascending=False)
    df = df.drop_duplicates(subset=["player_name"], keep="first").reset_index(drop=True)

    print(f"[fetch_understat] Clean rows after dedup: {len(df)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[fetch_understat] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()