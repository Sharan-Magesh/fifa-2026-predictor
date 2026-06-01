"""
fetch_matches.py

Fetches full international match history from martj42/international_results on GitHub.
Dataset: 45,000+ matches from 1872 to present, updated regularly.

This data serves two purposes:
    1. Calculate Elo ratings for all national teams (in team_features.py)
    2. Provide historical match outcomes for training the match outcome model

Columns we care about:
    date, home_team, away_team, home_score, away_score, tournament, neutral
"""

import requests
import pandas as pd
from pathlib import Path
from loguru import logger

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Tournaments to exclude from Elo calculation — results that don't reflect
# true competitive strength (walkovers, low-stakes friendlies distort ratings)
EXCLUDE_TOURNAMENTS = [
    "Friendly",
]

# Tournaments to keep — all competitive international football
# We weight World Cup qualifiers and finals more heavily in Elo later
COMPETITIVE_TOURNAMENTS = [
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "Copa América",
    "Africa Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Gold Cup",
    "AFC Asian Cup qualification",
    "Africa Cup of Nations qualification",
    "CONCACAF Gold Cup qualification",
    "Copa América qualification",
    "Confederations Cup",
    "Nations League",
]


def fetch_international_results() -> pd.DataFrame:
    """
    Downloads the full international results CSV from GitHub.

    Returns:
        DataFrame with all matches, cleaned and typed correctly.
    """
    logger.info(f"Fetching international results from GitHub...")

    try:
        response = requests.get(
            RESULTS_URL,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch match results: {e}")
        raise

    df = pd.read_csv(
        pd.io.common.StringIO(response.text),
        parse_dates=["date"],
    )

    logger.info(f"Downloaded {len(df)} matches ({df['date'].min().year}–{df['date'].max().year})")

    # Cast score columns to nullable integer — some matches have missing scores
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # Boolean column arrives as string TRUE/FALSE
    df["neutral"] = df["neutral"].map({"TRUE": True, "FALSE": False})

    # Drop rows with no score — can't use them for Elo or model training
    df = df.dropna(subset=["home_score", "away_score"])

    logger.info(f"After cleaning: {len(df)} matches with valid scores")

    # Save full dataset
    output_path = RAW_DIR / "international_results.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    return df


def get_competitive_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters to competitive matches only, excluding friendlies.

    Why exclude friendlies:
        Teams rotate squads, try new formations, and don't play at full
        intensity in friendlies. Including them adds noise to Elo calculation
        and weakens the signal from competitive results.

    Args:
        df: Full results DataFrame from fetch_international_results()

    Returns:
        Filtered DataFrame with only competitive matches.
    """
    competitive = df[df["tournament"] != "Friendly"].copy()
    logger.info(f"Competitive matches (non-friendly): {len(competitive)}")
    return competitive


def get_world_cup_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters to World Cup final tournament matches only.
    Used for model training — these are the highest-stakes, most reliable results.

    Args:
        df: Full results DataFrame

    Returns:
        DataFrame with only FIFA World Cup final matches (not qualifiers)
    """
    wc = df[df["tournament"] == "FIFA World Cup"].copy()
    logger.info(f"World Cup final matches: {len(wc)}")
    return wc


if __name__ == "__main__":
    df = fetch_international_results()

    print("\n--- Dataset Overview ---")
    print(f"Total matches: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Unique teams: {df['home_team'].nunique()}")
    print(f"\nTop tournaments by match count:")
    print(df["tournament"].value_counts().head(10).to_string())

    competitive = get_competitive_matches(df)
    print(f"\nCompetitive matches: {len(competitive)}")

    wc = get_world_cup_matches(df)
    print(f"World Cup final matches: {len(wc)}")