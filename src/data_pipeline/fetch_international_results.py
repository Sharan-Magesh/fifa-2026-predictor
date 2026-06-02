# src/data_pipeline/fetch_international_results.py

import os
import io
import requests
import pandas as pd

# Raw CSV served directly from GitHub — no API key, no scraping.
# martj42 maintains this actively; it was updated as recently as yesterday (June 2026).
# If this URL breaks, the Kaggle mirror is:
# https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
MARTJ42_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# 2004 cutoff — pre-2004 squads have no overlap with 2026 WC players.
# Keeping 20+ years of data gives us 5 full World Cup cycles (2006-2022)
# plus all qualification campaigns, friendlies, and continental tournaments.
CUTOFF_YEAR = 2004

OUTPUT_PATH = os.path.join("data", "processed", "international_results.csv")


def fetch_raw() -> pd.DataFrame:
    """
    Download the CSV directly into memory — no temp file needed.
    io.StringIO lets pandas read the response text as if it were a file object.
    """
    print(f"[fetch_international_results] Downloading from martj42 GitHub...")
    response = requests.get(MARTJ42_URL, timeout=30)
    response.raise_for_status()  # throws if 404 or 5xx

    df = pd.read_csv(io.StringIO(response.text))
    print(f"[fetch_international_results] Raw rows downloaded: {len(df)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse dates, apply 2004 cutoff, validate columns, add derived fields.
    """
    # Parse date — stored as YYYY-MM-DD string in the source
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows with unparseable dates (there are a few malformed entries)
    df = df.dropna(subset=["date"])

    # Apply cutoff
    df = df[df["date"].dt.year >= CUTOFF_YEAR].copy()
    print(f"[fetch_international_results] Rows after {CUTOFF_YEAR} cutoff: {len(df)}")

    # Validate expected columns exist
    expected = ["date", "home_team", "away_team", "home_score", "away_score",
                "tournament", "city", "country", "neutral"]
    missing = [c for c in expected if c in df.columns]
    # Note: we check presence, not absence — all should be there
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"[fetch_international_results] WARNING: Missing columns: {missing}")

    # Cast scores to int — occasional NaN from postponed/abandoned matches
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # neutral is stored as True/False string in some versions — normalise to bool
    df["neutral"] = df["neutral"].astype(str).str.lower().map(
        {"true": True, "false": False}
    ).fillna(False)

    # Derived columns used in feature engineering:

    # goal_diff from home team perspective
    df["goal_diff"] = df["home_score"] - df["away_score"]

    # result from home team perspective: W / D / L
    df["home_result"] = df["goal_diff"].apply(
        lambda x: "W" if x > 0 else ("D" if x == 0 else "L")
    )

    # total goals — used for over/under and xG calibration
    df["total_goals"] = df["home_score"] + df["away_score"]

    # year and month — used for recency weighting in features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # tournament type classification — World Cup matches weighted higher than friendlies
    df["tournament_type"] = df["tournament"].apply(classify_tournament)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def classify_tournament(tournament: str) -> str:
    """
    Classify each match into a tournament tier.
    Used in feature engineering to weight competitive matches more than friendlies.

    Tiers:
    - world_cup         : FIFA World Cup (highest weight)
    - world_cup_qual    : World Cup qualification
    - continental_final : Euros, Copa América, AFCON, Asian Cup, Gold Cup, etc.
    - continental_qual  : Qualification for the above
    - friendly          : International friendlies (lowest weight)
    - other             : Nations League, Confederations Cup, etc.
    """
    t = tournament.lower()

    if t == "fifa world cup":
        return "world_cup"

    if any(x in t for x in ["qualification", "qualifier", "qualifying"]):
        if "world cup" in t:
            return "world_cup_qual"
        else:
            return "continental_qual"

    if any(x in t for x in [
        "uefa euro",
        "copa américa", "copa america",
        "african cup of nations", "africa cup of nations", "afcon",
        "afc asian cup",
        "concacaf gold cup", "gold cup",
        "ofc nations cup",
        "confederations cup",       # FIFA major tournament
    ]):
        return "continental_final"

    # Nations League — competitive but below tournament level
    # Separate tier so feature engineering can weight appropriately
    if "nations league" in t:
        return "nations_league"

    if "friendly" in t:
        return "friendly"

    return "other"


def run():
    df = fetch_raw()
    df = clean(df)

    print(f"[fetch_international_results] Final rows: {len(df)}")
    print(f"[fetch_international_results] Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"[fetch_international_results] Tournament types:\n{df['tournament_type'].value_counts()}")
    print(f"[fetch_international_results] Unique teams: {pd.unique(df[['home_team','away_team']].values.ravel()).shape[0]}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[fetch_international_results] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()