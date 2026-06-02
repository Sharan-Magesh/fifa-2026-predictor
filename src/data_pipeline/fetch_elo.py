"""
fetch_elo.py

Calculates current Elo ratings for all national teams from historical
international match results.

Why we calculate Elo ourselves instead of scraping:
    - No external dependency that can break
    - We control the K-factor and weighting per tournament type
    - Better interview story: we understand every parameter

Elo formula:
    Expected score: E_a = 1 / (1 + 10^((R_b - R_a) / 400))
    New rating:     R_a' = R_a + K * (S_a - E_a)

    Where:
        R_a, R_b = current ratings of team A and B
        S_a      = actual result (1=win, 0.5=draw, 0=loss)
        K        = weight factor (higher for more important matches)
        400      = scaling constant (standard Elo)
"""

import pandas as pd
from pathlib import Path
from loguru import logger
from fetch_matches import fetch_international_results

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ELO = 1000.0

K_FACTORS = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 40,
    "Copa América": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "CONCACAF Gold Cup": 40,
    "UEFA Nations League": 40,
    "Friendly": 20,
}
DEFAULT_K = 30

# Actual 48 qualified teams by group — confirmed from official FIFA draw
# Source: FIFA draw December 5, 2025, Kennedy Center Washington D.C.
QUALIFIED_TEAMS = [
    # Group A
    "Mexico", "South Africa", "South Korea", "Czech Republic",
    # Group B
    "Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland",
    # Group C
    "Brazil", "Morocco", "Haiti", "Scotland",
    # Group D
    "United States", "Paraguay", "Australia", "Turkey",
    # Group E
    "Germany", "Curacao", "Ivory Coast", "Ecuador",
    # Group F
    "Netherlands", "Japan", "Sweden", "Tunisia",
    # Group G
    "Belgium", "Egypt", "Iran", "New Zealand",
    # Group H
    "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
    # Group I
    "France", "Senegal", "Iraq", "Norway",
    # Group J
    "Argentina", "Algeria", "Austria", "Jordan",
    # Group K
    "Portugal", "DR Congo", "Uzbekistan", "Colombia",
    # Group L
    "England", "Croatia", "Ghana", "Panama",
]

# Maps team names as they appear in the historical results CSV
# to our standard QUALIFIED_TEAMS names above.
# The martj42 dataset uses different naming conventions for some teams.
RESULTS_NAME_MAP = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Curaçao": "Curacao",
    "Cabo Verde": "Cape Verde",
}


def get_k_factor(tournament: str) -> float:
    for key, k in K_FACTORS.items():
        if key.lower() in tournament.lower():
            return k
    return DEFAULT_K


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def calculate_elo_ratings(df: pd.DataFrame) -> dict[str, float]:
    """
    Iterates through all matches chronologically and updates Elo ratings.

    Args:
        df: Full international results DataFrame with columns:
            date, home_team, away_team, home_score, away_score, tournament

    Returns:
        Dict mapping team name -> current Elo rating
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Normalize team names using the results name map
    df["home_team"] = df["home_team"].replace(RESULTS_NAME_MAP)
    df["away_team"] = df["away_team"].replace(RESULTS_NAME_MAP)

    ratings = {}

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_goals = row["home_score"]
        away_goals = row["away_score"]
        tournament = row["tournament"]

        if home not in ratings:
            ratings[home] = DEFAULT_ELO
        if away not in ratings:
            ratings[away] = DEFAULT_ELO

        r_home = ratings[home]
        r_away = ratings[away]

        e_home = expected_score(r_home, r_away)
        e_away = 1.0 - e_home

        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals < away_goals:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        k = get_k_factor(tournament)

        ratings[home] = r_home + k * (s_home - e_home)
        ratings[away] = r_away + k * (s_away - e_away)

    return ratings


def fetch_all_elo_ratings() -> pd.DataFrame:
    """
    Calculates Elo ratings from match history, filters to 48 qualified teams.

    Returns:
        DataFrame with columns [team, elo, group]
    """
    logger.info("Loading international match history...")
    df = fetch_international_results()

    logger.info("Calculating Elo ratings from match history...")
    ratings = calculate_elo_ratings(df)

    # Build full ratings DataFrame
    all_ratings = pd.DataFrame([
        {"team": team, "elo": round(elo, 1)}
        for team, elo in ratings.items()
    ])

    # Filter to qualified teams only
    qualified = all_ratings[all_ratings["team"].isin(QUALIFIED_TEAMS)].copy()

    # Add group column for reference
    group_map = {}
    groups = ["A","B","C","D","E","F","G","H","I","J","K","L"]
    for i, team in enumerate(QUALIFIED_TEAMS):
        group_map[team] = groups[i // 4]

    qualified["group"] = qualified["team"].map(group_map)
    qualified = qualified.sort_values("elo", ascending=False).reset_index(drop=True)

    # Log which qualified teams are missing from historical data
    missing = set(QUALIFIED_TEAMS) - set(qualified["team"].tolist())
    if missing:
        logger.warning(f"No historical data for: {missing}")
        logger.warning("These teams will need Elo assigned manually or via default")

    output_path = RAW_DIR / "elo_ratings.csv"
    qualified.to_csv(output_path, index=False)
    logger.info(f"Saved {len(qualified)}/48 team Elo ratings to {output_path}")

    return qualified


if __name__ == "__main__":
    ratings = fetch_all_elo_ratings()
    print(f"\nAll qualified teams by Elo ({len(ratings)}/48):")
    print(ratings.to_string(index=False))