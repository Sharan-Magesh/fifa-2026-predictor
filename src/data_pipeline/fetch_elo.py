"""
fetch_elo.py

Loads pre-tournament Elo ratings for all 48 WC 2026 teams from the
Kaggle dataset sourced from eloratings.net (CC BY-SA 4.0).

Dataset: afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings
Download: kaggle datasets download -d afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings -p data/raw --unzip

Why use eloratings.net instead of calculating our own:
    - eloratings.net is the gold standard — used by FiveThirtyEight, academic papers
    - Their ratings go back to 1872, giving stable long-run values
    - We calculate Elo MOMENTUM ourselves (change over last N matches)
      because that captures recent form, which the snapshot dataset doesn't

Elo formula (for reference — used in momentum calculation):
    E_a = 1 / (1 + 10^((R_b - R_a) / 400))
    R_a' = R_a + K * (S_a - E_a)
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Downloaded via: kaggle datasets download -d afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings
KAGGLE_ELO_PATH = RAW_DIR / "elo_ratings_wc2026.csv"
RESULTS_PATH = PROCESSED_DIR / "international_results.csv"

# Tournament starts June 11 2026 — use latest snapshot before that date
TOURNAMENT_START = "2026-06-11"

# Maps eloratings.net country names to our pipeline's standardised names
# Must match wc2026_groups.csv exactly
ELO_NAME_MAP = {
    "United States": "United States",
    "South Korea": "South Korea",
    "Ivory Coast": "Côte d'Ivoire",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Curaçao": "Curacao",
    "Cabo Verde": "Cape Verde",
    "DR Congo": "DR Congo",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
}

# K-factors for Elo momentum calculation
# Higher K = more weight on recent results
K_FACTORS = {
    "world_cup":        60,
    "continental_final": 50,
    "world_cup_qual":   40,
    "nations_league":   40,
    "continental_qual": 30,
    "friendly":         20,
    "other":            25,
}
DEFAULT_K = 30
DEFAULT_ELO = 1000.0

# How many recent matches to use for momentum calculation
MOMENTUM_WINDOW = 10


def load_pre_tournament_elo() -> pd.DataFrame:
    """
    Load the most recent Elo snapshot before the tournament start date.
    Uses eloratings.net data from the Kaggle dataset.
    """
    if not KAGGLE_ELO_PATH.exists():
        print(f"[fetch_elo] ERROR: {KAGGLE_ELO_PATH} not found.")
        print("Run: kaggle datasets download -d afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings -p data/raw --unzip")
        return pd.DataFrame()

    df = pd.read_csv(KAGGLE_ELO_PATH, parse_dates=["snapshot_date"])

    # Filter to snapshots before tournament start
    pre = df[df["snapshot_date"] < TOURNAMENT_START].copy()

    # Take most recent snapshot per team
    latest = (
        pre.sort_values("snapshot_date", ascending=False)
        .drop_duplicates(subset=["country"], keep="first")
        .copy()
    )

    print(f"[fetch_elo] Loaded {len(latest)} teams from eloratings.net snapshot ({latest['snapshot_date'].max().date()})")

    # Rename and normalise
    latest = latest.rename(columns={
        "country": "team",
        "rating":  "elo",
        "rank":    "elo_rank",
    })

    latest["team"] = latest["team"].replace(ELO_NAME_MAP)

    # Keep useful columns
    keep = ["team", "elo", "elo_rank", "confederation", "is_host",
            "matches_total", "wins", "losses", "draws",
            "goals_for", "goals_against"]
    keep = [c for c in keep if c in latest.columns]
    latest = latest[keep].reset_index(drop=True)

    return latest


def calculate_elo_momentum(df_results: pd.DataFrame, teams: list) -> pd.DataFrame:
    """
    Calculate Elo momentum for each team — change in Elo over last N matches.

    Why momentum matters:
        A team rated 1900 that won their last 8 matches is more dangerous
        than a team rated 1900 that lost their last 5. The absolute rating
        doesn't capture this — momentum does.

    Method:
        1. Run full Elo simulation on historical results
        2. For each team, compare their Elo after match N vs match N-MOMENTUM_WINDOW
        3. The difference is their momentum score
    """
    if df_results.empty:
        return pd.DataFrame()

    # Name map for martj42 dataset
    NAME_MAP = {
        "IR Iran": "Iran",
        "Korea Republic": "South Korea",
        "USA": "United States",
        "Côte d'Ivoire": "Côte d'Ivoire",
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Türkiye": "Turkey",
        "Czechia": "Czech Republic",
        "Curaçao": "Curacao",
        "Cabo Verde": "Cape Verde",
    }

    df = df_results.copy().sort_values("date").reset_index(drop=True)
    df["home_team"] = df["home_team"].replace(NAME_MAP)
    df["away_team"] = df["away_team"].replace(NAME_MAP)

    # Track rating history per team: {team: [rating_after_each_match]}
    ratings = {}
    history = {team: [] for team in teams}

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        tournament_type = row.get("tournament_type", "other")

        if home not in ratings:
            ratings[home] = DEFAULT_ELO
        if away not in ratings:
            ratings[away] = DEFAULT_ELO

        r_home = ratings[home]
        r_away = ratings[away]

        e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))
        e_away = 1.0 - e_home

        home_goals = row["home_score"]
        away_goals = row["away_score"]

        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals < away_goals:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        k = K_FACTORS.get(tournament_type, DEFAULT_K)

        ratings[home] = r_home + k * (s_home - e_home)
        ratings[away] = r_away + k * (s_away - e_away)

        # Record history for qualified teams only
        if home in history:
            history[home].append(ratings[home])
        if away in history:
            history[away].append(ratings[away])

    # Compute momentum: rating after last match minus rating MOMENTUM_WINDOW matches ago
    momentum_rows = []
    for team in teams:
        h = history[team]
        if len(h) >= MOMENTUM_WINDOW:
            momentum = h[-1] - h[-MOMENTUM_WINDOW]
        elif len(h) > 1:
            momentum = h[-1] - h[0]
        else:
            momentum = 0.0

        momentum_rows.append({
            "team": team,
            "elo_momentum": round(momentum, 1),
            "matches_in_data": len(h),
        })

    return pd.DataFrame(momentum_rows)


def run():
    print("[fetch_elo] Loading pre-tournament Elo ratings...")

    # --- Part 1: Absolute Elo from eloratings.net ---
    df_elo = load_pre_tournament_elo()
    if df_elo.empty:
        return

    # --- Part 2: Elo momentum from our historical results ---
    print("[fetch_elo] Calculating Elo momentum from match history...")

    if not RESULTS_PATH.exists():
        print(f"[fetch_elo] WARNING: {RESULTS_PATH} not found — skipping momentum calculation")
        df_final = df_elo
    else:
        df_results = pd.read_csv(RESULTS_PATH, parse_dates=["date"])
        teams = df_elo["team"].tolist()
        df_momentum = calculate_elo_momentum(df_results, teams)

        df_final = df_elo.merge(df_momentum, on="team", how="left")
        df_final["elo_momentum"] = df_final["elo_momentum"].fillna(0.0)

    # Sort by Elo descending
    df_final = df_final.sort_values("elo", ascending=False).reset_index(drop=True)

    output_path = PROCESSED_DIR / "elo_ratings.csv"
    df_final.to_csv(output_path, index=False)
    print(f"[fetch_elo] Saved {len(df_final)} teams to {output_path}")

    print(f"\n[fetch_elo] Top 10 by Elo:")
    print(df_final.head(10)[["team", "elo", "elo_rank", "elo_momentum"]].to_string())

    print(f"\n[fetch_elo] Top 5 by momentum (hottest teams right now):")
    print(df_final.nlargest(5, "elo_momentum")[["team", "elo", "elo_momentum"]].to_string())

    print(f"\n[fetch_elo] Bottom 5 by momentum (coldest teams):")
    print(df_final.nsmallest(5, "elo_momentum")[["team", "elo", "elo_momentum"]].to_string())


if __name__ == "__main__":
    run()