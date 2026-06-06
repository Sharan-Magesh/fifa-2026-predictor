# src/data_pipeline/fetch_fbref.py
#
# Fetches player standard stats from FBref via soccerdata for leagues
# NOT covered by Understat (which covers only the Big 5 + RFPL).
#
# Why FBref on top of Understat:
#   - MLS: 44 WC 2026 players (USA, Canada, Argentina captain, etc.)
#   - Primeira Liga: Gyökeres, Pedro Goncalves
#   - Eredivisie: Dutch players at home clubs
#   - Brasileirao: Brazil-based players
#   - Argentine Primera: Argentina/Uruguay-based players
#   - Scottish Premiership: Scotland players
#   - Belgian Pro League: Belgium players
#   - Turkish Super Lig: Turkey players
#   - Liga MX: Mexico players
#   - J1 League: Japan-based players
#
# Metric used: (G-PK)/90 — non-penalty goals per 90 minutes
# This is a proxy for npxg/90 when xG data isn't available.
# Less precise than Understat's npxG but covers ~95% of WC 2026 squads.
#
# In player_features.py:
#   Priority 1 — Understat npxg/90 (xG-based, more precise)
#   Priority 2 — FBref (G-PK)/90 (goal-based proxy, broader coverage)
#   Priority 3 — 0.0 (player has no club data in either source)

import os
import time
import pandas as pd
import soccerdata as sd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "fbref_players.csv"

# Leagues to fetch from FBref
# These are leagues NOT in Understat (epl, la_liga, bundesliga, serie_a, ligue_1, rfpl)
# Season 2025 = 2025/26 season in soccerdata convention
FBREF_LEAGUES = [
    "NLD-Eredivisie",
    "PRT-Primeira Liga",
    "BRA-Brasileirao",
    "ARG-Liga Profesional",
    "USA-MLS",
    "SCO-Premiership",
    "BEL-Pro League",
    "TUR-Super Lig",
    "MEX-Liga MX",
    "JPN-J1 League",
]

SEASON = 2025

# Minimum 90s played to be considered active
# 5 full games = reliable enough per-90 metrics
MIN_90S = 5.0


def fetch_league(league: str, season: int) -> pd.DataFrame:
    """
    Fetch standard stats for one league.
    Returns empty DataFrame on failure — don't crash the whole pipeline
    if one league fails.
    """
    try:
        fbref = sd.FBref(league, season)
        df = fbref.read_player_season_stats(stat_type="standard")

        if df.empty:
            print(f"  [fetch_fbref] WARNING: No data for {league}")
            return pd.DataFrame()

        # Reset MultiIndex — soccerdata returns (league, season, team, player)
        df = df.reset_index()

        # Rename columns — soccerdata uses MultiIndex columns for stats
        # Flatten them to single level
        df.columns = [
            "_".join(filter(None, [str(a), str(b)])).strip("_")
            if isinstance(df.columns, pd.MultiIndex) or b
            else a
            for a, b in df.columns
        ] if isinstance(df.columns[0], tuple) else df.columns.tolist()

        print(f"  [fetch_fbref] {league}: {len(df)} players, cols: {df.columns.tolist()[:8]}")
        return df

    except Exception as e:
        print(f"  [fetch_fbref] WARNING: Failed {league}: {e}")
        return pd.DataFrame()


def extract_key_columns(df: pd.DataFrame, league: str) -> pd.DataFrame:
    """
    Extract and standardise the columns we need.
    FBref MultiIndex columns are flattened — we pick what we want.

    Key columns:
        player      — player name
        team        — club
        nation      — nationality
        pos         — position
        age         — age
        90s         — 90 minutes played (playing time)
        Gls         — goals (Standard_Gls after flatten)
        PK          — penalty kicks scored
        G-PK/90     — non-penalty goals per 90 (our proxy for npxg/90)
        Ast         — assists
    """
    if df.empty:
        return pd.DataFrame()

    # Find the right column names after flattening
    # soccerdata returns columns like ('Performance', 'Gls') or ('', 'player')
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower == "player":
            col_map["player_name"] = col
        elif col_lower == "team":
            col_map["club"] = col
        elif col_lower == "nation" or col_lower == "('nation', '')":
            col_map["nation"] = col
        elif col_lower == "pos" or col_lower == "('pos', '')":
            col_map["position"] = col
        elif col_lower == "age" or col_lower == "('age', '')":
            col_map["age"] = col
        elif "90s" in col_lower and "per" not in col_lower:
            col_map["nineties"] = col
        elif col_lower in ["performance_gls", "gls", "('performance', 'gls')"]:
            col_map["goals"] = col
        elif col_lower in ["performance_pk", "pk", "('performance', 'pk')"]:
            col_map["penalty_goals"] = col
        elif col_lower in ["performance_ast", "ast", "('performance', 'ast')"]:
            col_map["assists"] = col
        elif col_lower in ["per_90_minutes_g-pk", "g-pk", "('per 90 minutes', 'g-pk')"]:
            col_map["npg_per90"] = col

    rows = []
    for _, row in df.iterrows():
        player_name = str(row.get(col_map.get("player_name", ""), "")).strip()
        if not player_name or player_name == "nan":
            continue

        nineties = pd.to_numeric(row.get(col_map.get("nineties", ""), 0), errors="coerce") or 0.0

        if nineties < MIN_90S:
            continue

        goals = pd.to_numeric(row.get(col_map.get("goals", ""), 0), errors="coerce") or 0.0
        pens = pd.to_numeric(row.get(col_map.get("penalty_goals", ""), 0), errors="coerce") or 0.0
        assists = pd.to_numeric(row.get(col_map.get("assists", ""), 0), errors="coerce") or 0.0

        np_goals = goals - pens
        npg_per90 = round(np_goals / nineties, 3) if nineties > 0 else 0.0

        rows.append({
            "player_name":   player_name,
            "club":          str(row.get(col_map.get("club", ""), "")).strip(),
            "league":        league,
            "position":      str(row.get(col_map.get("position", ""), "")).strip(),
            "age":           pd.to_numeric(row.get(col_map.get("age", ""), None), errors="coerce"),
            "minutes_90s":   nineties,
            "goals":         goals,
            "penalty_goals": pens,
            "assists":       assists,
            "npg_per90":     npg_per90,  # proxy for npxg/90
        })

    return pd.DataFrame(rows)


def run():
    print("[fetch_fbref] Fetching player stats from FBref for non-Understat leagues...")
    print(f"[fetch_fbref] Leagues: {FBREF_LEAGUES}")
    print(f"[fetch_fbref] Season: {SEASON} (= 2025/26)")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_dfs = []

    for league in FBREF_LEAGUES:
        print(f"\n[fetch_fbref] Fetching {league}...")
        df_raw = fetch_league(league, SEASON)

        if df_raw.empty:
            continue

        df_clean = extract_key_columns(df_raw, league)

        if not df_clean.empty:
            all_dfs.append(df_clean)
            print(f"  [fetch_fbref] {league}: {len(df_clean)} active players (>= {MIN_90S} 90s)")
        else:
            print(f"  [fetch_fbref] {league}: 0 players after cleaning")

        # Rate limit — FBref will block aggressive scrapers
        time.sleep(2)

    if not all_dfs:
        print("[fetch_fbref] ERROR: No data fetched from any league.")
        return

    df_final = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate — player on loan may appear in two leagues
    # Keep highest minutes record
    df_final = df_final.sort_values("minutes_90s", ascending=False)
    df_final = df_final.drop_duplicates(subset=["player_name"], keep="first").reset_index(drop=True)

    print(f"\n[fetch_fbref] Total unique players: {len(df_final)}")
    print(f"[fetch_fbref] League breakdown:\n{df_final['league'].value_counts().to_string()}")

    print(f"\n[fetch_fbref] Top 10 by npg_per90:")
    print(df_final.nlargest(10, "npg_per90")[
        ["player_name", "club", "league", "minutes_90s", "goals", "npg_per90"]
    ].to_string())

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[fetch_fbref] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()