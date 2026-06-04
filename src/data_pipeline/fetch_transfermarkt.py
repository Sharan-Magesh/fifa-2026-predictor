# src/data_pipeline/fetch_transfermarkt.py
#
# Instead of scraping Transfermarkt directly (blocked by Cloudflare in 2026),
# we use the dcaribou/transfermarkt-datasets repo — a maintained open-source
# pipeline that publishes clean CSVs from Transfermarkt data on GitHub.
#
# Repo: https://github.com/dcaribou/transfermarkt-datasets
# Data: published as gzipped CSVs on Cloudflare R2
# Coverage: players, valuations, clubs, international caps/goals, appearances
# Updated: automatically, tracks current season

import os
import io
import requests
import pandas as pd

# Confirmed R2 bucket URL — files are gzip compressed
BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"

# All files use .csv.gz extension
FILES = {
    "players":        "players.csv.gz",
    "valuations":     "player_valuations.csv.gz",
    "appearances":    "appearances.csv.gz",
    "clubs":          "clubs.csv.gz",
    "national_teams": "national_teams.csv.gz",
}

PROCESSED_DIR = os.path.join("data", "processed")


def download_csv(filename: str) -> pd.DataFrame:
    url = BASE_URL + filename
    print(f"  Downloading {filename}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    # gzip-compressed — pandas reads directly from bytes via BytesIO
    df = pd.read_csv(io.BytesIO(resp.content), compression="gzip")
    print(f"  {filename}: {len(df)} rows, cols: {list(df.columns[:8])}...")
    return df


def run():
    print("[fetch_transfermarkt] Downloading from dcaribou/transfermarkt-datasets...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    try:
        df_players    = download_csv(FILES["players"])
        df_valuations = download_csv(FILES["valuations"])
    except requests.RequestException as e:
        print(f"[fetch_transfermarkt] ERROR: Download failed: {e}")
        return

    # --- Build player profile table ---
    player_cols = {
        "player_id":                   "player_id",
        "name":                        "player_name",
        "position":                    "position",
        "sub_position":                "sub_position",
        "nationality":                 "nationality",
        "date_of_birth":               "date_of_birth",
        "current_club_id":             "current_club_id",
        "current_national_team_id":    "national_team_id",
        "international_caps":          "caps",
        "international_goals":         "international_goals",
        "market_value_in_eur":         "market_value_eur",
        "highest_market_value_in_eur": "peak_market_value_eur",
        "last_season":                 "last_season",
    }

    existing_cols = {k: v for k, v in player_cols.items() if k in df_players.columns}
    df_players = df_players[list(existing_cols.keys())].rename(columns=existing_cols)

    for col in ["market_value_eur", "peak_market_value_eur"]:
        if col in df_players.columns:
            df_players[col.replace("_eur", "_m")] = (
                pd.to_numeric(df_players[col], errors="coerce") / 1_000_000
            ).round(2)
            df_players.drop(columns=[col], inplace=True)

    # --- Latest valuation per player ---
    val_cols = [c for c in ["player_id", "date", "market_value_in_eur"] if c in df_valuations.columns]
    if val_cols:
        df_val = df_valuations[val_cols].copy()
        df_val["date"] = pd.to_datetime(df_val["date"], errors="coerce")
        df_val = df_val.sort_values("date", ascending=False).drop_duplicates(
            subset=["player_id"], keep="first"
        )
        df_val["latest_value_m"] = (
            pd.to_numeric(df_val["market_value_in_eur"], errors="coerce") / 1_000_000
        ).round(2)
        df_val = df_val[["player_id", "latest_value_m"]].rename(columns={"date": "valuation_date"})
        df_players = df_players.merge(df_val, on="player_id", how="left")

    # Filter to players with caps data only — reduces noise
    df_with_caps = df_players[df_players["caps"].notna() & (df_players["caps"] > 0)].copy()

    print(f"\n[fetch_transfermarkt] Total players: {len(df_players)}")
    print(f"[fetch_transfermarkt] Players with international caps: {len(df_with_caps)}")

    df_players.to_csv(os.path.join(PROCESSED_DIR, "transfermarkt_players.csv"), index=False)
    print(f"[fetch_transfermarkt] Saved transfermarkt_players.csv")

    # Sanity check — top 10 by caps
    print(f"\n[fetch_transfermarkt] Top 10 by caps:")
    available_cols = [c for c in ["player_name", "country_of_birth", "caps", "international_goals", "latest_value_m"] if c in df_with_caps.columns]
    print(df_with_caps.nlargest(10, "caps")[available_cols].to_string())

    print(f"\n[fetch_transfermarkt] Done.")


if __name__ == "__main__":
    run()