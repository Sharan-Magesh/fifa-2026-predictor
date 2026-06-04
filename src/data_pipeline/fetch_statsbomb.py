# src/data_pipeline/fetch_statsbomb.py
# Replaces the previous version that only cloned the repo.
# Now extracts player-level shot/xG stats from international tournaments.

import os
import json
import pandas as pd
from pathlib import Path

# Path to the cloned StatsBomb open-data repo
# Assumes the repo was cloned into data/statsbomb/ as per Step 1 structure
STATSBOMB_PATH = Path("data") / "statsbomb" / "open-data" / "data"

# Target competitions — confirmed IDs from StatsBomb competitions.json
# Format: (competition_id, season_id, label)
TARGET_COMPETITIONS = [
    (43,  106, "World Cup 2022"),
    (43,    3, "World Cup 2018"),
    (55,  282, "Euro 2024"),
    (223, 282, "Copa America 2024"),
]

# AFCON 2023 — including because it covers Morocco, Senegal, Ivory Coast,
# Egypt, DR Congo, Cape Verde, Ghana who are all at WC 2026
# competition_id=6 season_id=281 — verify this against competitions.json
# if it fails we skip gracefully
AFCON_COMPETITION = (1267, 107, "AFCON 2023")

OUTPUT_PATH = os.path.join("data", "processed", "statsbomb_player_stats.csv")


def load_matches(competition_id: int, season_id: int) -> list[dict]:
    """
    Load the matches JSON for a competition/season.
    File path pattern: data/matches/{competition_id}/{season_id}.json
    """
    matches_path = STATSBOMB_PATH / "matches" / str(competition_id) / f"{season_id}.json"

    if not matches_path.exists():
        print(f"  [WARNING] Matches file not found: {matches_path}")
        return []

    with open(matches_path, encoding="utf-8") as f:
        return json.load(f)


def load_events_for_match(match_id: int) -> list[dict]:
    """
    Load the events JSON for a single match.
    File path pattern: data/events/{match_id}.json
    Each event has a 'type' field — we only want type.name == 'Shot'
    """
    events_path = STATSBOMB_PATH / "events" / f"{match_id}.json"

    if not events_path.exists():
        return []

    with open(events_path, encoding="utf-8") as f:
        events = json.load(f)

    # Filter to shots only — this is the critical early filter for performance
    # Shot events have: player.name, team.name, shot.statsbomb_xg, shot.outcome.name
    shots = [e for e in events if e.get("type", {}).get("name") == "Shot"]
    return shots


def extract_shot_rows(shots: list[dict], match_id: int, competition_label: str) -> list[dict]:
    """
    Flatten shot events into tabular rows.
    StatsBomb shot events have nested JSON — we extract only what we need.
    """
    rows = []
    for shot in shots:
        player = shot.get("player", {})
        team = shot.get("team", {})
        shot_data = shot.get("shot", {})

        # shot.outcome.name can be: "Goal", "Saved", "Off T", "Post",
        # "Blocked", "Wayward", "Saved Off Target", "Saved To Post"
        outcome = shot_data.get("outcome", {}).get("name", "")
        xg = shot_data.get("statsbomb_xg", None)

        rows.append({
            "match_id": match_id,
            "competition": competition_label,
            "player_name": player.get("name", ""),
            "player_id": player.get("id", None),
            "team": team.get("name", ""),
            "xg": xg,
            "is_goal": 1 if outcome == "Goal" else 0,
            "outcome": outcome,
            # shot type — header, open play, free kick, penalty
            "shot_type": shot_data.get("type", {}).get("name", ""),
        })
    return rows


def fetch_competition(competition_id: int, season_id: int, label: str) -> pd.DataFrame:
    """
    Load all matches for a competition, then load shots for each match.
    Returns a DataFrame of all shot events across the competition.
    """
    print(f"  Processing: {label} (competition={competition_id}, season={season_id})")
    matches = load_matches(competition_id, season_id)

    if not matches:
        print(f"  Skipping {label} — no matches found")
        return pd.DataFrame()

    print(f"  Matches found: {len(matches)}")

    all_rows = []
    for match in matches:
        match_id = match["match_id"]
        shots = load_events_for_match(match_id)
        rows = extract_shot_rows(shots, match_id, label)
        all_rows.extend(rows)

    print(f"  Total shot events: {len(all_rows)}")
    return pd.DataFrame(all_rows)


def aggregate_player_stats(df_shots: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate shot-level rows into one row per player per competition.
    Then further aggregate to one row per player across all competitions.

    Why aggregate twice:
    1. Per-competition: preserves tournament-level context
    2. Cross-competition: gives overall international quality score

    We keep both in the output — feature engineering decides which to use.
    """
    if df_shots.empty:
        return pd.DataFrame()

    # Exclude penalties from xG aggregation — same reason as Understat npxG
    # Penalty xG is always ~0.76 regardless of the player, so it's not informative
    df_no_pen = df_shots[df_shots["shot_type"] != "Penalty"].copy()

    # Per player per competition
    per_comp = df_no_pen.groupby(["player_id", "player_name", "team", "competition"]).agg(
        shots=("xg", "count"),
        xg=("xg", "sum"),
        goals=("is_goal", "sum"),
        xg_per_shot=("xg", "mean"),
    ).reset_index()

    # Penalty goals (counted separately, not in xG)
    pen_goals = df_shots[df_shots["shot_type"] == "Penalty"].groupby(
        ["player_id", "player_name", "team", "competition"]
    )["is_goal"].sum().reset_index().rename(columns={"is_goal": "penalty_goals"})

    per_comp = per_comp.merge(pen_goals, on=["player_id", "player_name", "team", "competition"], how="left")
    per_comp["penalty_goals"] = per_comp["penalty_goals"].fillna(0).astype(int)
    per_comp["non_penalty_goals"] = per_comp["goals"] - per_comp["penalty_goals"]

    # Round floats
    per_comp["xg"] = per_comp["xg"].round(3)
    per_comp["xg_per_shot"] = per_comp["xg_per_shot"].round(3)

    # Cross-competition aggregate — one row per player
    # Team = most recent team (last alphabetically by competition name is a proxy)
    # This is imperfect but players rarely change national teams
    overall = df_no_pen.groupby(["player_id", "player_name", "team"]).agg(
        total_shots=("xg", "count"),
        total_xg=("xg", "sum"),
        total_goals=("is_goal", "sum"),
        competitions_appeared=("competition", "nunique"),
    ).reset_index()

    overall["total_xg"] = overall["total_xg"].round(3)
    overall["xg_per_shot_overall"] = (overall["total_xg"] / overall["total_shots"]).round(3)

    return per_comp, overall


def run():
    print("[fetch_statsbomb] Starting player stat extraction...")

    if not STATSBOMB_PATH.exists():
        print(f"[fetch_statsbomb] ERROR: StatsBomb repo not found at {STATSBOMB_PATH}")
        print("Run: git clone https://github.com/statsbomb/open-data.git data/statsbomb/open-data")
        return

    # Check AFCON competition ID — add to targets if it exists
    competitions_path = STATSBOMB_PATH / "competitions.json"
    targets = list(TARGET_COMPETITIONS)

    if competitions_path.exists():
        with open(competitions_path, encoding="utf-8") as f:
            comps = json.load(f)
        afcon_found = any(
            c["competition_id"] == AFCON_COMPETITION[0] and c["season_id"] == AFCON_COMPETITION[1]
            for c in comps
        )
        if afcon_found:
            targets.append(AFCON_COMPETITION)
            print(f"[fetch_statsbomb] AFCON 2023 confirmed — adding to targets")
        else:
            print(f"[fetch_statsbomb] AFCON 2023 not found in competitions.json — skipping")

    # Fetch all competitions
    all_shots = []
    for comp_id, season_id, label in targets:
        df = fetch_competition(comp_id, season_id, label)
        if not df.empty:
            all_shots.append(df)

    if not all_shots:
        print("[fetch_statsbomb] ERROR: No shot data extracted.")
        return

    df_all_shots = pd.concat(all_shots, ignore_index=True)
    print(f"\n[fetch_statsbomb] Total shot events across all competitions: {len(df_all_shots)}")
    print(f"[fetch_statsbomb] Unique players: {df_all_shots['player_name'].nunique()}")

    # Aggregate
    per_comp, overall = aggregate_player_stats(df_all_shots)

    print(f"[fetch_statsbomb] Per-competition rows: {len(per_comp)}")
    print(f"[fetch_statsbomb] Overall player rows: {len(overall)}")

    # Save both to processed/
    os.makedirs("data/processed", exist_ok=True)
    per_comp.to_csv("data/processed/statsbomb_player_stats_by_comp.csv", index=False)
    overall.to_csv(OUTPUT_PATH, index=False)

    print(f"[fetch_statsbomb] Saved per-comp stats to data/processed/statsbomb_player_stats_by_comp.csv")
    print(f"[fetch_statsbomb] Saved overall stats to {OUTPUT_PATH}")

    # Quick sanity output
    print(f"\n[fetch_statsbomb] Top 10 by total_xg (non-penalty):")
    print(overall.nlargest(10, "total_xg")[
        ["player_name", "team", "total_shots", "total_xg", "total_goals", "competitions_appeared"]
    ].to_string())


if __name__ == "__main__":
    run()