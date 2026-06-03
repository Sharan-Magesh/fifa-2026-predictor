# src/data_pipeline/fetch_wc2026_fixtures.py

import os
import requests
import pandas as pd
import json

# openfootball worldcup.json — free, no auth, public domain
# Confirmed structure: {"name": "World Cup 2026", "matches": [...]}
# Each match: {round, date, time, team1, team2, group, ground}
# Knockout matches use placeholder team refs (W101, L102 etc.) — we handle those separately
FIXTURES_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)
TEAMS_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.teams.json"
)

FIXTURES_OUTPUT = os.path.join("data", "processed", "wc2026_fixtures.csv")
GROUPS_OUTPUT = os.path.join("data", "processed", "wc2026_groups.csv")

# Confirmed group assignments from openfootball (verified June 2026)
# Hardcoded as fallback in case the teams JSON structure differs from expected
GROUPS_FALLBACK = {
    "Group A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "Group B": ["Canada", "Bosnia & Herzegovina", "Qatar", "Switzerland"],
    "Group C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "Group D": ["United States", "Paraguay", "Australia", "Turkey"],
    "Group E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "Group F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "Group G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "Group H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "Group I": ["France", "Senegal", "Iraq", "Norway"],
    "Group J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "Group K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "Group L": ["England", "Croatia", "Ghana", "Panama"],
}

# Note: openfootball uses "USA" — we map to "United States" to match martj42 dataset
# This normalisation must be consistent across ALL files in the pipeline
TEAM_NAME_MAP = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Ivory Coast": "Côte d'Ivoire",
    "DR Congo": "DR Congo",  # same in both sources, no change needed
    "Cape Verde": "Cape Verde",  # same
    "Curaçao": "Curaçao",  # same
}


def normalise_team_name(name: str) -> str:
    """
    Map openfootball team names to martj42/international_results names.
    This is critical — if names don't match, H2H lookups in feature engineering fail silently.
    """
    return TEAM_NAME_MAP.get(name, name)


def fetch_fixtures() -> pd.DataFrame:
    """
    Download and parse the fixtures JSON.
    Group stage matches have a 'group' field.
    Knockout matches use placeholder refs (W101, L102) for teams not yet determined.
    We keep both — group stage for simulation input, knockout structure for bracket logic.
    """
    print("[fetch_wc2026_fixtures] Downloading fixtures...")
    response = requests.get(FIXTURES_URL, timeout=30)
    response.raise_for_status()
    data = response.json()

    matches = data.get("matches", [])
    print(f"[fetch_wc2026_fixtures] Total matches in JSON: {len(matches)}")

    rows = []
    for m in matches:
        team1 = normalise_team_name(m.get("team1", ""))
        team2 = normalise_team_name(m.get("team2", ""))

        # Determine match stage
        # Group matches have a 'group' key; knockout matches don't
        group = m.get("group", None)
        stage = "group" if group else classify_knockout_stage(m.get("round", ""))

        rows.append({
            "round": m.get("round", ""),
            "date": m.get("date", ""),
            "time": m.get("time", ""),
            "team1": team1,
            "team2": team2,
            "group": group,
            "stage": stage,
            "venue": m.get("ground", ""),
            # Score fields — will be None until matches are played
            # Keeping these here so the same DataFrame structure works
            # for both prediction (pre-tournament) and tracking (live)
            "score1": m.get("score", {}).get("ft", [None, None])[0] if m.get("score") else None,
            "score2": m.get("score", {}).get("ft", [None, None])[1] if m.get("score") else None,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def classify_knockout_stage(round_str: str) -> str:
    """
    Map openfootball round strings to clean stage labels.
    These are the actual round names used in the JSON — verified from search results.
    """
    r = round_str.lower()
    if "round of 32" in r:
        return "round_of_32"
    elif "round of 16" in r:
        return "round_of_16"
    elif "quarterfinal" in r or "quarter-final" in r:
        return "quarterfinal"
    elif "semifinal" in r or "semi-final" in r:
        return "semifinal"
    elif "final" in r and "third" in r:
        return "third_place"
    elif "final" in r:
        return "final"
    else:
        return "group"


def build_groups_df() -> pd.DataFrame:
    """
    Build a clean team → group mapping DataFrame.
    Tries to fetch from openfootball teams JSON first.
    Falls back to hardcoded GROUPS_FALLBACK if the fetch fails or structure differs.

    This DataFrame is used by the simulation to know which teams share a group
    and therefore play each other in the group stage.
    """
    rows = []
    try:
        response = requests.get(TEAMS_URL, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Try to parse — structure may vary, so wrap in try/except
        # If it fails we fall through to the hardcoded fallback
        teams = data.get("teams", [])
        if teams:
            for t in teams:
                name = normalise_team_name(t.get("name", ""))
                group = t.get("group", "")
                if name and group:
                    rows.append({"team": name, "group": group})
            print(f"[fetch_wc2026_fixtures] Groups loaded from teams JSON: {len(rows)} teams")

    except Exception as e:
        print(f"[fetch_wc2026_fixtures] Teams JSON fetch failed: {e}. Using fallback.")

    # Use fallback if fetch failed or returned nothing useful
    if not rows:
        for group, teams in GROUPS_FALLBACK.items():
            for team in teams:
                rows.append({"team": normalise_team_name(team), "group": group})
        print(f"[fetch_wc2026_fixtures] Groups loaded from fallback: {len(rows)} teams")

    df = pd.DataFrame(rows).sort_values(["group", "team"]).reset_index(drop=True)
    return df


def run():
    os.makedirs(os.path.dirname(FIXTURES_OUTPUT), exist_ok=True)

    # --- Fixtures ---
    df_fixtures = fetch_fixtures()

    group_matches = df_fixtures[df_fixtures["stage"] == "group"]
    knockout_matches = df_fixtures[df_fixtures["stage"] != "group"]

    print(f"[fetch_wc2026_fixtures] Group stage matches: {len(group_matches)}")
    print(f"[fetch_wc2026_fixtures] Knockout matches: {len(knockout_matches)}")
    print(f"[fetch_wc2026_fixtures] Date range: {df_fixtures['date'].min().date()} → {df_fixtures['date'].max().date()}")

    df_fixtures.to_csv(FIXTURES_OUTPUT, index=False)
    print(f"[fetch_wc2026_fixtures] Fixtures saved to {FIXTURES_OUTPUT}")

    # --- Groups ---
    df_groups = build_groups_df()
    print(f"\n[fetch_wc2026_fixtures] Groups:\n{df_groups.groupby('group')['team'].apply(list).to_string()}")

    df_groups.to_csv(GROUPS_OUTPUT, index=False)
    print(f"[fetch_wc2026_fixtures] Groups saved to {GROUPS_OUTPUT}")


if __name__ == "__main__":
    run()