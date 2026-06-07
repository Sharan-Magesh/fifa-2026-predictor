# src/data_pipeline/fetch_squads.py
#
# Parses official WC 2026 squads from a locally saved Wikipedia HTML file.
# Source page: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads
#
# HOW TO GET THE INPUT FILE (one-time manual step):
#   1. Open https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads in Chrome
#   2. Ctrl+S -> "Webpage, Complete"
#   3. Save to: data/raw/wc2026_squads.html
#
# Output: data/processed/wc2026_squads.csv
# Columns: team, player_name, position, age, caps, goals, club

import re
import sys
import io
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

INPUT_PATH  = Path(__file__).resolve().parents[2] / "data" / "raw" / "wc2026_squads.html"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "wc2026_squads.csv"

POSITION_MAP = {
    "GK": "Goalkeeper",
    "DF": "Defender",
    "MF": "Midfielder",
    "FW": "Forward",
}

REQUIRED_COLS = {"Pos.", "Player", "Caps", "Goals", "Club"}

# Headings that exist on the page but are NOT team names
SKIP_HEADINGS = {
    "contents", "references", "notes", "squads", "see also",
    "external links", "group a", "group b", "group c", "group d",
    "group e", "group f", "group g", "group h", "group i", "group j",
    "group k", "group l", "statistics", "player representation by league",
    "player representation by club confederation", "average age of squads",
    "coach representation by country", "player representation by nationality",
}


def _extract_age(dob_str):
    match = re.search(r"aged\s+(\d+)", str(dob_str))
    return int(match.group(1)) if match else -1


def _clean_name(name):
    name = re.sub(r"\(captain\)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\[.*?\]", "", name)
    return name.strip()


def _extract_team_names(soup):
    """
    Extract team names in page order.
    Tries three strategies in order until one gives >= 40 names:
      1. <span class="mw-headline"> inside h2/h3  (live Wikipedia)
      2. Direct text of <h2>/<h3> tags            (some saved versions)
      3. id attribute of <h2>/<h3> tags            (Chrome saved pages)
    """
    strategies = [
        # Strategy 1: mw-headline span (standard Wikipedia)
        lambda soup: [
            re.sub(r"\[.*?\]", "", s.get_text(strip=True)).strip()
            for h in soup.find_all(["h2", "h3"])
            for s in [h.find("span", class_="mw-headline")]
            if s
        ],
        # Strategy 2: div.mw-heading span (newer MediaWiki)
        lambda soup: [
            re.sub(r"\[.*?\]", "", s.get_text(strip=True)).strip()
            for d in soup.find_all("div", class_="mw-heading")
            for s in d.find_all(["h2", "h3"])
        ],
        # Strategy 3: id on the heading tag itself (Chrome saved pages)
        lambda soup: [
            re.sub(r"_", " ", re.sub(r"\[.*?\]", "", h.get("id", ""))).strip()
            for h in soup.find_all(["h2", "h3"])
            if h.get("id")
        ],
        # Strategy 4: any heading text directly
        lambda soup: [
            re.sub(r"\[.*?\]", "", h.get_text(strip=True)).strip()
            for h in soup.find_all(["h2", "h3"])
        ],
    ]

    for i, strategy in enumerate(strategies):
        candidates = strategy(soup)
        filtered = [t for t in candidates if t and t.lower() not in SKIP_HEADINGS]
        print(f"[fetch_squads] Heading strategy {i+1}: {len(filtered)} candidates")
        if len(filtered) >= 40:
            return filtered

    return []


def fetch_squads():
    if not INPUT_PATH.exists():
        print(f"[fetch_squads] ERROR: Input file not found:")
        print(f"  {INPUT_PATH}")
        print()
        print("  1. Open https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads")
        print("  2. Ctrl+S -> 'Webpage, Complete'")
        print(f"  3. Save as: {INPUT_PATH}")
        sys.exit(1)

    print(f"[fetch_squads] Reading: {INPUT_PATH}")
    html_text = INPUT_PATH.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_text, "html.parser")

    # --- Extract team names ---
    team_names = _extract_team_names(soup)
    print(f"[fetch_squads] Team names extracted: {len(team_names)}")

    if len(team_names) < 40:
        print("[fetch_squads] ERROR: Could not extract enough team names.")
        print("  First 20 headings found:")
        all_headings = [h.get_text(strip=True) for h in soup.find_all(["h2", "h3"])]
        for h in all_headings[:20]:
            print(f"    {repr(h)}")
        sys.exit(1)

    # --- Parse squad tables ---
    all_tables = pd.read_html(io.StringIO(html_text), flavor="lxml")
    squad_tables = []
    for tbl in all_tables:
        col_set = {str(c).strip() for c in tbl.columns}
        if REQUIRED_COLS.issubset(col_set):
            squad_tables.append(tbl)

    print(f"[fetch_squads] Squad tables found: {len(squad_tables)}")

    if len(squad_tables) == 0:
        print("[fetch_squads] ERROR: No squad tables found.")
        print(f"  Total tables on page: {len(all_tables)}")
        if all_tables:
            print(f"  First table columns: {list(all_tables[0].columns)}")
        sys.exit(1)

    if len(squad_tables) != len(team_names):
        print(f"[fetch_squads] WARNING: {len(team_names)} names vs {len(squad_tables)} tables — zipping by min.")

    # --- Build records ---
    records = []
    for team, tbl in zip(team_names, squad_tables):
        for _, row in tbl.iterrows():
            pos_raw    = str(row.get("Pos.", "")).strip()
            player_raw = str(row.get("Player", "")).strip()
            dob_raw    = str(row.get("Date of birth (age)", "")).strip()
            caps_raw   = row.get("Caps", 0)
            goals_raw  = row.get("Goals", 0)
            club_raw   = str(row.get("Club", "")).strip()

            if pos_raw in ("Pos.", "") or player_raw in ("Player", ""):
                continue

            records.append({
                "team":        team,
                "player_name": _clean_name(player_raw),
                "position":    POSITION_MAP.get(pos_raw, pos_raw),
                "age":         _extract_age(dob_raw),
                "caps":        pd.to_numeric(caps_raw,  errors="coerce"),
                "goals":       pd.to_numeric(goals_raw, errors="coerce"),
                "club":        club_raw,
            })

    df = pd.DataFrame(records)
    df["caps"]  = df["caps"].fillna(0).astype(int)
    df["goals"] = df["goals"].fillna(0).astype(int)
    return df


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_squads()

    if df.empty:
        print("[fetch_squads] ERROR: DataFrame is empty.")
        sys.exit(1)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"[fetch_squads] Done.")
    print(f"  Total players : {len(df)}")
    print(f"  Teams         : {df['team'].nunique()}")
    print(f"  Output        : {OUTPUT_PATH}")
    print()
    print("[fetch_squads] Players per team:")
    counts = df.groupby("team").size().sort_values()
    for team, count in counts.items():
        flag = "  <<<" if count < 23 or count > 27 else ""
        print(f"  {team:<35} {count}{flag}")


if __name__ == "__main__":
    main()
