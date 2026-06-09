# src/features/player_features.py
#
# Builds a composite player quality score for every player at WC 2026.
#
# Ground truth: data/processed/wc2026_squads.csv (1244 players, 48 teams)
# All stat lookups filter DOWN to this list — no retired players, no guessing.
#
# Composite score:
#   score = 0.45 * intl_xg_norm + 0.35 * club_form_norm + 0.20 * experience_norm
#
#   intl_xg_norm    : recency-weighted StatsBomb xG/shot (2024=1.0, WC22=0.7, WC18=0.4)
#   club_form_norm  : Understat npxg/90 primary, FBref npg/90 fallback
#   experience_norm : caps from wc2026_squads.csv (already official), normalised 0-1

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# --- Score weights ---
W_INTL_XG    = 0.45
W_CLUB_FORM  = 0.35
W_EXPERIENCE = 0.20

# --- Normalisation ceilings ---
MAX_XG_PER_SHOT  = 0.25   # top-end xG/shot for elite strikers
MAX_NPXG_PER90   = 1.5    # top-end club npxg/90
MAX_CAPS         = 150    # normalise caps to 0-1

# --- Quality filters ---
MIN_SHOTS_FOR_XG   = 15    # ignore xG for players with < 5 weighted shots (defenders, GKs)
MIN_MINUTES_ACTIVE = 450  # ~5 full games — filters out fringe squad members

# --- Competition recency weights ---
# More recent = more predictive of WC 2026 performance
COMPETITION_RECENCY = {
    "Euro 2024":         1.0,
    "Copa America 2024": 1.0,
    "AFCON 2023":        1.0,
    "World Cup 2022":    0.7,
    "World Cup 2018":    0.4,
}

ATTACK_POSITIONS = {"Forward", "Midfielder"}
TOP_N_ATTACKERS  = 5

# StatsBomb stores legal names — map to common names used everywhere else
# These must match exactly what's in statsbomb_player_stats_by_comp.csv
PLAYER_NAME_OVERRIDES = {
    "Kylian Mbappé Lottin":                   "Kylian Mbappé",
    "Lionel Andrés Messi Cuccittini":         "Lionel Messi",
    "Neymar da Silva Santos Junior":           "Neymar",
    "Cristiano Ronaldo dos Santos Aveiro":     "Cristiano Ronaldo",
    "Vinicius José Paixão de Oliveira Júnior": "Vinicius Junior",
    "Vinícius Júnior":                         "Vinicius Junior",
    "Rodrygo Silva de Goes":                   "Rodrygo",
    "Pedri González López":                    "Pedri",
    "Gavi Páez":                               "Gavi",
    "Álvaro Borja Morata Martín":              "Álvaro Morata",
    "Lamine Yamal Nasraoui Ebana":             "Lamine Yamal",
    "Romelu Lukaku Menama":                    "Romelu Lukaku",
    "Raheem Shaquille Sterling":               "Raheem Sterling",
    "Lautaro Javier Martínez":                 "Lautaro Martinez",
    "Julián Álvarez":                          "Julian Alvarez",
    "Heung-Min Son":                           "Son Heung-min",
    "Kylian Mbappe-Lottin":                    "Kylian Mbappe",
    "Kylian Mbappé Lottin":                    "Kylian Mbappe",
    "Kylian Mbappé":                           "Kylian Mbappe",
    "Vinicius Jose Paixao de Oliveira Junior":  "Vinicius Junior",
    "Erling Braut Haaland":                    "Erling Haaland",
    "Pedri Gonzalez Lopez":                    "Pedri",
    "Lamine Yamal Nasraoui Ebana":             "Lamine Yamal",
    "Kylian Mbappe-Lottin":                    "Kylian Mbappe",
    "Kylian Mbappé Lottin":                    "Kylian Mbappe",
    "Kylian Mbappé":                           "Kylian Mbappe",
    "Vinicius Jose Paixao de Oliveira Junior":  "Vinicius Junior",
    "Erling Braut Haaland":                    "Erling Haaland",
    "Pedri Gonzalez Lopez":                    "Pedri",
    "Lamine Yamal Nasraoui Ebana":             "Lamine Yamal",
    "Gabriel Fernando de Jesus":               "Gabriel Jesus",
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_squads() -> pd.DataFrame:
    path = PROCESSED_DIR / "wc2026_squads.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run fetch_squads.py first: {path}")
    df = pd.read_csv(path)
    # Normalise team name column to match rest of pipeline
    df["team"] = df["team"].str.strip()
    return df


@lru_cache(maxsize=1)
def _load_statsbomb_by_comp() -> pd.DataFrame:
    path = PROCESSED_DIR / "statsbomb_player_stats_by_comp.csv"
    if not path.exists():
        print("  [player_features] WARNING: statsbomb_player_stats_by_comp.csv missing")
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _load_understat() -> pd.DataFrame:
    path = PROCESSED_DIR / "understat_players.csv"
    if not path.exists():
        print("  [player_features] WARNING: understat_players.csv missing")
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _load_fbref() -> pd.DataFrame:
    path = PROCESSED_DIR / "fbref_players.csv"
    if not path.exists():
        print("  [player_features] WARNING: fbref_players.csv missing, skipping FBref fallback")
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _load_transfermarkt() -> pd.DataFrame:
    path = PROCESSED_DIR / "transfermarkt_players.csv"
    if not path.exists():
        print("  [player_features] WARNING: transfermarkt_players.csv missing")
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    """
    Lowercase, strip accents-agnostic first+last name.
    Used for fuzzy joining across sources with different encodings.
    """
    if pd.isna(name):
        return ""
    s = str(name).strip()
    s = PLAYER_NAME_OVERRIDES.get(s, s)
    parts = s.split()
    if len(parts) >= 2:
        return f"{parts[0].lower()} {parts[1].lower()}"
    return s.lower()


# ---------------------------------------------------------------------------
# Step 1: international xG from StatsBomb (recency-weighted)
# ---------------------------------------------------------------------------

def _build_intl_xg() -> pd.DataFrame:
    """
    Returns one row per player with recency-weighted xG/shot.
    Players with < MIN_SHOTS_FOR_XG weighted shots get xg_per_shot = 0.
    This prevents defenders from being scored as attackers.
    """
    df = _load_statsbomb_by_comp()
    if df.empty:
        return pd.DataFrame(columns=["name_key", "intl_xg_per_shot", "weighted_shots"])

    df = df.copy()
    df["recency"] = df["competition"].map(COMPETITION_RECENCY).fillna(0.5)
    df["w_xg"]    = df["xg"]    * df["recency"]
    df["w_shots"] = df["shots"] * df["recency"]

    agg = df.groupby("player_name").agg(
        w_xg_total=("w_xg",    "sum"),
        w_shots_total=("w_shots", "sum"),
    ).reset_index()

    agg["intl_xg_per_shot"] = (
        agg["w_xg_total"] / agg["w_shots_total"].clip(lower=1)
    ).round(4)

    # Zero out low-shot players — they're GKs/defenders, not attackers
    agg.loc[agg["w_shots_total"] < MIN_SHOTS_FOR_XG, "intl_xg_per_shot"] = 0.0

    agg["name_key"] = agg["player_name"].apply(_norm_name)
    return agg[["name_key", "intl_xg_per_shot", "w_shots_total"]].rename(
        columns={"w_shots_total": "weighted_shots"}
    )


# ---------------------------------------------------------------------------
# Step 2: club form from Understat + FBref fallback
# ---------------------------------------------------------------------------

def _build_club_form() -> pd.DataFrame:
    """
    Returns one row per player with club_npxg_per90.
    Primary: Understat (npxg/90, seasons 2024+2025)
    Fallback: FBref (npg/90) for players not in Understat
    """
    us = _load_understat()
    fb = _load_fbref()

    results = []

    if not us.empty:
        us_active = us[us["minutes_played"] >= MIN_MINUTES_ACTIVE].copy()
        us_active["name_key"] = us_active["player_name"].apply(_norm_name)
        # Take highest npxg/90 per player (they may appear in multiple leagues)
        us_best = us_active.groupby("name_key")["npxg_per90"].max().reset_index()
        us_best.columns = ["name_key", "club_npxg_per90"]
        results.append(us_best)

    if not fb.empty and "npg_per90" in fb.columns:
        fb_active = fb[fb.get("minutes", pd.Series(0)) >= MIN_MINUTES_ACTIVE].copy() if "minutes" in fb.columns else fb.copy()
        fb_active["name_key"] = fb_active["player_name"].apply(_norm_name)
        fb_best = fb_active.groupby("name_key")["npg_per90"].max().reset_index()
        fb_best.columns = ["name_key", "fbref_npg_per90"]
        results.append(fb_best)

    if not results:
        return pd.DataFrame(columns=["name_key", "club_npxg_per90"])

    # Merge understat + fbref
    if len(results) == 2:
        club = results[0].merge(results[1], on="name_key", how="outer")
        # Understat is primary, FBref fills gaps
        club["club_npxg_per90"] = club["club_npxg_per90"].fillna(
            club["fbref_npg_per90"]
        ).fillna(0.0)
        club = club[["name_key", "club_npxg_per90"]]
    else:
        club = results[0]
        if "club_npxg_per90" not in club.columns:
            club = club.rename(columns={"fbref_npg_per90": "club_npxg_per90"})

    return club


# ---------------------------------------------------------------------------
# Step 3: market value from Transfermarkt
# ---------------------------------------------------------------------------

def _build_market_value() -> pd.DataFrame:
    tm = _load_transfermarkt()
    if tm.empty:
        return pd.DataFrame(columns=["name_key", "market_value_m"])
    tm = tm.copy()
    tm["name_key"] = tm["player_name"].apply(_norm_name)
    # Take highest value per name_key (handles duplicate entries)
    val = tm.groupby("name_key")["latest_value_m"].max().reset_index()
    val.columns = ["name_key", "market_value_m"]
    return val


# ---------------------------------------------------------------------------
# Main: build player quality table
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def build_player_quality_table() -> pd.DataFrame:
    """
    One row per player in wc2026_squads.csv.
    Looks up stats from all sources and computes composite score.

    Returns DataFrame with columns:
        team, player_name, position, age, caps, goals, club,
        intl_xg_per_shot, weighted_shots, club_npxg_per90, market_value_m,
        intl_xg_norm, club_form_norm, experience_norm, player_score
    """
    # --- Ground truth: official squads ---
    squads = _load_squads().copy()
    squads["name_key"] = squads["player_name"].apply(_norm_name)

    print(f"  [player_features] Squad players: {len(squads)} across {squads['team'].nunique()} teams")

    # --- Stat lookups ---
    intl_xg    = _build_intl_xg()
    club_form  = _build_club_form()
    mkt_value  = _build_market_value()

    # --- Join everything onto squad list ---
    df = squads.merge(intl_xg,   on="name_key", how="left")
    df = df.merge(club_form,     on="name_key", how="left")
    df = df.merge(mkt_value,     on="name_key", how="left")

    # --- Fill missing ---
    df["intl_xg_per_shot"] = df["intl_xg_per_shot"].fillna(0.0)
    df["weighted_shots"]   = df["weighted_shots"].fillna(0.0)
    df["club_npxg_per90"]  = df["club_npxg_per90"].fillna(0.0)
    df["market_value_m"]   = df["market_value_m"].fillna(0.0)

    # --- Normalise to 0-1 ---
    df["intl_xg_norm"] = (
        df["intl_xg_per_shot"].clip(0, MAX_XG_PER_SHOT) / MAX_XG_PER_SHOT
    ).round(4)

    df["club_form_norm"] = (
        df["club_npxg_per90"].clip(0, MAX_NPXG_PER90) / MAX_NPXG_PER90
    ).round(4)

    # caps comes directly from wc2026_squads.csv — official FIFA data, most reliable
    df["experience_norm"] = (
        df["caps"].clip(0, MAX_CAPS) / MAX_CAPS
    ).round(4)

    # --- Composite score ---
    df["player_score"] = (
        W_INTL_XG    * df["intl_xg_norm"] +
        W_CLUB_FORM  * df["club_form_norm"] +
        W_EXPERIENCE * df["experience_norm"]
    ).round(4)

    # Coverage stats
    intl_coverage = int((df["intl_xg_per_shot"] > 0).sum())
    club_coverage = int((df["club_npxg_per90"]  > 0).sum())
    print(f"  [player_features] intl xG coverage : {intl_coverage}/{len(df)} players")
    print(f"  [player_features] club form coverage: {club_coverage}/{len(df)} players")

    return df.sort_values("player_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Team-level helpers (used by match_features.py and API)
# ---------------------------------------------------------------------------

def get_team_player_features(team: str) -> dict:
    df = build_player_quality_table()
    players = df[df["team"] == team].copy()

    if players.empty:
        print(f"  [player_features] WARNING: {team} not found in squad data")
        return _default_player_features()

    players = players.sort_values("player_score", ascending=False)

    # Attack score based on forwards + midfielders, fall back to full squad
    attackers = players[players["position"].isin(ATTACK_POSITIONS)]
    pool = attackers if len(attackers) >= 3 else players
    top_n = pool.head(TOP_N_ATTACKERS)
    n = len(top_n)

    squad_attack_score = round(float(top_n["player_score"].mean()), 4) if n > 0 else 0.0
    star_player_score  = round(float(top_n["player_score"].iloc[0]), 4) if n > 0 else 0.0
    star_player_name   = str(top_n["player_name"].iloc[0]) if n > 0 else "Unknown"

    depth_score = round(float(top_n["player_score"].iloc[3:].mean()), 4) if n >= 4 else \
                  round(float(top_n["player_score"].iloc[-1]), 4) if n >= 2 else \
                  squad_attack_score

    return {
        "squad_attack_score":  squad_attack_score,
        "star_player_score":   star_player_score,
        "star_player_name":    star_player_name,
        "squad_depth_score":   depth_score,
        "depth_dropoff":       round(star_player_score - depth_score, 4),
        "avg_caps_top5":       round(float(top_n["caps"].mean()), 1),
        "total_squad_value_m": round(float(players["market_value_m"].sum()), 1),
        "active_player_count": int((players["club_npxg_per90"] > 0).sum()),
        "players_in_squad":    len(players),
    }


def _default_player_features() -> dict:
    df = build_player_quality_table()
    avg = float(df["player_score"].mean())
    return {
        "squad_attack_score":  round(avg * 0.7, 4),
        "star_player_score":   round(avg * 0.8, 4),
        "star_player_name":    "Unknown",
        "squad_depth_score":   round(avg * 0.6, 4),
        "depth_dropoff":       0.05,
        "avg_caps_top5":       20.0,
        "total_squad_value_m": 10.0,
        "active_player_count": 5,
        "players_in_squad":    0,
    }


def get_matchup_player_features(team_a: str, team_b: str) -> dict:
    fa = get_team_player_features(team_a)
    fb = get_team_player_features(team_b)
    features = {f"player_a_{k}": v for k, v in fa.items()}
    features.update({f"player_b_{k}": v for k, v in fb.items()})
    features["attack_score_diff"] = round(fa["squad_attack_score"] - fb["squad_attack_score"], 4)
    features["star_score_diff"]   = round(fa["star_player_score"]  - fb["star_player_score"],  4)
    features["depth_score_diff"]  = round(fa["squad_depth_score"]  - fb["squad_depth_score"],  4)
    features["value_diff_m"]      = round(fa["total_squad_value_m"] - fb["total_squad_value_m"], 1)
    return features


# ---------------------------------------------------------------------------
# Pipeline entry point — called by run_pipeline.py
# ---------------------------------------------------------------------------

def run():
    """
    Build player quality table and save to data/processed/player_scores.csv.

    Column rename before saving:
      player_name  -> player         (API routes expect "player")
      player_score -> composite_score (API routes expect "composite_score")
    name_key is an internal join key and is dropped.
    """
    print("  [player_features] Building player quality table...")
    df = build_player_quality_table()

    out = df.rename(columns={
        "player_name":  "player",
        "player_score": "composite_score",
    }).drop(columns=["name_key"], errors="ignore")

    out_path = PROCESSED_DIR / "player_scores.csv"
    out.to_csv(out_path, index=False)

    print(f"  [player_features] Saved {len(out)} rows -> {out_path}")
    print(f"  [player_features] Coverage: "
          f"{int((out['composite_score'] > 0).sum())} players with non-zero score")


# ---------------------------------------------------------------------------
# Run directly to validate output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Building player quality table ===\n")
    df = build_player_quality_table()

    print(f"\nTotal players scored: {len(df)}")
    print(f"\nTop 20 players by composite score:")
    print(df.head(20)[[
        "player_name", "team", "position", "age", "caps",
        "player_score", "intl_xg_norm", "club_form_norm", "experience_norm",
        "club_npxg_per90", "market_value_m"
    ]].to_string(index=False))

    print("\n=== Top 12 teams by squad attack score ===")
    teams = [
        "France", "Argentina", "England", "Brazil", "Spain", "Portugal",
        "Germany", "Morocco", "Japan", "Colombia", "Netherlands", "United States"
    ]
    rows = []
    for t in teams:
        f = get_team_player_features(t)
        rows.append({
            "team":         t,
            "attack_score": f["squad_attack_score"],
            "star_player":  f["star_player_name"],
            "star_score":   f["star_player_score"],
            "depth_dropoff":f["depth_dropoff"],
            "value_m":      f["total_squad_value_m"],
            "active":       f["active_player_count"],
        })
    print(pd.DataFrame(rows).sort_values("attack_score", ascending=False).to_string(index=False))

    print("\n=== France vs Argentina matchup features ===")
    m = get_matchup_player_features("France", "Argentina")
    for k, v in m.items():
        print(f"  {k}: {v}")