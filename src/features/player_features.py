# src/features/player_features.py

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

W_INTL_XG    = 0.45
W_CLUB_FORM  = 0.35
W_EXPERIENCE = 0.20

MAX_XG_PER_SHOT  = 0.25
MAX_NPXG_PER90   = 1.5
MAX_CAPS         = 150
MIN_SHOTS_FOR_XG = 5
MIN_MINUTES_ACTIVE = 450
TOP_N_ATTACKERS = 5
ATTACK_POSITIONS = ["Attack", "Midfield"]

# Manual overrides for star players whose names differ across sources
# StatsBomb uses full legal names; Understat and Transfermarkt use common names
PLAYER_NAME_OVERRIDES = {
    "Kylian Mbappé Lottin":                    "Kylian Mbappé",
    "Lionel Andrés Messi Cuccittini":          "Lionel Messi",
    "Neymar da Silva Santos Junior":            "Neymar",
    "Cristiano Ronaldo dos Santos Aveiro":      "Cristiano Ronaldo",
    "Vinicius José Paixão de Oliveira Júnior":  "Vinicius Junior",
    "Rodrygo Silva de Goes":                   "Rodrygo",
    "Pedri González López":                    "Pedri",
    "Gavi Páez":                               "Gavi",
    "Álvaro Borja Morata Martín":              "Álvaro Morata",
    "Lamine Yamal Nasraoui Ebana":             "Lamine Yamal",
    "Antoine Griezmann":                       "Antoine Griezmann",
    "Erling Braut Haaland":                    "Erling Haaland",
    "Bukayo Saka":                             "Bukayo Saka",
    "Jude Bellingham":                         "Jude Bellingham",
    "Romelu Lukaku Menama":                    "Romelu Lukaku",
    "Aleksandar Mitrović":                     "Aleksandar Mitrovic",
    "Olivier Giroud":                          "Olivier Giroud",
    "Raheem Shaquille Sterling":               "Raheem Sterling",
    "Mohamed Salah":                           "Mohamed Salah",
    "Sadio Mané":                              "Sadio Mane",
    "Richarlison de Andrade":                  "Richarlison",
    "Gabriel Fernando de Jesus":               "Gabriel Jesus",
    "Lautaro Javier Martínez":                 "Lautaro Martinez",
    "Julián Álvarez":                          "Julian Alvarez",
}


@lru_cache(maxsize=1)
def load_statsbomb() -> pd.DataFrame:
    path = PROCESSED_DIR / "statsbomb_player_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run fetch_statsbomb.py first: {path}")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_understat() -> pd.DataFrame:
    path = PROCESSED_DIR / "understat_players.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run fetch_understat.py first: {path}")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_transfermarkt() -> pd.DataFrame:
    path = PROCESSED_DIR / "transfermarkt_players.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run fetch_transfermarkt.py first: {path}")
    return pd.read_csv(path)


def normalise_player_name(name: str) -> str:
    """
    Normalise to first two words.
    "Kylian Mbappé Lottin" -> "Kylian Mbappé"  (after override: "Kylian Mbappé" -> "Kylian Mbappé")
    "Lionel Messi" -> "Lionel Messi" (after override applied first)
    Covers ~90% of join cases when combined with PLAYER_NAME_OVERRIDES.
    """
    if pd.isna(name):
        return str(name)
    parts = str(name).strip().split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return str(name).strip()


def apply_name_overrides(series: pd.Series) -> pd.Series:
    """Apply manual overrides then normalise."""
    return series.replace(PLAYER_NAME_OVERRIDES).apply(normalise_player_name)


@lru_cache(maxsize=1)
def build_player_quality_table() -> pd.DataFrame:
    """
    Master player quality table — one row per player.
    Joins StatsBomb + Understat + Transfermarkt on normalised name.

    Composite score:
        score = 0.45 * intl_xg_norm + 0.35 * club_form_norm + 0.20 * experience_norm

    Key calibration:
        MIN_SHOTS_FOR_XG=5 prevents defenders with 1 lucky shot
        from scoring as elite attackers.
        Name overrides handle the ~15% of star players whose full
        legal name (StatsBomb) differs from their common name.
    """
    df_sb = load_statsbomb()
    df_us = load_understat()
    df_tm = load_transfermarkt()

    # StatsBomb
    sb_cols = ["player_name", "team", "total_shots", "total_xg",
               "total_goals", "xg_per_shot_overall", "competitions_appeared"]
    sb = df_sb[sb_cols].copy().rename(columns={
        "team":                "national_team",
        "xg_per_shot_overall": "intl_xg_per_shot",
        "total_xg":            "intl_total_xg",
        "total_goals":         "intl_goals_statsbomb",
    })
    sb["intl_xg_per_shot"] = sb.apply(
        lambda r: r["intl_xg_per_shot"] if r["total_shots"] >= MIN_SHOTS_FOR_XG else 0.0,
        axis=1
    )
    sb["name_norm"] = apply_name_overrides(sb["player_name"])

    # Understat
    us_cols = ["player_name", "club", "league", "minutes_played",
               "npxg_per90", "xa_per90", "goals"]
    us = df_us[us_cols].copy().rename(columns={
        "npxg_per90": "club_npxg_per90",
        "xa_per90":   "club_xa_per90",
        "goals":      "club_goals",
    })
    us_active = us[us["minutes_played"] >= MIN_MINUTES_ACTIVE].copy()
    us_active["name_norm"] = apply_name_overrides(us_active["player_name"])

    # Transfermarkt
    tm_cols = ["player_name", "position", "caps", "international_goals",
               "latest_value_m", "date_of_birth"]
    tm_cols = [c for c in tm_cols if c in df_tm.columns]
    tm = df_tm[tm_cols].copy()
    tm = tm[tm["caps"].notna() & (tm["caps"] > 0)].copy()
    tm["name_norm"] = apply_name_overrides(tm["player_name"])

    # Join
    merged = sb.merge(
        us_active[["name_norm", "club_npxg_per90", "club_xa_per90",
                   "club_goals", "minutes_played"]],
        on="name_norm", how="left",
    )
    merged = merged.merge(
        tm[["name_norm", "position", "caps",
            "international_goals", "latest_value_m"]],
        on="name_norm", how="left",
    )

    # Fill missing
    merged["club_npxg_per90"]  = merged["club_npxg_per90"].fillna(0.0)
    merged["club_xa_per90"]    = merged["club_xa_per90"].fillna(0.0)
    merged["caps"]              = merged["caps"].fillna(0.0)
    merged["intl_xg_per_shot"] = merged["intl_xg_per_shot"].fillna(0.0)
    merged["latest_value_m"]   = merged["latest_value_m"].fillna(0.0)
    merged["position"]         = merged["position"].fillna("Unknown")

    # Normalise
    merged["intl_xg_norm"] = (
        merged["intl_xg_per_shot"].clip(0, MAX_XG_PER_SHOT) / MAX_XG_PER_SHOT
    ).round(4)
    merged["club_form_norm"] = (
        merged["club_npxg_per90"].clip(0, MAX_NPXG_PER90) / MAX_NPXG_PER90
    ).round(4)
    merged["experience_norm"] = (
        merged["caps"].clip(0, MAX_CAPS) / MAX_CAPS
    ).round(4)

    # Composite score
    merged["player_score"] = (
        W_INTL_XG    * merged["intl_xg_norm"] +
        W_CLUB_FORM  * merged["club_form_norm"] +
        W_EXPERIENCE * merged["experience_norm"]
    ).round(4)

    return merged.sort_values("player_score", ascending=False).reset_index(drop=True)


def get_team_player_features(team: str) -> dict:
    df = build_player_quality_table()
    team_players = df[df["national_team"] == team].copy()

    if team_players.empty:
        print(f"  [player_features] WARNING: {team} not in StatsBomb data, using defaults")
        return _default_player_features()

    team_players = team_players.sort_values("player_score", ascending=False)

    attackers = team_players[team_players["position"].isin(ATTACK_POSITIONS)]
    scoring_pool = attackers if len(attackers) >= 3 else team_players

    top_n = scoring_pool.head(TOP_N_ATTACKERS)
    n = len(top_n)

    squad_attack_score = round(top_n["player_score"].mean(), 4) if n > 0 else 0.0
    star_player_score  = round(top_n["player_score"].iloc[0], 4) if n > 0 else 0.0
    star_player_name   = top_n["player_name"].iloc[0] if n > 0 else "Unknown"

    if n >= 4:
        depth_score = round(top_n["player_score"].iloc[3:].mean(), 4)
    elif n >= 2:
        depth_score = round(top_n["player_score"].iloc[-1], 4)
    else:
        depth_score = squad_attack_score

    depth_dropoff = round(star_player_score - depth_score, 4)
    avg_caps = round(top_n["caps"].mean(), 1)

    df_tm = load_transfermarkt()
    tm_team = df_tm[df_tm["player_name"].isin(team_players["player_name"])]
    total_value = round(tm_team["latest_value_m"].sum(), 1)

    active_count = int((team_players["club_npxg_per90"] > 0).sum())

    return {
        "squad_attack_score":   squad_attack_score,
        "star_player_score":    star_player_score,
        "star_player_name":     star_player_name,
        "squad_depth_score":    depth_score,
        "depth_dropoff":        depth_dropoff,
        "avg_caps_top5":        avg_caps,
        "total_squad_value_m":  total_value,
        "active_player_count":  active_count,
        "players_in_statsbomb": len(team_players),
    }


def _default_player_features() -> dict:
    df = build_player_quality_table()
    avg_score = float(df["player_score"].mean())
    return {
        "squad_attack_score":   round(avg_score * 0.7, 4),
        "star_player_score":    round(avg_score * 0.8, 4),
        "star_player_name":     "Unknown",
        "squad_depth_score":    round(avg_score * 0.6, 4),
        "depth_dropoff":        0.05,
        "avg_caps_top5":        20.0,
        "total_squad_value_m":  10.0,
        "active_player_count":  5,
        "players_in_statsbomb": 0,
    }


def get_matchup_player_features(team_a: str, team_b: str) -> dict:
    feat_a = get_team_player_features(team_a)
    feat_b = get_team_player_features(team_b)

    features = {}
    for k, v in feat_a.items():
        features[f"player_a_{k}"] = v
    for k, v in feat_b.items():
        features[f"player_b_{k}"] = v

    features["attack_score_diff"] = round(
        feat_a["squad_attack_score"] - feat_b["squad_attack_score"], 4
    )
    features["star_score_diff"] = round(
        feat_a["star_player_score"] - feat_b["star_player_score"], 4
    )
    features["depth_score_diff"] = round(
        feat_a["squad_depth_score"] - feat_b["squad_depth_score"], 4
    )
    features["value_diff_m"] = round(
        feat_a["total_squad_value_m"] - feat_b["total_squad_value_m"], 1
    )

    return features


if __name__ == "__main__":
    print("=== Building player quality table ===")
    df = build_player_quality_table()
    print(f"Total players in table: {len(df)}")

    print(f"\nTop 20 players by composite score:")
    print(df.head(20)[["player_name", "national_team", "position", "player_score",
                        "intl_xg_norm", "club_form_norm", "experience_norm",
                        "total_shots", "caps", "club_npxg_per90"]].to_string())

    print("\n=== France vs Argentina player features ===")
    features = get_matchup_player_features("France", "Argentina")
    for k, v in features.items():
        print(f"  {k}: {v}")

    print("\n=== Top 10 teams by squad attack score ===")
    teams_of_interest = [
        "France", "Argentina", "England", "Brazil",
        "Spain", "Portugal", "Germany", "Morocco", "Japan", "Colombia"
    ]
    rows = []
    for t in teams_of_interest:
        f = get_team_player_features(t)
        rows.append({
            "team": t,
            "squad_attack": f["squad_attack_score"],
            "star_player": f["star_player_name"],
            "star_score": f["star_player_score"],
            "depth_dropoff": f["depth_dropoff"],
            "value_m": f["total_squad_value_m"],
        })
    print(pd.DataFrame(rows).sort_values("squad_attack", ascending=False).to_string())