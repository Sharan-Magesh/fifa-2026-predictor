# src/features/match_features.py
#
# Builds the final feature vector for a given match (team_a vs team_b).
# This is what gets fed directly into the XGBoost match outcome model.
#
# Sources:
#   - team_features.py  : Elo, form, H2H, tournament experience
#   - player_features.py: squad attack score, star player, depth, value
#
# Output: one dict per match with ~25 features, all numeric.

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache

from src.features.team_features import (
    get_fifa_ranking_feature,
    get_elo_features,
    get_team_recent_form,
    get_h2h_features,
    get_tournament_experience,
)
from src.features.player_features import (
    get_matchup_player_features,
)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def build_match_features(team_a: str, team_b: str) -> dict:
    """
    Build the complete feature vector for a match between team_a and team_b.

    Returns dict of ~25 numeric features ready for XGBoost inference.

    Feature groups:
        1. Elo differential           — raw team strength gap
        2. Elo momentum               — recent Elo trend
        3. Recent form                — win rate last 10 matches
        4. H2H record                 — historical head-to-head
        5. Tournament experience      — WC appearances, rounds reached
        6. Player quality differentials — attack, star, depth, value
        7. Composite strength index   — weighted summary
    """
    # --- Elo features (requires both teams) ---
    elo = get_elo_features(team_a, team_b)

    # --- Form features (per team) ---
    form_a = get_team_recent_form(team_a)
    form_b = get_team_recent_form(team_b)

    # --- H2H features ---
    h2h = get_h2h_features(team_a, team_b)

    # --- Tournament experience (per team) ---
    exp_a = get_tournament_experience(team_a)
    exp_b = get_tournament_experience(team_b)

    # --- Player features ---
    player = get_matchup_player_features(team_a, team_b)

    features = {}

    # 0. FIFA ranking points differential
    fifa_a = get_fifa_ranking_feature(team_a)
    fifa_b = get_fifa_ranking_feature(team_b)
    features["fifa_points_norm_a"]    = fifa_a["fifa_points_norm"]
    features["fifa_points_norm_b"]    = fifa_b["fifa_points_norm"]
    features["fifa_points_norm_diff"] = round(fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"], 4)

    # 1. Elo
    # Single strongest predictor of match outcome.
    # Differential captures relative strength directly.
    features["elo_a"]    = round(float(elo.get("elo_a",    1500.0)), 2)
    features["elo_b"]    = round(float(elo.get("elo_b",    1500.0)), 2)
    features["elo_diff"] = round(float(elo.get("elo_diff", 0.0)),    2)

    # 2. Elo momentum
    # Team on a winning streak is more dangerous than static Elo suggests.
    features["elo_momentum_a"]    = round(float(elo.get("momentum_a", 0.0)), 4)
    features["elo_momentum_b"]    = round(float(elo.get("momentum_b", 0.0)), 4)
    features["elo_momentum_diff"] = round(float(elo.get("momentum_diff", 0.0)) / 100.0, 4)

    # 3. Recent form
    # Short-term form captures fitness, tactical cohesion, confidence.
    features["form_a"]    = round(float(form_a.get("win_rate",  0.5)), 4)
    features["form_b"]    = round(float(form_b.get("win_rate",  0.5)), 4)
    features["form_diff"] = round(features["form_a"] - features["form_b"], 4)

    features["goals_scored_per_game_a"] = round(float(form_a.get("goals_scored_per_game",  1.2)), 4)
    features["goals_scored_per_game_b"] = round(float(form_b.get("goals_scored_per_game",  1.2)), 4)
    features["goals_conceded_per_game_a"] = round(float(form_a.get("goals_conceded_per_game", 1.0)), 4)
    features["goals_conceded_per_game_b"] = round(float(form_b.get("goals_conceded_per_game", 1.0)), 4)

    # 4. H2H
    # Some teams consistently beat specific opponents regardless of Elo.
    features["h2h_win_rate_a"]  = round(float(h2h.get("h2h_win_rate_a",  0.33)), 4)
    features["h2h_win_rate_b"]  = round(float(h2h.get("h2h_win_rate_b",  0.33)), 4)
    features["h2h_advantage"]   = round(float(h2h.get("h2h_win_rate_a",  0.33)) -
                                         float(h2h.get("h2h_win_rate_b",  0.33)), 4)
    features["h2h_matches"]     = int(h2h.get("h2h_matches", 0))

    # 5. Tournament experience
    # Teams used to WC pressure perform differently in knockout stages.
    features["wc_appearances_a"]     = int(exp_a.get("wc_matches_played", 0))
    features["wc_appearances_b"]     = int(exp_b.get("wc_matches_played", 0))
    features["tournament_exp_a"]     = round(float(exp_a.get("wc_experience_score", 0.0)), 4)
    features["tournament_exp_b"]     = round(float(exp_b.get("wc_experience_score", 0.0)), 4)
    features["tournament_exp_diff"]  = round(features["tournament_exp_a"] -
                                              features["tournament_exp_b"], 4)

    # 6. Player quality
    # XGBoost needs these explicitly — it can't infer player quality from Elo alone.
    features["squad_attack_a"]    = round(float(player.get("player_a_squad_attack_score", 0.0)), 4)
    features["squad_attack_b"]    = round(float(player.get("player_b_squad_attack_score", 0.0)), 4)
    features["attack_score_diff"] = round(float(player.get("attack_score_diff",           0.0)), 4)
    features["star_score_a"]      = round(float(player.get("player_a_star_player_score",  0.0)), 4)
    features["star_score_b"]      = round(float(player.get("player_b_star_player_score",  0.0)), 4)
    features["star_score_diff"]   = round(float(player.get("star_score_diff",             0.0)), 4)
    features["depth_score_diff"]  = round(float(player.get("depth_score_diff",            0.0)), 4)
    features["value_diff_m"]      = round(float(player.get("value_diff_m",                0.0)), 1)

    # 7. Composite strength index
    # Single summary feature combining Elo + player quality.
    features["strength_index_diff"] = round(
        0.6 * features["elo_diff"] / 400.0 +
        0.4 * features["attack_score_diff"],
        4
    )

    return features


def build_features_from_fixtures(fixtures_path: Path = None) -> pd.DataFrame:
    """
    Build feature vectors for all WC 2026 group stage fixtures.
    Used to generate inference inputs for the full tournament.
    """
    if fixtures_path is None:
        fixtures_path = PROCESSED_DIR / "wc2026_fixtures.csv"

    if not fixtures_path.exists():
        raise FileNotFoundError(f"Run fetch_wc2026_fixtures.py first: {fixtures_path}")

    fixtures = pd.read_csv(fixtures_path)
    print(f"[match_features] Building features for {len(fixtures)} fixtures...")

    records = []
    errors  = []

    for _, row in fixtures.iterrows():
        # Handle both column naming conventions
        team_a = str(row.get("team1", row.get("home_team", row.get("team_a", "")))).strip()
        team_b = str(row.get("team2", row.get("away_team", row.get("team_b", "")))).strip()

        if not team_a or not team_b:
            continue

        try:
            feats = build_match_features(team_a, team_b)
            feats["team_a"]     = team_a
            feats["team_b"]     = team_b
            feats["stage"]      = row.get("stage", "group")
            feats["match_date"] = row.get("date", "")
            feats["group"]      = row.get("group", "")
            records.append(feats)
        except Exception as e:
            errors.append(f"{team_a} vs {team_b}: {e}")

    if errors:
        print(f"[match_features] {len(errors)} errors:")
        for err in errors[:5]:
            print(f"  {err}")

    df = pd.DataFrame(records)
    print(f"[match_features] Done. {len(df)} feature vectors built.")
    return df


@lru_cache(maxsize=None)
def get_cached_match_features(team_a: str, team_b: str) -> dict:
    """Cached wrapper for API — avoids recomputing on every request."""
    return build_match_features(team_a, team_b)


if __name__ == "__main__":
    print("=== Testing match_features.py ===\n")

    test_matches = [
        ("France",    "Argentina"),
        ("England",   "Brazil"),
        ("Spain",     "Germany"),
        ("Morocco",   "Portugal"),
        ("Japan",     "United States"),
    ]

    for team_a, team_b in test_matches:
        print(f"\n--- {team_a} vs {team_b} ---")
        try:
            feats = build_match_features(team_a, team_b)
            for k, v in feats.items():
                print(f"  {k:<30} {v}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n\n=== Building full fixture feature table ===")
    try:
        df = build_features_from_fixtures()
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"ERROR: {e}")