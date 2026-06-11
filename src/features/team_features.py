# src/features/team_features.py
#
# Builds team-level features for each match.
# Called by match_features.py to produce the final model input matrix.

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# ---------------------------------------------------------------------------
# Canonical team names = wc2026_groups.csv / wc2026_fixtures.csv spelling
# ("Côte d'Ivoire", "Curaçao"). All other data sources are aliased to this
# spelling at lookup time via the dicts below.
# ---------------------------------------------------------------------------

# Maps team names -> elo_ratings.csv names.
# - "Curaçao" -> "Curacao": the Kaggle Elo dataset spells it without the cedilla.
# - "Ivory Coast" -> "Côte d'Ivoire": international_results.csv (used during
#   training) spells this team "Ivory Coast", but elo_ratings.csv uses the
#   canonical "Côte d'Ivoire". Without this alias, every historical match
#   involving this team falls back to the mean Elo rating during training.
TEAM_NAME_ALIASES = {
    "Curaçao": "Curacao",
    "Ivory Coast": "Côte d'Ivoire",
}

def _resolve_team_name(team: str) -> str:
    return TEAM_NAME_ALIASES.get(team, team)

# Maps canonical names -> wc2026_squads.csv names (only Côte d'Ivoire
# differs — the squads file uses "Ivory Coast"). Used by player_features.py.
SQUAD_NAME_ALIASES = {
    "Côte d'Ivoire": "Ivory Coast",
}

def _resolve_squad_team_name(team: str) -> str:
    return SQUAD_NAME_ALIASES.get(team, team)

# FIFA ranking points for all 48 WC 2026 teams.
# Source: live FIFA Men's World Ranking tracker (football-ranking.com),
# snapshot dated 2026-06-10 — the day before the tournament-opener
# official update (2026-06-11). Refresh after the official update lands.
FIFA_RANKING_LAST_UPDATED = "2026-06-10"
FIFA_RANKING_POINTS = {
    "Argentina": 1876.11, "Spain": 1873.87, "France": 1870.69, "England": 1827.05,
    "Portugal": 1766.17, "Brazil": 1765.86, "Morocco": 1755.44, "Netherlands": 1753.57,
    "Belgium": 1742.23, "Germany": 1735.77, "Croatia": 1714.87, "Colombia": 1698.35,
    "Mexico": 1687.48, "Senegal": 1685.24, "Uruguay": 1673.07, "United States": 1671.24,
    "Japan": 1661.58, "Switzerland": 1650.07, "Iran": 1619.58, "Turkey": 1605.73,
    "Ecuador": 1598.51, "Austria": 1597.41, "South Korea": 1591.63, "Australia": 1579.34,
    "Algeria": 1571.04, "Egypt": 1562.37, "Canada": 1559.48, "Norway": 1557.44,
    "Côte d'Ivoire": 1540.87, "Panama": 1539.15, "Sweden": 1509.79, "Czech Republic": 1505.74,
    "Paraguay": 1505.35, "Scotland": 1503.34, "DR Congo": 1477.06, "Tunisia": 1476.40,
    "Uzbekistan": 1458.73, "Iraq": 1451.16, "Qatar": 1450.31, "South Africa": 1432.71,
    "Saudi Arabia": 1422.71, "Jordan": 1387.73, "Bosnia and Herzegovina": 1387.22,
    "Cape Verde": 1371.11, "Curaçao": 1294.77, "Haiti": 1293.09, "New Zealand": 1275.58,
    "Ghana": 1346.88,
}

MAX_FIFA_POINTS = 2000.0


def get_fifa_ranking_feature(team: str) -> dict:
    """
    Normalised FIFA ranking points [0, 1].
    Argentina (1876) -> 0.938, New Zealand (1276) -> 0.638
    Used as a direct feature in match prediction — encodes
    current team strength as officially recognised by FIFA.
    """
    pts = FIFA_RANKING_POINTS.get(team, 1400.0)
    return {
        "fifa_points":      pts,
        "fifa_points_norm": round(pts / MAX_FIFA_POINTS, 4),
    }


# How many recent matches to use for form calculation
FORM_WINDOW = 10

# Minimum H2H matches required to use H2H win rate as a feature
# Below this threshold we fall back to Elo-based expected win rate
MIN_H2H_MATCHES = 3

# Tournament experience: how many WC matches is "experienced"
# Used to normalise the experience score to 0-1
MAX_WC_MATCHES = 100


@lru_cache(maxsize=1)
def load_elo() -> pd.DataFrame:
    """
    Load Elo ratings with momentum.
    Cached — we call this many times during feature generation,
    no need to re-read the CSV each time.
    """
    path = PROCESSED_DIR / "elo_ratings.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run fetch_elo.py first: {path}")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_results() -> pd.DataFrame:
    """
    Load international match results.
    Cached for same reason as load_elo.
    """
    path = PROCESSED_DIR / "international_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run fetch_international_results.py first: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_competitive_results() -> pd.DataFrame:
    """
    load_results() with friendlies filtered out, cached.

    get_team_recent_form() is called twice per training row (once per team)
    for ~3000 training rows, and previously re-ran the
    `df[df["tournament_type"] != "friendly"]` filter on the full ~21k-row
    table every single call. Caching the filtered frame once cuts that
    redundant work without changing the result.
    """
    df = load_results()
    return df[df["tournament_type"] != "friendly"].copy()


def get_elo_features(team_a: str, team_b: str) -> dict:
    """
    Feature 1: Elo differential
    Feature 2: Elo momentum differential

    Elo differential: R_a - R_b
        Positive = team_a is stronger
        Negative = team_b is stronger
        A 200-point difference = team_a wins ~76% of the time

    Momentum differential: mom_a - mom_b
        Captures which team is trending up vs down
        Independent of absolute rating — a 1800-rated team
        with +80 momentum may be more dangerous than a 1900-rated
        team with -50 momentum

    Interview answer: "Elo differential is our single strongest feature.
    It explains ~65% of variance in match outcomes on its own. We add
    momentum because Elo ratings lag — a team that just won 8 in a row
    won't have fully updated yet."
    """
    df_elo = load_elo()

    def get_team_elo(team: str) -> tuple:
        team = TEAM_NAME_ALIASES.get(team, team)
        row = df_elo[df_elo["team"] == team]
        if row.empty:
            # Unknown team — assign average Elo
            print(f"  [team_features] WARNING: {team} not in Elo table, using mean")
            return df_elo["elo"].mean(), 0.0
        return float(row["elo"].iloc[0]), float(row["elo_momentum"].iloc[0])

    elo_a, mom_a = get_team_elo(team_a)
    elo_b, mom_b = get_team_elo(team_b)

    return {
        "elo_a": elo_a,
        "elo_b": elo_b,
        "elo_diff": round(elo_a - elo_b, 1),
        "elo_win_prob_a": round(1 / (1 + 10 ** ((elo_b - elo_a) / 400)), 4),
        "momentum_a": mom_a,
        "momentum_b": mom_b,
        "momentum_diff": round(mom_a - mom_b, 1),
    }


@lru_cache(maxsize=1)
def _team_match_history_index() -> dict:
    """
    Precomputed per-team match histories from load_competitive_results(),
    each sorted by date with columns [date, team_goals, opp_goals, opponent]
    from that team's perspective.

    get_team_recent_form() is called twice per training row (~6000 times
    across the ~2900-row training set) and previously rebuilt this
    home/away split + concat from the full competitive-results table on
    every call. Building it once per team here yields identical per-team
    match sequences, just without the redundant work.
    """
    df = load_competitive_results()

    home = df.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "team_goals", "away_score": "opp_goals",
    })[["team", "date", "team_goals", "opp_goals", "opponent"]]

    away = df.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "team_goals", "home_score": "opp_goals",
    })[["team", "date", "team_goals", "opp_goals", "opponent"]]

    combined = pd.concat([home, away]).sort_values("date")
    return {team: group.drop(columns="team") for team, group in combined.groupby("team")}


@lru_cache(maxsize=1)
def _elo_lookup() -> dict:
    """team -> elo rating, precomputed once for opponent-strength weighting."""
    df_elo = load_elo()
    return dict(zip(df_elo["team"], df_elo["elo"]))


def get_team_recent_form(team: str, before_date: pd.Timestamp = None) -> dict:
    """
    Elo-weighted recent form over last FORM_WINDOW competitive matches.
    A win against Andorra (Elo 1100) counts less than a win against Germany (Elo 1923).
    Weight = opponent_elo / mean_elo, clipped to [0.5, 1.5].
    This fixes the CONMEBOL/UEFA confederation strength imbalance.
    """
    df_elo = load_elo()
    mean_elo = float(df_elo["elo"].mean())
    elo_lookup = _elo_lookup()

    matches = _team_match_history_index().get(team)

    if matches is not None and before_date is not None:
        matches = matches[matches["date"] < before_date]

    recent = matches.tail(FORM_WINDOW).copy() if matches is not None else matches

    if recent is None or recent.empty:
        return {"ppg": 1.0, "goals_scored_pg": 1.2, "goals_conceded_pg": 1.2,
                "form_matches": 0, "win_rate": 0.33, "clean_sheet_rate": 0.2}

    def opp_weight(opp_name):
        resolved = TEAM_NAME_ALIASES.get(opp_name, opp_name)
        opp_elo = elo_lookup.get(resolved, mean_elo)
        return max(0.5, min(1.5, opp_elo / mean_elo))

    recent["opp_weight"]  = recent["opponent"].apply(opp_weight)
    recent["points"]      = recent.apply(
        lambda r: 3 if r["team_goals"] > r["opp_goals"]
        else (1 if r["team_goals"] == r["opp_goals"] else 0), axis=1)
    recent["win"]         = (recent["team_goals"] > recent["opp_goals"]).astype(int)
    recent["clean_sheet"] = (recent["opp_goals"] == 0).astype(int)

    total_weight = recent["opp_weight"].sum()
    w_ppg        = (recent["points"] * recent["opp_weight"]).sum() / total_weight
    w_win_rate   = (recent["win"]    * recent["opp_weight"]).sum() / total_weight
    n = len(recent)

    return {
        "ppg":               round(w_ppg, 3),
        "goals_scored_pg":   round(recent["team_goals"].sum() / n, 3),
        "goals_conceded_pg": round(recent["opp_goals"].sum()  / n, 3),
        "form_matches":      n,
        "win_rate":          round(w_win_rate, 3),
        "clean_sheet_rate":  round(recent["clean_sheet"].sum() / n, 3),
    }

@lru_cache(maxsize=1)
def _h2h_index() -> dict:
    """
    Precomputed mapping of frozenset({team_a, team_b}) -> all matches between
    that pair, built once from load_results().

    get_h2h_features() is called twice per training row (~6000 times across
    the ~2900-row training set) and previously re-scanned the full ~21k-row
    results table with a boolean mask on every call. Grouping once here and
    doing a dict lookup instead is purely a performance change — the matches
    returned for a given pair are identical to the old boolean-mask filter.
    """
    df = load_results()
    pair_keys = df.apply(lambda r: frozenset((r["home_team"], r["away_team"])), axis=1)
    return {key: group for key, group in df.groupby(pair_keys)}


def get_h2h_features(team_a: str, team_b: str, before_date: pd.Timestamp = None) -> dict:
    """
    Feature 6: Head-to-head record

    H2H win rate: how often has team_a beaten team_b historically?

    Why H2H matters:
        Some matchups have genuine psychological or tactical history.
        Germany vs Argentina (5 WC meetings), Brazil vs France —
        these patterns show up in data. But we only use H2H when we
        have enough matches (MIN_H2H_MATCHES=3), otherwise we fall
        back to the Elo-implied win probability.

    Interview answer: "H2H is a weak signal for most matchups because
    international teams meet rarely. We use it as a small correction
    on top of Elo, not as a primary feature. The MIN_H2H_MATCHES
    threshold prevents us from over-indexing on single-game samples."
    """
    # All matches between these two teams (precomputed pair index)
    pair_matches = _h2h_index().get(frozenset((team_a, team_b)))
    h2h = pair_matches.copy() if pair_matches is not None else load_results().iloc[0:0].copy()

    if before_date is not None:
        h2h = h2h[h2h["date"] < before_date]

    n = len(h2h)

    if n < MIN_H2H_MATCHES:
        # Not enough history — use Elo-implied probability as fallback
        elo_features = get_elo_features(team_a, team_b)
        win_prob_a = elo_features["elo_win_prob_a"]
        return {
            "h2h_matches": n,
            "h2h_win_rate_a": win_prob_a,
            "h2h_win_rate_b": round(1.0 - win_prob_a, 4),
            "h2h_goals_a_pg": None,
            "h2h_goals_b_pg": None,
            "h2h_sufficient": False,
        }

    # Compute win rate from team_a perspective
    wins_a = 0
    wins_b = 0
    goals_a = 0
    goals_b = 0

    for _, row in h2h.iterrows():
        if row["home_team"] == team_a:
            ga, gb = row["home_score"], row["away_score"]
        else:
            ga, gb = row["away_score"], row["home_score"]

        goals_a += ga
        goals_b += gb
        if ga > gb:
            wins_a += 1
        elif gb > ga:
            wins_b += 1

    return {
        "h2h_matches": n,
        "h2h_win_rate_a": round(wins_a / n, 3),
        "h2h_win_rate_b": round(wins_b / n, 3),
        "h2h_goals_a_pg": round(goals_a / n, 3),
        "h2h_goals_b_pg": round(goals_b / n, 3),
        "h2h_sufficient": True,
    }


def get_tournament_experience(team: str) -> dict:
    """
    Feature 7: Tournament experience

    How many World Cup matches has this team played?
    Normalised to 0-1 scale.

    Why experience matters:
        First-time or rare World Cup teams (Haiti, Uzbekistan) tend to
        underperform their Elo rating at tournaments. The pressure,
        schedule, and tactical level are different from qualifiers.
        Teams like Brazil and Germany who've played 100+ WC matches
        have institutional knowledge that doesn't show in Elo.

    Interview answer: "Experience is a regularisation feature — it
    shrinks our confidence in predictions for teams with little
    World Cup data. A 1700-rated team with 5 WC matches is less
    predictable than a 1700-rated team with 50 WC matches."
    """
    df = load_results()

    wc_matches = df[
        (df["tournament_type"] == "world_cup") &
        ((df["home_team"] == team) | (df["away_team"] == team))
    ]

    n = len(wc_matches)

    return {
        "wc_matches_played": n,
        "wc_experience_score": round(min(n / MAX_WC_MATCHES, 1.0), 3),
    }


def get_all_team_features(team_a: str, team_b: str,
                           match_date: pd.Timestamp = None,
                           neutral: bool = True) -> dict:
    """
    Master function — assembles all team features for a matchup.
    Called by match_features.py for every match in the dataset.

    Returns a flat dict of all features from team_a's perspective.
    The model sees team_a as the "home" team in the feature vector.
    """
    features = {}

    # Elo features
    features.update(get_elo_features(team_a, team_b))

    # Recent form — both teams
    form_a = get_team_recent_form(team_a, before_date=match_date)
    form_b = get_team_recent_form(team_b, before_date=match_date)

    for k, v in form_a.items():
        features[f"form_a_{k}"] = v
    for k, v in form_b.items():
        features[f"form_b_{k}"] = v

    # Form differentials — model learns from deltas, not absolute values
    features["ppg_diff"] = round(form_a["ppg"] - form_b["ppg"], 3)
    features["goals_scored_diff"] = round(
        form_a["goals_scored_pg"] - form_b["goals_scored_pg"], 3
    )
    features["goals_conceded_diff"] = round(
        form_a["goals_conceded_pg"] - form_b["goals_conceded_pg"], 3
    )

    # H2H
    features.update(get_h2h_features(team_a, team_b, before_date=match_date))

    # Tournament experience — both teams
    exp_a = get_tournament_experience(team_a)
    exp_b = get_tournament_experience(team_b)
    for k, v in exp_a.items():
        features[f"exp_a_{k}"] = v
    for k, v in exp_b.items():
        features[f"exp_b_{k}"] = v
    features["experience_diff"] = round(
        exp_a["wc_experience_score"] - exp_b["wc_experience_score"], 3
    )

    # Match context
    features["neutral_venue"] = int(neutral)
    features["team_a"] = team_a
    features["team_b"] = team_b

    return features


if __name__ == "__main__":
    # Quick test — France vs Argentina
    print("=== France vs Argentina ===")
    f = get_all_team_features("France", "Argentina", neutral=True)
    for k, v in f.items():
        print(f"  {k}: {v}")
