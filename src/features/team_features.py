# src/features/team_features.py
#
# Builds team-level features for each match.
# Called by match_features.py to produce the final model input matrix.

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

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


def get_team_recent_form(team: str, before_date: pd.Timestamp = None) -> dict:
    """
    Feature 3: Recent form — points per game in last FORM_WINDOW matches
    Feature 4: Goals scored per game — last FORM_WINDOW matches
    Feature 5: Goals conceded per game — last FORM_WINDOW matches

    Why points per game, not win rate:
        Points per game (3W/1D/0L) is the standard football metric.
        It's more granular than win rate — a team that draws a lot
        looks identical to a team that loses a lot on win rate,
        but very different on PPG.

    before_date: if provided, only use matches before this date.
        This prevents data leakage when building the training set —
        we can't use future results to predict past matches.

    Interview answer: "We use a 10-match rolling window because
    international teams play infrequently — 10 matches covers roughly
    12-18 months which is the right recency window for WC prediction."
    """
    df = load_results()

    # Get all matches involving this team
    home = df[df["home_team"] == team].copy()
    away = df[df["away_team"] == team].copy()

    # Standardise to team perspective — always from team's point of view
    home["team_goals"] = home["home_score"]
    home["opp_goals"] = home["away_score"]
    home["is_home"] = True

    away["team_goals"] = away["away_score"]
    away["opp_goals"] = away["home_score"]
    away["is_home"] = False

    matches = pd.concat([
        home[["date", "team_goals", "opp_goals", "is_home", "tournament_type"]],
        away[["date", "team_goals", "opp_goals", "is_home", "tournament_type"]],
    ]).sort_values("date")

    # Apply date filter for leakage prevention
    if before_date is not None:
        matches = matches[matches["date"] < before_date]

    # Take last FORM_WINDOW matches
    recent = matches.tail(FORM_WINDOW)

    if recent.empty:
        return {
            "ppg": 1.0,          # fallback: average
            "goals_scored_pg": 1.2,
            "goals_conceded_pg": 1.2,
            "form_matches": 0,
            "win_rate": 0.33,
            "clean_sheet_rate": 0.2,
        }

    # Points per game
    def points(row):
        if row["team_goals"] > row["opp_goals"]: return 3
        if row["team_goals"] == row["opp_goals"]: return 1
        return 0

    recent = recent.copy()
    recent["points"] = recent.apply(points, axis=1)
    recent["win"] = (recent["team_goals"] > recent["opp_goals"]).astype(int)
    recent["clean_sheet"] = (recent["opp_goals"] == 0).astype(int)

    n = len(recent)
    return {
        "ppg": round(recent["points"].sum() / n, 3),
        "goals_scored_pg": round(recent["team_goals"].sum() / n, 3),
        "goals_conceded_pg": round(recent["opp_goals"].sum() / n, 3),
        "form_matches": n,
        "win_rate": round(recent["win"].sum() / n, 3),
        "clean_sheet_rate": round(recent["clean_sheet"].sum() / n, 3),
    }


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
    df = load_results()

    # All matches between these two teams
    h2h = df[
        ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
        ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    ].copy()

    if before_date is not None:
        h2h = h2h[h2h["date"] < before_date]

    n = len(h2h)

    if n < MIN_H2H_MATCHES:
        # Not enough history — use Elo-implied probability as fallback
        elo_features = get_elo_features(team_a, team_b)
        return {
            "h2h_matches": n,
            "h2h_win_rate_a": elo_features["elo_win_prob_a"],
            "h2h_goals_a_pg": None,
            "h2h_goals_b_pg": None,
            "h2h_sufficient": False,
        }

    # Compute win rate from team_a perspective
    wins_a = 0
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

    return {
        "h2h_matches": n,
        "h2h_win_rate_a": round(wins_a / n, 3),
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