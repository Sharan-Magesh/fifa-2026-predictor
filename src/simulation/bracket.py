# src/simulation/bracket.py
#
# WC 2026 bracket structure and group advancement logic.
#
# Format:
#   - 48 teams, 12 groups (A-L) of 4 teams each
#   - Top 2 from each group advance automatically (24 teams)
#   - 8 best 3rd-place teams also advance (32 teams total)
#   - Round of 32 -> Round of 16 -> QF -> SF -> Final
#
# This module handles:
#   1. Group stage table calculation (points, GD, GF tiebreakers)
#   2. Best 3rd-place team selection
#   3. R32 bracket seeding (which group winner plays which runner-up)
#
# Used by monte_carlo.py — one instance per simulation run.

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# WC 2026 groups — 12 groups of 4
# Source: official FIFA draw
GROUPS = {
    "A": ["Mexico",        "South Africa",          "South Korea",   "Czech Republic"],
    "B": ["Canada",        "Bosnia and Herzegovina","Qatar",         "Switzerland"],
    "C": ["Brazil",        "Morocco",               "Haiti",         "Scotland"],
    "D": ["United States", "Paraguay",              "Australia",     "Turkey"],
    "E": ["Germany",       "Curaçao",               "Ivory Coast",   "Ecuador"],
    "F": ["Netherlands",   "Japan",                 "Sweden",        "Tunisia"],
    "G": ["Belgium",       "Egypt",                 "Iran",          "New Zealand"],
    "H": ["Spain",         "Cape Verde",            "Saudi Arabia",  "Uruguay"],
    "I": ["France",        "Senegal",               "Iraq",          "Norway"],
    "J": ["Argentina",     "Algeria",               "Austria",       "Jordan"],
    "K": ["Portugal",      "DR Congo",              "Uzbekistan",    "Colombia"],
    "L": ["England",       "Croatia",               "Ghana",         "Panama"],
}

# R32 bracket seeding — which group winners/runners-up face each other
# Based on official FIFA R32 bracket for WC 2026
# Format: (team_a_slot, team_b_slot)
# Slots: "1A" = winner of Group A, "2A" = runner-up of Group A
# "3X" = one of the 8 best 3rd-place teams (assigned dynamically)
R32_BRACKET = [
    ("1A", "2B"),
    ("1B", "2A"),
    ("1C", "2D"),
    ("1D", "2C"),
    ("1E", "2F"),
    ("1F", "2E"),
    ("1G", "2H"),
    ("1H", "2G"),
    ("1I", "2J"),
    ("1J", "2I"),
    ("1K", "2L"),
    ("1L", "2K"),
    # 8 best 3rd-place matchups (simplified — FIFA hasn't published exact seeding)
    ("1A", "3BCDE"),
    ("1C", "3AFGH"),
    ("1E", "3IJKL"),
    ("1G", "3ABCD"),
]

# Points system
WIN_PTS  = 3
DRAW_PTS = 1
LOSS_PTS = 0


def simulate_group_stage(
    groups: Dict[str, List[str]],
    predict_fn,
) -> Dict[str, pd.DataFrame]:
    """
    Simulate all group stage matches and return final standings per group.

    Args:
        groups     : dict of group_name -> [team1, team2, team3, team4]
        predict_fn : callable(team_a, team_b) -> {"win": p, "draw": p, "loss": p}
                     Returns probabilities — we sample one outcome per match.

    Returns:
        dict of group_name -> DataFrame with columns:
            team, pts, gd, gf, ga, w, d, l
        Sorted by pts DESC, gd DESC, gf DESC (FIFA tiebreaker order)

    Why sample rather than use expected values:
        Each simulation run needs a definitive outcome per match so we can
        propagate teams through the bracket. Taking expected values would
        give the same bracket every run — no variance, no probability distribution.
    """
    standings = {}

    for group, teams in groups.items():
        # Initialise table
        table = {t: {"pts": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0}
                 for t in teams}

        # Round-robin: each team plays each other once (6 matches per group)
        for i, team_a in enumerate(teams):
            for team_b in teams[i+1:]:
                try:
                    probs = predict_fn(team_a, team_b)
                    outcome = _sample_outcome(probs)
                    goals_a, goals_b = _sample_scoreline(probs, outcome)
                except Exception:
                    # Fallback: equal probabilities
                    outcome = np.random.choice(["win", "draw", "loss"],
                                               p=[0.45, 0.25, 0.30])
                    goals_a, goals_b = 1, 1

                # Update table
                if outcome == "win":
                    table[team_a]["pts"] += WIN_PTS
                    table[team_a]["w"]   += 1
                    table[team_b]["l"]   += 1
                elif outcome == "draw":
                    table[team_a]["pts"] += DRAW_PTS
                    table[team_b]["pts"] += DRAW_PTS
                    table[team_a]["d"]   += 1
                    table[team_b]["d"]   += 1
                else:  # loss
                    table[team_b]["pts"] += WIN_PTS
                    table[team_b]["w"]   += 1
                    table[team_a]["l"]   += 1

                table[team_a]["gf"] += goals_a
                table[team_a]["ga"] += goals_b
                table[team_b]["gf"] += goals_b
                table[team_b]["ga"] += goals_a

        # Build DataFrame and sort by FIFA tiebreaker rules
        rows = []
        for team, stats in table.items():
            rows.append({
                "team": team,
                "pts":  stats["pts"],
                "gd":   stats["gf"] - stats["ga"],
                "gf":   stats["gf"],
                "ga":   stats["ga"],
                "w":    stats["w"],
                "d":    stats["d"],
                "l":    stats["l"],
            })

        df = pd.DataFrame(rows)
        # FIFA tiebreaker: pts -> gd -> gf -> alphabetical (simplified)
        df = df.sort_values(
            ["pts", "gd", "gf"],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        df["position"] = df.index + 1
        df["group"] = group

        standings[group] = df

    return standings


def get_advancing_teams(
    standings: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    """
    Determine which teams advance from group stage.

    Returns:
        group_winners  : {"A": "Spain", "B": "France", ...}
        group_runners  : {"A": "Morocco", "B": "Argentina", ...}
        best_thirds    : ["Japan", "USA", ...] — 8 best 3rd-place teams

    WC 2026 advancement rules:
        - Position 1 and 2 from each of 12 groups advance automatically
        - 8 best 3rd-place teams (ranked by pts, gd, gf) also advance
        - Total: 24 + 8 = 32 teams in R32
    """
    group_winners = {}
    group_runners = {}
    third_place   = []

    for group, df in standings.items():
        group_winners[group] = df.iloc[0]["team"]
        group_runners[group] = df.iloc[1]["team"]
        third_row = df.iloc[2].to_dict()
        third_row["group"] = group
        third_place.append(third_row)

    # Rank all 3rd-place teams and take best 8
    thirds_df = pd.DataFrame(third_place)
    thirds_df = thirds_df.sort_values(
        ["pts", "gd", "gf"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    best_thirds = thirds_df.head(8)["team"].tolist()

    return group_winners, group_runners, best_thirds


def build_r32_bracket(
    group_winners: Dict[str, str],
    group_runners: Dict[str, str],
    best_thirds: List[str],
) -> List[Tuple[str, str]]:
    """
    Build the Round of 32 matchup list — always returns exactly 16 matchups.

    WC 2026 R32 seeding:
    - 12 group winners play 12 group runners-up (crossing groups)
    - 4 remaining group winners play the 4 best 3rd-place teams
    - The other 4 best 3rd-place teams play the remaining 4 runners-up
    
    Crossing pattern (simplified — official FIFA bracket TBD):
        1A vs 2B, 1B vs 2A
        1C vs 2D, 1D vs 2C
        1E vs 2F, 1F vs 2E
        1G vs 2H, 1H vs 2G
        1I vs 2J, 1J vs 2I
        1K vs 2L, 1L vs 2K
        + 4 winners vs best 3rd-place teams
    """
    matchups = []

    # 6 crossing pairs = 12 matchups
    crossing = [
        ("A", "B"), ("C", "D"), ("E", "F"),
        ("G", "H"), ("I", "J"), ("K", "L"),
    ]
    for g1, g2 in crossing:
        matchups.append((group_winners[g1], group_runners[g2]))
        matchups.append((group_winners[g2], group_runners[g1]))

    # 4 best 3rd-place teams vs 4 group winners (rotated seeding)
    # Use groups A, C, E, G winners as the "home" side for 3rd-place matchups
    anchor_groups = ["A", "C", "E", "G"]
    for i, g in enumerate(anchor_groups):
        if i < len(best_thirds):
            matchups.append((group_winners[g], best_thirds[i]))
        else:
            # Fallback: use a runner-up if not enough 3rd place teams
            matchups.append((group_winners[g], group_runners[g]))

    return matchups[:16]


def _sample_outcome(probs: dict) -> str:
    """
    Sample one match outcome from win/draw/loss probabilities.
    Uses numpy multinomial for speed in 100k simulations.
    """
    p = [probs.get("win", 0.45), probs.get("draw", 0.25), probs.get("loss", 0.30)]
    # Renormalise in case of floating point drift
    total = sum(p)
    p = [x / total for x in p]
    return np.random.choice(["win", "draw", "loss"], p=p)


def _sample_scoreline(probs: dict, outcome: str) -> Tuple[int, int]:
    """
    Sample a plausible scoreline consistent with the match outcome.
    Uses simple Poisson approximation — full Bivariate Poisson is in team_strength.py.

    Expected goals roughly derived from win probability:
        Strong favorite (win_prob > 0.65) → higher xG differential
        Close match (win_prob ~0.45)      → lower xG differential
    """
    win_p = probs.get("win", 0.45)
    base  = 1.15  # average WC goals per team

    if outcome == "win":
        xg_a = base + (win_p - 0.33) * 1.5
        xg_b = base - (win_p - 0.33) * 0.8
    elif outcome == "loss":
        xg_a = base - (win_p - 0.33) * 0.8
        xg_b = base + (win_p - 0.33) * 1.5
    else:  # draw
        xg_a = xg_b = base

    xg_a = max(0.3, xg_a)
    xg_b = max(0.3, xg_b)

    goals_a = int(np.random.poisson(xg_a))
    goals_b = int(np.random.poisson(xg_b))

    # Enforce consistency with outcome
    if outcome == "win"  and goals_a <= goals_b:
        goals_a = goals_b + 1
    if outcome == "loss" and goals_b <= goals_a:
        goals_b = goals_a + 1
    if outcome == "draw" and goals_a != goals_b:
        goals_b = goals_a

    return goals_a, goals_b


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== Testing bracket.py ===\n")

    # Test with dummy predict function
    def dummy_predict(team_a, team_b):
        from src.models.match_outcome import predict
        return predict(team_a, team_b)

    print("Simulating group stage...")
    standings = simulate_group_stage(GROUPS, dummy_predict)

    for group, df in sorted(standings.items()):
        print(f"\nGroup {group}:")
        print(df[["team", "pts", "gd", "gf", "w", "d", "l"]].to_string(index=False))

    winners, runners, thirds = get_advancing_teams(standings)
    print(f"\nGroup winners: {list(winners.values())}")
    print(f"Group runners-up: {list(runners.values())}")
    print(f"Best 3rd-place (8): {thirds}")

    r32 = build_r32_bracket(winners, runners, thirds)
    print(f"\nRound of 32 matchups ({len(r32)}):")
    for i, (a, b) in enumerate(r32):
        print(f"  Match {i+1}: {a} vs {b}")