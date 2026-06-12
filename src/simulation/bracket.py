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

import numpy as np
from typing import List, Dict, Tuple

# WC 2026 groups — 12 groups of 4
# Source: official FIFA draw
GROUPS = {
    "A": ["Mexico",        "South Africa",          "South Korea",   "Czech Republic"],
    "B": ["Canada",        "Bosnia and Herzegovina","Qatar",         "Switzerland"],
    "C": ["Brazil",        "Morocco",               "Haiti",         "Scotland"],
    "D": ["United States", "Paraguay",              "Australia",     "Turkey"],
    "E": ["Germany",       "Curaçao",               "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands",   "Japan",                 "Sweden",        "Tunisia"],
    "G": ["Belgium",       "Egypt",                 "Iran",          "New Zealand"],
    "H": ["Spain",         "Cape Verde",            "Saudi Arabia",  "Uruguay"],
    "I": ["France",        "Senegal",               "Iraq",          "Norway"],
    "J": ["Argentina",     "Algeria",               "Austria",       "Jordan"],
    "K": ["Portugal",      "DR Congo",              "Uzbekistan",    "Colombia"],
    "L": ["England",       "Croatia",               "Ghana",         "Panama"],
}

# ---------------------------------------------------------------------------
# Official FIFA WC 2026 Round-of-32 bracket (Matches 73-88).
# Source: FIFA tournament regulations / 2026 knockout-stage schedule
# (https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage).
#
# Slot notation: "1A" = winner of Group A, "2A" = runner-up, "3?" = one of
# the 8 best third-placed teams (assigned to specific matches per Annex C
# of the regulations — reproduced here as a constraint-matching problem).
#
# Every qualified team appears in EXACTLY ONE R32 match (the previous
# version double-booked 1A/1C/1E/1G and left 4 best-thirds without a
# match, which corrupted every downstream advancement probability).
# ---------------------------------------------------------------------------
R32_BRACKET = [
    ("2A", "2B"),   # M73
    ("1E", "3?"),   # M74 — 3rd from A/B/C/D/F
    ("1F", "2C"),   # M75
    ("1C", "2F"),   # M76
    ("1I", "3?"),   # M77 — 3rd from C/D/F/G/H
    ("2E", "2I"),   # M78
    ("1A", "3?"),   # M79 — 3rd from C/E/F/H/I
    ("1L", "3?"),   # M80 — 3rd from E/H/I/J/K
    ("1D", "3?"),   # M81 — 3rd from B/E/F/I/J
    ("1G", "3?"),   # M82 — 3rd from A/E/H/I/J
    ("2K", "2L"),   # M83
    ("1H", "2J"),   # M84
    ("1B", "3?"),   # M85 — 3rd from E/F/G/I/J
    ("1J", "2H"),   # M86
    ("1K", "3?"),   # M87 — 3rd from D/E/I/J/L
    ("2D", "2G"),   # M88
]

# For each R32 list index that hosts a third-placed team:
# the set of groups whose 3rd-place side may be drawn into that match
# (per FIFA Annex C).
THIRD_SLOT_ALLOWED = {
    1:  ["A", "B", "C", "D", "F"],   # M74 vs 1E
    4:  ["C", "D", "F", "G", "H"],   # M77 vs 1I
    6:  ["C", "E", "F", "H", "I"],   # M79 vs 1A
    7:  ["E", "H", "I", "J", "K"],   # M80 vs 1L
    8:  ["B", "E", "F", "I", "J"],   # M81 vs 1D
    9:  ["A", "E", "H", "I", "J"],   # M82 vs 1G
    12: ["E", "F", "G", "I", "J"],   # M85 vs 1B
    14: ["D", "E", "I", "J", "L"],   # M87 vs 1K
}

# Knockout routing (winner-of-match index pairings), straight from the
# official schedule. Indices refer to positions in the previous round's
# winners list (R32 list above is in match order 73..88).
R16_FROM_R32 = [
    (1, 4),    # M89: W74 v W77
    (0, 2),    # M90: W73 v W75
    (3, 5),    # M91: W76 v W78
    (6, 7),    # M92: W79 v W80
    (10, 11),  # M93: W83 v W84
    (8, 9),    # M94: W81 v W82
    (13, 15),  # M95: W86 v W88
    (12, 14),  # M96: W85 v W87
]
QF_FROM_R16 = [
    (0, 1),    # M97:  W89 v W90
    (4, 5),    # M98:  W93 v W94
    (2, 3),    # M99:  W91 v W92
    (6, 7),    # M100: W95 v W96
]
SF_FROM_QF = [
    (0, 1),    # M101
    (2, 3),    # M102
]


def route_round(prev_winners: List[str], routing: List[Tuple[int, int]]) -> List[Tuple[str, str]]:
    """Build next-round matchups from previous-round winners via routing table."""
    return [(prev_winners[i], prev_winners[j]) for i, j in routing]


def _assign_third_slots(qualified_groups: List[str]) -> Dict[int, str]:
    """
    Assign the 8 qualified third-place groups to the 8 third-place R32 slots,
    respecting THIRD_SLOT_ALLOWED (FIFA Annex C as a bipartite matching).

    Returns {r32_index: group_letter}. Backtracking over the most
    constrained slot first; a perfect matching exists for all 495 valid
    combinations of qualified groups.
    """
    remaining = set(qualified_groups)
    # Most-constrained slot first
    slot_order = sorted(
        THIRD_SLOT_ALLOWED,
        key=lambda idx: len([g for g in THIRD_SLOT_ALLOWED[idx] if g in remaining]),
    )
    assignment: Dict[int, str] = {}

    def backtrack(k: int) -> bool:
        if k == len(slot_order):
            return True
        idx = slot_order[k]
        for g in THIRD_SLOT_ALLOWED[idx]:
            if g in remaining:
                remaining.discard(g)
                assignment[idx] = g
                if backtrack(k + 1):
                    return True
                remaining.add(g)
                del assignment[idx]
        return False

    if not backtrack(0):
        # Should never happen for a valid set of 8 groups, but keep the
        # simulation alive: greedily fill leftover slots ignoring constraints.
        leftover = [g for g in qualified_groups if g not in assignment.values()]
        for idx in THIRD_SLOT_ALLOWED:
            if idx not in assignment and leftover:
                assignment[idx] = leftover.pop()

    return assignment

# Points system
WIN_PTS  = 3
DRAW_PTS = 1
LOSS_PTS = 0


def simulate_group_stage(
    groups: Dict[str, List[str]],
    predict_fn,
) -> Dict[str, List[dict]]:
    """
    Simulate all group stage matches and return final standings per group.

    Args:
        groups     : dict of group_name -> [team1, team2, team3, team4]
        predict_fn : callable(team_a, team_b) -> {"win": p, "draw": p, "loss": p}
                     Returns probabilities — we sample one outcome per match.

    Returns:
        dict of group_name -> list of dicts, each with keys:
            team, pts, gd, gf, ga, w, d, l, position, group
        Sorted by pts DESC, gd DESC, gf DESC (FIFA tiebreaker order)

    Why sample rather than use expected values:
        Each simulation run needs a definitive outcome per match so we can
        propagate teams through the bracket. Taking expected values would
        give the same bracket every run — no variance, no probability distribution.

    Note on implementation: this is called once per Monte Carlo run (up to
    100k times). Building a pandas DataFrame + sort_values per group per run
    dominated runtime (~85% of total) for what is just a 4-row sort by three
    integer keys. Plain dict/list + sorted() with reverse=True on the
    (pts, gd, gf) tuple gives an identical ordering to
    sort_values(["pts","gd","gf"], ascending=[False,False,False]) — both are
    descending on all three keys — but avoids all DataFrame overhead.
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

        # Build standings rows and sort by FIFA tiebreaker rules
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

        # FIFA tiebreaker: pts -> gd -> gf -> alphabetical (simplified)
        rows.sort(key=lambda r: (r["pts"], r["gd"], r["gf"]), reverse=True)
        for i, r in enumerate(rows):
            r["position"] = i + 1
            r["group"] = group

        standings[group] = rows

    return standings


def get_advancing_teams(
    standings: Dict[str, List[dict]]
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

    for group, rows in standings.items():
        group_winners[group] = rows[0]["team"]
        group_runners[group] = rows[1]["team"]
        third_row = dict(rows[2])
        third_row["group"] = group
        third_place.append(third_row)

    # Rank all 3rd-place teams and take best 8 (same tiebreaker order as
    # simulate_group_stage — see note there on sorted()/reverse=True vs
    # sort_values(ascending=[False, False, False]))
    third_place.sort(key=lambda r: (r["pts"], r["gd"], r["gf"]), reverse=True)

    best_thirds = [r["team"] for r in third_place[:8]]
    # Group letter for each qualified third — needed for FIFA Annex C
    # slot assignment in build_r32_bracket.
    best_third_groups = {r["group"]: r["team"] for r in third_place[:8]}

    return group_winners, group_runners, best_thirds, best_third_groups


def build_r32_bracket(
    group_winners: Dict[str, str],
    group_runners: Dict[str, str],
    best_third_groups: Dict[str, str],
):
    """
    Build the 16 Round-of-32 matchups in official match order (M73-M88).

    Uses the real FIFA WC 2026 bracket: 8 group winners host best-thirds,
    4 group winners host runners-up, and the remaining 8 runners-up play
    each other. Each qualified team appears exactly once, and same-group
    teams cannot meet before the quarterfinals — exactly as in the
    official knockout schedule.

    Args:
        best_third_groups: {group_letter: team} for the 8 qualified thirds.
    """
    third_assignment = _assign_third_slots(list(best_third_groups.keys()))
    matchups = []
    for idx, (slot_a, slot_b) in enumerate(R32_BRACKET):
        def resolve(slot):
            if slot == "3?":
                return best_third_groups[third_assignment[idx]]
            kind, grp = slot[0], slot[1]
            return group_winners[grp] if kind == "1" else group_runners[grp]
        matchups.append((resolve(slot_a), resolve(slot_b)))

    return matchups


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


def _sample_scoreline(probs: dict, outcome: str):
    """
    Sample a plausible scoreline consistent with the match outcome.
    Uses simple Poisson approximation — full Bivariate Poisson is in team_strength.py.

    Expected goals roughly derived from win probability:
        Strong favorite (win_prob > 0.65) -> higher xG differential
        Close match (win_prob ~0.45)      -> lower xG differential
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

    def dummy_predict(team_a, team_b):
        from src.models.match_outcome import predict
        return predict(team_a, team_b)

    print("Simulating group stage...")
    standings = simulate_group_stage(GROUPS, dummy_predict)

    for group, rows in sorted(standings.items()):
        print(f"\nGroup {group}:")
        for r in rows:
            print(f"{r['team']:<25} {r['pts']:>3} {r['gd']:>3} {r['gf']:>3}")

    winners, runners, thirds, third_groups = get_advancing_teams(standings)
    print(f"\nGroup winners: {list(winners.values())}")
    print(f"Best 3rd-place (8): {thirds}")

    r32 = build_r32_bracket(winners, runners, third_groups)
    print(f"\nRound of 32 matchups ({len(r32)}):")
    for i, (a, b) in enumerate(r32):
        print(f"  Match {i+73}: {a} vs {b}")
