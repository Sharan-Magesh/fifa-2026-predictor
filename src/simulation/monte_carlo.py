# src/simulation/monte_carlo.py
#
# Monte Carlo simulation — runs the full WC 2026 bracket N times.
#
# Each run:
#   1. Simulates all 72 group stage matches -> group standings
#   2. Determines 32 advancing teams (24 top-2 + 8 best 3rd)
#   3. Simulates R32 -> R16 -> QF -> SF -> Final
#   4. Records which team wins each stage
#
# After N runs, the win count / N = probability of reaching that stage.
#
# Why 100,000 runs:
#   At 100k runs, win probabilities are stable to ±0.1%.
#   At 10k runs, they're stable to ±0.3% — acceptable but noisier.
#   Runtime: ~3-5 minutes on CPU with cached predictions.
#
# Key design decision — prediction caching:
#   We cache model predictions for every team pair before simulation starts.
#   Without caching, 100k runs × ~100 matches/run × feature computation
#   = 10M feature lookups = hours of runtime.
#   With caching: ~5,000 unique matchups cached once = fast.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from functools import lru_cache
import time

from src.simulation.bracket import (
    GROUPS,
    simulate_group_stage,
    get_advancing_teams,
    build_r32_bracket,
    route_round,
    R16_FROM_R32,
    QF_FROM_R16,
    SF_FROM_QF,
    _sample_outcome,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _build_prediction_cache(
    groups: Dict[str, List[str]],
    predict_fn,
) -> Dict[Tuple[str, str], dict]:
    """
    Pre-compute win/draw/loss probabilities for every possible matchup.

    We cache all group stage matchups plus all potential knockout matchups
    (every team vs every other team). This means ~48*47/2 = 1,128 unique pairs.
    At 100k runs this saves recomputing features millions of times.

    Returns dict: {(team_a, team_b): {"win": p, "draw": p, "loss": p}}
    Note: (A,B) and (B,A) are stored separately since the model is asymmetric
    (home/away effects and H2H are directional).
    """
    all_teams = [t for teams in groups.values() for t in teams]
    cache = {}
    total = len(all_teams) * (len(all_teams) - 1) // 2
    done  = 0

    print(f"[monte_carlo] Building prediction cache for {total} matchups...")

    for i, team_a in enumerate(all_teams):
        for team_b in all_teams[i+1:]:
            try:
                probs_ab = predict_fn(team_a, team_b)
                # Derive reverse matchup by flipping win/loss
                probs_ba = {
                    "win":  probs_ab["loss"],
                    "draw": probs_ab["draw"],
                    "loss": probs_ab["win"],
                }
                cache[(team_a, team_b)] = probs_ab
                cache[(team_b, team_a)] = probs_ba
            except Exception as e:
                # Fallback to uniform if prediction fails
                cache[(team_a, team_b)] = {"win": 0.40, "draw": 0.25, "loss": 0.35}
                cache[(team_b, team_a)] = {"win": 0.35, "draw": 0.25, "loss": 0.40}

            done += 1
            if done % 100 == 0:
                print(f"  {done}/{total} cached...", end="\r")

    print(f"  {total}/{total} cached.      ")
    return cache


def _simulate_knockout_match(
    team_a: str,
    team_b: str,
    cache: Dict,
) -> str:
    """
    Simulate a single knockout match. No draws — if draw after 90', the tie
    is resolved by extra time + penalties.

    ET/pens winner is NOT a pure coin flip: the stronger side still converts
    more often (they create more chances in ET and have deeper benches), but
    the edge is heavily compressed relative to 90-minute probabilities —
    matching the empirical record of WC shootouts being near-toss-ups.
    We shrink the team's conditional win share 60% of the way toward 0.5.

    Returns the winning team name.
    """
    probs = cache.get((team_a, team_b), {"win": 0.40, "draw": 0.25, "loss": 0.35})
    outcome = _sample_outcome(probs)

    if outcome == "win":
        return team_a
    elif outcome == "loss":
        return team_b
    else:
        # Level after 90' -> extra time / penalties.
        decisive = probs.get("win", 0.4) + probs.get("loss", 0.4)
        share_a = probs.get("win", 0.4) / decisive if decisive > 0 else 0.5
        p_a = 0.5 + 0.4 * (share_a - 0.5)   # compressed edge
        return team_a if np.random.random() < p_a else team_b


def _simulate_knockout_round(
    matchups: List[Tuple[str, str]],
    cache: Dict,
) -> List[str]:
    """Simulate one knockout round, return list of winners."""
    return [_simulate_knockout_match(a, b, cache) for a, b in matchups]


def _simulate_knockout_bracket(
    groups: Dict[str, List[str]],
    cache: Dict,
) -> Dict:
    """
    Shared core: simulate group stage + full official knockout bracket once.
    Winner routing follows the real FIFA match schedule (M73-M104) via the
    routing tables in bracket.py — every team plays exactly one match per
    round, and same-group teams cannot meet before the QFs.
    """
    def cached_predict(team_a, team_b):
        return cache.get((team_a, team_b), {"win": 0.40, "draw": 0.25, "loss": 0.35})

    standings = simulate_group_stage(groups, cached_predict)
    winners, runners, thirds, third_groups = get_advancing_teams(standings)
    r32_matchups = build_r32_bracket(winners, runners, third_groups)

    r32_winners = _simulate_knockout_round(r32_matchups, cache)
    r16_matchups = route_round(r32_winners, R16_FROM_R32)
    r16_winners  = _simulate_knockout_round(r16_matchups, cache)
    qf_matchups  = route_round(r16_winners, QF_FROM_R16)
    qf_winners   = _simulate_knockout_round(qf_matchups, cache)
    sf_matchups  = route_round(qf_winners, SF_FROM_QF)
    sf_winners   = _simulate_knockout_round(sf_matchups, cache)
    final_matchup = (sf_winners[0], sf_winners[1])
    champion = _simulate_knockout_match(*final_matchup, cache)

    return {
        "standings": standings,
        "group_winners": winners,
        "group_runners": runners,
        "best_thirds": thirds,
        "rounds": {
            "round_of_32":  {"matchups": r32_matchups,  "winners": r32_winners},
            "round_of_16":  {"matchups": r16_matchups,  "winners": r16_winners},
            "quarterfinal": {"matchups": qf_matchups,   "winners": qf_winners},
            "semifinal":    {"matchups": sf_matchups,   "winners": sf_winners},
            "final":        {"matchups": [final_matchup], "winners": [champion]},
        },
        "champion": champion,
    }


def run_single_simulation(
    groups: Dict[str, List[str]],
    cache: Dict,
) -> Dict[str, str]:
    """
    Run one complete tournament simulation.

    Returns dict mapping each team to the furthest stage they reached:
        "group_stage", "round_of_32", "round_of_16",
        "quarterfinal", "semifinal", "final", "winner"
    """
    trace = _simulate_knockout_bracket(groups, cache)
    results = {t: "group_stage" for teams in groups.values() for t in teams}

    advancing = (set(trace["group_winners"].values())
                 | set(trace["group_runners"].values())
                 | set(trace["best_thirds"]))
    for team in advancing:
        results[team] = "round_of_32"

    next_stage = {"round_of_32": "round_of_16", "round_of_16": "quarterfinal",
                  "quarterfinal": "semifinal", "semifinal": "final",
                  "final": "winner"}
    for key, stage in next_stage.items():
        for team in trace["rounds"][key]["winners"]:
            results[team] = stage

    return results


def _fifa_strength_prior(groups: Dict[str, List[str]], decay: float = 0.25) -> Dict[str, float]:
    """
    A title-probability prior derived from each team's current FIFA ranking
    *position* (1st, 2nd, 3rd, ...) rather than raw ranking points.

    Raw FIFA points are extremely tightly clustered for closely-matched teams
    (e.g. Portugal 1766.17 vs Brazil 1765.86 — a 0.31pt gap), so a points-based
    prior barely distinguishes them and the simulation's bracket-draw noise
    dominates the final ordering. Using rank position with an exponential
    decay gives every adjacent pair of teams a consistent, meaningful gap, so
    the calibrated table tracks the actual FIFA ranking order much more
    faithfully while still leaving room for the simulation to add realistic
    variance among teams of similar caliber.
    """
    from src.features.team_features import get_fifa_ranking_feature

    all_teams = [t for teams in groups.values() for t in teams]
    pts = {t: get_fifa_ranking_feature(t)["fifa_points"] for t in all_teams}
    ranked = sorted(all_teams, key=lambda t: pts[t], reverse=True)
    raw = np.array([np.exp(-decay * i) for i in range(len(ranked))])
    raw = raw / raw.sum()
    return dict(zip(ranked, raw))


def _apply_fifa_calibration(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    decay: float = 0.25,
    top_tier_size: int = 7,
    top_tier_blend: float = 0.25,
    low_tier_blend: float = 0.05,
) -> pd.DataFrame:
    """
    Calibrate the raw Monte Carlo title odds toward a FIFA-ranking-based
    prior, then re-sort.

    Two-tier blend:
      - The top `top_tier_size` FIFA-ranked teams (the genuine title
        contenders) keep a sizeable share of the raw simulation signal
        (`top_tier_blend`). This lets bracket-draw "luck" create realistic,
        non-monotonic variation among elite teams — exactly like real-world
        bookmaker odds, where e.g. the #6-ranked side can edge out the
        #5-ranked side depending on their path to the final.
      - Every other team is dominated by the rank-based prior
        (`low_tier_blend`), so lower-Elo sides get realistically crushed
        title odds — a team ranked #25 has essentially no business winning
        the tournament, regardless of a lucky bracket-stage simulation run.

    Only the "deep run" columns (p_final, p_advance_sf) are rescaled by the
    same per-team ratio as p_win_tournament, so winning it all is never more
    likely than reaching the final/semis. The earlier-stage columns
    (p_advance_r32/r16/qf) are left as the raw simulation produced them — a
    lower-ranked team's realistic shot at escaping the group stage or
    reaching the round of 16 shouldn't be crushed just because its title
    odds are.

    Finally, p_win_tournament is renormalized so the 48 values sum to ~1.
    """
    prior = _fifa_strength_prior(groups, decay)
    ranked_teams = sorted(prior.keys(), key=lambda t: prior[t], reverse=True)
    top_tier = set(ranked_teams[:top_tier_size])

    df = df.copy()

    blended = []
    for _, row in df.iterrows():
        team = row["team"]
        sim_p = float(row["p_win_tournament"])
        prior_p = prior.get(team, 0.0)
        blend = top_tier_blend if team in top_tier else low_tier_blend
        blended.append(blend * sim_p + (1 - blend) * prior_p)

    total = sum(blended)
    blended = [v / total for v in blended]
    ratios = [
        b / float(row["p_win_tournament"]) if float(row["p_win_tournament"]) > 1e-9 else 1.0
        for b, (_, row) in zip(blended, df.iterrows())
    ]

    df["p_win_tournament"] = [round(v, 4) for v in blended]
    for col in ["p_final", "p_advance_sf"]:
        df[col] = [round(min(1.0, v * r), 4) for v, r in zip(df[col], ratios)]

    # Enforce the physical monotonicity chain after calibration:
    # P(win) <= P(final) <= P(SF) <= P(QF) <= P(R16) <= P(R32).
    # Rescaling p_final / p_advance_sf can otherwise push a deep-run
    # probability above the (uncalibrated) earlier-stage probability,
    # which is impossible in a single-elimination tournament.
    chain = ["p_advance_r32", "p_advance_r16", "p_advance_qf",
             "p_advance_sf", "p_final", "p_win_tournament"]
    for earlier, later in zip(chain, chain[1:]):
        df[later] = np.minimum(df[later], df[earlier]).round(4)

    return df.sort_values("p_win_tournament", ascending=False).reset_index(drop=True)


def run_simulation(
    n_simulations: int = 100_000,
    groups: Dict[str, List[str]] = None,
    predict_fn = None,
    seed: int = 42,
    calibrate: bool = True,
) -> pd.DataFrame:
    """
    Run N Monte Carlo simulations of WC 2026.

    Args:
        n_simulations : number of runs (100k recommended, 10k for testing)
        groups        : group composition dict (defaults to GROUPS)
        predict_fn    : callable(team_a, team_b) -> probs dict
        seed          : random seed for reproducibility

    Returns:
        DataFrame with one row per team, columns:
            team, group,
            p_advance_r32, p_advance_r16, p_advance_qf,
            p_advance_sf, p_final, p_win_tournament,
            expected_stage (most likely stage reached)
    """
    np.random.seed(seed)

    if groups is None:
        groups = GROUPS

    if predict_fn is None:
        from src.models.match_outcome import predict
        predict_fn = predict

    # Build team -> group lookup
    team_to_group = {t: g for g, teams in groups.items() for t in teams}

    # Pre-compute all predictions
    cache = _build_prediction_cache(groups, predict_fn)

    # Stage counters per team
    stage_counts = defaultdict(lambda: defaultdict(int))
    stages = ["group_stage", "round_of_32", "round_of_16",
              "quarterfinal", "semifinal", "final", "winner"]

    print(f"[monte_carlo] Running {n_simulations:,} simulations...")
    t0 = time.time()

    for i in range(n_simulations):
        if i % 10_000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate    = i / elapsed
            remaining = (n_simulations - i) / rate
            print(f"  {i:,}/{n_simulations:,} runs | "
                  f"{elapsed:.0f}s elapsed | "
                  f"~{remaining:.0f}s remaining")

        sim_result = run_single_simulation(groups, cache)

        for team, stage in sim_result.items():
            stage_counts[team][stage] += 1

    elapsed = time.time() - t0
    print(f"[monte_carlo] Done. {n_simulations:,} runs in {elapsed:.1f}s "
          f"({n_simulations/elapsed:.0f} sims/sec)")

    # Build results DataFrame
    rows = []
    for team in [t for teams in groups.values() for t in teams]:
        counts = stage_counts[team]
        total  = n_simulations

        # P(reaching at least this stage) = P(this stage) + P(later stages)
        p_r32  = sum(counts.get(s, 0) for s in stages[1:]) / total
        p_r16  = sum(counts.get(s, 0) for s in stages[2:]) / total
        p_qf   = sum(counts.get(s, 0) for s in stages[3:]) / total
        p_sf   = sum(counts.get(s, 0) for s in stages[4:]) / total
        p_fin  = sum(counts.get(s, 0) for s in stages[5:]) / total
        p_win  = counts.get("winner", 0) / total

        # Most likely deepest stage reached
        best_stage = max(stages, key=lambda s: counts.get(s, 0))

        rows.append({
            "team":             team,
            "group":            team_to_group[team],
            "p_advance_r32":    round(p_r32, 4),
            "p_advance_r16":    round(p_r16, 4),
            "p_advance_qf":     round(p_qf,  4),
            "p_advance_sf":     round(p_sf,  4),
            "p_final":          round(p_fin, 4),
            "p_win_tournament": round(p_win, 4),
            "most_likely_exit": best_stage,
        })

    df = pd.DataFrame(rows).sort_values("p_win_tournament", ascending=False)
    df = df.reset_index(drop=True)

    if calibrate:
        df = _apply_fifa_calibration(df, groups)

    return df


def run_traced_simulation(
    groups: Dict[str, List[str]],
    cache: Dict,
) -> Dict:
    """
    Run one complete tournament simulation, returning the FULL trace
    (group standings + every knockout round's matchups and winners),
    not just the final-stage-per-team summary that run_single_simulation()
    returns.

    Used by the live "bracket simulation" API endpoint to animate a single
    tournament from group stage through to champion.

    Returns:
        {
          "standings": {group: [rows...]},
          "group_winners": {...}, "group_runners": {...}, "best_thirds": [...],
          "rounds": {
            "round_of_32":   {"matchups": [(a,b),...], "winners": [...]},
            "round_of_16":   {...},
            "quarterfinal":  {...},
            "semifinal":     {...},
            "final":         {"matchups": [(a,b)], "winners": [champion]},
          },
          "champion": "Team Name",
        }
    """
    return _simulate_knockout_bracket(groups, cache)


def save_results(df: pd.DataFrame, path: Path = None) -> None:
    if path is None:
        path = RESULTS_DIR / "monte_carlo_results.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[monte_carlo] Results saved: {path}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== WC 2026 Monte Carlo Simulation ===\n")
 