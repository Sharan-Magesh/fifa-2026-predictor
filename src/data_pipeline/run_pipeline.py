# src/data_pipeline/run_pipeline.py

import time
import os
import sys

from src.data_pipeline import (
    fetch_international_results,
    fetch_elo,
    fetch_wc2026_fixtures,
    fetch_statsbomb,
    fetch_understat,
    fetch_transfermarkt,
)
from src.features import player_features

# Order matters:
# 1. fetch_international_results — martj42 CSV, must run before fetch_elo
# 2. fetch_elo — reads international_results.csv, calculates Elo
# 3. fetch_wc2026_fixtures — groups and schedule
# 4. fetch_statsbomb — player xG from tournaments
# 5. fetch_understat — club xG current season
# 6. fetch_transfermarkt — market values, caps, goals
PIPELINE = [
    ("fetch_international_results",  fetch_international_results.run),
    ("fetch_elo",                    fetch_elo.run),
    ("fetch_wc2026_fixtures",        fetch_wc2026_fixtures.run),
    ("fetch_statsbomb",              fetch_statsbomb.run),
    ("fetch_understat",              fetch_understat.run),
    ("fetch_transfermarkt",          fetch_transfermarkt.run),
    ("player_features",              player_features.run),
]

EXPECTED_OUTPUTS = [
    "data/processed/international_results.csv",
    "data/processed/elo_ratings.csv",
    "data/processed/wc2026_fixtures.csv",
    "data/processed/wc2026_groups.csv",
    "data/processed/statsbomb_player_stats.csv",
    "data/processed/statsbomb_player_stats_by_comp.csv",
    "data/processed/understat_players.csv",
    "data/processed/transfermarkt_players.csv",
    "data/processed/player_scores.csv",
]


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def check_outputs() -> None:
    print("\n── Output file summary ──────────────────────")
    for path in EXPECTED_OUTPUTS:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  ✓  {path:<55} {size_kb:>8.1f} KB")
        else:
            print(f"  ✗  {path:<55} MISSING")
    print()


def run():
    print("=" * 60)
    print("  FIFA 2026 Predictor — Data Pipeline")
    print("=" * 60)

    results = []
    pipeline_start = time.time()

    for name, fetcher_fn in PIPELINE:
        print(f"\n--- {name}")
        start = time.time()
        try:
            fetcher_fn()
            duration = time.time() - start
            results.append((name, "OK", duration, None))
            print(f"--- done in {format_duration(duration)}")
        except Exception as e:
            duration = time.time() - start
            results.append((name, "FAIL", duration, str(e)))
            print(f"--- FAILED in {format_duration(duration)}: {e}")

    total = time.time() - pipeline_start

    print("\n" + "=" * 60)
    print("  Pipeline summary")
    print("=" * 60)
    for name, status, duration, error in results:
        line = f"  {status}  {name:<40} {format_duration(duration):>6}"
        if error:
            line += f"  <- {error[:50]}"
        print(line)

    passed = sum(1 for _, s, _, _ in results if s == "OK")
    failed = sum(1 for _, s, _, _ in results if s == "FAIL")
    print(f"\n  {passed}/{len(PIPELINE)} fetchers succeeded — total time: {format_duration(total)}")

    check_outputs()

    if failed > 0:
        print(f"  {failed} fetcher(s) failed — check output above")
        sys.exit(1)


if __name__ == "__main__":
    run()