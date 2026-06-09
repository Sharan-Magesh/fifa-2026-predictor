# src/models/train.py
#
# Master training script — retrains all models in one command.
# Run this whenever new match data is available.
#
# Usage:
#   python -m src.models.train
#   python -m src.models.train --fast   (uses last 5000 matches for quick iteration)

import sys
import argparse
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train all FIFA 2026 Predictor models")
    parser.add_argument("--fast", action="store_true",
                        help="Use last 5000 matches only (faster, lower accuracy)")
    parser.add_argument("--rows", type=int, default=None,
                        help="Override number of training rows")
    args = parser.parse_args()

    max_rows = args.rows or (5000 if args.fast else None)

    print("=" * 60)
    print("FIFA 2026 Predictor — Model Training")
    print("=" * 60)

    # --- Match outcome model ---
    print("\n[1/1] Training match outcome model (XGBoost)...")
    t0 = time.time()

    from src.models.match_outcome import (
        build_training_dataset,
        train,
        save_model,
    )

    # Train only on high-stakes competitive matches:
    # World Cup finals, continental finals (Euro/Copa/AFCON/Asian Cup),
    # and UEFA/CONMEBOL Nations League.
    # Excludes qualifiers and minor tournaments where weak nations
    # inflate form metrics for strong confederations.
    TOURNAMENT_TYPES = ["world_cup", "continental_final", "nations_league"]
    X, y, matches = build_training_dataset(
        max_rows=max_rows,
        tournament_types=TOURNAMENT_TYPES,
    )
    model = train(X, y)
    save_model(model)

    print(f"  Done in {time.time() - t0:.1f}s")

    print("\n" + "=" * 60)
    print("All models trained and saved.")
    print("=" * 60)

    # Quick sanity check
    print("\nSanity check — France vs Argentina:")
    from src.models.match_outcome import predict
    result = predict("France", "Argentina", model)
    print(f"  win={result['win']:.3f}  draw={result['draw']:.3f}  loss={result['loss']:.3f}")
    assert abs(result["win"] + result["draw"] + result["loss"] - 1.0) < 0.01, \
        "Probabilities don't sum to 1!"
    print("  Probabilities sum to 1.0 ✓")


if __name__ == "__main__":
    main()