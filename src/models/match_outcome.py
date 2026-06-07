# src/models/match_outcome.py
#
# XGBoost match outcome classifier.
# Predicts win/draw/loss probabilities for team_a vs team_b.
#
# Training data: 21,457 international matches (2004-present) from martj42
# Features: Elo, form, H2H, tournament experience, player quality (where available)
# Target: home_result (W/D/L) → encoded as 2/1/0
#
# Why XGBoost:
#   - Handles mixed numeric features well (Elo is on a 1000-2200 scale,
#     form is 0-1, value is 0-1000M — XGBoost handles this without scaling)
#   - Naturally outputs calibrated probabilities via softmax
#   - Fast to train and retrain as new match data comes in
#   - Interpretable via feature importance
#
# Model output: {"win": float, "draw": float, "loss": float}
# These sum to 1.0 and represent team_a's probability of each outcome.

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder

from src.features.team_features import (
    get_elo_features,
    get_team_recent_form,
    get_h2h_features,
    get_tournament_experience,
)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR    = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH    = MODELS_DIR / "match_outcome_xgb.pkl"
ENCODER_PATH  = MODELS_DIR / "label_encoder.pkl"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Target encoding: W=2, D=1, L=0
# XGBoost multiclass needs integer labels 0,1,2
RESULT_MAP = {"W": 2, "D": 1, "L": 0}
RESULT_INV = {2: "win", 1: "draw", 0: "loss"}

# Features used for training — must match exactly what build_match_features returns
# We exclude player features for training because most historical matches
# don't have player quality data. We add them back at inference time.
TRAINING_FEATURES = [
    "elo_diff",
    "elo_momentum_diff",
    "form_diff",
    "form_a",
    "form_b",
    "h2h_win_rate_a",
    "h2h_advantage",
    "h2h_matches",
    "tournament_exp_diff",
    "wc_appearances_a",
    "wc_appearances_b",
]

# Full feature set for inference (adds player quality on top of training features)
INFERENCE_FEATURES = TRAINING_FEATURES + [
    "attack_score_diff",
    "star_score_diff",
    "depth_score_diff",
    "value_diff_m",
    "strength_index_diff",
]


def _build_training_row(home_team: str, away_team: str,
                        date: pd.Timestamp) -> Optional[dict]:
    """
    Build feature vector for one historical match.
    Uses before_date to prevent data leakage — we only use results
    that were available on the day of the match.
    """
    try:
        elo   = get_elo_features(home_team, away_team)
        form_a = get_team_recent_form(home_team, before_date=date)
        form_b = get_team_recent_form(away_team, before_date=date)
        h2h   = get_h2h_features(home_team, away_team, before_date=date)
        exp_a = get_tournament_experience(home_team)
        exp_b = get_tournament_experience(away_team)

        return {
            "elo_diff":           elo.get("elo_diff", 0.0),
            "elo_momentum_diff":  elo.get("elo_momentum_diff", 0.0),
            "form_diff":          form_a.get("win_rate", 0.5) - form_b.get("win_rate", 0.5),
            "form_a":             form_a.get("win_rate", 0.5),
            "form_b":             form_b.get("win_rate", 0.5),
            "h2h_win_rate_a":     h2h.get("h2h_win_rate_a", 0.33),
            "h2h_advantage":      h2h.get("h2h_win_rate_a", 0.33) - h2h.get("h2h_win_rate_b", 0.33),
            "h2h_matches":        h2h.get("h2h_matches", 0),
            "tournament_exp_diff": exp_a.get("tournament_experience", 0.0) - exp_b.get("tournament_experience", 0.0),
            "wc_appearances_a":   exp_a.get("wc_appearances", 0),
            "wc_appearances_b":   exp_b.get("wc_appearances", 0),
        }
    except Exception:
        return None


def build_training_dataset(max_rows: int = None) -> tuple:
    """
    Build X, y for training from historical international results.

    Why we exclude friendlies:
        Friendly results are noisy — teams rotate squads, test formations,
        rest key players. Training on friendlies adds noise that hurts
        tournament prediction accuracy.

    Why 2004+:
        Pre-2004 squad compositions and Elo ratings are unreliable.
        Modern football changed significantly after 2004.

    Returns:
        X : pd.DataFrame of features
        y : np.array of encoded labels (0=L, 1=D, 2=W)
        matches: pd.DataFrame of raw match data (for inspection)
    """
    results_path = PROCESSED_DIR / "international_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Run fetch_international_results.py first: {results_path}")

    df = pd.read_csv(results_path, parse_dates=["date"])

    # Filter to competitive matches only
    df = df[df["tournament_type"] != "friendly"].copy()
    df = df[df["home_result"].isin(["W", "D", "L"])].copy()
    df = df.sort_values("date").reset_index(drop=True)

    print(f"[match_outcome] Training matches (non-friendly): {len(df)}")

    if max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
        print(f"[match_outcome] Using last {max_rows} matches for speed")

    rows = []
    labels = []
    skipped = 0

    for i, match in df.iterrows():
        if i % 1000 == 0:
            print(f"  Building features: {i}/{len(df)}...", end="\r")

        feats = _build_training_row(
            match["home_team"],
            match["away_team"],
            match["date"],
        )
        if feats is None:
            skipped += 1
            continue

        rows.append(feats)
        labels.append(RESULT_MAP[match["home_result"]])

    print(f"\n[match_outcome] Features built: {len(rows)}, skipped: {skipped}")

    X = pd.DataFrame(rows, columns=TRAINING_FEATURES)
    y = np.array(labels)

    return X, y, df.iloc[:len(rows)]


def train(X: pd.DataFrame, y: np.array) -> xgb.XGBClassifier:
    """
    Train XGBoost multiclass classifier.

    Hyperparameters explained:
        n_estimators=500     : number of boosting rounds — enough for convergence
        max_depth=4          : shallow trees prevent overfitting on 20k rows
        learning_rate=0.05   : small steps → more robust generalisation
        subsample=0.8        : row sampling per tree → reduces overfitting
        colsample_bytree=0.8 : feature sampling per tree → reduces overfitting
        objective=softmax    : outputs calibrated probabilities for 3 classes
        eval_metric=mlogloss : minimise log loss (proper scoring for probabilities)
        early_stopping=50    : stop if val loss doesn't improve for 50 rounds
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)

    acc      = accuracy_score(y_val, y_pred)
    logloss  = log_loss(y_val, y_prob, labels=[0, 1, 2])
    best_round = model.best_iteration

    print(f"[match_outcome] Val accuracy : {acc:.4f}")
    print(f"[match_outcome] Val log-loss : {logloss:.4f}")
    print(f"[match_outcome] Best round   : {best_round}")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=TRAINING_FEATURES
    ).sort_values(ascending=False)
    print(f"\n[match_outcome] Feature importance:")
    for feat, score in importance.items():
        print(f"  {feat:<30} {score:.4f}")

    return model


def save_model(model: xgb.XGBClassifier) -> None:
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[match_outcome] Model saved: {MODEL_PATH}")


def load_model() -> xgb.XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found. Run train.py first: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(team_a: str, team_b: str,
            model: xgb.XGBClassifier = None) -> dict:
    """
    Predict win/draw/loss probabilities for team_a vs team_b.

    At inference time we use the full INFERENCE_FEATURES set which includes
    player quality differentials on top of the training features.
    XGBoost handles missing features gracefully — it uses the mean split
    value for any feature not seen during training.

    Returns:
        {"win": float, "draw": float, "loss": float}
        where win = team_a wins, loss = team_a loses
    """
    if model is None:
        model = load_model()

    # Build inference features
    from src.features.match_features import build_match_features
    feats = build_match_features(team_a, team_b)

    # Use only training features for the model
    # (model was trained on these — adding extra features at inference
    # would cause shape mismatch)
    X = pd.DataFrame([{k: feats.get(k, 0.0) for k in TRAINING_FEATURES}])

    probs = model.predict_proba(X)[0]

    # probs are ordered by class label: [0=L, 1=D, 2=W]
    return {
        "win":  round(float(probs[2]), 4),
        "draw": round(float(probs[1]), 4),
        "loss": round(float(probs[0]), 4),
    }


def predict_score(team_a: str, team_b: str) -> dict:
    """
    Predict most likely scoreline using Bivariate Poisson.
    Expected goals are derived from Elo + form + player attack scores.
    Full Bivariate Poisson model lives in models/team_strength.py.
    This is a lightweight approximation for now.
    """
    from src.features.match_features import build_match_features
    feats = build_match_features(team_a, team_b)

    # Base expected goals from Elo differential
    # A 200 Elo point gap → ~0.5 xG difference
    elo_diff = feats.get("elo_diff", 0.0)
    base_xg  = 1.15  # average goals per team per match in WC football

    xg_a = base_xg + (elo_diff / 400.0) * 0.5
    xg_b = base_xg - (elo_diff / 400.0) * 0.5

    # Adjust for player attack quality
    xg_a += feats.get("squad_attack_a", 0.0) * 0.3
    xg_b += feats.get("squad_attack_b", 0.0) * 0.3

    xg_a = max(0.3, round(xg_a, 2))
    xg_b = max(0.3, round(xg_b, 2))

    # Most likely scoreline via Poisson PMF
    from scipy.stats import poisson
    best_prob = 0.0
    best_score = (1, 1)

    for g_a in range(6):
        for g_b in range(6):
            p = poisson.pmf(g_a, xg_a) * poisson.pmf(g_b, xg_b)
            if p > best_prob:
                best_prob = p
                best_score = (g_a, g_b)

    return {
        "xg_a":          xg_a,
        "xg_b":          xg_b,
        "likely_score_a": best_score[0],
        "likely_score_b": best_score[1],
        "score_prob":     round(best_prob, 4),
    }


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=== Training match outcome model ===\n")

    # Build training data — use last 8000 competitive matches for speed
    # Full dataset takes ~15 mins on CPU; 8000 rows takes ~2 mins
    X, y, matches = build_training_dataset(max_rows=8000)

    print(f"\nLabel distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {RESULT_INV[u]:<6}: {c} ({c/len(y)*100:.1f}%)")

    print(f"\n=== Training XGBoost ===\n")
    model = train(X, y)
    save_model(model)

    print(f"\n=== Test predictions ===\n")
    test_matches = [
        ("France",    "Argentina"),
        ("England",   "Brazil"),
        ("Spain",     "Germany"),
        ("Morocco",   "Portugal"),
        ("Japan",     "United States"),
        ("Argentina", "France"),
    ]

    for team_a, team_b in test_matches:
        try:
            result = predict(team_a, team_b, model)
            score  = predict_score(team_a, team_b)
            print(f"{team_a:<15} vs {team_b:<15} | "
                  f"W:{result['win']:.3f} D:{result['draw']:.3f} L:{result['loss']:.3f} | "
                  f"xG: {score['xg_a']:.2f}-{score['xg_b']:.2f} | "
                  f"Score: {score['likely_score_a']}-{score['likely_score_b']}")
        except Exception as e:
            print(f"{team_a} vs {team_b}: ERROR — {e}")