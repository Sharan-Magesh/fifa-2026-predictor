import re

path = "src/models/match_outcome.py"
f = open(path, "r", encoding="utf-8").read()

# Fix 1: Add tournament_types parameter to build_training_dataset
old = "def build_training_dataset(max_rows: int = None, min_year: int = 2004) -> tuple:"
new = "def build_training_dataset(max_rows: int = None, min_year: int = 2004, tournament_types: list = None) -> tuple:"
f = f.replace(old, new)

# Fix 2: Add filter after existing friendly filter
old2 = '    # Filter to competitive matches only\n    df = df[df["tournament_type"] != "friendly"].copy()\n    df = df[df["home_result"].isin(["W", "D", "L"])].copy()'
new2 = '''    # Filter to competitive matches only
    df = df[df["tournament_type"] != "friendly"].copy()
    df = df[df["home_result"].isin(["W", "D", "L"])].copy()
    # Optional: restrict to specific tournament types
    if tournament_types:
        df = df[df["tournament_type"].isin(tournament_types)].copy()'''
f = f.replace(old2, new2)

# Fix 3: Reduce max_depth and add regularisation for smaller datasets
old3 = '''    model = xgb.XGBClassifier(
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
    )'''
new3 = '''    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=3,          # shallower tree — less overfitting on small dataset
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,   # require 5 samples per leaf — reduces overfitting
        reg_alpha=0.1,        # L1 regularisation
        reg_lambda=1.5,       # L2 regularisation
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )'''
f = f.replace(old3, new3)

open(path, "w", encoding="utf-8").write(f)
print("match_outcome.py patched")

# Fix 4: Update train.py to use tournament filter
path2 = "src/models/train.py"
f2 = open(path2, "r", encoding="utf-8").read()
old4 = "    X, y, matches = build_training_dataset(max_rows=max_rows)"
new4 = """    # Train only on high-stakes competitive matches:
    # World Cup finals, continental finals (Euro/Copa/AFCON/Asian Cup),
    # and UEFA/CONMEBOL Nations League.
    # Excludes qualifiers and minor tournaments where weak nations
    # inflate form metrics for strong confederations.
    TOURNAMENT_TYPES = ["world_cup", "continental_final", "nations_league"]
    X, y, matches = build_training_dataset(
        max_rows=max_rows,
        tournament_types=TOURNAMENT_TYPES,
    )"""
f2 = f2.replace(old4, new4)
open(path2, "w", encoding="utf-8").write(f2)
print("train.py patched")
