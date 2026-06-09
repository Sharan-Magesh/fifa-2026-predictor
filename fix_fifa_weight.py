import re

# Step 1: Add more FIFA features to match_features.py
path_mf = "src/features/match_features.py"
f = open(path_mf, "r", encoding="utf-8").read()

old = '''    # 0. FIFA ranking points differential
    fifa_a = get_fifa_ranking_feature(team_a)
    fifa_b = get_fifa_ranking_feature(team_b)
    features["fifa_points_norm_a"]    = fifa_a["fifa_points_norm"]
    features["fifa_points_norm_b"]    = fifa_b["fifa_points_norm"]
    features["fifa_points_norm_diff"] = round(fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"], 4)'''

new = '''    # 0. FIFA ranking points
    # Three separate features to give the model multiple ways to learn
    # the non-linear relationship between FIFA points and win probability.
    # A 200-point gap (Argentina vs Mexico) should be very different from
    # a 20-point gap (France vs Spain).
    fifa_a = get_fifa_ranking_feature(team_a)
    fifa_b = get_fifa_ranking_feature(team_b)
    diff = fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"]
    features["fifa_points_norm_a"]    = fifa_a["fifa_points_norm"]
    features["fifa_points_norm_b"]    = fifa_b["fifa_points_norm"]
    features["fifa_points_norm_diff"] = round(diff, 4)
    # Squared diff amplifies large gaps — 0.1 diff -> 0.01, 0.5 diff -> 0.25
    features["fifa_points_norm_diff_sq"] = round(diff * abs(diff), 4)'''

if old in f:
    f = f.replace(old, new)
    open(path_mf, "w", encoding="utf-8").write(f)
    print("match_features.py patched")
else:
    print("ERROR: anchor not found in match_features.py")

# Step 2: Add fifa_points_norm_diff_sq to TRAINING_FEATURES
path_mo = "src/models/match_outcome.py"
f2 = open(path_mo, "r", encoding="utf-8").read()

old2 = '''TRAINING_FEATURES = [
    "fifa_points_norm_diff",  # NEW: direct FIFA strength signal
    "elo_diff",'''

new2 = '''TRAINING_FEATURES = [
    "fifa_points_norm_diff",     # direct FIFA strength signal
    "fifa_points_norm_diff_sq",  # amplifies large gaps non-linearly
    "fifa_points_norm_a",        # absolute team strength
    "fifa_points_norm_b",
    "elo_diff",'''

if old2 in f2:
    f2 = f2.replace(old2, new2)
    open(path_mo, "w", encoding="utf-8").write(f2)
    print("match_outcome.py TRAINING_FEATURES patched")
else:
    print("ERROR: TRAINING_FEATURES anchor not found")

# Step 3: Add these features to _build_training_row
old3 = '            "fifa_points_norm_diff": round(fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"], 4),'
new3 = '''            "fifa_points_norm_diff":     round(fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"], 4),
            "fifa_points_norm_diff_sq":  round((fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"]) * abs(fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"]), 4),
            "fifa_points_norm_a":        fifa_a["fifa_points_norm"],
            "fifa_points_norm_b":        fifa_b["fifa_points_norm"],'''

if old3 in f2:
    f2 = f2.replace(old3, new3)
    open(path_mo, "w", encoding="utf-8").write(f2)
    print("match_outcome.py training row patched")
else:
    print("ERROR: training row anchor not found")
    # Show context
    idx = f2.find("fifa_points_norm_diff")
    print(repr(f2[idx:idx+200]))
