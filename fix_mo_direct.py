import re

path = "src/models/match_outcome.py"
f = open(path, "r", encoding="utf-8").read()

# Fix 1: TRAINING_FEATURES - find and replace
old_features = '    "fifa_points_norm_diff",  # NEW: direct FIFA strength signal\n    "elo_diff",'
new_features = '    "fifa_points_norm_diff",\n    "fifa_points_norm_diff_sq",\n    "fifa_points_norm_a",\n    "fifa_points_norm_b",\n    "elo_diff",'

if old_features in f:
    f = f.replace(old_features, new_features)
    print("TRAINING_FEATURES fixed")
else:
    # Try alternate
    idx = f.find('"fifa_points_norm_diff"')
    print("current fifa line:", repr(f[idx:idx+60]))

# Fix 2: _build_training_row — find the fifa line and add new ones after it
old_row = '            "fifa_points_norm_diff": round(fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"], 4),'
new_row = '''            "fifa_points_norm_diff":    round(fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"], 4),
            "fifa_points_norm_diff_sq": round((fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"]) * abs(fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"]), 4),
            "fifa_points_norm_a":       fifa_a["fifa_points_norm"],
            "fifa_points_norm_b":       fifa_b["fifa_points_norm"],'''

if old_row in f:
    f = f.replace(old_row, new_row)
    print("training row fixed")
else:
    idx = f.find('"fifa_points_norm_diff"')
    # Find the one in the return dict (not TRAINING_FEATURES)
    idx2 = f.find('"fifa_points_norm_diff"', idx+1)
    if idx2 > 0:
        end = f.find('\n', idx2)
        print("training row line:", repr(f[idx2:end]))
    else:
        print("ERROR: training row fifa line not found")

open(path, "w", encoding="utf-8").write(f)
print("saved")
