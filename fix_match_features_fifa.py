import re

path = "src/features/match_features.py"
f = open(path, "r", encoding="utf-8").read()

# Add import if missing
if "get_fifa_ranking_feature" not in f:
    f = f.replace(
        "from src.features.team_features import (",
        "from src.features.team_features import (\n    get_fifa_ranking_feature,"
    )
    print("import added")

# Find where features dict starts being built and inject FIFA block
old = '    features = {}\n\n    # 1. Elo'
new = '''    features = {}

    # 0. FIFA ranking points differential
    fifa_a = get_fifa_ranking_feature(team_a)
    fifa_b = get_fifa_ranking_feature(team_b)
    features["fifa_points_norm_a"]    = fifa_a["fifa_points_norm"]
    features["fifa_points_norm_b"]    = fifa_b["fifa_points_norm"]
    features["fifa_points_norm_diff"] = round(fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"], 4)

    # 1. Elo'''

if old in f:
    f = f.replace(old, new)
    open(path, "w", encoding="utf-8").write(f)
    print("match_features.py patched")
else:
    # Show what's around the features = {} line
    idx = f.find("features = {}")
    print("anchor not found, context:")
    print(repr(f[idx:idx+100]))
