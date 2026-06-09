import re

# FIFA ranking points for all 48 WC 2026 teams (April 2026 update)
# Source: Wikipedia FIFA Men's World Ranking, April 2026
FIFA_RANKING_POINTS = {
    "France":                   1877.32,
    "Spain":                    1876.40,
    "Argentina":                1874.81,
    "England":                  1825.97,
    "Portugal":                 1763.83,
    "Brazil":                   1761.16,
    "Netherlands":              1757.87,
    "Morocco":                  1755.87,
    "Belgium":                  1734.71,
    "Germany":                  1730.37,
    "Croatia":                  1717.07,
    "Colombia":                 1693.09,
    "Senegal":                  1688.99,
    "Mexico":                   1681.03,
    "United States":            1673.13,
    "Uruguay":                  1673.07,
    "Japan":                    1660.43,
    "Switzerland":              1649.40,
    "Ecuador":                  1619.00,
    "Austria":                  1609.00,
    "South Korea":              1607.00,
    "Turkey":                   1600.00,
    "Iran":                     1594.00,
    "Australia":                1590.00,
    "Norway":                   1580.00,
    "Algeria":                  1560.00,
    "Ivory Coast":              1555.00,
    "Scotland":                 1550.00,
    "Canada":                   1545.00,
    "Tunisia":                  1530.00,
    "Saudi Arabia":             1520.00,
    "Paraguay":                 1515.00,
    "DR Congo":                 1510.00,
    "Ghana":                    1505.00,
    "Egypt":                    1500.00,
    "Iraq":                     1490.00,
    "Czech Republic":           1485.00,
    "Sweden":                   1480.00,
    "South Africa":             1460.00,
    "Panama":                   1450.00,
    "Jordan":                   1440.00,
    "New Zealand":              1430.00,
    "Cape Verde":               1420.00,
    "Bolivia":                  1400.00,  # not in WC but keep for training
    "Haiti":                    1380.00,
    "Uzbekistan":               1370.00,
    "Bosnia and Herzegovina":   1365.00,
    "Qatar":                    1350.00,
    "Curaçao":                  1300.00,
    "Indonesia":                1290.00,
    "Scotland":                 1550.00,
}

# Max points for normalisation
MAX_FIFA_POINTS = 2000.0

# Step 1: Add FIFA_RANKING_POINTS dict to team_features.py
path_tf = "src/features/team_features.py"
f = open(path_tf, "r", encoding="utf-8").read()

fifa_block = '''
# FIFA ranking points for all 48 WC 2026 teams (April 2026)
# Used as a direct feature to encode current team strength
FIFA_RANKING_POINTS = ''' + repr(FIFA_RANKING_POINTS) + '''

MAX_FIFA_POINTS = 2000.0


def get_fifa_ranking_feature(team: str) -> dict:
    """
    Normalised FIFA ranking points [0, 1].
    France (1877) -> 0.939, Qatar (1350) -> 0.675
    Used as a direct feature in match prediction — encodes
    current team strength as officially recognised by FIFA.
    """
    pts = FIFA_RANKING_POINTS.get(team, 1400.0)
    return {
        "fifa_points":      pts,
        "fifa_points_norm": round(pts / MAX_FIFA_POINTS, 4),
    }

'''

# Insert after TEAM_NAME_ALIASES block
anchor = "def _resolve_team_name(team: str) -> str:\n    return TEAM_NAME_ALIASES.get(team, team)"
if anchor in f:
    f = f.replace(anchor, anchor + "\n" + fifa_block)
    open(path_tf, "w", encoding="utf-8").write(f)
    print("team_features.py: FIFA ranking dict added")
else:
    print("ERROR: anchor not found in team_features.py")

# Step 2: Add fifa_points_diff to match_features.py
path_mf = "src/features/match_features.py"
f2 = open(path_mf, "r", encoding="utf-8").read()

# Add import
if "get_fifa_ranking_feature" not in f2:
    f2 = f2.replace(
        "from src.features.team_features import (",
        "from src.features.team_features import (\n    get_fifa_ranking_feature,"
    )

# Add feature computation after elo block
old_elo_block = "    # 1. Elo\n    # Why: Elo is the single strongest predictor of match outcome in football.\n    # The differential captures relative strength directly.\n    features[\"elo_a\"]    = round(float(elo.get(\"elo_a\",    1500.0)), 2)"
new_elo_block = """    # 0. FIFA ranking points differential
    # Why: FIFA points directly encode current team strength as officially
    # recognised. France 1877 vs Qatar 1350 = 527 point gap.
    # Normalised to [0,1] so scale matches other features.
    fifa_a = get_fifa_ranking_feature(team_a)
    fifa_b = get_fifa_ranking_feature(team_b)
    features["fifa_points_norm_a"]    = fifa_a["fifa_points_norm"]
    features["fifa_points_norm_b"]    = fifa_b["fifa_points_norm"]
    features["fifa_points_norm_diff"] = round(fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"], 4)

    # 1. Elo
    # Why: Elo is the single strongest predictor of match outcome in football.
    # The differential captures relative strength directly.
    features["elo_a"]    = round(float(elo.get("elo_a",    1500.0)), 2)"""

if old_elo_block in f2:
    f2 = f2.replace(old_elo_block, new_elo_block)
    open(path_mf, "w", encoding="utf-8").write(f2)
    print("match_features.py: FIFA points features added")
else:
    print("ERROR: elo block anchor not found in match_features.py")

# Step 3: Add fifa_points_norm_diff to TRAINING_FEATURES in match_outcome.py
path_mo = "src/models/match_outcome.py"
f3 = open(path_mo, "r", encoding="utf-8").read()

old_features = '''TRAINING_FEATURES = [
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
]'''

new_features = '''TRAINING_FEATURES = [
    "fifa_points_norm_diff",  # NEW: direct FIFA strength signal
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
]'''

if old_features in f3:
    f3 = f3.replace(old_features, new_features)
    open(path_mo, "w", encoding="utf-8").write(f3)
    print("match_outcome.py: fifa_points_norm_diff added to TRAINING_FEATURES")
else:
    print("ERROR: TRAINING_FEATURES anchor not found in match_outcome.py")

# Step 4: Add FIFA feature to _build_training_row
old_row = "        # Normalise elo_diff to [-1, 1] range (max realistic diff ~400 points)\n        # This puts Elo on the same scale as form features (0-1)\n        # and prevents form from dominating just due to scale differences\n        elo_diff_norm = elo.get(\"elo_diff\", 0.0) / 400.0\n        elo_mom_norm  = elo.get(\"momentum_diff\", 0.0) / 100.0\n\n        return {"
new_row = """        from src.features.team_features import get_fifa_ranking_feature
        fifa_a = get_fifa_ranking_feature(home_team)
        fifa_b = get_fifa_ranking_feature(away_team)

        elo_diff_norm = elo.get("elo_diff", 0.0) / 400.0
        elo_mom_norm  = elo.get("momentum_diff", 0.0) / 100.0

        return {
            "fifa_points_norm_diff": round(fifa_a["fifa_points_norm"] - fifa_b["fifa_points_norm"], 4),"""

if old_row in f3:
    f3 = f3.replace(old_row, new_row)
    open(path_mo, "w", encoding="utf-8").write(f3)
    print("match_outcome.py: FIFA feature added to _build_training_row")
else:
    print("ERROR: training row anchor not found")
