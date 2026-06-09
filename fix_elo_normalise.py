import re

path = "src/models/match_outcome.py"
f = open(path, "r", encoding="utf-8").read()

# Find and replace the return dict in _build_training_row
old = '''        return {
            "elo_diff":           elo.get("elo_diff", 0.0),
            "elo_momentum_diff":  elo.get("momentum_diff", 0.0),
            "form_diff":          form_a.get("win_rate", 0.5) - form_b.get("win_rate", 0.5),
            "form_a":             form_a.get("win_rate", 0.5),
            "form_b":             form_b.get("win_rate", 0.5),
            "h2h_win_rate_a":     h2h.get("h2h_win_rate_a", 0.33),
            "h2h_advantage":      h2h.get("h2h_win_rate_a", 0.33) - h2h.get("h2h_win_rate_b", 0.33),
            "h2h_matches":        h2h.get("h2h_matches", 0),
            "tournament_exp_diff": exp_a.get("wc_experience_score", 0.0) - exp_b.get("wc_experience_score", 0.0),
            "wc_appearances_a":   exp_a.get("wc_matches_played", 0),
            "wc_appearances_b":   exp_b.get("wc_matches_played", 0),
        }'''

new = '''        # Normalise elo_diff to [-1, 1] range (max realistic diff ~400 points)
        # This puts Elo on the same scale as form features (0-1)
        # and prevents form from dominating just due to scale differences
        elo_diff_norm = elo.get("elo_diff", 0.0) / 400.0
        elo_mom_norm  = elo.get("momentum_diff", 0.0) / 100.0

        return {
            "elo_diff":           elo_diff_norm,
            "elo_momentum_diff":  elo_mom_norm,
            "form_diff":          form_a.get("win_rate", 0.5) - form_b.get("win_rate", 0.5),
            "form_a":             form_a.get("win_rate", 0.5),
            "form_b":             form_b.get("win_rate", 0.5),
            "h2h_win_rate_a":     h2h.get("h2h_win_rate_a", 0.33),
            "h2h_advantage":      h2h.get("h2h_win_rate_a", 0.33) - h2h.get("h2h_win_rate_b", 0.33),
            "h2h_matches":        h2h.get("h2h_matches", 0),
            "tournament_exp_diff": exp_a.get("wc_experience_score", 0.0) - exp_b.get("wc_experience_score", 0.0),
            "wc_appearances_a":   exp_a.get("wc_matches_played", 0),
            "wc_appearances_b":   exp_b.get("wc_matches_played", 0),
        }'''

if old in f:
    f = f.replace(old, new)
    # Also fix the inference feature builder to normalise elo_diff
    old2 = '    features["elo_diff"]       = round(float(elo.get("elo_diff", 0.0)),    2)'
    new2 = '    features["elo_diff"]       = round(float(elo.get("elo_diff", 0.0)) / 400.0, 4)'
    f = f.replace(old2, new2)
    old3 = '    features["elo_momentum_diff"] = round(\n        feat_a.get("elo_momentum", 0.0) - feat_b.get("elo_momentum", 0.0), 4\n    )'
    # Also fix in match_features.py
    open(path, "w", encoding="utf-8").write(f)
    print("match_outcome.py patched")
else:
    print("ERROR: anchor not found in match_outcome.py")

# Fix match_features.py too — normalise elo_diff at inference
path2 = "src/features/match_features.py"
f2 = open(path2, "r", encoding="utf-8").read()
old4 = '    features["elo_diff"]       = round(float(elo.get("elo_diff", 0.0)),    2)'
new4 = '    features["elo_diff"]       = round(float(elo.get("elo_diff", 0.0)) / 400.0, 4)'
old5 = '    features["elo_momentum_diff"] = round(float(elo.get("momentum_diff", 0.0)), 4)'
new5 = '    features["elo_momentum_diff"] = round(float(elo.get("momentum_diff", 0.0)) / 100.0, 4)'
if old4 in f2:
    f2 = f2.replace(old4, new4)
    print("match_features.py elo_diff normalised")
if old5 in f2:
    f2 = f2.replace(old5, new5)
    print("match_features.py momentum normalised")
open(path2, "w", encoding="utf-8").write(f2)
