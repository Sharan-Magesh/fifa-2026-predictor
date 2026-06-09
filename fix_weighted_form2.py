import re

path = "src/features/team_features.py"
f = open(path, "r", encoding="utf-8").read()

# Find the function start and end
start = f.find("def get_team_recent_form(team: str, before_date: pd.Timestamp = None) -> dict:")
if start == -1:
    print("ERROR: function not found")
    exit()

# Find the next function definition after this one
end = f.find("\ndef ", start + 10)
if end == -1:
    end = len(f)

old_func = f[start:end]

new_func = '''def get_team_recent_form(team: str, before_date: pd.Timestamp = None) -> dict:
    """
    Elo-weighted recent form over last FORM_WINDOW competitive matches.
    A win against Andorra (Elo 1100) counts less than a win against Germany (Elo 1923).
    Weight = opponent_elo / mean_elo, clipped to [0.5, 1.5].
    This fixes the CONMEBOL/UEFA confederation strength imbalance.
    """
    df     = load_results()
    df_elo = load_elo()
    mean_elo = float(df_elo["elo"].mean())

    df = df[df["tournament_type"] != "friendly"].copy()

    home = df[df["home_team"] == team].copy()
    away = df[df["away_team"] == team].copy()

    home["team_goals"] = home["home_score"]
    home["opp_goals"]  = home["away_score"]
    home["opponent"]   = home["away_team"]
    away["team_goals"] = away["away_score"]
    away["opp_goals"]  = away["home_score"]
    away["opponent"]   = away["home_team"]

    matches = pd.concat([
        home[["date", "team_goals", "opp_goals", "opponent"]],
        away[["date", "team_goals", "opp_goals", "opponent"]],
    ]).sort_values("date")

    if before_date is not None:
        matches = matches[matches["date"] < before_date]

    recent = matches.tail(FORM_WINDOW).copy()

    if recent.empty:
        return {"ppg": 1.0, "goals_scored_pg": 1.2, "goals_conceded_pg": 1.2,
                "form_matches": 0, "win_rate": 0.33, "clean_sheet_rate": 0.2}

    def opp_weight(opp_name):
        resolved = TEAM_NAME_ALIASES.get(opp_name, opp_name)
        row = df_elo[df_elo["team"] == resolved]
        opp_elo = float(row["elo"].iloc[0]) if not row.empty else mean_elo
        return max(0.5, min(1.5, opp_elo / mean_elo))

    recent["opp_weight"]  = recent["opponent"].apply(opp_weight)
    recent["points"]      = recent.apply(
        lambda r: 3 if r["team_goals"] > r["opp_goals"]
        else (1 if r["team_goals"] == r["opp_goals"] else 0), axis=1)
    recent["win"]         = (recent["team_goals"] > recent["opp_goals"]).astype(int)
    recent["clean_sheet"] = (recent["opp_goals"] == 0).astype(int)

    total_weight = recent["opp_weight"].sum()
    w_ppg        = (recent["points"] * recent["opp_weight"]).sum() / total_weight
    w_win_rate   = (recent["win"]    * recent["opp_weight"]).sum() / total_weight
    n = len(recent)

    return {
        "ppg":               round(w_ppg, 3),
        "goals_scored_pg":   round(recent["team_goals"].sum() / n, 3),
        "goals_conceded_pg": round(recent["opp_goals"].sum()  / n, 3),
        "form_matches":      n,
        "win_rate":          round(w_win_rate, 3),
        "clean_sheet_rate":  round(recent["clean_sheet"].sum() / n, 3),
    }
'''

f = f[:start] + new_func + f[end:]
open(path, "w", encoding="utf-8").write(f)
print("patched")
