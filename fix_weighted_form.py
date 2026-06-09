path = "src/features/team_features.py"
f = open(path, "r", encoding="utf-8").read()

old = '''def get_team_recent_form(team: str, before_date: pd.Timestamp = None) -> dict:
    """
    Feature 3: Recent form — points per game in last FORM_WINDOW matches
    Feature 4: Goals scored per game — last FORM_WINDOW matches
    Feature 5: Goals conceded per game — last FORM_WINDOW matches
    Why points per game, not win rate:
        Points per game (3W/1D/0L) is the standard football metric.
        It\'s more granular than win rate — a team that draws a lot
        looks identical to a team that loses a lot on win rate,
        but very different on PPG.
    before_date: if provided, only use matches before this date.
        This prevents data leakage when building the training set —
        we can\'t use future results to predict past matches.
    Interview answer: "We use a 10-match rolling window because
    international teams play infrequently — 10 matches covers roughly
    12-18 months which is the right recency window for WC prediction."
    """
    df = load_results()
    # Get all matches involving this team
    df = df[df["tournament_type"] != "friendly"].copy()
    home = df[df["home_team"] == team].copy()
    away = df[df["away_team"] == team].copy()
    # Standardise to team perspective — always from team\'s point of view
    home["team_goals"] = home["home_score"]
    home["opp_goals"] = home["away_score"]
    home["is_home"] = True
    away["team_goals"] = away["away_score"]
    away["opp_goals"] = away["home_score"]
    away["is_home"] = False
    matches = pd.concat([
        home[["date", "team_goals", "opp_goals", "is_home", "tournament_type"]],
        away[["date", "team_goals", "opp_goals", "is_home", "tournament_type"]],
    ]).sort_values("date")
    # Apply date filter for leakage prevention
    if before_date is not None:
        matches = matches[matches["date"] < before_date]
    # Take last FORM_WINDOW matches
    recent = matches.tail(FORM_WINDOW)
    if recent.empty:
        return {
            "ppg": 1.0,          # fallback: average
            "goals_scored_pg": 1.2,
            "goals_conceded_pg": 1.2,
            "form_matches": 0,
            "win_rate": 0.33,
            "clean_sheet_rate": 0.2,
        }
    # Points per game
    def points(row):
        if row["team_goals"] > row["opp_goals"]: return 3
        if row["team_goals"] == row["opp_goals"]: return 1
        return 0
    recent = recent.copy()
    recent["points"] = recent.apply(points, axis=1)
    recent["win"] = (recent["team_goals"] > recent["opp_goals"]).astype(int)
    recent["clean_sheet"] = (recent["opp_goals"] == 0).astype(int)
    n = len(recent)
    return {
        "ppg": round(recent["points"].sum() / n, 3),
        "goals_scored_pg": round(recent["team_goals"].sum() / n, 3),
        "goals_conceded_pg": round(recent["opp_goals"].sum() / n, 3),
        "form_matches": n,
        "win_rate": round(recent["win"].sum() / n, 3),
        "clean_sheet_rate": round(recent["clean_sheet"].sum() / n, 3),
    }'''

new = '''def get_team_recent_form(team: str, before_date: pd.Timestamp = None) -> dict:
    """
    Feature 3: Elo-weighted recent form over last FORM_WINDOW competitive matches.

    Why Elo-weighted:
        Raw win rate is misleading in international football because
        confederation strength varies massively. England beating Andorra
        and Brazil beating Argentina are both "wins" but carry completely
        different information. We weight each match result by the
        opponent\'s Elo rating normalised to [0.5, 1.5]:
            opponent_elo=2100 (world class) -> weight ~1.4
            opponent_elo=1500 (average)     -> weight ~1.0
            opponent_elo=1000 (weak)        -> weight ~0.6
        This means CONMEBOL teams (all tough opponents) get fair credit
        and UEFA teams beating minnows in qualification don\'t get inflated.

    before_date: prevents data leakage in training.
    """
    df = load_results()
    df_elo = load_elo()
    mean_elo = df_elo["elo"].mean()

    # Competitive matches only
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
        return {
            "ppg":               1.0,
            "goals_scored_pg":   1.2,
            "goals_conceded_pg": 1.2,
            "form_matches":      0,
            "win_rate":          0.33,
            "clean_sheet_rate":  0.2,
        }

    # Compute opponent Elo weight for each match
    def opp_weight(opp_name):
        resolved = TEAM_NAME_ALIASES.get(opp_name, opp_name)
        row = df_elo[df_elo["team"] == resolved]
        opp_elo = float(row["elo"].iloc[0]) if not row.empty else mean_elo
        # Normalise: weight = opp_elo / mean_elo, clipped to [0.5, 1.5]
        return max(0.5, min(1.5, opp_elo / mean_elo))

    recent["opp_weight"] = recent["opponent"].apply(opp_weight)

    def points(row):
        if row["team_goals"] > row["opp_goals"]:  return 3
        if row["team_goals"] == row["opp_goals"]: return 1
        return 0

    recent["points"]      = recent.apply(points, axis=1)
    recent["win"]         = (recent["team_goals"] > recent["opp_goals"]).astype(int)
    recent["clean_sheet"] = (recent["opp_goals"] == 0).astype(int)

    # Weighted metrics
    total_weight = recent["opp_weight"].sum()
    w_ppg      = (recent["points"] * recent["opp_weight"]).sum() / total_weight
    w_win_rate = (recent["win"]    * recent["opp_weight"]).sum() / total_weight
    n = len(recent)

    return {
        "ppg":               round(w_ppg, 3),
        "goals_scored_pg":   round(recent["team_goals"].sum() / n, 3),
        "goals_conceded_pg": round(recent["opp_goals"].sum()  / n, 3),
        "form_matches":      n,
        "win_rate":          round(w_win_rate, 3),
        "clean_sheet_rate":  round(recent["clean_sheet"].sum() / n, 3),
    }'''

if old in f:
    f = f.replace(old, new)
    open(path, "w", encoding="utf-8").write(f)
    print("patched")
else:
    print("ERROR: anchor not found")
