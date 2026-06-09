path = "src/features/team_features.py"
f = open(path, "r", encoding="utf-8").read()

old = '    home = df[df["home_team"] == team].copy()\n    away = df[df["away_team"] == team].copy()'
new = '    df = df[df["tournament_type"] != "friendly"].copy()\n    home = df[df["home_team"] == team].copy()\n    away = df[df["away_team"] == team].copy()'

if old in f:
    f = f.replace(old, new)
    open(path, "w", encoding="utf-8").write(f)
    print("patched")
else:
    print("ERROR: anchor not found")
